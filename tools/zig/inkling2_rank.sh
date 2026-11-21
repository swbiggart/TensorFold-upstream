#!/usr/bin/env bash
# inkling2_rank.sh RANK LOG SECONDS CMD...: one host's rank of a multi-host run (tf-inkling2-cuda --world W --rank R)
# over NCCL on RoCE, with the settings the four-GB10 switchless ring needs (Ring algorithm, cuMem and runtime connect
# on, a per-rank graph file INK2_HOME/nccl/graph-rankR.xml). NCCL bootstraps on the LAN at MASTER:NCCL_PORT (default
# 29521). CMD runs under `timeout SECONDS` with its output in LOG; MemAvailable is sampled every 0.5 s into LOG.mem and
# its value before, minimum and after are printed at the end. The exit status is CMD's.
# The NCCL settings follow MiaAI-Lab's four-Spark DeepSeek recipe (scripts/nodes.sh) and upstream's
# docs/recipes/glm-5.3-flash-cuda.md (see THIRD_PARTY_NOTICES.md).
# Needs INK2_HOME (bin; lib with libnccl.so.2 and libcublasLt.so.13; nccl), MASTER, NCCL_IB_HCA, NCCL_SOCKET_IFNAME.
set -u
r=$1; log=$2; secs=$3; shift 3
d=${INK2_HOME:?INK2_HOME=the rank folder root}
export LD_LIBRARY_PATH=$d/lib
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NCCL_COMM_ID=${MASTER:?MASTER=rank 0 LAN address}:${NCCL_PORT:-29521}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:?NCCL_SOCKET_IFNAME=the LAN netdev}
export NCCL_IB_HCA=${NCCL_IB_HCA:?NCCL_IB_HCA=the RoCE HCAs}
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} NCCL_IB_MERGE_NICS=0 NCCL_CROSS_NIC=1 NCCL_IB_SUBNET_AWARE_ROUTING=1 NCCL_IB_SUBNET_PREFIX_LEN=24
export NCCL_ALGO=Ring NCCL_CUMEM_ENABLE=1 NCCL_RUNTIME_CONNECT=1 NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1
export NCCL_MIN_NCHANNELS=4 NCCL_MAX_NCHANNELS=4
[ -f "$d/nccl/graph-rank$r.xml" ] && export NCCL_GRAPH_FILE=$d/nccl/graph-rank$r.xml
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
mkdir -p "$(dirname "$log")"
grep MemAvailable /proc/meminfo > "$log.mem"
( while :; do echo "$(date +%s.%N | cut -c1-14) $(awk '/MemAvailable/{print $2}' /proc/meminfo)"; sleep 0.5; done ) >> "$log.mem" &
mon=$!
start=$(date +%s.%N)
timeout "$secs" "$@" > "$log" 2>&1
rc=$?
kill $mon 2>/dev/null; wait $mon 2>/dev/null
end=$(date +%s.%N)
min=$(awk 'NR>1 && ($2 < m || m == "") {m = $2} END {print m}' "$log.mem")
echo "rank $r exit $rc in $(echo "$end - $start" | bc) s; MemAvailable before $(head -1 "$log.mem" | awk '{print $2}') kB, min $min kB, after $(awk '/MemAvailable/{print $2}' /proc/meminfo) kB"
exit $rc
