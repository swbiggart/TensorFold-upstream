//! tf-inkling2-decode: the serving engine's decode path against whole-prompt forwards. On slot 0 a prompt of T
//! tokens runs in 2048-row chunks, then S greedy decode steps of one row each (the per-stream state carried across
//! forwards: global keys and values, the sliding ring, the four convolution states a layer). Then on slot 1 the
//! whole sequence (prompt plus the S drawn tokens) runs as one prompt (chunks of --chunk rows), and each decode
//! step's logits are compared with the same row of that forward: top-1, KL, largest difference, bit equality.
//! At W > 1 (--world W --rank R, NCCL_COMM_ID on every rank) rank 0 checks and the other ranks follow its steps.
//!
//! One load, many experiments (--exp): the check, the check with eager decode steps, the DSpark gates, the round
//! prices, the per-op budget and graph-against-eager timing all run on the weights loaded once, each printing its
//! own result line. The budget (on by default) times graph and eager decode steps alternated in this process, then
//! profiles eager steps op by op (CUDA events between the forward's ops, on every rank: cuda_forward.zig Prof) and
//! prints each op's ms a step, the exchanges' total, and the step times. Profiling and the slots past the check's
//! (3 for the budget and the A/B) touch no slot the check reads: the check's bits are the same with or without them.
//! --depths D1,D2,..: the check at several depths in one pass. Slot 0 takes the prompt to D1, decodes S steps, then
//! the prompt from where it stands (the last drawn token first) to D2, decodes S steps, and so on; the whole sequence
//! (the prompt with each depth's drawn tokens in place) runs once on slot 1 and every depth's rows are compared. Each
//! depth prints its prefill time (cumulative: positions 0 .. D, the decode steps left out) and its decode steps' ms
//! (median); the DSpark gates decode slot 2 the same way, each depth's verify rounds ending exactly at its S tokens.
//! --exp deep (deep-draft: the drafter loaded): the check at one long depth with ONE prefill (a 512k-position slot
//! holds no second whole sequence, and its prefill takes most of an hour): slot 0 takes the prompt but its last token
//! (P - 1 positions), its state is marked on every rank (cuda_engine.zig `mark`: the convolution states and
//! position), then S greedy decode steps from position P - 1 give the decode rows; the slot is rewound to the mark and
//! the same S tokens run as one prompt forward of S rows whose rows are compared with them, bit for bit. The rows are
//! those of a whole-sequence forward cut at P - 1 (chunk invariance), the decode hash the check's at --prompt P.
//! deep-draft then rewinds again and decodes the same S tokens in drafted rounds (real drafts, then mixed): each must
//! draw exactly the plain tokens; its rounds' times give DSpark at depth.
//! --fill synth: no prefill; slot 0's state for positions 0 .. P - 2 is written with synthetic values on every rank
//! (cuda_synth.zig: every layer's keys and values, the sliding rings, the convolution states, the draft caches), so
//! the same check runs at a 1M-position depth in minutes; after it every rank reads its global layers' positions
//! 0 .. P - 2 back against the fill (nothing wrote below the depth).
//! --price-depths D1,D2,.. (deep-draft): the round prices at depth with the same one prefill: as slot 0's prefill
//! passes each D (below the deep depth) its state is marked, the price table runs there (`price`: windows of 1..8 rows
//! and drafts, --depth-reps each), and the slot is rewound to the mark and prefilled on (the price's rows are rewritten,
//! so the deep check's bits do not change); after deep-draft the table runs once more at the deep depth itself.
//! Ranks over processes (W > 1) keep the server's lifecycle (cuda_native.zig open): liveness lines (cuda/tp_watch.zig)
//! before the NCCL group, so a rank that fails anywhere (opening the group, the transport vote, loading, a step) stops
//! every rank at once with its reason; an error before the load aborts this rank's communicator, one after it ends the
//! process at once (tp_watch.fatal); the followers sleep until rank 0's first header; rank 0 ends them only after a
//! run without error, and every rank passes a last barrier before it exits. The lines no longer stop a rank once the
//! end is under way (a closing line is then expected), so the last barrier is bounded instead: a rank still in it after
//! `last_barrier_s` seconds (TF_INK2_LAST_BARRIER_S) ends its process with status 1. TF_INK2_DECODE_FAULT (tests):
//! `open:R`, `vote:R` or `load:R` fails rank R before its NCCL group opens, before the transport vote, or after its
//! load; `mid:R:MS` ends rank R as a crash would, MS ms after its load (its reason on its lines, then exit); `end:R`
//! ends rank R as a crash would after the end header, before the last barrier. Rank 0's steps also take
//! TF_INK2_SERVE_FAULT (cuda_follow.zig).
//! Exit status: 0 every gate run passed (every row's top-1 equal and KL under --kl-max; drafted equals plain; the
//! eager check's hash equal to the graph one's), 1 otherwise or failed, 2 usage.
const std = @import("std");
const cuda = @import("cuda");
const ink = @import("inkling2");
const ic = @import("inkling2_cuda");

const engine = ic.engine;

const usage =
    \\usage: tf-inkling2-decode <model dir> [options]
    \\  --layers N      layers [0, N) (default: TF_INK2_LAYERS, else all 48)
    \\  --ids FILE      prompt ids from an oracle prompt file (its `ids`, repeated to the length); default: fixed
    \\                  pseudo-random ids
    \\  --prompt T      prompt tokens (default 1000)
    \\  --steps S       decode steps (default 64)
    \\  --chunk C       the whole-sequence forward's chunk rows (default 2048)
    \\  --kl-max K      the largest KL a row may have (default 1e-3)
    \\  --world W       ranks (default 1); with --rank R, one process a GPU over NCCL (NCCL_COMM_ID on every rank)
    \\  --rank R
    \\  --exp LIST      experiments on the one load, in order, comma-separated (default check,budget; every rank
    \\                  must be given the same list):
    \\                    check        the decode check above (decode steps as CUDA graphs unless TF_INK2_GRAPH=0)
    \\                    check-eager  the check again with eager decode steps (slots 0 and 1 start over); its hash
    \\                                 must equal the first check's
    \\                    draft-real   the DSpark gate (loads the drafter; after a check): slot 2 decodes the same S
    \\                                 tokens in verify rounds (anchor + --drafts rows, greedy acceptance, rollback of
    \\                                 the rest) with the drafter's proposals and must draw exactly the plain tokens
    \\                    draft-mixed  the same with the plain tokens as drafts, one wrong at a pseudo-random depth
    \\                                 (every acceptance length and rollback depth)
    \\                    price        the round-price table (loads the drafter; after a check): --price N verify
    \\                                 windows of each width 1..8 (default 20), then N drafts
    \\                    budget       on slot 3 (the prompt, then decode steps): --budget-steps N graph and N eager
    \\                                 steps alternated 3 times, then N eager steps profiled op by op on every rank
    \\                    ab           graph against eager decode steps on slot 3, alternated --ab-reps times,
    \\                                 --ab-steps each (defaults 10 and 8)
    \\                    fuse         after a check: --ab N more greedy steps (default 200) on slot 0 (the ops
    \\                                 TF_INK2_FUSE picks, by default the one-row latency forms) and slot 1 (always the
    \\                                 release's ops), alternating which goes first; each step's logits bit-equal, ms a
    \\                                 step each and the paired difference (TF_INK2_FUSE=0: an A/A control of two slots)
    \\                    resume       the prompt cache (after a check): for each --cuts c, slot 4 prefills [0, c)
    \\                                 (chunks of 2048 cut at c), saves its state (cuda_snap.zig), runs on past c;
    \\                                 slot 5 (every byte poisoned first) and slot 4 itself restore it, prefill
    \\                                 [c, T) in 2048-row chunks and decode S - 1 steps: each run's decode logits
    \\                                 xxh64 must equal the check's; save and restore ms are printed
    \\                    resume-draft the DSpark gate after a resume (loads the drafter; after a check): slot 6
    \\                                 decodes the plain tokens in verify rounds from a cold prefill, then for each
    \\                                 --draft-cuts c restores slot 4's state at c, prefills [c, T) and decodes the
    \\                                 same rounds, drafts made apart (draft-real) and in the verify's step
    \\                                 (draft-defer): drafted equals plain, and the rounds and proposals equal the
    \\                                 cold run's
    \\                    resume-inside the prompt cache's captures inside a chunk (stage 1b; after a check): slot 4
    \\                                 prefills [0, T) in 2048-row chunks cut as the server's lanes cut them (the
    \\                                 --cuts inside a chunk captured by its forward, at most 8, a cut on a chunk's
    \\                                 end saved after it); each state is then restored into slot 5 (poisoned
    \\                                 first), prefills [c, T) and decodes S - 1 steps: each run's decode logits
    \\                                 xxh64 must equal the check's. resume-draft also resumes from captures
    \\                    resume-streams batched decode after resumes (after a check): --streams S slots (default
    \\                                 4) from 8 each restore a state saved at another cut (T - 1, 2049, 2048, 1000,
    \\                                 ...; poisoned first), prefill to T and decode S - 1 steps stepped together in
    \\                                 batched forwards (graphs from a key's second sighting, then again eager):
    \\                                 every slot's decode logits xxh64 must equal the check's
    \\  --draft MODE    real|mixed: adds draft-MODE to the list (as before --exp)
    \\  --drafts D      drafts a round (default 7)
    \\  --price N       adds price (N windows a width) to the list
    \\  --ab N          the fuse experiment's steps; without --exp, adds fuse after the check (as before --exp)
    \\  --budget-steps N  (default 8)   --ab-reps N (default 10)   --ab-steps N (default 8)
    \\  --cuts LIST    resume's cuts (default 1,2,3,8,9,1000,1023,1024,2047,2048,2049 and T - 1; those below T)
    \\  --draft-cuts LIST  resume-draft's cuts (default 1000 and T - 1)
    \\  --resume-back 0|1  resume also restores into the source slot (default 1)
    \\  --src-rows N   resume-inside's source prefill in chunks of up to N rows (default 2048; 512 = the server's
    \\                 chunks beside decoders with --decode-share)
    \\  --poison B     the byte resume poisons a slot with before a restore (default 60: bf16 0.0115, finite;
    \\                 255: NaN, for information), or none
    \\  --solo R        one rank alone (kernel work on one GPU): rank R of --world W's plan (its weights, checked
    \\                  against the W manifest), no group: its exchanges move nothing, so its numbers are not the
    \\                  model's; budget and ab only (default budget). Rank 0 embeds and reads its logits; another
    \\                  rank decodes a fixed token, as a follower would (it gets rank 0's embedded row)
    \\  --streams S     adds streams (S slots, 2..8; default 4 when listed): the batched-decode gate: S slots at
    \\                  different positions decode alone, then stepped together in batched forwards (one row each,
    \\                  then windows of 1-3 rows; with the drafter, drafted windows too); every slot's logits rows must
    \\                  equal its solo run's bit for bit (cuda_batch_check.zig)
    \\  --op-check 1    adds op-check: the state kernels' op check (window history commit, slot rollback; with the
    \\                  drafter its context writes at a device position) on rank 0
    \\  --win-ab N      adds win-ab: on slot 3, verify windows of 1, 2, 4 and 8 rows (the head over every row; with
    \\                  the drafter a keep of 1 after each, as a DSpark round) as captured graphs and eagerly,
    \\                  alternated N times a width (default 10): each width's medians and paired difference
    \\  --shared-price N  adds shared-price (the drafter loaded): forwards of 1 .. 32 rows in all as the fewest streams
    \\                  of at most 8 rows (slots 3 .. 6), N timed a size (default 10): the lane core's shared prices
    \\  --argmax N      adds argmax (B2 C4, rank 0): the device argmax's op check (10,000 rows: ties, NaNs, -inf)
    \\                  against the host's greedy loop, then on slot 0 (after the check) windows of 1 and 8 rows
    \\                  with the logits downloaded and drawn on the host against the argmax rows alone, alternated N
    \\                  times a width (default 20)
    \\  --exp deep      the check at --prompt P with one prefill (slot 0 only; see the top): decode rows against one
    \\                  prompt forward of the same --steps S rows over the same positions
    \\  --exp deep-draft  deep with the drafter loaded, then drafted equals plain from the same point (real, mixed)
    \\  --fill F        deep's state before the depth: real (default: the prompt's prefill) or synth (synthetic
    \\                  values written on every rank, no forward; then read back: see the top)
    \\  --price-depths D,..  (deep-draft, not with --fill synth) the round prices (windows of 1..8 rows, drafts) at each depth D in its
    \\                  prefill (marked, priced, rewound) and at the deep depth after it; --depth-reps N a width
    \\                  (default 10)
    \\  --exp scratch   (last) the op scratch each kind of forward takes against cuda_scratch.zig's need for it
    \\                  (cuda_mem_check.zig): a prompt chunk, 1-64-row forwards and captured steps on a fresh and on
    \\                  the deepest slot, a batched forward; rank 0's peaks by kind
    \\  --exp chunks   (after a check) chunk invariance: the prompt prefilled on slot 2 with each of a set of cut lists
    \\                  (2048, 512, 256, 128, 64, 37, odd lists, seeded random lists; above 4,096 tokens 256 and a
    \\                  random list with forced cuts), each then the check's decode steps: every list's decode logits
    \\                  must hash to the check's (cuda_batch_check.zig chunks)
    \\  --exp mixprice  the mixed step's price: prompt chunks of 64 .. 1,024 rows on slot 2 at each --mix-depths band
    \\                  beside --mix-dec 1-row decode streams (slots 4 ..), alone then the decoders, against one forward of
    \\                  both (the fold), --mix-reps pairs a cell (cuda_batch_check.zig mixPrice)
    \\  --mix-reps N (default 3)   --mix-depths D,.. (default 0,8192,32768)   --mix-dec R,.. (default 1,4)
    \\  --context N     every slot's positions at least N (default: what the experiments need): a captured step's
    \\                  attention grid covers the slot (the server's is --context, 32768 in the recipe)
    \\  --depths D,..   the check (and the DSpark gates) at each depth in one pass (increasing, more than S apart;
    \\                  the prompt is the last, --prompt ignored): see the top of cuda_decode_main.zig
    \\  --digests F     the content-digest manifest the loaded bytes must match, or none (default TF_INK2_DIGESTS,
    \\                  else <model dir>/inkling2-digests-w<W>.tsv, needed at W > 1); TF_INK2_MODE picks the experts' mode
    \\The GPU is the first visible device (set CUDA_VISIBLE_DEVICES).
    \\
;

pub const Exp = enum { check, @"check-eager", @"draft-real", @"draft-mixed", price, budget, ab, fuse, streams, @"op-check", @"win-ab", @"shared-price", @"draft-defer", chunks, mixprice, @"resume", @"resume-draft", @"resume-streams", @"resume-inside", scratch, argmax, deep, @"deep-draft" };

const max_exps = 16;
const max_cuts = 32;
const max_depths = 16;

const Opts = struct {
    dir: []const u8,
    layers: u32 = 0,
    ids: ?[]const u8 = null,
    prompt: usize = 1000,
    steps: usize = 64,
    chunk: usize = engine.prompt_rows,
    kl_max: f64 = 1e-3,
    world: u32 = 1,
    rank: u32 = 0,
    drafts: usize = 7,
    price: usize = 0,
    fuse_steps: usize = 0,
    budget_steps: usize = 8,
    ab_reps: usize = 10,
    ab_steps: usize = 8,
    digests: ?[]const u8 = null,
    streams: usize = 0,
    op_check: bool = false,
    win_ab: usize = 0,
    shared_price: usize = 0,
    argmax: usize = 0,
    context: usize = 0,
    /// deep's state before the depth: the prompt's prefill, or synthetic values (--fill synth)
    synth: bool = false,
    solo: ?u32 = null,
    exps: [max_exps]Exp = undefined,
    nexps: usize = 0,
    /// the check's depths (--depths), else the prompt alone
    depths: [max_depths]usize = undefined,
    ndepths: usize = 0,
    /// mixprice: pairs a cell, its depth bands and decode streams
    mix_reps: usize = 3,
    mix_depths: [max_depths]usize = undefined,
    mix_ndepths: usize = 0,
    mix_decs: [max_depths]usize = undefined,
    mix_ndecs: usize = 0,
    cuts: ?[]const u8 = null,
    draft_cuts: ?[]const u8 = null,
    resume_back: bool = true,
    poison: ?u8 = 60,
    /// resume-inside: the source prefill's chunk cap (the server's --decode-share caps a chunk beside decoders at 512)
    src_rows: usize = engine.prompt_rows,
    /// deep-draft: the depths its prefill prices the rounds at (--price-depths), and windows a width there
    price_depths: [max_depths]usize = undefined,
    nprice_depths: usize = 0,
    depth_reps: usize = 10,

    fn list(o: *const Opts) []const Exp {
        return o.exps[0..o.nexps];
    }

    fn depthList(o: *const Opts) []const usize {
        return o.depths[0..o.ndepths];
    }

    fn has(o: *const Opts, e: Exp) bool {
        return std.mem.indexOfScalar(Exp, o.list(), e) != null;
    }

    fn add(o: *Opts, e: Exp) !void {
        if (o.nexps == max_exps) return error.TooManyExperiments;
        o.exps[o.nexps] = e;
        o.nexps += 1;
    }

    /// The drafter is loaded (every rank): a DSpark gate, the price table or the argmax tail (its windows keep 1) is
    /// asked for.
    fn spec(o: *const Opts) bool {
        return o.has(.@"draft-real") or o.has(.@"draft-mixed") or o.has(.price) or o.has(.@"shared-price") or o.has(.@"draft-defer") or o.has(.@"resume-draft") or o.has(.argmax) or o.has(.@"deep-draft");
    }

    /// A cut list (`text`, else `default`), those in (0, T), ascending and once each.
    fn cutList(o: *const Opts, text: ?[]const u8, default: []const usize, out: *[max_cuts]usize) ![]usize {
        var n: usize = 0;
        var buf: [max_cuts]usize = undefined;
        if (text) |t| {
            var it = std.mem.splitScalar(u8, t, ',');
            while (it.next()) |w| {
                if (n == max_cuts) return error.TooManyCuts;
                buf[n] = try std.fmt.parseInt(usize, w, 10);
                n += 1;
            }
        } else {
            for (default) |c| {
                buf[n] = c;
                n += 1;
            }
        }
        std.mem.sort(usize, buf[0..n], {}, std.sort.asc(usize));
        var m: usize = 0;
        for (buf[0..n]) |c| if (c > 0 and c < o.prompt and (m == 0 or out[m - 1] != c)) {
            out[m] = c;
            m += 1;
        };
        return out[0..m];
    }

    /// Decode steps slot 3 takes past its prompt (the budget's, then the A/B's, then the windows' A/B).
    fn benchSteps(o: *const Opts) usize {
        var n: usize = 0;
        if (o.has(.budget)) n += 7 * o.budget_steps;
        if (o.has(.ab)) n += 2 * o.ab_reps * o.ab_steps;
        if (o.has(.@"win-ab")) n += (2 * o.win_ab + 2) * win_rows_sum;
        return n;
    }
};

fn parse(args: []const [:0]const u8) !?Opts {
    if (args.len < 2) return null;
    var o: Opts = .{ .dir = args[1] };
    var exp: ?[]const u8 = null;
    var draft: ?Exp = null;
    var i: usize = 2;
    while (i + 1 < args.len) : (i += 2) {
        const k = args[i];
        const v = args[i + 1];
        if (std.mem.eql(u8, k, "--fill")) {
            o.synth = if (std.mem.eql(u8, v, "synth")) true else if (std.mem.eql(u8, v, "real")) false else return null;
            continue;
        }
        if (std.mem.eql(u8, k, "--context")) {
            o.context = try std.fmt.parseInt(usize, v, 10);
            continue;
        }
        if (std.mem.eql(u8, k, "--mix-depths") or std.mem.eql(u8, k, "--mix-dec")) {
            const dec = std.mem.eql(u8, k, "--mix-dec");
            const list = if (dec) &o.mix_decs else &o.mix_depths;
            const n = if (dec) &o.mix_ndecs else &o.mix_ndepths;
            n.* = 0;
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| {
                if (n.* == max_depths) return null;
                list[n.*] = try std.fmt.parseInt(usize, x, 10);
                n.* += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, k, "--mix-reps")) {
            o.mix_reps = try std.fmt.parseInt(usize, v, 10);
            continue;
        }
        if (std.mem.eql(u8, k, "--price-depths")) {
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| {
                if (o.nprice_depths == max_depths) return null;
                o.price_depths[o.nprice_depths] = try std.fmt.parseInt(usize, x, 10);
                o.nprice_depths += 1;
            }
            continue;
        }
        if (std.mem.eql(u8, k, "--depth-reps")) {
            o.depth_reps = try std.fmt.parseInt(usize, v, 10);
            continue;
        }
        if (std.mem.eql(u8, k, "--depths")) {
            var it = std.mem.splitScalar(u8, v, ',');
            while (it.next()) |x| {
                if (o.ndepths == max_depths) return null;
                o.depths[o.ndepths] = try std.fmt.parseInt(usize, x, 10);
                o.ndepths += 1;
            }
            continue;
        }
        const eql = std.mem.eql;
        const int = std.fmt.parseInt;
        if (eql(u8, k, "--layers")) o.layers = try int(u32, v, 10) else if (eql(u8, k, "--ids")) o.ids = v else if (eql(u8, k, "--prompt")) o.prompt = try int(usize, v, 10) else if (eql(u8, k, "--steps")) o.steps = try int(usize, v, 10) else if (eql(u8, k, "--chunk")) o.chunk = try int(usize, v, 10) else if (eql(u8, k, "--kl-max")) o.kl_max = try std.fmt.parseFloat(f64, v) else if (eql(u8, k, "--world")) o.world = try int(u32, v, 10) else if (eql(u8, k, "--rank")) o.rank = try int(u32, v, 10) else if (eql(u8, k, "--digests")) o.digests = v else if (eql(u8, k, "--draft")) {
            draft = if (eql(u8, v, "real")) .@"draft-real" else if (eql(u8, v, "mixed")) .@"draft-mixed" else return null;
        } else if (eql(u8, k, "--drafts")) o.drafts = try int(usize, v, 10) else if (eql(u8, k, "--price")) o.price = try int(usize, v, 10) else if (eql(u8, k, "--exp")) exp = v else if (eql(u8, k, "--budget-steps")) o.budget_steps = try int(usize, v, 10) else if (eql(u8, k, "--ab-reps")) o.ab_reps = try int(usize, v, 10) else if (eql(u8, k, "--ab-steps")) o.ab_steps = try int(usize, v, 10) else if (eql(u8, k, "--ab")) o.fuse_steps = try int(usize, v, 10) else if (eql(u8, k, "--solo")) o.solo = try int(u32, v, 10) else if (eql(u8, k, "--streams")) o.streams = try int(usize, v, 10) else if (eql(u8, k, "--op-check")) o.op_check = eql(u8, v, "1") else if (eql(u8, k, "--win-ab")) o.win_ab = try int(usize, v, 10) else if (eql(u8, k, "--shared-price")) o.shared_price = try int(usize, v, 10) else if (eql(u8, k, "--argmax")) o.argmax = try int(usize, v, 10) else if (eql(u8, k, "--src-rows")) o.src_rows = try int(usize, v, 10) else if (eql(u8, k, "--cuts")) o.cuts = v else if (eql(u8, k, "--draft-cuts")) o.draft_cuts = v else if (eql(u8, k, "--resume-back")) {
            o.resume_back = !eql(u8, v, "0");
        } else if (eql(u8, k, "--poison")) {
            o.poison = if (eql(u8, v, "none")) null else try int(u8, v, 10);
        } else return null;
    }
    if (i != args.len) return null;
    if (exp) |list| {
        var it = std.mem.splitScalar(u8, list, ',');
        while (it.next()) |name| try o.add(std.meta.stringToEnum(Exp, name) orelse return null);
    } else if (o.solo != null) {
        try o.add(.budget);
    } else {
        if (o.op_check) try o.add(.@"op-check");
        try o.add(.check);
        if (o.streams > 0) try o.add(.streams);
        if (draft) |d| try o.add(d);
        if (o.price > 0) try o.add(.price);
        if (o.fuse_steps > 0) try o.add(.fuse);
        try o.add(.budget);
    }
    if (exp != null and draft != null) try o.add(draft.?);
    if (exp != null and o.streams > 0 and !o.has(.streams) and !o.has(.@"resume-streams")) try o.add(.streams);
    if (exp != null and o.op_check and !o.has(.@"op-check")) try o.add(.@"op-check");
    if ((o.has(.streams) or o.has(.@"resume-streams")) and o.streams == 0) o.streams = 4;
    if (exp != null and o.win_ab > 0 and !o.has(.@"win-ab")) try o.add(.@"win-ab");
    if (exp == null and o.win_ab > 0) try o.add(.@"win-ab");
    if (o.has(.@"win-ab") and o.win_ab == 0) o.win_ab = 10;
    if (o.shared_price > 0 and !o.has(.@"shared-price")) try o.add(.@"shared-price");
    if (o.has(.@"shared-price") and o.shared_price == 0) o.shared_price = 10;
    if (o.argmax > 0 and !o.has(.argmax)) try o.add(.argmax);
    if (o.has(.argmax) and o.argmax == 0) o.argmax = 20;
    if (o.has(.price) and o.price == 0) o.price = 20;
    if (o.has(.fuse) and o.fuse_steps == 0) o.fuse_steps = 200;
    if (o.nexps == 0) return null;
    // the DSpark gates, the price table, the fuse A/B and the resumes follow a check (its plain tokens and hash, slot
    // 0's position); the resumes compare with one depth's check
    for (o.list(), 0..) |e, j| if (e == .@"draft-real" or e == .@"draft-mixed" or e == .price or e == .fuse or e == .argmax or e == .@"resume" or e == .@"resume-draft" or e == .@"resume-streams" or e == .@"resume-inside") {
        if (std.mem.indexOfScalar(Exp, o.list()[0..j], .check) == null) return null;
        if ((e == .@"resume" or e == .@"resume-draft" or e == .@"resume-streams" or e == .@"resume-inside") and o.ndepths > 1) return null;
    };
    // a synthetic fill is deep's alone (one depth, no other experiment on the filled slot)
    if (o.synth) for (o.list()) |e| if (e != .deep and e != .@"deep-draft") return null;
    if (o.solo) |r| {
        if (o.world < 2 or r >= o.world or (o.rank != 0 and o.rank != r)) return null;
        o.rank = r;
        for (o.list()) |e| if (e != .budget and e != .ab) return null;
    }
    if (o.mix_ndepths == 0) {
        o.mix_depths[0..3].* = .{ 0, 8192, 32768 };
        o.mix_ndepths = 3;
    }
    if (o.mix_ndecs == 0) {
        o.mix_decs[0..2].* = .{ 1, 4 };
        o.mix_ndecs = 2;
    }
    if (o.ndepths == 0) {
        o.depths[0] = o.prompt;
        o.ndepths = 1;
    }
    for (o.depthList(), 0..) |d, j| if (d == 0 or (j > 0 and d <= o.depthList()[j - 1] + o.steps)) return null;
    o.prompt = o.depthList()[o.ndepths - 1];
    if (o.prompt == 0 or o.steps == 0 or o.chunk == 0 or o.chunk > engine.prompt_rows or o.world == 0 or o.rank >= o.world) return null;
    if (o.nprice_depths > 0) {
        if (!o.has(.@"deep-draft") or o.depth_reps == 0 or o.synth) return null; // a synthetic fill has no prefill to price in
        for (o.price_depths[0..o.nprice_depths], 0..) |d, j| if (d == 0 or d + 1 >= o.prompt or (j > 0 and d <= o.price_depths[j - 1])) return null;
    }
    if (o.drafts == 0 or o.drafts > ic.dspark.drafts or o.budget_steps == 0 or o.ab_reps == 0 or o.ab_steps == 0) return null;
    if (o.streams == 1 or o.streams > engine.max_segs) return null;
    return o;
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.arena.allocator();
    const args = try init.minimal.args.toSlice(a);
    const o = (parse(args) catch null) orelse {
        std.debug.print("{s}", .{usage});
        return 2;
    };
    return run(init.gpa, init.io, o) catch |e| {
        std.debug.print("FAIL tf-inkling2-decode: rank {d}: {t}\n", .{ o.rank, e });
        cuda.tp_watch.fail(@errorName(e)); // ranks over processes: every other rank stops too (a no-op without lines)
        return 1;
    };
}

fn promptIds(gpa: std.mem.Allocator, io: std.Io, o: Opts) ![]u32 {
    const out = try gpa.alloc(u32, o.prompt);
    errdefer gpa.free(out);
    if (o.ids) |path| {
        var r = try ink.st.Reader.open(gpa, io, path);
        defer r.close();
        const raw = try r.alloc(gpa, "ids", .i64);
        defer gpa.free(raw);
        const n = raw.len / 8;
        if (n == 0) return error.NoIds;
        for (out, 0..) |*t, i| t.* = @intCast(std.mem.readInt(i64, raw[8 * (i % n) ..][0..8], .little));
        return out;
    }
    var rng = std.Random.DefaultPrng.init(0x1d2c3b4a);
    for (out) |*t| t.* = rng.random().uintLessThan(u32, 199_000) + 100;
    return out;
}

fn argmax(row: []const f32) u32 {
    var best: usize = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = i;
    };
    return @intCast(best);
}

/// KL(p || q) of the softmaxes of logit rows p and q (fp64).
fn kl(p: []const f32, q: []const f32) f64 {
    var mp: f64 = -std.math.inf(f64);
    var mq: f64 = -std.math.inf(f64);
    for (p, q) |x, y| {
        mp = @max(mp, x);
        mq = @max(mq, y);
    }
    var zp: f64 = 0;
    var zq: f64 = 0;
    for (p, q) |x, y| {
        zp += @exp(@as(f64, x) - mp);
        zq += @exp(@as(f64, y) - mq);
    }
    const lzp = @log(zp) + mp;
    const lzq = @log(zq) + mq;
    var s: f64 = 0;
    for (p, q) |x, y| {
        const lp = @as(f64, x) - lzp;
        const lq = @as(f64, y) - lzq;
        s += @exp(lp) * (lp - lq);
    }
    return @max(s, 0);
}

fn run(gpa: std.mem.Allocator, io: std.Io, o: Opts) !u8 {
    var driver = try cuda.Driver.open();
    defer driver.close();
    if (cuda.Context.scheduleFromEnv()) |s| try cuda.Context.setSchedule(&driver, 0, s); // TF_CUDA_SCHED (host waits only)
    var ctx = try cuda.Context.init(&driver, 0);
    defer ctx.deinit();
    var group: ?cuda.tp_group.Group = null;
    defer if (group) |*g| g.close();
    const fault = Fault.read(o.rank);
    if (o.world > 1 and o.solo == null) {
        // liveness lines first (NCCL_COMM_ID's port + 1): from here a rank that ends early stops the others
        cuda.tp_watch.start(io, o.rank, o.world) catch |err| {
            std.debug.print("FAIL tf-inkling2-decode: rank {d} of {d}: liveness lines: {t}{s}{s}\n", .{ o.rank, o.world, err, if (cuda.tp_watch.explain(err) != null) ": " else "", cuda.tp_watch.explain(err) orelse "" });
            return err;
        };
        try fault.at(.open);
        group = try cuda.tp_group.Group.open(&driver, o.rank, o.world);
    }
    // an error from here aborts the communicator (no kernel waits on a peer), and main tells the others by name
    errdefer if (group) |*g| g.abort();
    if (group) |*g| {
        try fault.at(.vote);
        try ic.wire.openAll(gpa, io, &driver, g); // the transport settings' vote, then the wires
    }
    const T = o.prompt;
    const S = o.steps;
    // the check's context as before the experiments list (a slot's capacity sets a captured step's attention grid);
    // more only when the budget's and the A/B's slot needs it
    const need = @max(T + S + 64 + 10 * o.price + (if (o.has(.fuse)) o.fuse_steps else 0) + (if (o.has(.@"shared-price")) ic.batch_check.sharedPositions(o.shared_price) else 0) + (if (o.has(.argmax)) ic.argmax_check.tailPositions(o.argmax) else 0), T + o.benchSteps() + 8, if (o.has(.mixprice)) ic.batch_check.mixPositions(o.mix_reps, o.mix_depths[0..o.mix_ndepths], o.mix_decs[0..o.mix_ndecs]) else 0);
    const context = @max(need, o.context);
    const e = try engine.Engine.init(gpa, io, &driver, o.dir, .{ .rank = o.rank, .world = o.world, .layers = o.layers, .context = context, .group = if (group) |*g| g else null, .digests = o.digests, .drafter = o.spec(), .solo = o.solo != null, .scratch_gib = ic.scratch.gib() });
    defer e.deinit();
    // on pages, one pool for the slots the experiments bind (every rank alike), so a kept prompt state resumes on
    // another slot by its pages (the prompt cache on pages)
    try e.toolPool(toolSlots(o), toolSpare(o) * ic.pages.pagesFor(context));
    // a failure from here, ranks over processes: the others hear it and this process ends at once (its stream may
    // hold a kernel waiting on a peer, which freeing the engine or a normal exit would wait on), as the server's
    // followers do
    return loaded(gpa, io, &driver, o, e, if (group) |*g| g else null, fault, context) catch |err| {
        if (group) |*g| {
            std.debug.print("FAIL tf-inkling2-decode: rank {d}: {t}; every rank stops\n", .{ o.rank, err });
            cuda.tp_watch.fatal(@errorName(err)); // ends the process (lines open)
            g.abort();
        }
        return err;
    };
}

/// The slots the experiments bind at most (the tool pool's, Engine.toolPool): the check 0 and 1; the DSpark gates,
/// chunks and mixprice's chunk 2; the budget, the A/B, win-ab and the batched gates from 3 (--streams of them, 4 for
/// shared-price), mixprice's decoders from 4; the resumes 4 .. 6 and resume-streams 8 .. 8 + --streams; scratch a
/// fresh slot past them all; deep alone 0.
fn toolSlots(o: Opts) u32 {
    var n: usize = 2;
    var only_deep = true;
    for (o.exps[0..o.nexps]) |x| switch (x) {
        .deep, .@"deep-draft" => n = @max(n, 1),
        .check, .@"check-eager", .fuse, .price, .argmax, .scratch, .@"op-check" => {
            only_deep = false;
        },
        .@"draft-real", .@"draft-mixed", .@"draft-defer", .chunks => {
            only_deep = false;
            n = @max(n, 3);
        },
        .budget, .ab, .@"win-ab" => {
            only_deep = false;
            n = @max(n, 4);
        },
        .streams => {
            only_deep = false;
            n = @max(n, ic.batch_check.first_slot + @max(o.streams, 4));
        },
        .@"shared-price" => {
            only_deep = false;
            n = @max(n, ic.batch_check.first_slot + 4);
        },
        .mixprice => {
            only_deep = false;
            var most: usize = 0;
            for (o.mix_decs[0..o.mix_ndecs]) |d| most = @max(most, d);
            n = @max(n, ic.batch_check.first_slot + 1 + most);
        },
        .@"resume", .@"resume-inside" => {
            only_deep = false;
            n = @max(n, resume_dst + 1);
        },
        .@"resume-draft" => {
            only_deep = false;
            n = @max(n, resume_draft + 1);
        },
        .@"resume-streams" => {
            only_deep = false;
            n = @max(n, 8 + @max(o.streams, 2));
        },
    };
    // scratch (run last) takes a fresh slot past every other experiment's
    if (o.has(.scratch)) n += 1;
    return @intCast(if (only_deep) 1 else n);
}

/// Whole contexts of pages past the slots' on the tool pool: what kept prompt states hold while their source slots
/// run on (a state each with resume-streams, else one).
fn toolSpare(o: Opts) usize {
    if (o.has(.@"resume-streams")) return @max(o.streams, 2) + 1;
    if (o.has(.@"resume") or o.has(.@"resume-inside") or o.has(.@"resume-draft")) return 2;
    return 0;
}

/// The experiments on the loaded engine (rank 0), or following rank 0 (the others).
fn loaded(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, o: Opts, e: *engine.Engine, group: ?*cuda.tp_group.Group, fault: Fault, context: usize) !u8 {
    const T = o.prompt;
    const S = o.steps;
    try fault.at(.load);
    if (fault.kind == .mid) _ = try std.Thread.spawn(.{}, Fault.crash, .{ fault, io });
    e.asleep = group != null and cuda.tp_watch.open(); // the followers sleep until rank 0's first header
    if (o.solo != null) try soloZero(e);
    if (o.has(.fuse)) ic.forward.unfused_state = try e.bind(1);
    var prof: ic.forward.Prof = .{ .d = driver, .stream = e.stream };
    defer prof.deinit();
    if (o.has(.budget)) e.prof = &prof;
    if (o.rank != 0 and o.solo == null) {
        try ic.follow.follow(e); // rank 0 ended it (this rank's lines then end on purpose)
        if (prof.ends > 0) printBudget(&prof, o.rank, e.layers, null);
        fault.atEnd();
        try lastBarrier(io, driver, e, group);
        return 0;
    }
    var x: ic.follow.Leader = .{ .e = e };
    const long = try gpa.alloc(u32, T + S + ic.dspark.block); // deep-draft's last rounds read up to a block past S
    defer gpa.free(long);
    @memset(long[T + S ..], 0);
    const seq = long[0 .. T + S];
    {
        const p = try promptIds(gpa, io, o);
        defer gpa.free(p);
        @memcpy(seq[0..T], p);
    }
    var ok = true;
    var first_hash: ?u64 = null;
    var bench: Bench = .{};
    for (o.list()) |exp| switch (exp) {
        .check, .@"check-eager" => {
            x.eager = exp == .@"check-eager";
            defer x.eager = false;
            const c0 = e.counts;
            const h = try check(gpa, io, &x, o, seq, &ok);
            printCountsSince(e, @tagName(exp), c0);
            if (first_hash) |f| {
                const same = f == h;
                ok = ok and same;
                std.debug.print("{s} {s}: decode logits xxh64 {x:0>16} {s} the first check's\n", .{ if (same) "PASS" else "FAIL", @tagName(exp), h, if (same) "equals" else "DIFFERS FROM" });
            } else first_hash = h;
        },
        .@"draft-real", .@"draft-mixed", .@"draft-defer" => {
            const c0 = e.counts;
            ok = try draftGate(gpa, io, x, o, seq, switch (exp) {
                .@"draft-real" => .real,
                .@"draft-mixed" => .mixed,
                else => .deferred,
            }) and ok;
            printCountsSince(e, @tagName(exp), c0);
        },
        .price => try price(io, x, o, o.price, seq),
        .budget => try budget(io, &x, o, seq, &bench, &prof),
        .ab => try ab(io, &x, o, seq, &bench),
        .fuse => ok = try abFuse(gpa, io, x, o, seq) and ok,
        .streams => ok = try ic.batch_check.streams(gpa, io, x, seq, T, S, o.streams) and ok,
        .@"op-check" => ok = try ic.batch_check.opCheck(gpa, e) and ok,
        .@"win-ab" => try winAb(io, &x, o, seq, &bench),
        .@"shared-price" => try ic.batch_check.sharedPrice(io, x, seq, T, o.shared_price),
        .chunks => ok = try ic.batch_check.chunks(gpa, io, x, seq, T, S, if (o.ndepths == 1) first_hash else null) and ok,
        .mixprice => try ic.batch_check.mixPrice(io, x, seq, T, o.mix_reps, o.mix_depths[0..o.mix_ndepths], o.mix_decs[0..o.mix_ndecs]),
        .@"resume" => ok = try resumeGate(gpa, io, x, o, seq, first_hash.?) and ok,
        .@"resume-draft" => ok = try resumeDraft(gpa, io, x, o, seq) and ok,
        .@"resume-streams" => ok = try resumeStreams(gpa, x, o, seq, first_hash.?) and ok,
        .@"resume-inside" => ok = try resumeInside(gpa, io, x, o, seq, first_hash.?) and ok,
        .scratch => ok = try ic.mem_check.run(gpa, x, seq) and ok,
        .deep, .@"deep-draft" => ok = try deep(gpa, io, &x, o, long) and ok,
        .argmax => {
            ok = try ic.argmax_check.opCheck(gpa, e) and ok;
            ok = try ic.argmax_check.tail(gpa, x, seq, o.argmax) and ok;
        },
    };
    printCounts(e, "the run", context);
    if (e.tool_pool) ok = poolEnd(e) and ok;
    // the other ranks end only after a run without error (on an error main stops them by name instead: an end header
    // could wait on a rank inside a collective this one left)
    try x.end();
    fault.atEnd();
    try lastBarrier(io, driver, e, group);
    return if (ok) 0 else 1;
}

/// The last barrier's bound in seconds (TF_INK2_LAST_BARRIER_S, default 120; a healthy barrier takes milliseconds).
const last_barrier_s: u64 = 120;

/// Every rank has ended on purpose (rank 0 set it before its end header, the others on it) before any exits, so no
/// line closing after this stops a rank that is still finishing. Since the lines then ignore a rank that fails or
/// vanishes, the barrier is bounded: a watch thread ends this process with status 1 if the gather has not returned
/// within the bound (a peer that died after the end header, or failed inside the gather).
fn lastBarrier(io: std.Io, d: *const cuda.Driver, e: *engine.Engine, group: ?*cuda.tp_group.Group) !void {
    const g = group orelse return;
    const Watch = struct {
        var done = std.atomic.Value(bool).init(false);
        fn run(wio: std.Io, rank: u32, limit_s: u64) void {
            std.Io.sleep(wio, .fromMilliseconds(@intCast(limit_s * 1000)), .awake) catch {};
            if (done.load(.acquire)) return;
            std.debug.print("FAIL tf-inkling2-decode: rank {d}: the last barrier did not complete within {d} s (a rank ended after the end header): exiting\n", .{ rank, limit_s });
            std.c._exit(1);
        }
    };
    const limit_s = if (std.c.getenv("TF_INK2_LAST_BARRIER_S")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch last_barrier_s else last_barrier_s;
    const t = try std.Thread.spawn(.{}, Watch.run, .{ io, e.o.rank, limit_s });
    t.detach();
    var all: [ink.plan.max_world]u32 = undefined;
    try g.gather(u32, d, e.stream, e.o.rank, all[0..g.world]);
    Watch.done.store(true, .release);
}

/// TF_INK2_DECODE_FAULT (tests of the failure paths; see the top): `open:R`, `vote:R`, `load:R`, `mid:R:MS`, `end:R`.
const Fault = struct {
    kind: enum { none, open, vote, load, mid, end } = .none,
    after_ms: u64 = 0,

    /// This rank's fault, if the variable names it.
    fn read(rank: u32) Fault {
        const v = std.mem.span(std.c.getenv("TF_INK2_DECODE_FAULT") orelse return .{});
        var it = std.mem.splitScalar(u8, v, ':');
        const name = it.next() orelse return .{};
        const r = std.fmt.parseInt(u32, it.next() orelse return .{}, 10) catch return .{};
        if (r != rank) return .{};
        const kind = std.meta.stringToEnum(@FieldType(Fault, "kind"), name) orelse return .{};
        const after = if (kind == .mid) std.fmt.parseInt(u64, it.next() orelse return .{}, 10) catch return .{} else 0;
        return .{ .kind = kind, .after_ms = after };
    }

    /// error.InducedFault at stage `k` if this rank's fault is there.
    fn at(f: Fault, k: @FieldType(Fault, "kind")) !void {
        if (f.kind != k) return;
        std.debug.print("rank: TF_INK2_DECODE_FAULT fails this rank at {t}\n", .{k});
        return error.InducedFault;
    }

    /// `end`: this rank's process gone after the end header, before the last barrier (as a rank that crashes there).
    fn atEnd(f: Fault) void {
        if (f.kind != .end) return;
        std.debug.print("rank: TF_INK2_DECODE_FAULT ends this rank after the end header\n", .{});
        std.c._exit(1);
    }

    /// `mid`: after ms, this rank's reason on its lines and its process gone (as a rank that crashes mid-run).
    fn crash(f: Fault, io: std.Io) void {
        std.Io.sleep(io, .fromMilliseconds(@intCast(f.after_ms)), .awake) catch {};
        std.debug.print("rank: TF_INK2_DECODE_FAULT ends this rank mid-run after {d} ms\n", .{f.after_ms});
        cuda.tp_watch.fail("InducedFault");
        cuda.tp_watch.stop();
    }
};

/// --solo: the buffers other ranks would fill (their partials and gathered columns, their head shares, and a
/// rank above 0's embedded rows) start as zeros, so the lone rank computes on finite values (kernel times as a real
/// rank's: the same shapes, the same experts held; the routes follow its own values).
fn soloZero(e: *engine.Engine) !void {
    for (e.act.own.items) |b| try b.fill8(0, null);
    if (e.parts) |*p| {
        if (p.own) |own| for (own) |b| try b.fill8(0, null);
        if (p.split) |sp| for (sp.own) |b| try b.fill8(0, null);
    }
    for (e.hbufs) |hb| if (hb) |b| try b.fill8(0, null);
}

/// The decode check (see the top); its result line, its decode logits' xxh64 returned. Failing rows clear `ok`.
fn check(gpa: std.mem.Allocator, io: std.Io, x: *const ic.follow.Leader, o: Opts, seq: []u32, ok: *bool) !u64 {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const T = o.prompt;
    const S = o.steps;
    const U = e.vocab();
    const depths = o.depthList();
    // decode: at each depth P (in order) the prompt up to P in chunks, then S one-row steps; depth j's dec row k is the
    // logits at position P - 1 + k
    const dec = try gpa.alloc(f32, (depths.len * S + 1) * U);
    defer gpa.free(dec);
    const step_ms = try gpa.alloc(f64, @max(S - 1, 1));
    defer gpa.free(step_ms);
    var prefill_s: f64 = 0;
    var decode_s: f64 = 0;
    var at: usize = 0;
    for (depths, 0..) |P, j| {
        const rows_j = dec[j * S * U ..];
        var t0 = clock.now(io).toNanoseconds();
        while (at < P) {
            const n = @min(P - at, engine.prompt_rows);
            const last = at + n == P;
            try x.forward(0, at, seq[at..][0..n], if (last) .{ .lo = n - 1, .n = 1 } else null);
            at += n;
            if (last) @memcpy(rows_j[0..U], try e.logits());
        }
        const pre = secs(clock.now(io).toNanoseconds() - t0);
        prefill_s += pre;
        t0 = clock.now(io).toNanoseconds();
        for (0..S) |k| {
            const t1 = clock.now(io).toNanoseconds();
            seq[P + k] = argmax(rows_j[k * U ..][0..U]);
            if (k + 1 == S) break; // the last drawn token is fed by the next depth's prompt, or never
            try x.forward(0, P + k, seq[P + k ..][0..1], .{ .lo = 0, .n = 1 });
            @memcpy(rows_j[(k + 1) * U ..][0..U], try e.logits());
            step_ms[k] = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t1)) * 1e-6;
        }
        decode_s += secs(clock.now(io).toNanoseconds() - t0);
        at = P + S - 1;
        {
            const med = median(step_ms[0 .. S - 1]);
            var mn: f64 = std.math.inf(f64);
            for (step_ms[0 .. S - 1]) |v| mn = @min(mn, v);
            std.debug.print("depth {d}: prefill {d:.2} s (positions {d}..{d}), {d:.2} s cumulative; {d} decode steps at positions {d}..{d}{s}: {d:.2} ms a step (median; min {d:.2}), {d:.2} tok/s\n", .{ P, pre, if (j == 0) 0 else depths[j - 1] + S - 1, P - 1, prefill_s, S - 1, P, P + S - 2, if (x.eager) ", eager" else ", graphs", med, mn, 1e3 / med });
        }
    }
    // the decode rows' bits, to compare builds (a head or graph change must leave them unchanged)
    const dec_hash = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(dec[0 .. depths.len * S * U]));
    // the whole sequence but its last token as one prompt on slot 1: each depth's rows P - 1 .. P + S - 2 against dec
    const L = T + S - 1;
    var rows: usize = 0;
    var top1: usize = 0;
    var bits: usize = 0;
    var kl_max: f64 = 0;
    var kl_sum: f64 = 0;
    var diff_max: f32 = 0;
    var worst: usize = 0;
    var bits_at: [max_depths]usize = @splat(0);
    const t0 = clock.now(io).toNanoseconds();
    at = 0;
    while (at < L) {
        const n = @min(L - at, o.chunk);
        try x.forward(1, at, seq[at..][0..n], null);
        // rows of interest in this forward: each depth's, in order
        for (depths, 0..) |P, dj| {
            var r = @max(at, P - 1);
            const end = @min(at + n, P + S - 1);
            while (r < end) {
                const m = @min(end - r, engine.head_rows);
                const full = try x.head(.{ .lo = r - at, .n = m });
                for (0..m) |i| {
                    const j = r + i - (P - 1);
                    const fr = full[i * U ..][0..U];
                    const dr = dec[(dj * S + j) * U ..][0..U];
                    const same_top = argmax(fr) == argmax(dr);
                    const d = kl(fr, dr);
                    var md: f32 = 0;
                    for (fr, dr) |p, q| md = @max(md, @abs(p - q));
                    const same_bits = std.mem.eql(u8, std.mem.sliceAsBytes(fr), std.mem.sliceAsBytes(dr));
                    rows += 1;
                    top1 += @intFromBool(same_top);
                    bits += @intFromBool(same_bits);
                    bits_at[dj] += @intFromBool(same_bits);
                    kl_sum += d;
                    if (d > kl_max) {
                        kl_max = d;
                        worst = r + i;
                    }
                    diff_max = @max(diff_max, md);
                    if (!same_top or d > o.kl_max or j < 2 or r + i == 1023 or r + i == 1024 or (r + i + 1) % 3071 <= 1) std.debug.print("  position {d}: top-1 {s} (decode {d}, whole {d}), KL {e:.3}, max |diff| {e:.3}{s}\n", .{ r + i, if (same_top) "equal" else "DIFFERS", argmax(dr), argmax(fr), d, md, if (md == 0) ", bit-equal" else "" });
                }
                r += m;
            }
        }
        at += n;
    }
    const check_s = secs(clock.now(io).toNanoseconds() - t0);
    const pass = top1 == rows and kl_max <= o.kl_max;
    ok.* = ok.* and pass;
    if (depths.len > 1) {
        std.debug.print("depths:", .{});
        for (depths, 0..) |P, dj| std.debug.print(" {d} bit-equal {d}/{d};", .{ P, bits_at[dj], S });
        std.debug.print(" the whole sequence ({d} positions) in {d:.2} s\n", .{ L, check_s });
    }
    std.debug.print("{s}: layers 0..{d}, W={d}, {t} mode, prompt {d} tokens ({d:.2} s), {d} decode steps ({d:.1} ms a step{s}), whole sequence in chunks of {d} ({d:.2} s); positions {d} .. {d}: top-1 equal {d}/{d}, bit-equal {d}/{d}, KL max {e:.3} (position {d}) mean {e:.3}, max |diff| {e:.3}; decode logits xxh64 {x:0>16}\n", .{
        if (pass) "PASS" else "FAIL", e.layers - 1, o.world, e.k.moe.mode, T, prefill_s, depths.len * (S - 1), decode_s * 1e3 / @as(f64, @floatFromInt(@max(depths.len * (S - 1), 1))), if (x.eager) ", eager" else "", o.chunk, check_s, depths[0] - 1, L - 1, top1, rows, bits, rows, kl_max, worst, kl_sum / @as(f64, @floatFromInt(@max(rows, 1))), diff_max, dec_hash,
    });
    return dec_hash;
}

/// --exp deep / deep-draft (see the top): the check at depth P = --prompt with one prefill on slot 0; true if it passed.
fn deep(gpa: std.mem.Allocator, io: std.Io, x: *const ic.follow.Leader, o: Opts, seq: []u32) !bool {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const T = o.prompt;
    const S = o.steps;
    const U = e.vocab();
    const slot: u32 = 0;
    if (T < 2) return error.DeepNeedsAPrompt;
    const Q = T - 1; // the decode steps start at the prompt's last token
    // 1. the prompt but its last token, in prompt_rows chunks (progress every 64 chunks), or the synthetic fill; at
    // each --price-depths D the chunk ends there, the slot is marked, the rounds priced, and the slot rewound to D (the
    // prefill's clock paused)
    var t0 = clock.now(io).toNanoseconds();
    var priced_ns: i96 = 0;
    var at: usize = if (o.synth) Q else 0;
    var chunks: usize = 0;
    var next_price: usize = 0;
    if (o.synth) try x.synth(slot, Q);
    while (at < Q) {
        const stop = if (next_price < o.nprice_depths) o.price_depths[next_price] else Q;
        const n = @min(stop - at, engine.prompt_rows);
        try x.forward(slot, at, seq[at..][0..n], null);
        at += n;
        chunks += 1;
        if (chunks % 64 == 0) {
            try e.stream.synchronize();
            std.debug.print("deep: prefill {d} positions in {d:.1} s\n", .{ at, secs(clock.now(io).toNanoseconds() - t0 - priced_ns) });
        }
        if (next_price < o.nprice_depths and at == o.price_depths[next_price]) {
            try e.stream.synchronize();
            const tp = clock.now(io).toNanoseconds();
            try x.mark(slot);
            std.debug.print("round price at depth {d}:\n", .{at});
            try price(io, x.*, o, o.depth_reps, seq);
            try x.rewind(slot);
            try e.stream.synchronize();
            priced_ns += clock.now(io).toNanoseconds() - tp;
            next_price += 1;
        }
    }
    try e.stream.synchronize();
    const prefill_s = secs(clock.now(io).toNanoseconds() - t0 - priced_ns);
    if (o.synth) {
        std.debug.print("deep: synthetic fill of positions 0 .. {d} on every rank in {d:.2} s\n", .{ Q - 1, prefill_s });
    } else std.debug.print("deep: prefill {d} positions (0 .. {d}) in {d:.2} s ({d:.0} tokens/s)\n", .{ Q, Q - 1, prefill_s, @as(f64, @floatFromInt(Q)) / prefill_s });
    try x.mark(slot);
    // 2. S greedy decode steps from Q (the decode rows: positions Q .. Q + S - 1), as the check's steps
    const dec = try gpa.alloc(f32, S * U);
    defer gpa.free(dec);
    const step_ms = try gpa.alloc(f64, S);
    defer gpa.free(step_ms);
    for (0..S) |k| {
        const t1 = clock.now(io).toNanoseconds();
        try x.forward(slot, Q + k, seq[Q + k ..][0..1], .{ .lo = 0, .n = 1 });
        @memcpy(dec[k * U ..][0..U], try e.logits());
        step_ms[k] = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t1)) * 1e-6;
        seq[Q + k + 1] = argmax(dec[k * U ..][0..U]);
    }
    const dec_hash = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(dec));
    const first_ms = step_ms[0];
    const med = if (S > 1) median(step_ms[1..]) else first_ms;
    // 3. back to the mark; the same S tokens as one prompt forward over the same positions, its rows against dec
    try x.rewind(slot);
    t0 = clock.now(io).toNanoseconds();
    try x.forward(slot, Q, seq[Q..][0..S], null);
    var rows: usize = 0;
    var top1: usize = 0;
    var bits: usize = 0;
    var kl_max: f64 = 0;
    var diff_max: f32 = 0;
    var r: usize = 0;
    while (r < S) {
        const m = @min(S - r, engine.head_rows);
        const full = try x.head(.{ .lo = r, .n = m });
        for (0..m) |i| {
            const fr = full[i * U ..][0..U];
            const dr = dec[(r + i) * U ..][0..U];
            const same_top = argmax(fr) == argmax(dr);
            const d = kl(fr, dr);
            var md: f32 = 0;
            for (fr, dr) |p, q| md = @max(md, @abs(p - q));
            const same_bits = std.mem.eql(u8, std.mem.sliceAsBytes(fr), std.mem.sliceAsBytes(dr));
            rows += 1;
            top1 += @intFromBool(same_top);
            bits += @intFromBool(same_bits);
            kl_max = @max(kl_max, d);
            diff_max = @max(diff_max, md);
            if (!same_top or !same_bits or r + i == 0 or r + i + 1 == S) std.debug.print("  position {d}: top-1 {s} (decode {d}, prompt {d}), KL {e:.3}, max |diff| {e:.3}{s}\n", .{ Q + r + i, if (same_top) "equal" else "DIFFERS", argmax(dr), argmax(fr), d, md, if (same_bits) ", bit-equal" else "" });
        }
        r += m;
    }
    const chunk_s = secs(clock.now(io).toNanoseconds() - t0);
    const pass = top1 == rows and bits == rows and kl_max <= o.kl_max; // exact: every row bit-equal
    std.debug.print("{s} deep: layers 0..{d}, W={d}, {t} mode{s}, depth {d} ({s} {d:.2} s), {d} decode steps at positions {d}..{d}: {d:.2} ms a step (median; the first {d:.2}), {d:.2} tok/s; one {d}-row prompt forward over them ({d:.2} s with its heads): top-1 equal {d}/{d}, bit-equal {d}/{d}, KL max {e:.3}, max |diff| {e:.3}; decode logits xxh64 {x:0>16}\n", .{
        if (pass) "PASS" else "FAIL", e.layers - 1, o.world, e.k.moe.mode, if (e.drafter != null) ", drafter loaded" else "", T, if (o.synth) "synthetic fill" else "prefill", prefill_s, S, Q, Q + S - 1, med, first_ms, 1e3 / med, S, chunk_s, top1, rows, bits, rows, kl_max, diff_max, dec_hash,
    });
    std.debug.print("deep tokens (positions {d}..):", .{T});
    for (seq[T .. T + S]) |t| std.debug.print(" {d}", .{t});
    std.debug.print("\n", .{});
    var ok = pass;
    if (e.drafter == null) return deepEnd(x, o, slot, Q, ok);
    // 4. drafted equals plain from the same point: back to the mark, the decode step at Q, then verify rounds
    const got = try gpa.alloc(u32, S + ic.dspark.block);
    defer gpa.free(got);
    for ([_]bool{ false, true }) |mixed| {
        try x.rewind(slot);
        try x.forward(slot, Q, seq[Q..][0..1], .{ .lo = 0, .n = 1 });
        got[0] = argmax(try e.logits());
        var made: usize = 1;
        var pos: usize = T;
        var keep_n: usize = 1;
        var rounds: usize = 0;
        var hist: [ic.dspark.block]usize = @splat(0);
        var rng = std.Random.DefaultPrng.init(0x5eed);
        var draft_ns: i96 = 0;
        var verify_ns: i96 = 0;
        const D = o.drafts;
        while (made < S) {
            const anchor = got[made - 1];
            var props: [ic.dspark.drafts]u32 = @splat(0);
            var t2 = clock.now(io).toNanoseconds();
            try x.draft(slot, keep_n, anchor, &props);
            draft_ns += clock.now(io).toNanoseconds() - t2;
            if (mixed) {
                const wrong = rng.random().uintAtMost(usize, D);
                for (0..D) |i| {
                    const truth = if (made + i < S) seq[T + made + i] else 0;
                    props[i] = if (i == wrong) (truth + 1) % @as(u32, @intCast(U)) else truth;
                }
            }
            var win: [ic.dspark.block]u32 = undefined;
            win[0] = anchor;
            @memcpy(win[1 .. D + 1], props[0..D]);
            t2 = clock.now(io).toNanoseconds();
            try x.forward(slot, pos, win[0 .. D + 1], .{ .lo = 0, .n = D + 1 });
            const lg = try e.logits();
            verify_ns += clock.now(io).toNanoseconds() - t2;
            var k: usize = 0;
            while (k < D and argmax(lg[k * U ..][0..U]) == props[k]) k += 1;
            for (0..k) |i| got[made + i] = props[i];
            got[made + k] = argmax(lg[k * U ..][0..U]);
            made += k + 1;
            pos += k + 1;
            keep_n = k + 1;
            rounds += 1;
            hist[k] += 1;
        }
        try x.keep(slot, keep_n);
        var same: usize = 0;
        while (same < S and got[same] == seq[T + same]) same += 1;
        const dpass = same == S;
        ok = ok and dpass;
        const spent = draft_ns + verify_ns;
        std.debug.print("{s} deep drafted equals plain ({s} drafts, {d} a round) at depth {d}: {d}/{d} tokens equal; {d} rounds, {d:.3} tokens a round; accepted drafts", .{ if (dpass) "PASS" else "FAIL", if (mixed) "mixed" else "real", D, T, same, S, rounds, @as(f64, @floatFromInt(made - 1)) / @as(f64, @floatFromInt(@max(rounds, 1))) });
        for (hist[0 .. D + 1], 0..) |h, i| std.debug.print(" {d}:{d}", .{ i, h });
        std.debug.print("; {d:.2} ms a draft (with keep), {d:.2} ms a verify (with the head), {d:.2} ms a round, {d:.2} tok/s\n", .{ ms(draft_ns, rounds), ms(verify_ns, rounds), ms(spent, rounds), @as(f64, @floatFromInt(made - 1)) / @max(secs(spent), 1e-9) });
    }
    if (o.nprice_depths > 0) {
        // the prices at the deep depth itself, from the mark
        try x.rewind(slot);
        std.debug.print("round price at depth {d}:\n", .{Q});
        try price(io, x.*, o, o.depth_reps, seq);
    }
    return deepEnd(x, o, slot, Q, ok);
}

/// deep's end: with --fill synth, every rank reads its global layers' positions 0 .. Q - 1 back against the fill
/// (each logs its own; rank 0's counts here).
fn deepEnd(x: *const ic.follow.Leader, o: Opts, slot: u32, Q: usize, ok: bool) !bool {
    if (!o.synth) return ok;
    const intact = try x.synthCheck(slot, Q);
    std.debug.print("{s} deep synthetic fill intact after the experiment (rank 0; each rank logs its own)\n", .{if (intact) "PASS" else "FAIL"});
    return ok and intact;
}

/// Slot 3, the budget's and the A/B's: the prompt once, then greedy decode steps from where the last left it.
const Bench = struct {
    const slot: u32 = 3;
    pos: usize = 0,
    token: u32 = 0,

    /// The prompt on the slot, the first time.
    fn ready(b: *Bench, x: *const ic.follow.Leader, o: Opts, seq: []const u32) !void {
        if (b.pos > 0) return;
        const T = o.prompt;
        var at: usize = 0;
        while (at < T) {
            const n = @min(T - at, engine.prompt_rows);
            try x.forward(slot, at, seq[at..][0..n], if (at + n == T) .{ .lo = n - 1, .n = 1 } else null);
            at += n;
        }
        b.pos = T;
        b.token = if (x.e.o.rank == 0) argmax(try x.e.logits()) else seq[T - 1]; // --solo: only rank 0 has logits
    }

    /// n decode steps (eager or as graphs, by `x.eager`); the wall time of each into `ms` (n of them) if given.
    fn steps(b: *Bench, io: std.Io, x: *const ic.follow.Leader, n: usize, times: ?[]f64) !void {
        const clock = std.Io.Clock.awake;
        for (0..n) |k| {
            const t0 = clock.now(io).toNanoseconds();
            try x.forward(slot, b.pos, &.{b.token}, .{ .lo = 0, .n = 1 });
            if (x.e.o.rank != 0) try x.e.stream.synchronize(); // --solo above rank 0: no head download waits for it
            const lg = if (x.e.o.rank == 0) try x.e.logits() else null;
            if (times) |m| m[k] = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-6;
            if (lg) |l| b.token = argmax(l);
            b.pos += 1;
        }
    }
};

const Stat = struct {
    mean: f64,
    median: f64,
    min: f64,

    fn of(v: []f64) Stat {
        std.mem.sort(f64, v, {}, std.sort.asc(f64));
        var sum: f64 = 0;
        for (v) |x| sum += x;
        return .{ .mean = sum / @as(f64, @floatFromInt(v.len)), .median = v[v.len / 2], .min = v[0] };
    }
};

/// Graph and eager steps alternated `reps` times, `n` each: the times of each kind (in `g` and `e`, reps * n each).
fn alternate(io: std.Io, x: *ic.follow.Leader, b: *Bench, reps: usize, n: usize, g: []f64, e: []f64) !void {
    for (0..reps) |r| {
        x.eager = false;
        try b.steps(io, x, n, g[r * n ..][0..n]);
        x.eager = true;
        try b.steps(io, x, n, e[r * n ..][0..n]);
    }
    x.eager = false;
}

/// The budget: graph and eager steps alternated, then eager steps profiled op by op (on every rank).
fn budget(io: std.Io, x: *ic.follow.Leader, o: Opts, seq: []const u32, b: *Bench, prof: *ic.forward.Prof) !void {
    const N = o.budget_steps;
    const reps = 3;
    try b.ready(x, o, seq);
    var buf: [3 * 3 * 64]f64 = undefined;
    if (N > 64) return error.BudgetStepsAbove64;
    const g = buf[0 .. reps * N];
    const ea = buf[reps * N ..][0 .. reps * N];
    const pf = buf[2 * reps * N ..][0..N];
    const pos0 = b.pos;
    try alternate(io, x, b, reps, N, g, ea);
    prof.reset();
    x.e.prof_on = true;
    defer x.e.prof_on = false;
    try b.steps(io, x, N, pf);
    x.e.prof_on = false;
    const sg = Stat.of(g);
    const se = Stat.of(ea);
    const sp = Stat.of(pf);
    printBudget(prof, x.e.o.rank, x.e.layers, .{ .graph = sg.median, .eager = se.median, .profiled = sp.median });
    std.debug.print("budget: positions {d}..{d}, {d} graph and {d} eager steps alternated {d} times{s}: graph {d:.2} ms a step (median; mean {d:.2}, min {d:.2}), eager {d:.2} ms (mean {d:.2}, min {d:.2}), eager profiled {d:.2} ms; the per-op table above is rank {d}'s{s} ({d} profiled steps)\n", .{ pos0, b.pos - 1, N, N, reps, if (x.e.drafter != null) " (the drafter is loaded)" else "", sg.median, sg.mean, sg.min, se.median, se.mean, se.min, sp.median, x.e.o.rank, if (x.e.o.solo) " alone (--solo: exchanges move nothing)" else "", prof.ends });
}

/// Graph against eager steps, alternated --ab-reps times, --ab-steps each, in this one process.
fn ab(io: std.Io, x: *ic.follow.Leader, o: Opts, seq: []const u32, b: *Bench) !void {
    try b.ready(x, o, seq);
    const gpa = std.heap.page_allocator;
    const n = o.ab_reps * o.ab_steps;
    const g = try gpa.alloc(f64, n);
    defer gpa.free(g);
    const ea = try gpa.alloc(f64, n);
    defer gpa.free(ea);
    const pos0 = b.pos;
    try alternate(io, x, b, o.ab_reps, o.ab_steps, g, ea);
    // each rep's difference of means (eager - graph), for its spread
    var d_min: f64 = std.math.inf(f64);
    var d_max: f64 = -std.math.inf(f64);
    for (0..o.ab_reps) |r| {
        var sg: f64 = 0;
        var se: f64 = 0;
        for (g[r * o.ab_steps ..][0..o.ab_steps], ea[r * o.ab_steps ..][0..o.ab_steps]) |u, v| {
            sg += u;
            se += v;
        }
        const d = (se - sg) / @as(f64, @floatFromInt(o.ab_steps));
        d_min = @min(d_min, d);
        d_max = @max(d_max, d);
    }
    const sg = Stat.of(g);
    const se = Stat.of(ea);
    std.debug.print("ab graph vs eager: positions {d}..{d}, {d} reps of {d} steps each{s}: graph {d:.2} ms a step (median; mean {d:.2}, min {d:.2}), eager {d:.2} ms (mean {d:.2}, min {d:.2}); eager - graph {d:.2} ms (median), per rep {d:.2} .. {d:.2} ms\n", .{ pos0, b.pos - 1, o.ab_reps, o.ab_steps, if (x.e.drafter != null) " (the drafter is loaded)" else "", sg.median, sg.mean, sg.min, se.median, se.mean, se.min, se.median - sg.median, d_min, d_max });
}

/// The windows' A/B: widths 1, 2, 4 and 8.
const win_widths = [_]usize{ 1, 2, 4, 8 };
const win_rows_sum = 1 + 2 + 4 + 8;

/// win-ab (--win-ab N): on slot 3 past its prompt, for each width R in 1, 2, 4, 8: a window of R rows (the head over
/// every row) as a captured graph and eagerly, alternated N times (which goes first alternating), each timed from the
/// call to its logits on the host; with the drafter a keep of 1 row after each window (a DSpark round's verify; the
/// keep is not timed), else the slot moves on R positions. The first window of each kind is not timed (the graph's
/// capture). Prints each width's medians and the paired difference (eager - graph) with its mean and standard error.
fn winAb(io: std.Io, x: *ic.follow.Leader, o: Opts, seq: []const u32, b: *Bench) !void {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    try b.ready(x, o, seq);
    const gpa = std.heap.page_allocator;
    const n = o.win_ab;
    const g = try gpa.alloc(f64, n);
    defer gpa.free(g);
    const ea = try gpa.alloc(f64, n);
    defer gpa.free(ea);
    const d = try gpa.alloc(f64, n);
    defer gpa.free(d);
    const spec = e.drafter != null;
    std.debug.print("win-ab: slot {d} from position {d}, {d} pairs a width{s}; ms from the call to the logits on the host\n", .{ Bench.slot, b.pos, n, if (spec) ", a keep of 1 after each window (the drafter is loaded)" else "" });
    for (win_widths) |R| {
        var ids: [8]u32 = undefined;
        for (0..2 * n + 2) |i| {
            const graph = i % 2 == (i / 2) % 2; // pairs alternate which kind goes first
            for (ids[0..R], 0..) |*t, j| t.* = seq[(b.pos + j) % o.prompt];
            x.eager = !graph;
            const t0 = clock.now(io).toNanoseconds();
            try x.forward(Bench.slot, b.pos, ids[0..R], .{ .lo = 0, .n = R });
            if (e.o.rank == 0) _ = try e.logits();
            const ms_ = @as(f64, @floatFromInt(clock.now(io).toNanoseconds() - t0)) * 1e-6;
            x.eager = false;
            if (spec) {
                try x.keep(Bench.slot, 1);
                b.pos += 1;
            } else b.pos += R;
            if (i < 2) continue; // the capture, and the first eager window
            const k = (i - 2) / 2;
            if (graph) g[k] = ms_ else ea[k] = ms_;
        }
        for (d, g, ea) |*v, u, w| v.* = w - u;
        var mean: f64 = 0;
        for (d) |v| mean += v;
        mean /= @floatFromInt(n);
        var ss: f64 = 0;
        for (d) |v| ss += (v - mean) * (v - mean);
        const se = if (n > 1) @sqrt(ss / @as(f64, @floatFromInt(n - 1)) / @as(f64, @floatFromInt(n))) else 0;
        const sd = Stat.of(d);
        const sg = Stat.of(g);
        const sea = Stat.of(ea);
        std.debug.print("  width {d}: graph {d:.2} ms (median; min {d:.2}), eager {d:.2} ms (min {d:.2}); eager - graph {d:.2} ms (median of pairs), mean {d:.2} +- {d:.2} (se)\n", .{ R, sg.median, sg.min, sea.median, sea.min, sd.median, mean, se });
    }
    b.token = seq[b.pos % o.prompt];
}

/// The engine's forward counts: graph, eager, captured (their ms and device MiB), evicted.
fn printCounts(e: *const engine.Engine, what: []const u8, context: usize) void {
    const c = e.counts;
    std.debug.print("forwards in {s} (context {d}): {d} as graphs, {d} eager, {d} captures ({d:.0} ms, {d:.1} MiB in all), {d} evicted, {d} held\n", .{ what, context, c.graph, c.eager, c.captured, c.capture_ms, @as(f64, @floatFromInt(c.graph_bytes)) / (1 << 20), c.evicted, e.graphs.items.len });
    if (c.one_ring + c.one_paged > 0) std.debug.print("one-row stream forwards on pages in {s}: {d} on the run's ring kernels, {d} on the page table, {d} mode switches; {d} captured nodes pointed at a moved run in {d} replays\n", .{ what, c.one_ring, c.one_paged, c.switches, c.repointed, c.repoint_replays });
}

/// The tool pool after the experiments (G5's idle accounting): every slot's pages given back (rank 0's leases; the
/// other ranks hold no counts), then nothing promised, no stream's page, and free + kept = total, with no kept state
/// left (each experiment drops its own). The resumes' adoption counts with it.
fn poolEnd(e: *engine.Engine) bool {
    const p = e.pool orelse return true;
    const a = &(p.alloc orelse return true);
    for (0..p.slots) |slot| a.finish(&p.leases[slot]);
    const c = a.census();
    const idle = a.idle() and c.kept == 0 and e.snap.live.items.len == 0;
    const st = a.stats;
    std.debug.print("POOL {s} at the end: promised {d}, leased {d}, free {d} + kept {d} == total {d} ({d} kept states left); resumes on kept pages {d} ({d} pages adopted, {d} tables one run); covers {d}, one run {d}, scattered {d}, grown {d} ({d} in place)\n", .{ if (idle) "idle" else "NOT IDLE", c.promised, a.leased(), c.free, c.kept, c.total, e.snap.live.items.len, st.resumed, st.resumed_pages, st.resumed_one_run, st.covers, st.one_run, st.fallbacks, st.grown, st.grown_in_place });
    return idle;
}

fn printCountsSince(e: *const engine.Engine, what: []const u8, c0: engine.Counts) void {
    const c = e.counts;
    std.debug.print("forwards in {s}: {d} as graphs, {d} eager, {d} captured\n", .{ what, c.graph - c0.graph, c.eager - c0.eager, c.captured - c0.captured });
}

const Steps = struct { graph: f64, eager: f64, profiled: f64 };

/// A rank's per-op budget: each label's stream ms a profiled step summed over the layers, the exchanges' total
/// (tp.*), the layers' total, and (rank 0) the step times.
fn printBudget(p: *const ic.forward.Prof, rank: u32, layers: u32, steps: ?Steps) void {
    const n: f64 = @floatFromInt(@max(p.ends, 1));
    var total: f64 = 0;
    var tp: f64 = 0;
    var per_layer: [ic.forward.Prof.rows]f64 = @splat(0);
    std.debug.print("budget rank {d}: {d} profiled eager steps, stream ms a step by op (all layers; us a layer)\n", .{ rank, p.ends });
    for (p.labels[0..p.nlabels], 0..) |name, j| {
        var sum: f64 = 0;
        var used: u32 = 0;
        for (0..ic.forward.Prof.rows) |l| {
            sum += p.ms[l][j];
            per_layer[l] += p.ms[l][j] / n;
            used += @intFromBool(p.ms[l][j] > 0);
        }
        sum /= n;
        total += sum;
        if (std.mem.startsWith(u8, name, "tp.")) tp += sum;
        std.debug.print("budget op {s:<16} {d:>8.3} ms {d:>9.1} us\n", .{ name, sum, sum * 1e3 / @as(f64, @floatFromInt(@max(used, 1))) });
    }
    var lo: f64 = std.math.inf(f64);
    var hi: f64 = 0;
    for (per_layer[0..@min(layers, ic.forward.Prof.rows - 1)]) |v| {
        lo = @min(lo, v);
        hi = @max(hi, v);
    }
    std.debug.print("budget rank {d}: forward body {d:.3} ms a step (exchanges tp.* {d:.3} ms, the rest {d:.3} ms); a layer {d:.3} .. {d:.3} ms", .{ rank, total, tp, total - tp, lo, hi });
    if (steps) |s| std.debug.print("; step: graph {d:.2} ms, eager {d:.2} ms, eager profiled {d:.2} ms (head, host and launch gaps outside the body {d:.2} ms)", .{ s.graph, s.eager, s.profiled, s.profiled - total });
    std.debug.print("\n", .{});
}

fn secs(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-9;
}

/// The DSpark gate: slot 2 decodes in verify rounds and must draw the plain decode's S tokens (seq[P..P+S)) at each
/// depth P (one depth unless --depths); before the last depth, a depth's last rounds verify only as many drafts as
/// tokens remain, so slot 2 ends the depth where slot 0 did and takes the next depth's prompt from there.
/// draft-defer (mode deferred): draft-real with each round's draft made in its verify's step (Leader.forwardDrafted:
/// one header, the keep, the block, then the window reading the drafts on the device): the same rounds and accepts.
fn draftGate(gpa: std.mem.Allocator, io: std.Io, x: ic.follow.Leader, o: Opts, seq: []const u32, mode: enum { real, mixed, deferred }) !bool {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const S = o.steps;
    const U = e.vocab();
    const slot: u32 = 2;
    const depths = o.depthList();
    const got = try gpa.alloc(u32, S + ic.dspark.block);
    defer gpa.free(got);
    var at: usize = 0;
    var rounds: usize = 0;
    var tokens: usize = 0;
    var hist: [ic.dspark.block]usize = @splat(0);
    var rng = std.Random.DefaultPrng.init(0x5eed);
    const D = o.drafts;
    var draft_ns: i96 = 0;
    var verify_ns: i96 = 0;
    var pass = true;
    var same_at: [max_depths]usize = @splat(0);
    for (depths, 0..) |T, dj| {
        var last: usize = 0;
        var first: u32 = 0;
        while (at < T) {
            const n = @min(T - at, engine.prompt_rows);
            try x.forward(slot, at, seq[at..][0..n], if (at + n == T) .{ .lo = n - 1, .n = 1 } else null);
            at += n;
            last = n;
            if (at == T) first = argmax(try e.logits());
        }
        got[0] = first;
        var made: usize = 1;
        var pos: usize = T; // the anchor's position
        var keep_n: usize = last; // rows of the last forward the next draft keeps
        while (made < S) {
            const anchor = got[made - 1];
            // drafts this round: before the last depth, at most the tokens the depth still needs past the anchor's
            const Dr = if (dj + 1 == depths.len) D else @min(D, S - 1 - made);
            var props: [ic.dspark.drafts]u32 = @splat(0);
            var t0 = clock.now(io).toNanoseconds();
            if (mode != .deferred) try x.draft(slot, keep_n, anchor, &props);
            draft_ns += clock.now(io).toNanoseconds() - t0;
            if (mode == .mixed) {
                const wrong = rng.random().uintAtMost(usize, D);
                for (0..D) |i| {
                    const truth = if (T + made + i < seq.len) seq[T + made + i] else 0;
                    props[i] = if (i == wrong) (truth + 1) % @as(u32, @intCast(U)) else truth;
                }
            }
            var rows: [ic.dspark.block]u32 = undefined;
            rows[0] = anchor;
            @memcpy(rows[1 .. Dr + 1], props[0..Dr]);
            t0 = clock.now(io).toNanoseconds();
            if (mode == .deferred) {
                const seg = [1]engine.Segment{.{ .slot = slot, .pos0 = pos, .rows = Dr + 1 }};
                try x.forwardDrafted(&seg, rows[0 .. Dr + 1], &.{.{ .n = keep_n, .draft = true }});
                try e.proposals(&.{&props});
                @memcpy(rows[1 .. Dr + 1], props[0..Dr]);
            } else try x.forward(slot, pos, rows[0 .. Dr + 1], .{ .lo = 0, .n = Dr + 1 });
            const lg = try e.logits();
            verify_ns += clock.now(io).toNanoseconds() - t0;
            var k: usize = 0;
            while (k < Dr and argmax(lg[k * U ..][0..U]) == props[k]) k += 1;
            for (0..k) |i| got[made + i] = props[i];
            got[made + k] = argmax(lg[k * U ..][0..U]);
            made += k + 1;
            pos += k + 1;
            keep_n = k + 1;
            rounds += 1;
            hist[k] += 1;
        }
        try x.keep(slot, keep_n);
        var same: usize = 0;
        while (same < S and got[same] == seq[T + same]) same += 1;
        same_at[dj] = same;
        pass = pass and same == S;
        tokens += made;
        at = T + S - 1;
    }
    std.debug.print("{s} drafted equals plain ({s} drafts, {d} a round): ", .{ if (pass) "PASS" else "FAIL", @tagName(mode), D });
    for (depths, 0..) |T, dj| {
        if (depths.len > 1) std.debug.print("depth {d} ", .{T});
        std.debug.print("{d}/{d} tokens equal{s}; ", .{ same_at[dj], S, if (same_at[dj] == S) "" else " (first difference there)" });
    }
    std.debug.print("{d} rounds, {d:.3} tokens a round; accepted drafts", .{ rounds, @as(f64, @floatFromInt(tokens)) / @as(f64, @floatFromInt(@max(rounds, 1))) });
    for (hist[0 .. D + 1], 0..) |h, i| std.debug.print(" {d}:{d}", .{ i, h });
    if (mode == .deferred) std.debug.print("; {d:.2} ms a round's step (keep, draft, verify with the head)\n", .{ms(verify_ns, rounds)}) else std.debug.print("; {d:.2} ms a draft (with keep), {d:.2} ms a verify (with the head)\n", .{ ms(draft_ns, rounds), ms(verify_ns, rounds) });
    return pass;
}

/// The resume gates' DSpark rounds: drafts made apart (as draft-real) or in the verify's step (as draft-defer).
const DraftMode = enum { real, deferred };

/// A DSpark run's outcome: drafted equals plain, and its rounds (compared across a cold and a resumed run).
const DraftSum = struct {
    pass: bool,
    rounds: usize,
    /// Tokens equal to the plain decode's (from the first), and drawn in all.
    made: usize,
    made_all: usize,
    hist: [ic.dspark.block]usize,
    draft_ms: f64,
    verify_ms: f64,
    /// xxh64 of every round's proposals (the drafter's own, before acceptance).
    proposals: u64,

    /// The same rounds: the same proposals, every round accepting as many.
    fn same(a: DraftSum, b: DraftSum) bool {
        return a.rounds == b.rounds and a.made_all == b.made_all and std.mem.eql(usize, &a.hist, &b.hist) and a.proposals == b.proposals;
    }
};

fn printDraft(r: DraftSum, mode: DraftMode, o: Opts, what: []const u8) void {
    std.debug.print("{s} drafted equals plain ({s} drafts, {d} a round{s}): {d}/{d} tokens equal{s}; {d} rounds, {d:.3} tokens a round; accepted drafts", .{ if (r.pass) "PASS" else "FAIL", @tagName(mode), o.drafts, what, r.made, o.steps, if (r.pass) "" else " (first difference there)", r.rounds, @as(f64, @floatFromInt(r.made_all)) / @as(f64, @floatFromInt(@max(r.rounds, 1))) });
    for (r.hist[0 .. o.drafts + 1], 0..) |h, i| std.debug.print(" {d}:{d}", .{ i, h });
    std.debug.print("; {d:.2} ms a draft (with keep), {d:.2} ms a verify (with the head); proposals xxh64 {x:0>16}\n", .{ r.draft_ms, r.verify_ms, r.proposals });
}

/// Verify rounds on `slot` from its prompt's end (`first` drawn there, the last forward `last` rows): the drafter's
/// proposals, greedy acceptance, rollback of the rest; the S tokens it draws against the plain decode's. `deferred`:
/// each round's keep and draft in its verify's step (Leader.forwardDrafted), as the server's lanes run them.
fn draftRounds(gpa: std.mem.Allocator, io: std.Io, x: ic.follow.Leader, o: Opts, seq: []const u32, mode: DraftMode, slot: u32, first: u32, last: usize) !DraftSum {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const T = o.prompt;
    const S = o.steps;
    const U = e.vocab();
    const got = try gpa.alloc(u32, S + ic.dspark.block);
    defer gpa.free(got);
    got[0] = first;
    var made: usize = 1;
    var pos: usize = T; // the anchor's position
    var keep_n: usize = last; // rows of the last forward the next draft keeps
    var rounds: usize = 0;
    var hist: [ic.dspark.block]usize = @splat(0);
    const D = o.drafts;
    var draft_ns: i96 = 0;
    var verify_ns: i96 = 0;
    var proposals = std.hash.XxHash64.init(0);
    while (made < S) {
        const anchor = got[made - 1];
        var props: [ic.dspark.drafts]u32 = @splat(0);
        var t0 = clock.now(io).toNanoseconds();
        if (mode == .real) try x.draft(slot, keep_n, anchor, &props);
        draft_ns += clock.now(io).toNanoseconds() - t0;
        var rows: [ic.dspark.block]u32 = undefined;
        rows[0] = anchor;
        @memcpy(rows[1 .. D + 1], props[0..D]);
        t0 = clock.now(io).toNanoseconds();
        if (mode == .deferred) {
            const seg = [1]engine.Segment{.{ .slot = slot, .pos0 = pos, .rows = D + 1 }};
            try x.forwardDrafted(&seg, rows[0 .. D + 1], &.{.{ .n = keep_n, .draft = true }});
            try e.proposals(&.{&props});
            @memcpy(rows[1 .. D + 1], props[0..D]);
        } else try x.forward(slot, pos, rows[0 .. D + 1], .{ .lo = 0, .n = D + 1 });
        proposals.update(std.mem.sliceAsBytes(&props));
        const lg = try e.logits();
        verify_ns += clock.now(io).toNanoseconds() - t0;
        var k: usize = 0;
        while (k < D and argmax(lg[k * U ..][0..U]) == props[k]) k += 1;
        for (0..k) |i| got[made + i] = props[i];
        got[made + k] = argmax(lg[k * U ..][0..U]);
        made += k + 1;
        pos += k + 1;
        keep_n = k + 1;
        rounds += 1;
        hist[k] += 1;
    }
    try x.keep(slot, keep_n);
    var same: usize = 0;
    while (same < S and got[same] == seq[T + same]) same += 1;
    return .{ .pass = same == S, .rounds = rounds, .made = same, .made_all = made, .hist = hist, .draft_ms = ms(draft_ns, rounds), .verify_ms = ms(verify_ns, rounds), .proposals = proposals.final() };
}

/// resume's and resume-inside's cuts unless --cuts: suffixes of 1, 2, 3, 8, 9 rows, the ring's edges, the chunk's.
fn defaultCuts(T: usize) [12]usize {
    return .{ 1, 2, 3, 8, 9, 1000, 1023, 1024, 2047, 2048, 2049, T - 1 };
}

/// Slot `slot` prefilled [0, to) in 2048-row chunks cut as the server's lanes cut them (Lanes.chunk, stage 1b): the
/// marks inside a chunk captured by its forward (more than max_caps end it at the next), and every mark the forward
/// passed kept after it into kept[i] (marks ascending, kept[i] for marks[i]).
fn prefillMarks(x: ic.follow.Leader, slot: u32, seq: []const u32, to: usize, rows: usize, marks: []const u32, kept: []ic.snap.Kept) !void {
    var at: usize = 0;
    while (at < to) {
        var caps: [ic.snap.max_caps]u32 = undefined;
        const in = ic.snap.inside(marks, at, @min(to - at, rows), &caps);
        if (in.n > 0) try ic.snap.capture(x, slot, at, in.rows, caps[0..in.n]);
        try x.forward(slot, at, seq[at..][0..in.rows], null);
        for (marks, 0..) |m, i| if (m > at and m <= at + in.rows) try ic.snap.save(x, slot, m, &kept[i]);
        at += in.rows;
    }
}

/// The capture gate (see `resume-inside` in the usage).
fn resumeInside(gpa: std.mem.Allocator, io: std.Io, x: ic.follow.Leader, o: Opts, seq: []const u32, cold: u64) !bool {
    _ = io;
    const e = x.e;
    const T = o.prompt;
    const S = o.steps;
    const U = e.vocab();
    try resumeArena(x, o);
    var buf: [max_cuts]usize = undefined;
    const dc = defaultCuts(T);
    const cuts = try o.cutList(o.cuts, &dc, &buf);
    var marks: [max_cuts]u32 = undefined;
    for (cuts, marks[0..cuts.len]) |c, *m| m.* = @intCast(c);
    var kept: [max_cuts]ic.snap.Kept = undefined;
    const cap0 = e.snap.captured;
    try prefillMarks(x, resume_src, seq, T, @min(@max(o.src_rows, 1), engine.prompt_rows), marks[0..cuts.len], kept[0..cuts.len]);
    defer for (kept[0..cuts.len]) |*k| ic.snap.drop(x, k);
    const captured = e.snap.captured - cap0;
    const dec = try gpa.alloc(f32, S * U);
    defer gpa.free(dec);
    var pass = true;
    for (cuts, kept[0..cuts.len]) |c, k| {
        try ic.snap.restore(x, resume_dst, k, o.poison);
        const h = try warmRun(x, o, seq, resume_dst, c, dec);
        const same = h == cold;
        pass = pass and same;
        std.debug.print("  resume-inside at {d} ({s}): decode logits xxh64 {x:0>16} {s}\n", .{ c, if (c % engine.prompt_rows == 0) "a chunk's end" else "inside a chunk or at a cut chunk's end", h, if (same) "equal" else "DIFFERS" });
    }
    std.debug.print("{s} resume-inside: {d} states from one prefill (chunks of up to {d} rows; {d} captured inside a chunk; layers 0..{d}, W={d}, prompt {d}, {d} steps{s}): every decode logits xxh64 {s} the check's {x:0>16}\n", .{ if (pass) "PASS" else "FAIL", cuts.len, @min(@max(o.src_rows, 1), engine.prompt_rows), captured, e.layers - 1, o.world, T, S, if (e.drafter != null) ", drafter loaded" else "", if (pass) "equals" else "NOT ALL EQUAL TO", cold });
    return pass and captured > 0;
}

/// The prompt cache's slots in `resume`: the source, the other slot a state is restored into, the DSpark one.
const resume_src: u32 = 4;
const resume_dst: u32 = 5;
const resume_draft: u32 = 6;

/// Prompt rows [from, to) of `slot` in chunks of up to 2048 from `from` (as a lane cuts them at a mark); the head
/// over the last row when `to` is the prompt's end. Returns the last chunk's rows.
fn prefillRange(x: ic.follow.Leader, slot: u32, seq: []const u32, from: usize, to: usize, head_at_end: bool) !usize {
    var at = from;
    var last: usize = 0;
    while (at < to) {
        const n = @min(to - at, engine.prompt_rows);
        try x.forward(slot, at, seq[at..][0..n], if (head_at_end and at + n == to) .{ .lo = n - 1, .n = 1 } else null);
        at += n;
        last = n;
    }
    return last;
}

/// The arena for the resume experiments (every rank, made once): room for two states at the prompt's length, or
/// one a stream with resume-streams.
fn resumeArena(x: ic.follow.Leader, o: Opts) !void {
    if (x.e.snap.buf != null) return;
    var n: u64 = if (o.has(.@"resume-streams")) @max(2, o.streams) else 2;
    if (o.has(.@"resume-inside")) {
        var buf: [max_cuts]usize = undefined;
        const dc = defaultCuts(o.prompt);
        n = @max(n, (try o.cutList(o.cuts, &dc, &buf)).len + 1);
    }
    _ = try ic.snap.arena(x, n * ic.snap.bytes(x.e, o.prompt) + ic.snap.capBytes(x.e));
}

/// The stream drained (every earlier step done on this rank); the time now in ns.
fn drained(io: std.Io, x: ic.follow.Leader) !i96 {
    try x.e.stream.synchronize();
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

/// The resume gate (see `resume` in the usage): every warm run's decode logits bit-equal to the cold check's.
fn resumeGate(gpa: std.mem.Allocator, io: std.Io, x: ic.follow.Leader, o: Opts, seq: []const u32, cold: u64) !bool {
    const e = x.e;
    const T = o.prompt;
    const S = o.steps;
    const U = e.vocab();
    try resumeArena(x, o);
    var buf: [max_cuts]usize = undefined;
    const dc = defaultCuts(T);
    const cuts = try o.cutList(o.cuts, &dc, &buf);
    const dec = try gpa.alloc(f32, S * U);
    defer gpa.free(dec);
    var pass = true;
    var runs: usize = 0;
    for (cuts) |c| {
        _ = try prefillRange(x, resume_src, seq, 0, c, false);
        const t0 = try drained(io, x);
        var k: ic.snap.Kept = undefined;
        try ic.snap.save(x, resume_src, c, &k);
        const save_ms = ms(try drained(io, x) - t0, 1);
        defer ic.snap.drop(x, &k);
        // the source runs on past the cut: the kept copy must not change with it
        _ = try prefillRange(x, resume_src, seq, c, @min(T, c + engine.prompt_rows), false);
        const dests = [2]u32{ resume_dst, resume_src };
        for (dests[0..if (o.resume_back) 2 else 1]) |slot| {
            const t1 = try drained(io, x);
            try ic.snap.restore(x, slot, k, if (slot == resume_dst) o.poison else null);
            const restore_ms = ms(try drained(io, x) - t1, 1);
            const h = try warmRun(x, o, seq, slot, c, dec);
            const same = h == cold;
            pass = pass and same;
            runs += 1;
            std.debug.print("  resume at {d} into slot {d}{s}: {d:.1} MiB, save {d:.2} ms, restore {d:.2} ms{s}; decode logits xxh64 {x:0>16} {s}\n", .{ c, slot, if (slot == resume_dst) " (poisoned)" else " (its source)", @as(f64, @floatFromInt(k.bytes)) / (1 << 20), save_ms, restore_ms, if (slot == resume_dst and o.poison != null) " with the poisoning" else "", h, if (same) "equal" else "DIFFERS" });
        }
    }
    std.debug.print("{s} resume: {d} warm runs at {d} cuts (layers 0..{d}, W={d}, prompt {d}, {d} steps{s}{s}): every decode logits xxh64 {s} the check's {x:0>16}\n", .{ if (pass) "PASS" else "FAIL", runs, cuts.len, e.layers - 1, o.world, T, S, if (x.eager) ", eager" else "", if (e.drafter != null) ", drafter loaded" else "", if (pass) "equals" else "NOT ALL EQUAL TO", cold });
    return pass;
}

/// A restored slot standing at `c`: the prompt's rest [c, T) and S - 1 greedy decode steps; the decode logits'
/// xxh64 (the first token's and every step's, as the check hashes them).
fn warmRun(x: ic.follow.Leader, o: Opts, seq: []const u32, slot: u32, c: usize, dec: []f32) !u64 {
    const e = x.e;
    const T = o.prompt;
    const S = o.steps;
    const U = e.vocab();
    _ = try prefillRange(x, slot, seq, c, T, true);
    @memcpy(dec[0..U], try e.logits());
    for (0..S - 1) |k| {
        const tok = [1]u32{argmax(dec[k * U ..][0..U])};
        try x.forward(slot, T + k, &tok, .{ .lo = 0, .n = 1 });
        @memcpy(dec[(k + 1) * U ..][0..U], try e.logits());
    }
    return std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(dec[0 .. S * U]));
}

/// DSpark after a resume (see `resume-draft` in the usage): for each mode, a cold run on slot 6, then a run resumed at
/// each cut; each must draw the plain tokens and repeat the cold run's rounds and proposals.
fn resumeDraft(gpa: std.mem.Allocator, io: std.Io, x: ic.follow.Leader, o: Opts, seq: []const u32) !bool {
    const e = x.e;
    const T = o.prompt;
    try resumeArena(x, o);
    var buf: [max_cuts]usize = undefined;
    const cuts = try o.cutList(o.draft_cuts, &.{ 1000, T - 1 }, &buf);
    var pass = true;
    for ([_]DraftMode{ .real, .deferred }) |mode| {
        const cl = try prefillRange(x, resume_draft, seq, 0, T, true);
        const cold = try draftRounds(gpa, io, x, o, seq, mode, resume_draft, argmax(try e.logits()), cl);
        printDraft(cold, mode, o, ", cold");
        pass = pass and cold.pass;
        // each cut saved where a forward ended there (stage 1), then captured inside a chunk (stage 1b)
        for (cuts) |c| for ([_]bool{ false, true }) |in_chunk| {
            var k: ic.snap.Kept = undefined;
            if (in_chunk) {
                const m = [1]u32{@intCast(c)};
                try prefillMarks(x, resume_src, seq, T, engine.prompt_rows, &m, (&k)[0..1]);
            } else {
                _ = try prefillRange(x, resume_src, seq, 0, c, false);
                try ic.snap.save(x, resume_src, c, &k);
            }
            defer ic.snap.drop(x, &k);
            try ic.snap.restore(x, resume_draft, k, o.poison);
            const last = try prefillRange(x, resume_draft, seq, c, T, true);
            const first = argmax(try e.logits());
            const r = try draftRounds(gpa, io, x, o, seq, mode, resume_draft, first, last);
            var what: [64]u8 = undefined;
            const how = if (in_chunk) "captured" else "saved";
            printDraft(r, mode, o, try std.fmt.bufPrint(&what, ", resumed at {d} ({s})", .{ c, how }));
            const same = r.pass and r.same(cold);
            pass = pass and same;
            std.debug.print("{s} resume-draft ({s}) at {d} ({s}): drafted equals plain {s}, rounds and proposals {s} the cold run's ({d} rounds, {d} drawn)\n", .{ if (same) "PASS" else "FAIL", @tagName(mode), c, how, if (r.pass) "yes" else "NO", if (r.same(cold)) "equal" else "DIFFER FROM", cold.rounds, cold.made_all });
        };
    }
    return pass;
}

/// Batched decode after resumes (see `resume-streams` in the usage): slot 4 saves a state at each stream's cut, slot
/// 8 + j restores stream j's (poisoned first) and prefills [c_j, T); then all streams' S - 1 decode steps run
/// together in forwards of one row a stream (pass G: graphs from the key's second sighting; pass E: every forward
/// eager, the slots restored again). Every stream's decode logits xxh64 must equal the cold check's.
fn resumeStreams(gpa: std.mem.Allocator, x: ic.follow.Leader, o: Opts, seq: []const u32, cold: u64) !bool {
    const e = x.e;
    const T = o.prompt;
    const S = o.steps;
    const U = e.vocab();
    const n = o.streams;
    const first: u32 = 8;
    const pick = [_]usize{ T - 1, 2049, 2048, 1000, 1, 1024, 3, 2047 };
    var cuts: [engine.max_segs]usize = undefined;
    for (0..n) |j| cuts[j] = @max(1, @min(pick[j], T - 1));
    try resumeArena(x, o);
    var kept: [engine.max_segs]ic.snap.Kept = undefined;
    var nk: usize = 0;
    defer for (kept[0..nk]) |*k| ic.snap.drop(x, k);
    for (cuts[0..n]) |c| {
        _ = try prefillRange(x, resume_src, seq, 0, c, false);
        try ic.snap.save(x, resume_src, c, &kept[nk]);
        nk += 1;
    }
    const dec = try gpa.alloc(f32, n * S * U);
    defer gpa.free(dec);
    var pass = true;
    for ([_]bool{ false, true }) |eager| {
        var xp = x;
        xp.eager = eager;
        const c0 = e.counts;
        for (0..n) |j| {
            const slot = first + @as(u32, @intCast(j));
            try ic.snap.restore(x, slot, kept[j], o.poison);
            _ = try prefillRange(x, slot, seq, cuts[j], T, true);
            @memcpy(dec[j * S * U ..][0..U], try e.logits());
        }
        for (0..S - 1) |k| {
            var segs: [engine.max_segs]engine.Segment = undefined;
            var ids: [engine.max_segs]u32 = undefined;
            for (0..n) |j| {
                segs[j] = .{ .slot = first + @as(u32, @intCast(j)), .pos0 = T + k, .rows = 1 };
                ids[j] = argmax(dec[(j * S + k) * U ..][0..U]);
            }
            try xp.forwardMany(segs[0..n], ids[0..n], .{ .lo = 0, .n = n });
            const lg = try e.logits();
            for (0..n) |j| @memcpy(dec[(j * S + k + 1) * U ..][0..U], lg[j * U ..][0..U]);
        }
        const c = e.counts;
        for (0..n) |j| {
            const h = std.hash.XxHash64.hash(0, std.mem.sliceAsBytes(dec[j * S * U ..][0 .. S * U]));
            const same = h == cold;
            pass = pass and same;
            std.debug.print("  resume-streams pass {c}, slot {d} resumed at {d}: decode logits xxh64 {x:0>16} {s}\n", .{ @as(u8, if (eager) 'E' else 'G'), first + j, cuts[j], h, if (same) "equal" else "DIFFERS" });
        }
        std.debug.print("  resume-streams pass {c}: {d} batched forwards, {d} as graphs ({d} captured), {d} eager\n", .{ @as(u8, if (eager) 'E' else 'G'), S - 1, c.graph - c0.graph, c.captured - c0.captured, c.eager - c0.eager });
    }
    std.debug.print("{s} resume-streams: {d} streams resumed at different cuts, decoded together (graphs, then eager): every decode logits xxh64 {s} the check's {x:0>16}\n", .{ if (pass) "PASS" else "FAIL", n, if (pass) "equals" else "NOT ALL EQUAL TO", cold });
    return pass;
}

/// The round-price table: verify windows of 1..8 rows (forward + head logits of every row, then keep 1), N each,
/// and N drafts; slot 0 continues past its decode (positions advance one a window).
fn price(io: std.Io, x: ic.follow.Leader, o: Opts, reps: usize, seq: []const u32) !void {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const slot: u32 = 0;
    var pos = e.pos(slot);
    var rows: [ic.dspark.block]u32 = undefined;
    for (&rows, 0..) |*r, i| r.* = seq[(pos + i) % seq.len];
    // warm-up
    for (1..ic.dspark.block + 1) |R| {
        try x.forward(slot, pos, rows[0..R], .{ .lo = 0, .n = R });
        _ = try e.logits();
        try x.keep(slot, 1);
        pos += 1;
    }
    std.debug.print("round price (ms), {d} reps, layers 0..{d}, W={d}: rows forward+head keep\n", .{ reps, e.layers - 1, o.world });
    // TF_INK2_ATTN_AB=1: windows at odd positions take the previous attention kernels, at even ones the row passes
    const attn_ab = e.k.attn.ab;
    if (attn_ab) std.debug.print("round price, TF_INK2_ATTN_AB=1: rows forward+head (previous attention kernels / row passes, the windows alternating)\n", .{});
    for (1..ic.dspark.block + 1) |R| {
        var fwd_ns: i96 = 0;
        var keep_ns: i96 = 0;
        var by: [2]i96 = .{ 0, 0 };
        var nby: [2]usize = .{ 0, 0 };
        for (0..reps) |_| {
            var t0 = clock.now(io).toNanoseconds();
            try x.forward(slot, pos, rows[0..R], .{ .lo = 0, .n = R });
            _ = try e.logits();
            const dt = clock.now(io).toNanoseconds() - t0;
            fwd_ns += dt;
            by[pos % 2] += dt;
            nby[pos % 2] += 1;
            t0 = clock.now(io).toNanoseconds();
            try x.keep(slot, 1);
            try e.stream.synchronize();
            keep_ns += clock.now(io).toNanoseconds() - t0;
            pos += 1;
        }
        std.debug.print("  {d} {d:.3} {d:.3}\n", .{ R, ms(fwd_ns, reps), ms(keep_ns, reps) });
        if (attn_ab) std.debug.print("  attn-ab {d} prev {d:.3} rows {d:.3} diff {d:.3}\n", .{ R, ms(by[1], @max(nby[1], 1)), ms(by[0], @max(nby[0], 1)), ms(by[0], @max(nby[0], 1)) - ms(by[1], @max(nby[1], 1)) });
    }
    var props: [ic.dspark.drafts]u32 = undefined;
    var dn: i96 = 0;
    for (0..reps) |_| {
        try x.forward(slot, pos, rows[0..1], .{ .lo = 0, .n = 1 });
        _ = try e.logits();
        const t0 = clock.now(io).toNanoseconds();
        try x.draft(slot, 1, rows[1], &props);
        dn += clock.now(io).toNanoseconds() - t0;
        pos += 1;
    }
    std.debug.print("  draft {d:.3}\n", .{ms(dn, reps)});
}

fn ms(ns: i96, n: usize) f64 {
    return @as(f64, @floatFromInt(ns)) * 1e-6 / @as(f64, @floatFromInt(@max(n, 1)));
}

/// fuse (--ab N steps): after the check slots 0 and 1 hold the same positions (slot 0: the prompt and the decode steps; slot 1:
/// the whole sequence), so N more greedy steps run on both from the last drawn token, slot 0 with the fused ops and
/// slot 1 without (forward.unfused_state), in alternating order; each step's logits must be the same bits. The first
/// two steps capture slot 1's graph and are not timed. Times: each variant's mean and median, and the paired
/// difference (unfused - fused, the two steps of one round) with its standard error.
fn abFuse(gpa: std.mem.Allocator, io: std.Io, x: ic.follow.Leader, o: Opts, seq: []const u32) !bool {
    const e = x.e;
    const clock = std.Io.Clock.awake;
    const U = e.vocab();
    const L = o.prompt + o.steps - 1;
    if (e.pos(0) != L or e.pos(1) != L) return error.AbSlotsApart;
    const rows = try gpa.alloc(f32, 2 * U);
    defer gpa.free(rows);
    const t = try gpa.alloc(f64, 2 * o.fuse_steps);
    defer gpa.free(t);
    const warm = 2;
    var same: usize = 0;
    var tok = seq[L];
    for (0..o.fuse_steps + warm) |i| {
        const first: u32 = @intCast(i % 2);
        for ([2]u32{ first, 1 - first }) |slot| {
            const ids = [1]u32{tok};
            const t0 = clock.now(io).toNanoseconds();
            try x.forward(slot, L + i, &ids, .{ .lo = 0, .n = 1 });
            const lg = try e.logits();
            if (i >= warm) t[slot * o.fuse_steps + i - warm] = ms(clock.now(io).toNanoseconds() - t0, 1);
            @memcpy(rows[slot * U ..][0..U], lg[0..U]);
        }
        if (i >= warm) same += @intFromBool(std.mem.eql(u8, std.mem.sliceAsBytes(rows[0..U]), std.mem.sliceAsBytes(rows[U..])));
        tok = argmax(rows[0..U]);
    }
    const pass = same == o.fuse_steps;
    const n: f64 = @floatFromInt(o.fuse_steps);
    var sum: [2]f64 = .{ 0, 0 };
    var dsum: f64 = 0;
    var dsq: f64 = 0;
    for (0..o.fuse_steps) |i| {
        sum[0] += t[i];
        sum[1] += t[o.fuse_steps + i];
        const d = t[o.fuse_steps + i] - t[i];
        dsum += d;
        dsq += d * d;
    }
    const dmean = dsum / n;
    const se = if (o.fuse_steps > 1) @sqrt(@max(dsq / n - dmean * dmean, 0) / (n - 1)) else 0;
    const med0 = median(t[0..o.fuse_steps]);
    const med1 = median(t[o.fuse_steps..]);
    const v = if (std.c.getenv("TF_INK2_FUSE")) |p| std.mem.span(p) else "lat";
    std.debug.print("{s} fuse A/B: {d} steps on slot 0 (TF_INK2_FUSE={s}) and slot 1 (the release's ops), logits bit-equal {d}/{d}; ms a step slot 0 {d:.3} (median {d:.3}), slot 1 {d:.3} (median {d:.3}); saved {d:.3} +- {d:.3} ms (paired mean +- standard error), medians {d:.3}\n", .{ if (pass) "PASS" else "FAIL", o.fuse_steps, v, same, o.fuse_steps, sum[0] / n, med0, sum[1] / n, med1, dmean, se, med1 - med0 });
    return pass;
}

/// The median of `xs` (sorted in place).
fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const m = xs.len / 2;
    return if (xs.len % 2 == 1) xs[m] else (xs[m - 1] + xs[m]) / 2;
}
