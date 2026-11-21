"""recipe/inkling2-tp4/smoke.py's token gate (the yardstick: the server forced along the oracle's replies, scored
against the oracle's top-1 and top-2 over 445 positions), its forcing client against a stand-in /v1/completions, the
free-running departures (information), and tools/zig/inkling2_tokengate.py and inkling2_tolerance.py --yardstick
offline. Standard library only (python3 -I -m pytest tools/tests/test_inkling2_tokengate.py, or -m unittest)."""

import contextlib
import io
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "zig"))
import inkling2_tokengate as D  # noqa: E402
import inkling2_tolerance as T  # noqa: E402

S = D.smoke
EOS = 200006
N, TOP1, TOP2 = S.YARDSTICK_POSITIONS, S.YARDSTICK_TOP1, S.YARDSTICK_TOP2


def row(top1, top2, l1, gap, pre=None):
    return {"top1": top1, "top2": top2, "top1_logit": l1, "top2_logit": None if gap is None else l1 - gap, "gap": gap,
            "gap_pre": pre, "step": D.bf16_spacing(l1)}


def entry(ids, margins=None):
    e = {"ids": ids, "text": ""}
    if margins is not None:
        e["margins"] = margins
    return e


def clear(ids, logit=20.0, gap=3.0):
    """Margins far from any tie, the oracle's picks first and token 1 second."""
    return [row(t, 1, logit, gap) for t in ids]


# the gate's shape: chat_short 11, tool_call 48 and 7 extended prompts, 445 positions
SIZES = {"chat_short": 11, "tool_call": 48, "ext_json": 64, "ext_german": 64, "ext_long": 26, "ext_thinking": 64,
         "ext_code": 64, "ext_tool": 64, "ext_multi_turn": 40}


def gate_set():
    """{name: (prompt ids, oracle entry)} with 445 positions; every reply token distinct, runner-up 1."""
    out, base = {}, 1000
    for i, (name, n) in enumerate(SIZES.items()):
        ref = list(range(base, base + n))
        base += n
        out[name] = ([10 + i, 50 + i], entry(ref, clear(ref)))
    return out


def forced_score(off_top1_in_top2=0, outside_top2=0):
    """A forced run over the gate set with that many positions at the oracle's runner-up and outside its top-2."""
    gs = gate_set()
    rows, k = [], 0
    for name, (_, e) in gs.items():
        f = list(e["ids"])
        for j in range(len(f)):
            if k < off_top1_in_top2:
                f[j] = 1  # the runner-up
            elif k < off_top1_in_top2 + outside_top2:
                f[j] = 7  # not in the oracle's top-2
            k += 1
        rows.append((name, e["ids"], e["margins"], f))
    return S.yardstick_score(rows)


class Yardstick(unittest.TestCase):
    def test_bars(self):
        self.assertEqual((N, TOP1, TOP2), (445, 438, 443))
        self.assertEqual(sum(SIZES.values()), N)

    def test_all_equal_passes(self):
        sc = forced_score()
        self.assertEqual((sc["positions"], sc["top1"], sc["top2"]), (445, 445, 445))
        self.assertTrue(S.yardstick_verdict(sc)[0])

    def test_top1_at_just_over_just_under(self):
        for off, ok in ((7, True), (6, True), (8, False)):  # 438 (at the bar), 439 (just over), 437 (just under)
            sc = forced_score(off_top1_in_top2=off)
            self.assertEqual((sc["top1"], sc["top2"]), (445 - off, 445))
            passed, line = S.yardstick_verdict(sc)
            self.assertEqual(passed, ok, line)
        self.assertIn("YARDSTICK: top-1 437/445 (< 438), in the oracle's top-2 445/445 (>= 443): FAIL",
                      S.yardstick_verdict(forced_score(off_top1_in_top2=8))[1])

    def test_top2_at_just_over_just_under(self):
        for out, ok in ((2, True), (1, True), (3, False)):  # 443 (at the bar), 444 (just over), 442 (just under)
            sc = forced_score(outside_top2=out)
            self.assertEqual((sc["top1"], sc["top2"]), (445 - out, 445 - out))
            self.assertEqual(S.yardstick_verdict(sc)[0], ok)
        line = S.yardstick_verdict(forced_score(outside_top2=3))[1]
        self.assertIn("in the oracle's top-2 442/445 (< 443): FAIL", line)

    def test_mixed_at_both_bars(self):
        sc = forced_score(off_top1_in_top2=5, outside_top2=2)  # 438 top-1, 443 top-2
        self.assertEqual((sc["top1"], sc["top2"]), (438, 443))
        self.assertTrue(S.yardstick_verdict(sc)[0])
        sc = forced_score(off_top1_in_top2=4, outside_top2=3)  # 438 top-1, 442 top-2
        self.assertFalse(S.yardstick_verdict(sc)[0])

    def test_a_token_outside_the_top_2(self):
        m = row(179692, 220, 27.796, 0.1086)
        self.assertEqual(S.yardstick_row(m, 179692, 179692), (True, True))
        self.assertEqual(S.yardstick_row(m, 179692, 220), (False, True))
        self.assertEqual(S.yardstick_row(m, 179692, 13), (False, False))
        lines = []
        sc = S.yardstick_score([("tool_call", [5, 179692], [row(5, 1, 20.0, 3.0), m], [5, 13])])
        S.yardstick_lines(sc, lines.append)
        self.assertEqual(lines, ["  tool_call j=1: server 13, oracle 179692 (outside the oracle's top-2; the oracle's "
                                 "runner-up is 220, not the server's token)"])
        sc = S.yardstick_score([("tool_call", [5, 179692], [row(5, 1, 20.0, 3.0), m], [5, 220])])
        lines = []
        S.yardstick_lines(sc, lines.append)
        self.assertEqual(lines, ["  tool_call j=1: server 220, oracle 179692 (in the oracle's top-2; oracle gap "
                                 "0.1086; upstream one bf16 step 0.125: near tie)"])

    def test_exact_tie_at_the_oracle_top_listed_either_way(self):
        # topk may list an exact tie at the top in either order; both tokens are the oracle's top-2
        m = row(151272, 155262, 28.35, 0.0)
        self.assertEqual(S.yardstick_row(m, 155262, 151272), (False, True))
        self.assertEqual(S.yardstick_row(m, 155262, 155262), (True, True))

    def test_other_position_counts_never_pass(self):
        gs = gate_set()
        rows = [(n, e["ids"], e["margins"], e["ids"]) for n, (_, e) in gs.items() if n != "ext_long"]
        sc = S.yardstick_score(rows)
        self.assertEqual(sc["positions"], 419)
        passed, line = S.yardstick_verdict(sc)
        self.assertFalse(passed)
        self.assertIn("the bars are for 445 positions", line)

    def test_refused_inputs(self):
        with self.assertRaises(SystemExit):
            S.yardstick_score([("x", [1, 2], None, [1, 2])])  # no margins
        with self.assertRaises(SystemExit):
            S.yardstick_score([("x", [1, 2], clear([1, 2]), [1])])  # a forced token missing


class Departures(unittest.TestCase):
    """Free-running greedy replies: information with margins, v0.1's strict rule without."""

    def test_equal_and_eos(self):
        ref = [5, 6, EOS]
        self.assertEqual(D.compare(ref, clear(ref), [5, 6, EOS])[:3], ("EQUAL", None, 3))
        self.assertEqual(D.compare(ref, clear(ref), [5, 6])[:3], ("EQUAL", None, 3))  # EOS not returned

    def test_without_margins_strict(self):
        self.assertEqual(D.compare([5, 6, 7], None, [5, 9, 7])[:3], ("DIFFER", 1, 1))
        self.assertEqual(D.compare([5, 6, 7, 8], None, [5, 6])[:3], ("DIFFER", 2, 2))

    def test_with_margins_information(self):
        ref = [5, 6, 7, 8]
        m = clear(ref)
        m[2] = row(7, 9, 27.8, 0.1)
        st, j, n, note = D.compare(ref, m, [5, 6, 9, 4])
        self.assertEqual((st, j, n), ("DEPARTS", 2, 2))
        self.assertIn("oracle gap 0.1000; upstream one bf16 step 0.125: near tie; information", note)
        m[2] = row(7, 9, 27.8, 0.2)
        self.assertIn("upstream one bf16 step 0.125: over", D.compare(ref, m, [5, 6, 9, 4])[3])
        self.assertIn("the oracle's runner-up is 9, not the server's token", D.compare(ref, m, [5, 6, 3])[3])
        self.assertEqual(D.compare(ref, m, [5, 6])[:3], ("DEPARTS", 2, 2))  # stopped early

    def test_verdict_is_information(self):
        v = D.verdict(row(179692, 220, 27.796, 0.146), 179692, 220)
        self.assertEqual((v["upstream_one_step_info"], v["in_oracle_top2"]), (False, True))
        self.assertTrue(v["oracle_gap_under_0p5"])
        v = D.verdict(row(179692, 220, 27.796, 0.072), 179692, 220,
                      server_row={"top1": 220, "top2": 179692, "top1_logit": 27.7604, "gap": 0.0})
        self.assertTrue(v["upstream_two_sided_info"])
        self.assertNotIn("verdict", v)

    def test_check_never_fails_on_a_departure(self):
        cs = list(range(100, 110)) + [EOS]
        tc = list(range(200, 248))
        dec = {"chat_short": entry(cs, clear(cs)), "tool_call": entry(tc, clear(tc))}
        ext = {"ext_a": entry([1, 2, 3], clear([1, 2, 3])), "ext_b": entry([4, 5], clear([4, 5]))}
        srv = {"chat_short": cs, "tool_call": tc[:10] + [220] + tc[11:], "ext_a": [1, 9], "ext_b": [4, 5]}
        lines = []
        r = D.check(dec, srv, ["chat_short", "tool_call"], ext, ["ext_a", "ext_b"], srv, out=lines.append)
        self.assertEqual(r["fails"], 0)
        self.assertEqual(r["results"]["tool_call"], ("DEPARTS", 10, 48))
        self.assertIn("EXT summary: 2 prompts, 1 EQUAL, 1 DEPARTS, 0 DIFFER; 3 of 5 oracle tokens equal before the "
                      "first departure (60.0%)", lines)
        dec["tool_call"] = entry(tc)  # no margins: v0.1's strict rule
        self.assertEqual(D.check(dec, srv, ["tool_call"], out=lines.append)["fails"], 1)


class Offline(unittest.TestCase):
    def test_score(self):
        gs = gate_set()
        dec = {n: e for n, (_, e) in gs.items() if not n.startswith("ext_")}
        ext = {n: e for n, (_, e) in gs.items() if n.startswith("ext_")}
        forced = {n: list(e["ids"]) for n, (_, e) in gs.items()}
        forced["tool_call"][10] = 1
        forced["ext_code"][3] = 7
        lines = []
        r = D.score(dec, forced, ["chat_short", "tool_call"], ext, list(ext), out=lines.append)
        self.assertTrue(r["passed"])
        self.assertEqual(lines[-1], "YARDSTICK: top-1 443/445 (>= 438), in the oracle's top-2 444/445 (>= 443): PASS")
        self.assertIn("tool_call: 48 positions: top-1 47, in the oracle's top-2 48", lines)

    def test_order_from_oracle_data(self):
        a = entry([1] * 30, clear([1] * 30))
        a["margins"][5] = row(1, 2, 20.0, 0.05)  # step 0.125 at 20: near tie at 5
        b = entry([1] * 30, clear([1] * 30))  # none: 30
        c = entry([1] * 40, clear([1] * 40))
        c["margins"][12] = row(1, 2, 20.0, 0.3)  # under 0.5 only
        self.assertEqual(D.order({"ext_a": a, "ext_b": b, "ext_c": c, "_wall_s": 1.0}), ["ext_c", "ext_b", "ext_a"])
        self.assertEqual(D.first_near_tie(c, 0.5), 12)
        with self.assertRaises(SystemExit):
            D.order({"x": entry([1, 2])})

    def test_tolerance_yardstick_counts_ties_both_ways(self):
        def r(tok, top, logits):
            return {"token": tok, "top": top, "top_logits": logits, "argmax": top[0], "margin": logits[0] - logits[1]}
        pos = {}
        for j in range(3):
            pos[("p", j)] = {f: r(10 + j, [10 + j, 1, 2], [9.0, 5.0, 4.0]) for f in T.FORMS}
        pos[("p", 1)][("eager", "tf")] = r(11, [11, 2, 1], [7.0, 7.0, 4.0])  # an exact tie at the top, argmax 11
        pos[("p", 2)][("eager", "tf")] = r(12, [2, 12, 1], [7.0, 6.0, 4.0])  # outside the oracle's top-2
        y = T.yardstick(pos)
        self.assertEqual(y["eager/tf"], {"top1": (1, 2, 2), "top2": (1, 2, 2)})  # (low, argmax, high)
        self.assertEqual(y["fp32/decode"], {"top1": (3, 3, 3), "top2": (3, 3, 3)})


class Stand_in:
    """A /v1/completions stand-in: replies with the oracle's next token after (prompt ids + reply[:j]), except where
    `wrong` says otherwise; records every request body."""

    def __init__(self, prompts, wrong=None, free=None):
        self.prompts, self.wrong, self.free, self.bodies = prompts, wrong or {}, free or {}, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.bodies.append(body)
                p = body["prompt"]
                ids = None
                for name, (pids, e) in outer.prompts.items():
                    if p[: len(pids)] == pids and p[len(pids):] == e["ids"][: len(p) - len(pids)]:
                        j = len(p) - len(pids)
                        ids = ([outer.wrong.get((name, j), e["ids"][j])] if body["max_tokens"] == 1
                               else outer.free.get(name, e["ids"])[: body["max_tokens"]])
                        break
                out = json.dumps({"choices": [{"text": "", "finish_reason": "length"}],
                                  "tensorfold": {"token_ids": ids if ids is not None else []}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def write_release_data(d, gs):
    def put(name, obj):
        with open(os.path.join(d, name), "w") as f:
            json.dump(obj, f)
    put("decode.json", {n: e for n, (_, e) in gs.items() if not n.startswith("ext_")})
    os.mkdir(os.path.join(d, "ext"))
    put("ext/decode.json", {n: e for n, (_, e) in gs.items() if n.startswith("ext_")})
    for n, (pids, _) in gs.items():
        put(("ext/" if n.startswith("ext_") else "") + n + ".ids.json", pids)
    with open(os.path.join(d, "ext", "ORDER"), "w") as f:
        f.write("".join(n + "\n" for n in gs if n.startswith("ext_")))


def run_smoke(argv):
    buf, old = io.StringIO(), sys.argv
    sys.argv = ["smoke.py"] + argv
    try:
        with contextlib.redirect_stdout(buf):
            rc = S.main()
    finally:
        sys.argv = old
    return rc, buf.getvalue()


class SmokeHTTP(unittest.TestCase):
    """smoke.py yardstick and oracle end to end against the stand-in."""

    def run_yardstick(self, wrong):
        gs = gate_set()
        st = Stand_in(gs, wrong)
        try:
            with tempfile.TemporaryDirectory() as d:
                write_release_data(d, gs)
                out = os.path.join(d, "forced.json")
                rc, text = run_smoke([st.url, "yardstick", os.path.join(d, "decode.json"),
                                      "chat_short=" + os.path.join(d, "chat_short.ids.json"),
                                      "tool_call=" + os.path.join(d, "tool_call.ids.json"),
                                      "--ext", os.path.join(d, "ext"), "--json", out])
                with open(out) as f:
                    saved = json.load(f)
                with open(os.path.join(d, "decode.json")) as f:
                    dec = json.load(f)
                lines = []
                offline = D.score(dec, saved["forced"],
                                  ["chat_short", "tool_call"], *D.read_ext(os.path.join(d, "ext")), out=lines.append)
        finally:
            st.close()
        return gs, st, rc, text, saved, offline

    def test_yardstick_requests_are_release_completions(self):
        gs, st, rc, text, saved, offline = self.run_yardstick({("tool_call", 10): 1})
        self.assertEqual(rc, 0, text)
        self.assertEqual(len(st.bodies), 445)
        for b in st.bodies:
            self.assertEqual({k: b[k] for k in ("max_tokens", "temperature", "ignore_eos", "return_token_ids")},
                             {"max_tokens": 1, "temperature": 0, "ignore_eos": True, "return_token_ids": True})
            self.assertEqual(set(b), {"model", "prompt", "max_tokens", "temperature", "ignore_eos", "return_token_ids"})
        pids, e = gs["tool_call"]
        self.assertIn({"model": "model", "prompt": pids + e["ids"][:10], "max_tokens": 1, "temperature": 0,
                       "ignore_eos": True, "return_token_ids": True}, st.bodies)
        self.assertIn("tool_call: prompt 2 tokens, 48 positions: top-1 47, in the oracle's top-2 48", text)
        self.assertIn("YARDSTICK: top-1 444/445 (>= 438), in the oracle's top-2 445/445 (>= 443): PASS", text)
        self.assertEqual((saved["top1"], saved["top2"], saved["passed"]), (444, 445, True))
        self.assertEqual(saved["forced"]["tool_call"][10], 1)
        self.assertEqual((offline["top1"], offline["top2"], offline["passed"]), (444, 445, True))

    def test_yardstick_fails_just_under(self):
        wrong = {("ext_json", j): 7 for j in range(3)}  # three tokens outside the top-2: 442
        _, _, rc, text, saved, offline = self.run_yardstick(wrong)
        self.assertEqual(rc, 1)
        self.assertIn("in the oracle's top-2 442/445 (< 443): FAIL", text)
        self.assertIn("  ext_json j=0: server 7, oracle 1059 (outside the oracle's top-2;", text)
        self.assertFalse(offline["passed"])

    def test_force_one_refuses_a_missing_token(self):
        st = Stand_in({})
        try:
            with self.assertRaises(SystemExit):
                S.force_one(st.url, [1, 2, 3])
        finally:
            st.close()

    def test_oracle_free_running_is_information(self):
        gs = gate_set()
        _, tc = gs["tool_call"]
        st = Stand_in(gs, free={"tool_call": tc["ids"][:10] + [1] + tc["ids"][11:]})
        try:
            with tempfile.TemporaryDirectory() as d:
                write_release_data(d, gs)
                args = [st.url, "oracle", os.path.join(d, "decode.json"),
                        "chat_short=" + os.path.join(d, "chat_short.ids.json"),
                        "tool_call=" + os.path.join(d, "tool_call.ids.json")]
                rc, text = run_smoke(args + ["--ext", os.path.join(d, "ext")])
                self.assertEqual(rc, 0, text)
                self.assertIn("tool_call: prompt 2 tokens; oracle 48 tokens, server 48 (finish length): DEPARTS at 10",
                              text)
                self.assertIn("EXT summary: 7 prompts, 7 EQUAL, 0 DEPARTS, 0 DIFFER; 386 of 386", text)
                with open(os.path.join(d, "decode.json"), "w") as f:  # v0.1's oracle file: no margins, strict
                    json.dump({n: {"ids": e["ids"], "text": ""} for n, (_, e) in gs.items()}, f)
                rc, text = run_smoke(args)
                self.assertEqual(rc, 1)
                self.assertIn("DIFFER at 10", text)
        finally:
            st.close()


if __name__ == "__main__":
    unittest.main()
