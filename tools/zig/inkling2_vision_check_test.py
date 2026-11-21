"""inkling2_vision_check.py v5 on small made files: a fixture passes only on nonempty tensors with one shape, one
dtype and one sha256 across its variants, and the default tower rows must be bf16 [T, 4608].
python3 -I inkling2_vision_check_test.py (needs torch and safetensors, as the checker does)"""
import argparse
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_vision_check as vc  # noqa: E402

BF = torch.bfloat16


def rows(n: int, width: int = 4608, dtype=BF) -> torch.Tensor:
    return (torch.arange(n * width, dtype=torch.float32).reshape(n, width) % 251).to(dtype)


class V5(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def file(self, variant: str, t: torch.Tensor, key: str = "rows", fixture: str = "cat.png") -> str:
        p = os.path.join(self.tmp.name, f"{fixture}.{variant}.safetensors")
        save_file({key: t.contiguous()}, p)
        return p

    def v5(self, files, key: str = "rows", variants: str = ""):
        a = argparse.Namespace(files=files, key=key, variants=variants, json="", threads=1)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            code = vc.cmd_v5(a)
        return code, json.loads(out.getvalue())

    def test_equal_tower_rows_pass(self):
        code, rep = self.v5([self.file("rb32", rows(3)), self.file("rb64", rows(3))])
        self.assertEqual((code, rep["verdict"]), (0, "PASS"))

    def test_empty_rows_fail(self):
        code, rep = self.v5([self.file("rb32", rows(0)), self.file("rb64", rows(0))])
        self.assertEqual((code, rep["verdict"]), (1, "FAIL"))
        code, rep = self.v5([self.file("l0", rows(0, 8), key="layer.0"), self.file("l1", rows(0, 8), key="layer.0")],
                            key="layer.0")
        self.assertEqual((code, rep["verdict"]), (1, "FAIL"))

    def test_same_bytes_other_shape_fail(self):
        t = rows(4, 8)
        code, rep = self.v5([self.file("a", t, key="layer.3"), self.file("b", t.reshape(8, 4), key="layer.3")],
                            key="layer.3")
        self.assertEqual((code, rep["verdict"]), (1, "FAIL"))

    def test_same_bytes_other_dtype_fail(self):
        t = rows(4, 8)
        code, rep = self.v5([self.file("a", t, key="layer.3"), self.file("b", t.view(torch.int16), key="layer.3")],
                            key="layer.3")
        self.assertEqual((code, rep["verdict"]), (1, "FAIL"))

    def test_tower_rows_must_be_bf16_at_4608(self):
        f32 = rows(3, dtype=torch.float32)
        code, rep = self.v5([self.file("a", f32), self.file("b", f32)])
        self.assertEqual((code, rep["verdict"]), (1, "FAIL"))
        code, rep = self.v5([self.file("a", rows(3, 4096)), self.file("b", rows(3, 4096))])
        self.assertEqual((code, rep["verdict"]), (1, "FAIL"))
        code, rep = self.v5([self.file("a", rows(3).reshape(-1)), self.file("b", rows(3).reshape(-1))])
        self.assertEqual((code, rep["verdict"]), (1, "FAIL"))

    def test_other_key_any_width(self):
        code, rep = self.v5([self.file("a", rows(2, 8), key="layer.3"), self.file("b", rows(2, 8), key="layer.3")],
                            key="layer.3")
        self.assertEqual((code, rep["verdict"]), (0, "PASS"))


if __name__ == "__main__":
    unittest.main()
