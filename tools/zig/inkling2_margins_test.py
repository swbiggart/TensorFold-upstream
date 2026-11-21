"""inkling2_margins.py's top-1 agreement: the reply token against the oracle's greedy pick (torch's argmax, its tie
winner the first maximal ID), not against membership of the maximum. Needs torch and safetensors (CPU is enough)."""
import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inkling2_margins as M  # noqa: E402


class AgreementTests(unittest.TestCase):
    def test_tie_at_the_maximum_agrees_only_with_the_first_id(self):
        # softcapped logits of IDs 0 and 1 both 27.760387, the others smaller, reply token 1
        lg = torch.full((1, 8), 3.0)
        lg[0, 0] = lg[0, 1] = 27.760387
        picks, agree, at_max = M.agreement(lg, [1])
        self.assertEqual(picks, [0])
        self.assertEqual(agree, 0)
        self.assertEqual(at_max, 1)
        self.assertEqual(M.agreement(lg, [0]), ([0], 1, 1))

    def test_rows_with_one_maximum(self):
        lg = torch.tensor([[0.5, 2.0, 1.0], [4.0, -1.0, 3.5], [1.0, 1.0, 1.25]])
        self.assertEqual(M.agreement(lg, [1, 2, 2]), ([1, 0, 2], 2, 2))
        self.assertEqual(M.agreement(lg, [0, 0, 0]), ([1, 0, 2], 1, 1))


if __name__ == "__main__":
    unittest.main()
