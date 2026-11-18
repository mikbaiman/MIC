"""Regressions for domain-adaptation heads across feature-pyramid levels."""
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "det"))
import maskrcnn_benchmark

maskrcnn_benchmark.__path__.insert(0, str(ROOT / "uv/det/build/lib/maskrcnn_benchmark"))
from maskrcnn_benchmark.modeling.da_heads.da_heads import DAInsHead


class DomainHeadTests(unittest.TestCase):
    def make_head(self):
        head = DAInsHead(2).eval()
        with torch.no_grad():
            for level in range(4):
                for layer in (1, 2, 3):
                    module = getattr(head, "da_ins_fc{}_level{}".format(layer, level))
                    module.weight.zero_()
                    module.bias.fill_(level + 1 if layer == 3 else 0)
        return head

    def test_all_levels_produce_outputs_and_gradients(self):
        head = self.make_head()
        result = head(torch.zeros(4, 2), torch.arange(4))
        torch.testing.assert_close(result[:, 0], torch.arange(1., 5.))
        result.sum().backward()
        for level in range(4):
            bias = getattr(head, "da_ins_fc3_level{}".format(level)).bias
            self.assertIsNotNone(bias.grad)
            self.assertEqual(bias.grad.item(), 1)

    def test_later_levels_work_without_level_zero_proposals(self):
        head = self.make_head()
        result = head(torch.zeros(2, 2), torch.tensor([2, 3]))
        torch.testing.assert_close(result[:, 0], torch.tensor([3., 4.]))


if __name__ == "__main__":
    unittest.main()
