"""Feature distance must remain finite when a small crop has no thing pixels."""
import os
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'uv/.cache/matplotlib'))
sys.path.insert(0, str(ROOT / 'seg'))

import torch
from mmseg.models.uda.dacs import DACS


class FeatureDistanceTests(unittest.TestCase):
    def test_empty_mask_has_zero_loss_and_gradient(self):
        student = torch.ones(1, 2, 2, 2, requires_grad=True)
        reference = torch.zeros_like(student)
        mask = torch.zeros(1, 1, 2, 2, dtype=torch.bool)
        loss = DACS.masked_feat_dist(None, student, reference, mask)
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertTrue(torch.equal(student.grad, torch.zeros_like(student)))

    def test_nonempty_mask_preserves_mean_l2_distance(self):
        student = torch.ones(1, 2, 2, 2, requires_grad=True)
        reference = torch.zeros_like(student)
        mask = torch.ones(1, 1, 2, 2, dtype=torch.bool)
        loss = DACS.masked_feat_dist(None, student, reference, mask)
        self.assertAlmostEqual(loss.item(), 2 ** 0.5, places=6)
        loss.backward()
        self.assertTrue(torch.isfinite(student.grad).all())


if __name__ == '__main__':
    unittest.main()
