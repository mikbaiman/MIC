"""Regression checks for the optimizer/scheduler used by both classifiers."""
from pathlib import Path
import sys
import unittest
import warnings
import io
import tempfile
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "cls"))
from common.utils.sam import SAM
from common.utils.logger import CompleteLogger


class SchedulerTests(unittest.TestCase):
    def test_schedule_tracks_real_sam_update(self):
        weight = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = SAM([weight], torch.optim.SGD, lr=0.1, rho=0.05)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer.base_optimizer, lambda step: 0.5 ** step)
        self.assertIs(optimizer.param_groups, optimizer.base_optimizer.param_groups)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            for _ in range(2):
                optimizer.zero_grad()
                weight.sum().backward()
                optimizer.first_step(zero_grad=True)
                (2 * weight).sum().backward()
                optimizer.second_step(zero_grad=True)
                scheduler.step()
        self.assertAlmostEqual(weight.item(), 0.7, places=6)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.025)
        self.assertEqual(scheduler.last_epoch, 2)
        self.assertFalse(any("before `optimizer.step()`" in str(w.message) for w in caught))


class LoggerTests(unittest.TestCase):
    def test_close_restores_borrowed_streams(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sys, "stdout", stdout), patch.object(sys, "stderr", stderr):
                logger = CompleteLogger(directory)
                print("training completed")
                logger.close()
                logger.close()
                self.assertIs(sys.stdout, stdout)
                self.assertIs(sys.stderr, stderr)
                self.assertFalse(stdout.closed)
                self.assertFalse(stderr.closed)
                print("output after logger shutdown")
            logs = list(Path(directory).glob("*.txt"))
            self.assertIn("training completed", logs[0].read_text())
            self.assertIn("output after logger shutdown", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
