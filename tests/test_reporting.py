import contextlib
import io
import math
import unittest

import torch

import loomformer as lf


class ReportingTests(unittest.TestCase):
    def test_sft_eval_omits_bytes_metric(self):
        line = lf.format_eval_status(1, 2.0, 2.0 / math.log(2.0), float("nan"))
        self.assertIn("bit/tok:", line)
        self.assertNotIn("bpb:", line)

    def test_unknown_dataset_size_omits_data_token_scale(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            lf.print_training_scale(
                run_tokens=1024,
                data_tokens=0,
                cumulative_target_tokens=1024,
                model=torch.nn.Linear(2, 2),
            )
        self.assertIn("tok/param", output.getvalue())
        self.assertNotIn("data-tok/param", output.getvalue())


if __name__ == "__main__":
    unittest.main()
