from pathlib import Path
import tempfile
import unittest

import loomformer as lf
import yaml
from tests.attention_matrix import attention_cases


PT_TEMPLATE = Path(__file__).with_name("test_pt.yaml")


class AttentionMatrixConfigTests(unittest.TestCase):
    def test_matrix_is_complete_unique_and_parses_as_real_configs(self):
        cases = attention_cases()
        self.assertEqual(len(cases), 30)
        signatures = {
            (
                None
                if case["overrides"]["attn_layers"] is None
                else tuple(case["overrides"]["attn_layers"]),
                case["overrides"]["attn_token_stride"],
                case["overrides"]["attn_token_schedule"],
                case["overrides"]["grad_checkpointing"],
            )
            for case in cases
        }
        self.assertEqual(len(signatures), 30)

        layer_modes = (None, (1, 3), (1, 2))
        token_modes = (
            (1, "shared"),
            (2, "shared"),
            (2, "staggered"),
            (3, "shared"),
            (3, "staggered"),
        )
        expected = {
            (layers, stride, schedule, checkpoint)
            for layers in layer_modes
            for stride, schedule in token_modes
            for checkpoint in (False, True)
        }
        self.assertEqual(signatures, expected)

        with tempfile.TemporaryDirectory() as directory:
            for case in cases:
                with self.subTest(case=case["name"]):
                    source = yaml.safe_load(
                        PT_TEMPLATE.read_text(encoding="utf-8"))
                    source.update(case["overrides"])
                    path = Path(directory) / f"{case['name']}.yaml"
                    path.write_text(
                        yaml.safe_dump(source, sort_keys=False),
                        encoding="utf-8",
                    )
                    cfg = lf.Config.from_yaml(str(path))
                    lf.apply_config(cfg)
                    expected_layers = case["overrides"]["attn_layers"]
                    if expected_layers is None:
                        expected_layers = [1, 2, 3, 4]
                    self.assertEqual(cfg.attn_layers, expected_layers)
                    self.assertEqual(
                        cfg.attn_token_stride,
                        case["overrides"]["attn_token_stride"],
                    )
                    self.assertEqual(
                        cfg.attn_token_schedule,
                        case["overrides"]["attn_token_schedule"],
                    )
                    self.assertEqual(
                        cfg.grad_checkpointing,
                        case["overrides"]["grad_checkpointing"],
                    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
