import unittest

import graph_helper
import loomformer as lf


class GraphRequirementTests(unittest.TestCase):
    def test_disabled_tria_ops_are_not_reported_missing(self):
        cfg = lf.Config(
            model_dim=24,
            head_dim=4,
            graph=True,
            compile=True,
            tria_carry_enabled=False,
            tria_temporal_enabled=False,
            use_cuda_tria=False,
            use_cuda_phase_sin=False,
            use_cuda_beta_space=False,
            use_cuda_pvpowlu=False,
            use_cuda_depth_attn=False,
        )
        lf.apply_config(cfg)
        _registered, missing, inactive = graph_helper.registration_summary()
        tria_ops = {
            "tria_init", "tria_init_gate", "tria_step", "tria_step_gate",
            "gate_slot_mix", "slot_attention_pool", "final_ca_sparse",
            "temporal_carry",
        }
        self.assertTrue(tria_ops.isdisjoint(missing))
        self.assertTrue(tria_ops.issubset(set(inactive)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
