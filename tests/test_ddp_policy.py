import unittest

import loomformer as lf


class DDPPolicyTests(unittest.TestCase):
    def test_static_graph_requires_eager_single_accumulation(self):
        eager = lf.Config(
            grad_accum_steps=1,
            compile=False,
            tria_carry_enabled=True,
            tria_temporal_enabled=True,
        )
        self.assertEqual(lf.ddp_static_graph_policy(eager), (True, ""))

        accumulated = lf.Config(
            grad_accum_steps=2,
            compile=False,
            tria_carry_enabled=True,
            tria_temporal_enabled=True,
        )
        self.assertEqual(
            lf.ddp_static_graph_policy(accumulated),
            (False, "grad_accum_steps > 1 needs no_sync"),
        )

        compiled = lf.Config(
            grad_accum_steps=1,
            compile=True,
            tria_carry_enabled=True,
            tria_temporal_enabled=True,
        )
        self.assertEqual(
            lf.ddp_static_graph_policy(compiled),
            (False, "compiled depth-replay eager island"),
        )


if __name__ == "__main__":
    unittest.main()
