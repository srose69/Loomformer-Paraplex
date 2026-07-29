import unittest
from unittest import mock

import loomformer as lf
import torch


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

    def test_mutable_buffers_use_one_explicit_broadcast(self):
        class TinyFFN(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer(
                    "beta_anchor", torch.tensor(float(value))
                )

        class TinyBlock(torch.nn.Module):
            def __init__(self, value):
                super().__init__()
                self.ffn = TinyFFN(value)

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = torch.nn.ModuleList(
                    [TinyBlock(1.0), TinyBlock(2.0)]
                )

        model = TinyModel()

        def emulate_rank_zero_broadcast(packed, src):
            self.assertEqual(src, 0)
            packed.copy_(torch.tensor([3.0, 4.0]))

        with (
            mock.patch.object(lf.dist, "is_available", return_value=True),
            mock.patch.object(lf.dist, "is_initialized", return_value=True),
            mock.patch.object(
                lf.dist,
                "broadcast",
                side_effect=emulate_rank_zero_broadcast,
            ) as broadcast,
        ):
            lf.ddp_sync_mutable_buffers(model)

        broadcast.assert_called_once()
        self.assertEqual(model.blocks[0].ffn.beta_anchor.item(), 3.0)
        self.assertEqual(model.blocks[1].ffn.beta_anchor.item(), 4.0)


if __name__ == "__main__":
    unittest.main()
