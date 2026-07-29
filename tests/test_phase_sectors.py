import unittest

import torch
import torch.nn.functional as F

import loomformer as lf


def _sector_reference(
    ffn: lf.ParaplexFFN,
    u: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    sectors: str,
) -> torch.Tensor:
    """Direct compact definition: one independent GEMM per query-head group."""
    outputs = []
    rows_per_head = lf.HIDDEN_PER_Q_HEAD
    for head in range(lf.N_Q_HEADS):
        row0 = head * rows_per_head
        row1 = row0 + rows_per_head
        if sectors == "head":
            visible = torch.cat(
                (q[:, :, head], k[:, :, head], c[:, :, head], u, d[:, :, head]),
                dim=-1,
            )
        else:
            visible = torch.cat(
                (
                    q[:, :, head],
                    k.flatten(2),
                    c.flatten(2),
                    u,
                    d.flatten(2),
                ),
                dim=-1,
            )
        outputs.append(F.linear(visible, ffn.w1_imag[row0:row1]))
    return torch.cat(outputs, dim=-1)


def _inputs(device: torch.device, dtype: torch.dtype):
    shape = (2, 3, lf.N_Q_HEADS, lf.HEAD_DIM)
    return (
        torch.randn(2, 3, lf.N, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True),
        torch.randn(shape, device=device, dtype=dtype, requires_grad=True),
    )


class PhaseSectorParityTests(unittest.TestCase):
    @staticmethod
    def _configure(sectors: str, *, cuda_shape: bool = False) -> None:
        cfg = lf.Config(
            model_dim=16 if cuda_shape else 24,
            n_q_heads=4 if cuda_shape else 6,
            head_dim=4,
            n_kv_heads=2 if cuda_shape else 3,
            hidden=32 if cuda_shape else 66,
            layers=1,
            phase_sectors=sectors,
            use_cuda_beta_space=True,
        )
        lf.apply_config(cfg)

    def test_dense_expansion_matches_compact_semantics_and_gradients(self):
        for sectors in ("head", "open"):
            with self.subTest(sectors=sectors):
                self._configure(sectors)
                ffn = lf.ParaplexFFN().double()
                actual_inputs = _inputs(torch.device("cpu"), torch.float64)
                reference_inputs = tuple(
                    value.detach().clone().requires_grad_() for value in actual_inputs
                )
                reference_ffn = lf.ParaplexFFN().double()
                reference_ffn.load_state_dict(ffn.state_dict())

                actual = ffn._beta_space(*actual_inputs)
                expected = _sector_reference(
                    reference_ffn, *reference_inputs, sectors)
                torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

                probe = torch.randn_like(actual)
                actual.backward(probe)
                expected.backward(probe)
                for got, want in zip(actual_inputs, reference_inputs):
                    torch.testing.assert_close(
                        got.grad, want.grad, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(
                    ffn.w1_imag.grad,
                    reference_ffn.w1_imag.grad,
                    rtol=1e-12,
                    atol=1e-12,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_compact_kernel_matches_sector_reference_forward_and_backward(self):
        for sectors in ("head", "open"):
            with self.subTest(sectors=sectors):
                self._configure(sectors, cuda_shape=True)
                ffn = lf.ParaplexFFN().cuda()
                actual_inputs = _inputs(torch.device("cuda"), torch.float32)
                reference_inputs = tuple(
                    value.detach().clone().requires_grad_() for value in actual_inputs
                )
                reference_ffn = lf.ParaplexFFN().cuda()
                reference_ffn.load_state_dict(ffn.state_dict())

                actual = ffn._beta_space(*actual_inputs)
                expected = _sector_reference(
                    reference_ffn, *reference_inputs, sectors)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

                probe = torch.randn_like(actual)
                actual.backward(probe)
                expected.backward(probe)
                for got, want in zip(actual_inputs, reference_inputs):
                    torch.testing.assert_close(
                        got.grad, want.grad, rtol=3e-5, atol=3e-5)
                torch.testing.assert_close(
                    ffn.w1_imag.grad,
                    reference_ffn.w1_imag.grad,
                    rtol=3e-5,
                    atol=3e-5,
                )


if __name__ == "__main__":
    unittest.main()
