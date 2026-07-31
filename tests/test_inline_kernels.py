import unittest

import torch

from inline_kernels.capped_residual import capped_residual, capped_residual_reference
from inline_kernels.depth_history import (
    depth_attention,
    depth_history_append,
    depth_history_init,
)
from inline_kernels.fixed_rms import fixed_rms
from inline_kernels.sample_hold import sample_hold


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class InlineKernelParityTests(unittest.TestCase):
    def test_depth_history_arena_forward_backward(self):
        for dtype, atol in ((torch.float32, 5e-6), (torch.float16, 3e-3),
                            (torch.bfloat16, 2e-2)):
            batch, tokens, heads, dim, count = 2, 3, 4, 8, 6
            q = torch.randn(
                heads, dim, device="cuda", dtype=dtype, requires_grad=True)
            keys = [
                torch.randn(
                    batch, tokens, heads, dim, device="cuda",
                    dtype=dtype, requires_grad=True)
                for _ in range(count)
            ]
            values = [
                torch.randn_like(keys[0], requires_grad=True)
                for _ in range(count)
            ]
            state = depth_history_init(keys[0], values[0], count)
            outputs = []
            for index in range(1, count):
                outputs.append(depth_attention(q, state))
                state = depth_history_append(
                    state, keys[index], values[index])
            outputs.append(depth_attention(q, state))
            output_grads = [torch.randn_like(x) for x in outputs]
            sum(
                (x * grad).sum()
                for x, grad in zip(outputs, output_grads)
            ).backward()
            got_grads = [
                q.grad.detach(), *[x.grad.detach() for x in keys],
                *[x.grad.detach() for x in values],
            ]

            qr = q.detach().clone().requires_grad_()
            kr = [x.detach().clone().requires_grad_() for x in keys]
            vr = [x.detach().clone().requires_grad_() for x in values]
            references = []
            for length in range(1, count + 1):
                key = torch.stack(kr[:length], dim=2)
                value = torch.stack(vr[:length], dim=2)
                score = (
                    qr.view(1, 1, heads, dim).float().unsqueeze(2)
                    * key.float()
                ).sum(-1) / dim ** .5
                probability = score.softmax(dim=2)
                references.append(
                    (probability[..., None] * value.float()).sum(2).to(dtype))
            sum(
                (x * grad).sum()
                for x, grad in zip(references, output_grads)
            ).backward()
            expected_grads = [qr.grad, *[x.grad for x in kr], *[x.grad for x in vr]]
            for actual, expected in zip(outputs, references):
                torch.testing.assert_close(actual, expected, rtol=0, atol=atol)
            for actual, expected in zip(got_grads, expected_grads):
                torch.testing.assert_close(actual, expected, rtol=0, atol=atol)

    def test_capped_residual_forward_backward(self):
        for dtype, atol in ((torch.float32, 5e-6), (torch.float16, 3e-3),
                            (torch.bfloat16, 2e-2)):
            a = (torch.randn(7, 96, device="cuda", dtype=dtype) * 3).requires_grad_()
            b = (torch.randn(7, 96, device="cuda", dtype=dtype) * .3).requires_grad_()
            grad = torch.randn_like(a)
            out = capped_residual(a, b, 1.0)
            (out * grad).sum().backward()
            got = out.detach(), a.grad.detach(), b.grad.detach()
            ar = a.detach().clone().requires_grad_()
            br = b.detach().clone().requires_grad_()
            ref = capped_residual_reference(ar, br, 1.0)
            (ref * grad).sum().backward()
            expected = ref.detach(), ar.grad.detach(), br.grad.detach()
            for actual, wanted in zip(got, expected):
                torch.testing.assert_close(actual, wanted, rtol=0, atol=atol)

    def test_fixed_rms_group_mapping_and_backward(self):
        for dtype, atol in ((torch.float32, 5e-6), (torch.float16, 3e-3),
                            (torch.bfloat16, 2e-2)):
            x = torch.randn(
                2, 3, 2, 6, 16, device="cuda", dtype=dtype, requires_grad=True)
            targets = torch.tensor((.7, .2), device="cuda")
            grad = torch.randn_like(x)
            out = fixed_rms(x, targets)
            (out * grad).sum().backward()
            got_grad = x.grad.detach()
            xr = x.detach().clone().requires_grad_()
            work = xr.float() if dtype != torch.float32 else xr
            scale = torch.rsqrt(work.square().mean(-1, keepdim=True) + 1e-6)
            ref = xr * (scale * targets.view(1, 1, 2, 1, 1)).to(dtype)
            (ref * grad).sum().backward()
            torch.testing.assert_close(out, ref, rtol=0, atol=atol)
            torch.testing.assert_close(got_grad, xr.grad, rtol=0, atol=atol)

    def test_sample_hold_routes_every_gradient(self):
        batch, tokens, heads, dim = 2, 7, 3, 4
        ranks = torch.tensor(
            ((-1, 0, -1, 1, -1, 2, -1), (-1, 3, -1, 4, -1, 5, -1)),
            device="cuda", dtype=torch.int32)
        selected = [
            torch.randn(6, heads, dim, device="cuda", requires_grad=True)
            for _ in range(3)
        ]
        residual = torch.randn(
            6, heads * dim, device="cuda", requires_grad=True)
        inherited = [
            torch.randn(
                batch, tokens, heads, dim, device="cuda", requires_grad=True)
            for _ in range(3)
        ]
        out = sample_hold(*selected, residual, inherited, ranks)
        grads = [torch.randn_like(x) for x in out]
        sum((x * grad).sum() for x, grad in zip(out, grads)).backward()
        got = [x.grad.detach().clone() for x in (*selected, residual, *inherited)]

        selected_ref = [x.detach().clone().requires_grad_() for x in selected]
        residual_ref = residual.detach().clone().requires_grad_()
        inherited_ref = [x.detach().clone().requires_grad_() for x in inherited]
        mask = ranks.ge(0)
        safe = ranks.flatten().clamp_min(0).long()
        own = [
            x.index_select(0, safe).view(batch, tokens, heads, dim)
            for x in selected_ref
        ]
        choose = mask[..., None, None]
        ref = tuple(
            torch.where(choose, x, y) for x, y in zip(own, inherited_ref))
        own_residual = residual_ref.index_select(0, safe).view(
            batch, tokens, heads * dim)
        ref += (torch.where(
            mask[..., None], own_residual, torch.zeros_like(own_residual)),)
        sum((x * grad).sum() for x, grad in zip(ref, grads)).backward()
        expected = [x.grad for x in (
            *selected_ref, residual_ref, *inherited_ref)]
        for actual, wanted in zip(out, ref):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
        for actual, wanted in zip(got, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
