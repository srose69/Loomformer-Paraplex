import unittest
from dataclasses import replace
from unittest import mock

import torch

import loomformer as lf
import tria


def _position_ids(segment_ids: torch.Tensor) -> torch.Tensor:
    batch, length = segment_ids.shape
    out = torch.empty(batch, length, dtype=torch.long, device=segment_ids.device)
    for b in range(batch):
        pos = 0
        for t in range(length):
            if t == 0 or segment_ids[b, t] != segment_ids[b, t - 1]:
                pos = 0
            out[b, t] = pos
            pos += 1
    return out


class PackedAttentionTests(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_packed_gather_matches_reference_forward_and_backward(self):
        seg = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
             [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]],
            dtype=torch.int32,
        )
        layout = lf.packed_layout_from_segment_ids(seg)
        cpu_plan = lf.build_packed_chunk_layout(
            layout, 4, 8, ((0, 4), (4, 8)))
        cuda_plan = cpu_plan.to(torch.device("cuda"))
        for dtype in (torch.float32, torch.float16):
            source = [
                torch.randn(2, 4, 2, 3, device="cuda", dtype=dtype)
                for _ in range(2)
            ]
            chunks = [x.detach().clone().requires_grad_() for x in source]
            refs = [x.detach().cpu().requires_grad_() for x in source]
            got = lf._pack_selected_chunk_history(tuple(chunks), cuda_plan)
            expected = lf._pack_selected_chunk_history(tuple(refs), cpu_plan)
            torch.testing.assert_close(got, expected.cuda())
            grad = torch.randn_like(got)
            got.backward(grad)
            expected.backward(grad.cpu())
            for chunk, ref in zip(chunks, refs):
                torch.testing.assert_close(chunk.grad, ref.grad.cuda())

            values = [torch.randn_like(x) for x in source]
            keys = [x.detach().clone().requires_grad_() for x in source]
            vals = [x.detach().clone().requires_grad_() for x in values]
            key_refs = [x.detach().cpu().requires_grad_() for x in source]
            val_refs = [x.detach().cpu().requires_grad_() for x in values]
            fused = lf._pack_selected_chunk_kv(
                tuple(keys), tuple(vals), cuda_plan)
            fused_ref = lf._pack_selected_chunk_kv(
                tuple(key_refs), tuple(val_refs), cpu_plan)
            torch.testing.assert_close(fused, fused_ref.cuda())
            fused_grad = torch.randn_like(fused)
            fused.backward(fused_grad)
            fused_ref.backward(fused_grad.cpu())
            for value, ref in zip((*keys, *vals), (*key_refs, *val_refs)):
                torch.testing.assert_close(value.grad, ref.grad.cuda())

    def test_layout_is_linear_and_chunk_gather_is_document_major(self):
        seg = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
             [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]],
            dtype=torch.int32,
        )
        layout = lf.packed_layout_from_segment_ids(seg)
        packed = lf.build_packed_chunk_layout(
            layout, 4, 8, ((0, 4), (4, 8)))
        chunks = []
        for start, end in ((0, 4), (4, 8)):
            values = torch.empty(2, end - start, 1, 1)
            for batch_idx in range(2):
                values[batch_idx, :, 0, 0] = (
                    batch_idx * 100 + torch.arange(start, end))
            chunks.append(values)
        chunks = [x.requires_grad_() for x in chunks]
        got = lf._pack_selected_chunk_history(tuple(chunks), packed)[:, 0, 0]
        expected = torch.tensor(
            [3, 4, 5, 6, 7, 102, 103, 104, 105, 106, 107],
            dtype=got.dtype,
        )
        self.assertTrue(torch.equal(got, expected))
        self.assertEqual(packed.cu_seqlens_q.tolist(), [0, 2, 4, 5, 8])
        self.assertEqual(packed.cu_seqlens_k.tolist(), [0, 3, 5, 8, 11])
        self.assertEqual(layout.segment_ids.numel(), seg.numel())
        got.square().sum().backward()
        self.assertTrue(all(x.grad is not None for x in chunks))
        k_chunks = [x.detach().clone().requires_grad_() for x in chunks]
        v_chunks = [(x.detach() + 1000).requires_grad_() for x in chunks]
        fused = lf._pack_selected_chunk_kv(
            tuple(k_chunks), tuple(v_chunks), packed)
        self.assertTrue(torch.equal(fused[:, 0, 0], expected))
        self.assertTrue(torch.equal(fused[:, 0, 1], expected + 1000))
        fused.square().sum().backward()
        self.assertTrue(all(x.grad is not None for x in (*k_chunks, *v_chunks)))

    def test_packed_sdpa_matches_dense_reference_forward_and_backward(self):
        cfg = lf.Config(
            vocab=64, seq_len=10, batch_size=2, model_dim=16,
            n_q_heads=2, head_dim=8, n_kv_heads=1, hidden=32,
            layers=1, attn_impl="sdpa", amp_dtype="fp32",
        )
        lf.apply_config(cfg)
        packed_attn = lf.GroupedQueryCausalSelfAttention()
        dense_attn = lf.GroupedQueryCausalSelfAttention()
        dense_attn.load_state_dict(packed_attn.state_dict())
        seg = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2, 2],
             [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]],
            dtype=torch.int32,
        )
        layout = lf.packed_layout_from_segment_ids(seg)
        position_ids = _position_ids(seg)
        causal = torch.tril(torch.ones(10, 10, dtype=torch.bool))
        dense = ((seg[:, :, None] == seg[:, None, :]) & causal).unsqueeze(1)
        x1 = torch.randn(2, 10, 16, requires_grad=True)
        x2 = x1.detach().clone().requires_grad_(True)
        out1 = packed_attn(x1, layout, position_ids)[0]
        out2 = dense_attn(x2, dense, position_ids)[0]
        self.assertTrue(torch.allclose(out1, out2, atol=2e-6, rtol=2e-5))
        out1.square().sum().backward()
        out2.square().sum().backward()
        self.assertTrue(torch.allclose(x1.grad, x2.grad, atol=3e-6, rtol=3e-5))

    def test_tria_final_ca_uses_same_document_without_dense_mask(self):
        batch, length, dim = 2, 9, 8
        seg = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 2, 2, 2],
             [0, 0, 1, 1, 1, 1, 2, 2, 2]],
            dtype=torch.int32,
        )
        layout = lf.packed_layout_from_segment_ids(seg)
        pos = torch.arange(length)
        dense = (
            (seg[:, :, None] == seg[:, None, :])
            & (pos[None, :] <= pos[:, None])
        ).unsqueeze(1)
        compact_ca = tria.TriaFinalCrossAttention(dim, raw_gamma_init=0.4)
        dense_ca = tria.TriaFinalCrossAttention(dim, raw_gamma_init=0.4)
        dense_ca.load_state_dict(compact_ca.state_dict())
        a1 = torch.randn(batch, length, dim, requires_grad=True)
        b1 = torch.randn(batch, length, dim, requires_grad=True)
        a2 = a1.detach().clone().requires_grad_(True)
        b2 = b1.detach().clone().requires_grad_(True)
        fire = torch.zeros(batch, length, dtype=torch.bool)
        fire[:, [2, 5, 8]] = True
        out1 = compact_ca(a1, b1, layout, carry_key_mask=fire)
        out2 = dense_ca(a2, b2, dense, carry_key_mask=fire)
        self.assertTrue(torch.allclose(out1, out2, atol=2e-6, rtol=2e-5))
        out1.sum().backward()
        out2.sum().backward()
        self.assertTrue(torch.allclose(a1.grad, a2.grad, atol=2e-6, rtol=2e-5))
        self.assertTrue(torch.allclose(b1.grad, b2.grad, atol=2e-6, rtol=2e-5))

    def test_chunked_model_consumes_precomputed_plans(self):
        cfg = lf.Config(
            vocab=32, seq_len=8, batch_size=1, model_dim=16,
            n_q_heads=2, head_dim=8, n_kv_heads=1, hidden=32,
            layers=1, attn_impl="sdpa", amp_dtype="fp32",
            tria_carry_enabled=True, tria_temporal_enabled=True,
            tria_temporal_auto=False, tria_temporal_window=4,
            grad_checkpointing=True,
            use_cuda_tria=False,
        )
        lf.apply_config(cfg)
        x = torch.tensor([[1, 2, 3, 7, 4, 5, 7, 6]])
        position_ids, layout = lf.build_doc_reset_state(x, eos_id=7)
        stops = lf.temporal_chunk_stops(
            x, window=4, hard_fire_enabled=True,
            carry_token_id=None, compiling=False)
        ranges = []
        plans = []
        start = 0
        for stop in stops:
            end = stop + 1
            ranges.append((start, end))
            plans.append(lf.build_packed_chunk_layout(
                layout, start, end, tuple(ranges)))
            start = end
        layout = replace(layout, chunk_plans=tuple(plans))
        model = lf.Model(cfg).train()
        self.assertFalse(any(
            "causal_mask" in name for name, _buffer in model.named_buffers()))
        labels = torch.randint(0, cfg.vocab, x.shape)
        with mock.patch.object(
            lf, "build_packed_chunk_layout",
            side_effect=AssertionError("GPU/runtime plan rebuild was used"),
        ):
            loss = model(
                x, attn_mask=layout, position_ids=position_ids, labels=labels)
            loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
