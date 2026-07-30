from dataclasses import asdict, replace
import math
from pathlib import Path
import sys
import tempfile
import unittest

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import loomformer as lf
import loompack


def _cfg(**overrides):
    cfg = lf.Config(
        vocab=64,
        seq_len=16,
        batch_size=2,
        model_dim=32,
        n_q_heads=4,
        head_dim=8,
        n_kv_heads=2,
        hidden=64,
        layers=4,
        attn_layers=[1, 3],
        attn_token_stride=2,
        attn_token_schedule="staggered",
        attn_impl="sdpa",
        tria_carry_enabled=False,
    )
    return replace(cfg, **overrides)


def _oracle(mixer, z, positions, segments, stride, offset, inherited=None):
    B, T, _ = z.shape
    selected = positions.remainder(stride).eq(offset)
    indices = selected.reshape(-1).nonzero().flatten()
    zs = z.reshape(B * T, lf.N).index_select(0, indices)
    qkv = F.linear(zs, mixer.qkv_weight)
    qp, kp, vp = torch.split(qkv, (lf.N, lf.KV_DIM, lf.KV_DIM), dim=-1)
    q = qp.view(-1, lf.N_Q_HEADS, lf.HEAD_DIM)
    k = kp.view(-1, lf.N_KV_HEADS, lf.HEAD_DIM)
    v = vp.view(-1, lf.N_KV_HEADS, lf.HEAD_DIM)
    selected_positions = positions.reshape(-1).index_select(0, indices)
    q, k = mixer.rope(
        q.unsqueeze(0), k.unsqueeze(0), selected_positions.unsqueeze(0))
    q, k = q.squeeze(0), k.squeeze(0)

    kctx = torch.zeros_like(q)
    value = torch.zeros_like(q)
    flat_segments = segments.reshape(-1)
    selected_segments = flat_segments.index_select(0, indices)
    selected_rows = indices.div(T, rounding_mode="floor")
    for row in range(B):
        row_mask = selected_rows.eq(row)
        for segment in torch.unique_consecutive(
            selected_segments[row_mask]).tolist():
            take = row_mask & selected_segments.eq(segment)
            local = take.nonzero().flatten()
            qd = q.index_select(0, local)
            kd = k.index_select(0, local).repeat_interleave(
                lf.GQA_GROUP_SIZE, dim=1)
            vd = v.index_select(0, local).repeat_interleave(
                lf.GQA_GROUP_SIZE, dim=1)
            scores = torch.einsum("ihd,jhd->hij", qd, kd) / math.sqrt(
                lf.HEAD_DIM)
            causal = torch.ones(
                qd.shape[0], qd.shape[0], dtype=torch.bool,
                device=z.device).tril()
            weights = torch.softmax(
                scores.masked_fill(~causal, float("-inf")).float(), dim=-1
            ).to(z.dtype)
            kctx.index_copy_(0, local, torch.einsum("hij,jhd->ihd", weights, kd))
            value.index_copy_(0, local, torch.einsum("hij,jhd->ihd", weights, vd))

    shape = (B * T, lf.N_Q_HEADS, lf.HEAD_DIM)

    def scatter(x):
        return x.new_zeros(shape).index_copy(0, indices, x).view(
            B, T, lf.N_Q_HEADS, lf.HEAD_DIM)

    own = tuple(scatter(x) for x in (q, kctx, value))
    if inherited is None:
        held = [torch.zeros_like(own[0]) for _ in range(3)]
        for row in range(B):
            last = None
            last_segment = None
            for token in range(T):
                segment = int(segments[row, token])
                if segment != last_segment:
                    last = None
                    last_segment = segment
                if bool(selected[row, token]):
                    last = token
                if last is None:
                    raise AssertionError("first active sparse layer must select document starts")
                for target, source in zip(held, own):
                    target[row, token] = source[row, last]
        full = tuple(held)
    else:
        choose = selected.unsqueeze(-1).unsqueeze(-1)
        full = tuple(
            torch.where(choose, own_value, inherited_value)
            for own_value, inherited_value in zip(own, inherited)
        )
    residual_selected = F.linear(value.reshape(-1, lf.N), mixer.o.weight)
    residual = z.new_zeros(B * T, lf.N).index_copy(
        0, indices, residual_selected).view(B, T, lf.N)
    return residual, *full, k, v, indices


class AttentionSparsityTests(unittest.TestCase):
    def test_config_resolution_validation_and_roundtrip(self):
        dense = _cfg(attn_layers=None, attn_token_stride=1)
        lf.apply_config(dense)
        self.assertEqual(dense.attn_layers, [1, 2, 3, 4])
        restored = lf.Config(**asdict(dense))
        self.assertEqual(restored.attn_layers, dense.attn_layers)
        self.assertEqual(restored.attn_token_stride, 1)
        self.assertEqual(restored.attn_token_schedule, "staggered")

        invalid = (
            {"attn_layers": []},
            {"attn_layers": [2, 3]},
            {"attn_layers": [1, 1]},
            {"attn_layers": [1, 5]},
            {"attn_token_stride": 0},
            {"attn_token_schedule": "random"},
        )
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                lf.apply_config(_cfg(**overrides))

    def test_constructor_matrix_has_exact_parameters_and_offsets(self):
        cases = (
            (None, 1, "shared", [3072] * 4, [None] * 4),
            ([1, 3], 1, "shared", [3072, 0, 3072, 0], [None] * 4),
            ([1, 3], 2, "shared", [3072, 0, 3072, 0], [0, None, 0, None]),
            ([1, 3], 2, "staggered", [3072, 0, 3072, 0], [0, None, 1, None]),
        )
        for layers, stride, schedule, counts, offsets in cases:
            with self.subTest(layers=layers, stride=stride, schedule=schedule):
                cfg = _cfg(
                    attn_layers=layers,
                    attn_token_stride=stride,
                    attn_token_schedule=schedule,
                )
                lf.apply_config(cfg)
                model = lf.Model(cfg)
                self.assertEqual(
                    [sum(p.numel() for p in block.attn.parameters())
                     for block in model.blocks],
                    counts,
                )
                self.assertEqual(
                    [getattr(block.attn, "token_offset", None)
                     for block in model.blocks],
                    offsets,
                )

    def test_sparse_forward_and_backward_match_independent_oracle(self):
        torch.manual_seed(41)
        cfg = _cfg(layers=2, attn_layers=[1])
        lf.apply_config(cfg)
        actual = lf.StridedGroupedQueryCausalSelfAttention(2, 0)
        oracle_module = lf.StridedGroupedQueryCausalSelfAttention(2, 0)
        oracle_module.load_state_dict(actual.state_dict())
        z_actual = torch.randn(2, 8, 32, requires_grad=True)
        z_oracle = z_actual.detach().clone().requires_grad_(True)
        segments = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 1, 1], [3, 3, 4, 4, 4, 5, 5, 5]],
            dtype=torch.int32,
        )
        layout = lf.packed_layout_from_segment_ids(segments)
        got = actual(
            z_actual, attn_mask=layout, position_ids=layout.position_ids.long())
        expected = _oracle(
            oracle_module,
            z_oracle,
            layout.position_ids.long(),
            segments,
            2,
            0,
        )[:4]
        for left, right in zip(got, expected):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
        self.assertTrue(torch.equal(
            got[0][~layout.position_ids.remainder(2).eq(0)],
            torch.zeros_like(got[0][~layout.position_ids.remainder(2).eq(0)]),
        ))

        generators = []
        for tensor in got:
            generators.append(torch.randn_like(tensor))
        loss_actual = sum((x * g).sum() for x, g in zip(got, generators))
        loss_oracle = sum((x * g).sum() for x, g in zip(expected, generators))
        loss_actual.backward()
        loss_oracle.backward()
        torch.testing.assert_close(
            z_actual.grad, z_oracle.grad, atol=3e-6, rtol=3e-6)
        torch.testing.assert_close(
            actual.qkv_weight.grad,
            oracle_module.qkv_weight.grad,
            atol=4e-6,
            rtol=4e-6,
        )
        torch.testing.assert_close(
            actual.o.weight.grad,
            oracle_module.o.weight.grad,
            atol=3e-6,
            rtol=3e-6,
        )

    def test_staggered_gaps_use_exact_inherited_context(self):
        torch.manual_seed(43)
        cfg = _cfg(layers=2, attn_layers=[1, 2])
        lf.apply_config(cfg)
        mixer = lf.StridedGroupedQueryCausalSelfAttention(2, 1)
        z = torch.randn(2, 8, 32)
        positions = torch.arange(8).view(1, -1).expand(2, -1)
        segments = torch.zeros(2, 8, dtype=torch.int32)
        inherited = tuple(torch.randn(2, 8, 4, 8) for _ in range(3))
        got = mixer(
            z, position_ids=positions, inherited_context=inherited)
        expected = _oracle(
            mixer, z, positions, segments, 2, 1, inherited=inherited)[:4]
        for left, right in zip(got, expected):
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-6)
        for actual, source in zip(got[1:], inherited):
            torch.testing.assert_close(actual[:, 0::2], source[:, 0::2])

    def test_empty_offset_keeps_all_attention_parameters_in_graph(self):
        cfg = _cfg(layers=2, attn_layers=[1, 2], attn_token_stride=3)
        lf.apply_config(cfg)
        mixer = lf.StridedGroupedQueryCausalSelfAttention(3, 2)
        z = torch.randn(2, 2, 32, requires_grad=True)
        inherited = tuple(torch.randn(2, 2, 4, 8) for _ in range(3))
        residual, q, kctx, value = mixer(
            z,
            position_ids=torch.arange(2).view(1, -1).expand(2, -1),
            inherited_context=inherited,
        )
        loss = residual.square().sum() + q.square().sum()
        loss += kctx.square().sum() + value.square().sum()
        loss.backward()
        self.assertIsNotNone(mixer.qkv_weight.grad)
        self.assertIsNotNone(mixer.o.weight.grad)
        self.assertEqual(torch.count_nonzero(mixer.qkv_weight.grad).item(), 0)
        self.assertEqual(torch.count_nonzero(mixer.o.weight.grad).item(), 0)

        chunk_mixer = lf.StridedGroupedQueryCausalSelfAttention(3, 2)
        chunk_z = torch.randn(2, 2, 32, requires_grad=True)
        segments = torch.tensor([[0, 1], [0, 1]], dtype=torch.int32)
        layout = lf.packed_layout_from_segment_ids(segments)
        plan = lf.build_packed_chunk_layout(layout, 0, 2, ((0, 2),))
        chunk_out = chunk_mixer.forward_chunk(
            chunk_z,
            (),
            (),
            layout.position_ids.long(),
            layout,
            plan,
            inherited_context=inherited,
        )
        sum(tensor.square().sum() for tensor in chunk_out[:4]).backward()
        self.assertIsNotNone(chunk_mixer.qkv_weight.grad)
        self.assertIsNotNone(chunk_mixer.o.weight.grad)
        self.assertEqual(
            torch.count_nonzero(chunk_mixer.qkv_weight.grad).item(), 0)
        self.assertEqual(
            torch.count_nonzero(chunk_mixer.o.weight.grad).item(), 0)

    def test_incremental_matches_flat_and_cache_is_exact_compact_kv(self):
        torch.manual_seed(47)
        cfg = _cfg(layers=2, attn_layers=[1])
        lf.apply_config(cfg)
        mixer = lf.StridedGroupedQueryCausalSelfAttention(2, 0).eval()
        z = torch.randn(2, 12, 32)
        positions = torch.arange(12).view(1, -1).expand(2, -1)
        full = mixer(z, position_ids=positions)
        cache_k = cache_v = held = None
        cache_pos = 0
        pieces = [[] for _ in range(4)]
        for token in range(12):
            out = mixer.step(
                z[:, token:token + 1],
                token,
                cache_k,
                cache_v,
                cache_pos,
                held_context=held,
            )
            for index in range(4):
                pieces[index].append(out[index])
            cache_k, cache_v, cache_pos, held = out[4:]
        for expected, values in zip(full, pieces):
            torch.testing.assert_close(
                torch.cat(values, dim=1), expected, atol=2e-6, rtol=2e-6)
        self.assertEqual(cache_pos, 6)
        self.assertEqual(cache_k.shape[1], 8)

        selected_z = z[:, 0::2]
        qkv = F.linear(selected_z, mixer.qkv_weight)
        _, expected_k, expected_v = torch.split(
            qkv, (lf.N, lf.KV_DIM, lf.KV_DIM), dim=-1)
        expected_k = expected_k.view(2, 6, lf.N_KV_HEADS, lf.HEAD_DIM)
        dummy_q = torch.zeros(
            2, 6, lf.N_Q_HEADS, lf.HEAD_DIM, dtype=z.dtype)
        _, expected_k = mixer.rope(dummy_q, expected_k, positions[:, 0::2])
        expected_v = expected_v.view(2, 6, lf.N_KV_HEADS, lf.HEAD_DIM)
        torch.testing.assert_close(cache_k[:, :cache_pos], expected_k)
        torch.testing.assert_close(cache_v[:, :cache_pos], expected_v)

    def test_compact_chunk_attention_matches_flat_across_documents(self):
        torch.manual_seed(53)
        cfg = _cfg(layers=2, attn_layers=[1])
        lf.apply_config(cfg)
        mixer = lf.StridedGroupedQueryCausalSelfAttention(2, 0)
        z = torch.randn(2, 8, 32)
        segments = torch.tensor(
            [[0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 1, 1, 1, 2, 2, 2]],
            dtype=torch.int32,
        )
        layout = lf.packed_layout_from_segment_ids(segments)
        positions = layout.position_ids.long()
        flat = mixer(z, attn_mask=layout, position_ids=positions)
        plan0 = lf.build_packed_chunk_layout(layout, 0, 4, ((0, 4),))
        out0 = mixer.forward_chunk(
            z[:, :4], (), (), positions[:, :4], layout, plan0)
        meta0 = mixer._chunk_layout(positions[:, :4], layout, plan0)
        held0 = tuple(x[:, -1] for x in out0[1:4])
        plan1 = lf.build_packed_chunk_layout(
            layout, 4, 8, ((0, 4), (4, 8)))
        out1 = mixer.forward_chunk(
            z[:, 4:],
            (out0[4],),
            (out0[5],),
            positions[:, 4:],
            layout,
            plan1,
            held_context=held0,
            past_document_chunks=(meta0.documents,),
            past_position_chunks=(meta0.positions,),
        )
        for expected, first, second in zip(flat, out0[:4], out1[:4]):
            torch.testing.assert_close(
                torch.cat((first, second), dim=1), expected,
                atol=2e-6, rtol=2e-6)
        expected_selected = int(positions.remainder(2).eq(0).sum())
        self.assertEqual(out0[4].shape[1] + out1[4].shape[1], expected_selected)
        self.assertEqual(out0[5].shape[1] + out1[5].shape[1], expected_selected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_staggered_chunk_backward_crosses_empty_history_piece(self):
        cfg = _cfg(layers=2, attn_layers=[1, 2])
        lf.apply_config(cfg)
        mixer = lf.StridedGroupedQueryCausalSelfAttention(2, 1).cuda()

        def plan(start, end):
            empty = torch.empty(0, dtype=torch.int32, device="cuda")
            return lf.PackedChunkLayout(
                start=start,
                end=end,
                selectors=empty,
                destinations=empty,
                piece_sizes=(),
                piece_offsets=torch.zeros(
                    1, dtype=torch.int32, device="cuda"),
                cu_seqlens_q=torch.tensor(
                    [0, 1], dtype=torch.int32, device="cuda"),
                cu_seqlens_k=torch.tensor(
                    [0, 1], dtype=torch.int32, device="cuda"),
                max_seqlen_q=1,
                max_seqlen_k=1,
            )

        z0 = torch.randn(1, 1, 32, device="cuda", requires_grad=True)
        z1 = torch.randn(1, 1, 32, device="cuda", requires_grad=True)
        inherited0 = tuple(
            torch.randn(1, 1, 4, 8, device="cuda") for _ in range(3))
        inherited1 = tuple(
            torch.randn(1, 1, 4, 8, device="cuda") for _ in range(3))
        pos0 = torch.tensor([[0]], device="cuda")
        pos1 = torch.tensor([[1]], device="cuda")
        first_plan = plan(0, 1)
        out0 = mixer.forward_chunk(
            z0, (), (), pos0, None, first_plan,
            inherited_context=inherited0)
        meta0 = mixer._chunk_layout(pos0, None, first_plan)
        self.assertEqual(out0[4].shape[1], 0)
        out1 = mixer.forward_chunk(
            z1,
            (out0[4],),
            (out0[5],),
            pos1,
            None,
            plan(1, 2),
            inherited_context=inherited1,
            past_document_chunks=(meta0.documents,),
            past_position_chunks=(meta0.positions,),
        )
        sum(tensor.square().sum() for tensor in out1[:4]).backward()
        self.assertIsNotNone(z0.grad)
        self.assertEqual(torch.count_nonzero(z0.grad).item(), 0)
        self.assertIsNotNone(z1.grad)
        self.assertGreater(torch.count_nonzero(z1.grad).item(), 0)

    def test_dense_checkpoint_conversion_is_explicit_and_strict(self):
        dense_cfg = _cfg(attn_layers=None, attn_token_stride=1)
        lf.apply_config(dense_cfg)
        dense_state = lf.Model(dense_cfg).state_dict()
        sparse_cfg = _cfg()
        lf.apply_config(sparse_cfg)
        sparse = lf.Model(sparse_cfg)
        result = sparse.load_state_dict(
            lf.canonicalize_model_state_dict(dense_state), strict=True)
        self.assertEqual(result.missing_keys, [])
        self.assertEqual(result.unexpected_keys, [])
        for layer in (0, 2):
            torch.testing.assert_close(
                sparse.blocks[layer].attn.qkv_weight,
                dense_state[f"blocks.{layer}.attn.qkv_weight"],
            )
        with self.assertRaisesRegex(ValueError, "init_checkpoint"):
            lf.assert_resume_attention_config(
                sparse_cfg, asdict(dense_cfg))

    def test_aio_manifest_contains_attention_schedule(self):
        cfg = _cfg()
        lf.apply_config(cfg)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            tokenizer = root / "tokenizer.json"
            template = root / "chat_template.jinja"
            output = root / "model.aio"
            torch.save(
                {
                    "cfg": asdict(cfg),
                    "model_kind": "loomformer",
                    "ffn_type": "paraplex",
                    "ablation": False,
                    "model": {"weight": torch.ones(2, 2)},
                },
                checkpoint,
            )
            tokenizer.write_bytes(b"{}")
            template.write_text("", encoding="utf-8")
            manifest = loompack.build_package(
                str(checkpoint),
                str(tokenizer),
                str(template),
                str(output),
                "none",
                1,
            )
        packaged = manifest["model"]["config"]
        self.assertEqual(packaged["attn_layers"], [1, 3])
        self.assertEqual(packaged["attn_token_stride"], 2)
        self.assertEqual(packaged["attn_token_schedule"], "staggered")


if __name__ == "__main__":
    unittest.main(verbosity=2)
