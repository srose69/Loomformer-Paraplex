import asyncio
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

import loomformer as lf


class _EvalStream:
    def __init__(self, batches):
        self._batches = iter(batches)
        self._lock = threading.Lock()

    def sample_device_batch(self):
        with self._lock:
            return next(self._batches)


class _KnownLossModel(torch.nn.Module):
    def forward(self, idx, **kwargs):
        del kwargs
        return idx.new_tensor(2.0 if int(idx[0, 0]) == 1 else 8.0,
                              dtype=torch.float32)


class _TinyTokenizer:
    vocab_size = 32

    def encode(self, text):
        return [{"a": 1, "b": 2, "c": 3, "d": 4,
                 "e": 5, "f": 6, "g": 7, "h": 8,
                 "i": 9}[c] for c in text]

    def special_id(self, token):
        return {"<bos>": 30, "<eos>": 31}.get(token)


class _DocumentAwareLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, idx, **kwargs):
        self.calls.append((idx.detach().clone(), kwargs))
        return idx.new_tensor(2.0, dtype=torch.float32)


class LossAccountingTests(unittest.TestCase):
    def test_fused_ce_matches_explicit_masked_mean_and_gradients(self):
        torch.manual_seed(7)
        hidden = torch.randn(11, 5, requires_grad=True)
        weight = torch.randn(17, 5, requires_grad=True)
        targets = torch.tensor([1, -100, 4, 3, -100, 8, 2, 6, -100, 9, 0])

        fused = lf._fused_linear_cross_entropy_eager(
            hidden, weight, targets, -100, 3)
        fused_grads = torch.autograd.grad(fused, (hidden, weight))

        hidden_ref = hidden.detach().requires_grad_(True)
        weight_ref = weight.detach().requires_grad_(True)
        explicit = F.cross_entropy(
            F.linear(hidden_ref, weight_ref).reshape(-1, weight.shape[0]),
            targets,
            ignore_index=-100,
        )
        explicit_grads = torch.autograd.grad(explicit, (hidden_ref, weight_ref))

        torch.testing.assert_close(fused, explicit, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(
            fused_grads[0], explicit_grads[0], rtol=2e-6, atol=2e-6)
        torch.testing.assert_close(
            fused_grads[1], explicit_grads[1], rtol=2e-6, atol=2e-6)

    def test_shifted_sft_mask_is_the_only_target_selector(self):
        ids = torch.tensor([
            [3, 4, 5, 6, 7, 8],
            [9, 10, 11, 12, 13, 14],
        ])
        loss_mask = torch.tensor([
            [0, 0, 1, 1, 0, 1],
            [0, 1, 0, 1, 1, 0],
        ])
        x, y, positions, layout = lf.split_train_batch(
            (ids, loss_mask), eos_id=None)

        self.assertIsNone(layout)
        torch.testing.assert_close(x, ids[:, :-1])
        torch.testing.assert_close(
            positions, torch.arange(x.shape[1]).unsqueeze(0).expand_as(x))
        self.assertTrue(torch.equal(
            y.ne(lf.IGNORE_INDEX), loss_mask[:, 1:].bool()))
        torch.testing.assert_close(
            y[y.ne(lf.IGNORE_INDEX)], ids[:, 1:][loss_mask[:, 1:].bool()])

    def test_eval_batches_are_weighted_by_supervised_targets(self):
        cfg = lf.Config(eval_batches=2)
        batches = [
            torch.tensor([[1, 2, 3]]),
            torch.tensor([[9, 4, 5, 6, 7]]),
        ]
        loss = asyncio.run(lf.eval_loss_async(
            _KnownLossModel(), _EvalStream(batches), cfg, torch.device("cpu")))
        self.assertAlmostEqual(loss, (2.0 * 2 + 8.0 * 4) / 6, places=7)

    def test_full_raw_eval_preserves_pt_document_boundaries(self):
        cfg = lf.Config(
            dataset_format="jsonl",
            text_field="text",
            seq_len=4,
            batch_size=2,
            tria_temporal_enabled=False,
        )
        model = _DocumentAwareLossModel()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "docs.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for text in ("ab", "c", "defghi"):
                    f.write(json.dumps({"text": text}) + "\n")
            with mock.patch.object(
                    lf, "build_tokenizer", return_value=_TinyTokenizer()):
                result = lf.eval_full_model(
                    model, cfg, path, torch.device("cpu"), eval_batch_size=2)

        self.assertEqual(result["total_tokens"], 11.0)
        self.assertEqual(result["loss_nats"], 2.0)
        self.assertEqual(len(model.calls), 2)
        idx, kwargs = model.calls[0]
        torch.testing.assert_close(idx, torch.tensor([
            [30, 1, 2, 31],
            [30, 31, 4, 5],
        ]))
        torch.testing.assert_close(kwargs["labels"], torch.tensor([
            [1, 2, 31, 3],
            [31, 4, 5, 6],
        ]))
        torch.testing.assert_close(kwargs["position_ids"], torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 0, 1],
        ]))
        self.assertIsInstance(kwargs["attn_mask"], lf.PackedAttentionLayout)
        torch.testing.assert_close(
            kwargs["attn_mask"].segment_ids,
            torch.tensor([[0, 0, 0, 0], [0, 0, 1, 1]], dtype=torch.int32),
        )
        tail_idx, tail_kwargs = model.calls[1]
        torch.testing.assert_close(tail_idx, torch.tensor([[30, 7, 8]]))
        torch.testing.assert_close(tail_kwargs["labels"], torch.tensor([[7, 8, 9]]))


if __name__ == "__main__":
    unittest.main()
