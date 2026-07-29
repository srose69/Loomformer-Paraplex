import asyncio
import json
import os
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

import loomformer as lf
from synthetic_tokenizer import SPECIAL_TOKENS, build_synthetic_bpe


class SFTOnTheFlyDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tokenizer_temp = None
        if list(lf.DEFAULT_SPECIAL_TOKENS) != SPECIAL_TOKENS:
            raise AssertionError(
                "synthetic tokenizer special tokens drifted from LoomFormer"
            )
        tokenizer = os.environ.get("LOOM_TEST_TOKENIZER")
        vocab = os.environ.get("LOOM_TEST_VOCAB")
        if bool(tokenizer) != bool(vocab):
            raise RuntimeError(
                "LOOM_TEST_TOKENIZER and LOOM_TEST_VOCAB must be set together"
            )
        if tokenizer and vocab:
            if not os.path.isfile(tokenizer):
                raise RuntimeError(
                    f"LOOM_TEST_TOKENIZER does not exist: {tokenizer}"
                )
            cls.tokenizer = tokenizer
            cls.vocab = int(vocab)
        else:
            # Keep direct unittest discovery self-contained as well. The
            # complete matrix supplies the same tokenizer it builds before
            # launching the suite, so all subsequent PT/SFT cases share it.
            cls._tokenizer_temp = tempfile.TemporaryDirectory(
                prefix="loomformer-test-tokenizer."
            )
            cls.tokenizer = os.path.join(
                cls._tokenizer_temp.name, "tokenizer.json"
            )
            cls.vocab = build_synthetic_bpe(cls.tokenizer, vocab_size=256)

        from tokenizers import Tokenizer

        stored = Tokenizer.from_file(cls.tokenizer)
        if stored.get_vocab_size() != cls.vocab:
            raise AssertionError(
                f"synthetic tokenizer/config vocab mismatch: "
                f"{stored.get_vocab_size()} != {cls.vocab}"
            )
        if len(stored.encode("q" * 256).ids) <= 16:
            raise AssertionError(
                "synthetic BPE no longer exercises the overlength rejection"
            )

    @classmethod
    def tearDownClass(cls):
        if cls._tokenizer_temp is not None:
            cls._tokenizer_temp.cleanup()

    def test_directory_split_is_destructive_idempotent_and_parquet_only(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": f"question {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ]
            }
            for i in range(20)
        ]
        with tempfile.TemporaryDirectory() as dataset:
            for shard in range(2):
                pq.write_table(
                    pa.Table.from_pylist(rows[shard * 10:(shard + 1) * 10]),
                    os.path.join(dataset, f"data-{shard}.parquet"),
                )
            # Hub snapshots may contain an alternate JSONL export. SFT directory
            # discovery must not train on the same examples twice.
            with open(os.path.join(dataset, "duplicate.jsonl"), "w") as f:
                f.write(json.dumps(rows[0]) + "\n")

            cfg = lf.Config(
                dataset_format="sft",
                train_dataset=dataset,
                auto_val_split_pct=20.0,
                dataset_cache="otf",
                tokenizer=self.tokenizer,
                vocab=self.vocab,
                seq_len=64,
                batch_size=1,
                prefetch_batches=1,
                gpu_prefetch_batches=1,
            )
            val_path = lf.maybe_auto_val_split(cfg, dataset)
            self.assertIsNotNone(val_path)
            self.assertEqual(
                [
                    pq.ParquetFile(
                        os.path.join(dataset, f"data-{shard}.parquet")
                    ).metadata.num_rows
                    for shard in range(2)
                ],
                [8, 8],
            )
            self.assertEqual(pq.ParquetFile(val_path).metadata.num_rows, 4)

            cfg.val_dataset = None
            self.assertEqual(lf.maybe_auto_val_split(cfg, dataset), val_path)
            self.assertEqual(
                pq.ParquetFile(os.path.join(dataset, "data-0.parquet"))
                .metadata.num_rows,
                8,
            )

            stream = lf.make_stream(dataset, cfg, lf.device_auto("cpu"))
            try:
                self.assertEqual(len(stream._files), 2)
                self.assertTrue(all(path.endswith(".parquet") for path in stream._files))
                self.assertEqual(stream._assigned_rows, 16)
                batches = list(stream.iter_eval_batches(batch_size=3))
                self.assertGreater(len(batches), 0)
                for ids, mask, layout in batches:
                    self.assertEqual(ids.shape, mask.shape)
                    self.assertEqual(ids.shape[1], cfg.seq_len + 1)
                    self.assertEqual(
                        tuple(layout.segment_ids.shape),
                        (ids.shape[0], cfg.seq_len),
                    )
            finally:
                stream.close()

    def test_otf_fails_fast_when_every_example_exceeds_sequence_length(self):
        rows = [
            {
                "messages": [
                    {"role": "user", "content": "q" * 256},
                    {"role": "assistant", "content": "a" * 256},
                ]
            }
            for _ in range(4)
        ]
        with tempfile.TemporaryDirectory() as dataset:
            pq.write_table(
                pa.Table.from_pylist(rows),
                os.path.join(dataset, "data.parquet"),
            )
            cfg = lf.Config(
                dataset_format="sft",
                train_dataset=dataset,
                dataset_cache="otf",
                tokenizer=self.tokenizer,
                vocab=self.vocab,
                seq_len=16,
                batch_size=1,
                prefetch_batches=1,
                gpu_prefetch_batches=1,
            )
            stream = lf.make_stream(dataset, cfg, lf.device_auto("cpu"))
            try:
                with self.assertRaisesRegex(
                    RuntimeError, "data producer failed"
                ) as raised:
                    asyncio.run(asyncio.wait_for(stream.prime(), timeout=5.0))
                self.assertIsInstance(raised.exception.__cause__, ValueError)
                self.assertIn(
                    "zero trainable examples",
                    str(raised.exception.__cause__),
                )
            finally:
                stream.close()


if __name__ == "__main__":
    unittest.main()
