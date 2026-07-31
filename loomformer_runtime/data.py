from __future__ import annotations

import asyncio
import contextlib
import glob
import json
import os
import queue
import threading
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist

from .distributed import ddp_is_distributed, ddp_is_main, ddp_print, ddp_rank, ddp_world_size
from .tokenization import _tok_special_id

if TYPE_CHECKING:
    from loomformer import Config

class TokenStream:
    def __init__(self, path: str, cfg: Config, device: torch.device, bos_id: Optional[int] = None):
        cache = str(getattr(cfg, "dataset_cache", "mmap") or "mmap").lower()
        if cache not in ("mmap", "ram"):
            raise ValueError(f"dataset_cache must be 'mmap' or 'ram', got {cache!r}")
        mm = np.memmap(path, dtype=np.uint16, mode="r")
        self.data = np.array(mm, dtype=np.uint16, copy=True) if cache == "ram" else mm
        self.cfg = cfg
        self.device = device
        self._bos_id = bos_id
        if len(self.data) <= cfg.seq_len + 1:
            raise ValueError(f"dataset too short: {len(self.data)} tokens for seq_len={cfg.seq_len}")

    def _sample_batch(self) -> torch.Tensor:
        B, T = self.cfg.batch_size, self.cfg.seq_len
        content_need = T + 1 - (1 if self._bos_id is not None else 0)
        ix = np.random.randint(0, len(self.data) - content_need - 1, size=B)
        rows = [self.data[i : i + content_need].astype(np.int64) for i in ix]
        if self._bos_id is not None:
            rows = [np.concatenate(([self._bos_id], r)) for r in rows]
        xb = np.stack(rows)
        return torch.from_numpy(xb)

    def sample_device_batch(self) -> torch.Tensor:
        b = self._sample_batch()
        if self.device.type == "cuda":
            b = b.pin_memory()
        return b.to(self.device, non_blocking=True)

    async def _produce(self, queue: "asyncio.Queue", n: int):
        loop = asyncio.get_event_loop()
        for _ in range(n):
            batch = await loop.run_in_executor(None, self._sample_batch)
            await queue.put(batch.pin_memory() if self.device.type == "cuda" else batch)
        await queue.put(None)

    async def batches(self, n: int):
        queue: asyncio.Queue = asyncio.Queue(maxsize=max(1, int(getattr(self.cfg, "prefetch_batches", 256))))
        producer = asyncio.create_task(self._produce(queue, n))
        while True:
            batch = await queue.get()
            if batch is None:
                break
            yield batch.to(self.device, non_blocking=True)
        await producer


class RawCorpus:

    _EXTS = {"txt": (".txt",), "jsonl": (".jsonl", ".ndjson"),
             "parquet": (".parquet",), "arrow": (".arrow", ".feather")}

    def __init__(self, path: str, fmt: str = "auto", text_field: str = "text"):
        self.text_field = text_field
        files = self._resolve_files(path, fmt)
        if not files:
            raise ValueError(f"no corpus files found at {path!r} (format={fmt!r})")
        self.fmt = fmt if fmt != "auto" else self._infer_format(files[0])
        self._files = files
        self._cache: Dict[Tuple[str, int], object] = {}
        docs: List[Tuple[int, object, int]] = []  # (file_idx, row/offset key, char length)
        indexer = {"txt": self._index_txt, "jsonl": self._index_jsonl,
                   "parquet": self._index_parquet, "arrow": self._index_arrow}[self.fmt]
        for fi, p in enumerate(files):
            docs.extend(indexer(fi, p))
        if not docs:
            raise ValueError(f"corpus at {path!r} indexed to zero documents")
        self._docs = docs
        self._cum = np.cumsum([d[2] for d in docs])
        self.total_chars = int(self._cum[-1])

    def __len__(self) -> int:
        return len(self._docs)

    def iter_texts(self):
        for fi, key, length in self._docs:
            txt = self._read_doc_text(fi, key, length)
            if txt:
                yield txt

    @staticmethod
    def _resolve_files(path: str, fmt: str) -> List[str]:
        if os.path.isfile(path):
            return [path]
        exts = RawCorpus._EXTS.get(fmt, [e for v in RawCorpus._EXTS.values() for e in v])
        found = []
        for e in exts:
            found.extend(glob.glob(os.path.join(path, f"*{e}")))
        return sorted(set(found))

    @staticmethod
    def _infer_format(path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        for fmt, exts in RawCorpus._EXTS.items():
            if ext in exts:
                return fmt
        raise ValueError(f"cannot infer corpus format from extension {ext!r} ({path!r})")

    def _index_txt(self, fi: int, p: str) -> List[Tuple[int, None, int]]:
        return [(fi, None, os.path.getsize(p))]

    def _index_jsonl(self, fi: int, p: str) -> List[Tuple[int, int, int]]:
        out = []
        with open(p, "rb") as f:
            offset = 0
            for line in f:
                s = line.decode("utf-8", errors="replace").strip()
                if s:
                    try:
                        txt = json.loads(s).get(self.text_field, "")
                        if txt:
                            out.append((fi, offset, len(txt)))
                    except Exception:
                        pass
                offset += len(line)
        return out

    def _index_parquet(self, fi: int, p: str) -> List[Tuple[int, int, int]]:
        import pyarrow.parquet as pq
        import pyarrow.compute as pc
        pf = pq.ParquetFile(p)
        out = []
        row_base = 0
        for rg in range(pf.num_row_groups):
            col = pf.read_row_group(rg, columns=[self.text_field]).column(self.text_field)
            lens = pc.utf8_length(col).to_numpy(zero_copy_only=False)
            out.extend((fi, row_base + i, int(l)) for i, l in enumerate(lens) if l > 0)
            row_base += len(lens)
        return out

    def _index_arrow(self, fi: int, p: str) -> List[Tuple[int, int, int]]:
        import pyarrow.compute as pc
        # HuggingFace datasets usually store .arrow shards as Arrow IPC STREAMS,
        # while pyarrow/Feather-style files are Arrow IPC FILES with a footer.
        # The extension alone does not distinguish them, so use the shared reader
        # that tries open_file first and falls back to open_stream.
        table, _container = _read_arrow_table_with_container(p)
        col = table.column(self.text_field)
        lens = pc.utf8_length(col).to_numpy(zero_copy_only=False)
        return [(fi, i, int(l)) for i, l in enumerate(lens) if l > 0]

    def _read_doc_text(self, fi: int, key, length: int) -> str:
        p = self._files[fi]
        if self.fmt == "txt":
            with open(p, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        if self.fmt == "jsonl":
            with open(p, "rb") as f:
                f.seek(key)
                line = f.readline()
            return json.loads(line.decode("utf-8", errors="replace")).get(self.text_field, "")
        if self.fmt in ("parquet", "arrow"):
            ck = ("table", fi)
            table = self._cache.get(ck)
            if table is None:
                if self.fmt == "parquet":
                    import pyarrow.parquet as pq
                    table = pq.read_table(p, columns=[self.text_field])
                else:
                    table, _container = _read_arrow_table_with_container(p)
                self._cache[ck] = table
            return str(table.column(self.text_field)[key])
        raise ValueError(self.fmt)

    def iter_sampled_texts(self, docs):
        """Yield sampled texts, reading only the required Parquet row groups."""
        if self.fmt != "parquet":
            for fi, key, length in docs:
                yield self._read_doc_text(fi, key, length)
            return

        import pyarrow.parquet as pq

        by_file: Dict[int, List[int]] = {}
        for fi, key, _length in docs:
            by_file.setdefault(fi, []).append(int(key))

        for fi, row_indices in by_file.items():
            pf = pq.ParquetFile(self._files[fi])
            rows = iter(sorted(row_indices))
            wanted = next(rows, None)
            row_base = 0
            for rg in range(pf.num_row_groups):
                row_count = pf.metadata.row_group(rg).num_rows
                row_end = row_base + row_count
                if wanted is not None and wanted < row_end:
                    col = pf.read_row_group(rg, columns=[self.text_field]).column(self.text_field)
                    while wanted is not None and wanted < row_end:
                        yield str(col[wanted - row_base])
                        wanted = next(rows, None)
                row_base = row_end
                if wanted is None:
                    break

    def sample_window_spans(self, min_chars: int, rng: np.random.Generator) -> List[str]:
        pos = int(rng.integers(0, self.total_chars))
        doc_i = int(np.searchsorted(self._cum, pos, side="right"))
        doc_i = min(doc_i, len(self._docs) - 1)
        fi, key, length = self._docs[doc_i]
        prev_cum = self._cum[doc_i] - length
        start = max(0, pos - prev_cum)
        spans = [self._read_doc_text(fi, key, length)[start:]]
        total = len(spans[0])
        j = doc_i + 1
        while total < min_chars and j < len(self._docs):
            fi2, key2, length2 = self._docs[j]
            text2 = self._read_doc_text(fi2, key2, length2)
            spans.append(text2)
            total += len(text2)
            j += 1
        return spans


def _auto_val_split_pct(cfg: Config) -> float:
    return float(getattr(cfg, "auto_val_split_pct", 0.0) or 0.0)


def _split_count(n: int, pct: float) -> int:
    if n <= 1:
        return 0
    k = int(round(n * (pct / 100.0)))
    if k <= 0:
        k = 1
    return min(k, n - 1)


def _concat_arrow_tables(tables):
    import pyarrow as pa
    if not tables:
        raise ValueError("auto val split produced no validation rows")
    try:
        return pa.concat_tables(tables, promote_options="default")
    except TypeError:
        return pa.concat_tables(tables, promote=True)


def _read_arrow_table_with_container(path: str):
    import pyarrow as pa
    ext = os.path.splitext(path)[1].lower()
    if ext == ".feather":
        import pyarrow.feather as feather
        return feather.read_table(path), "feather"
    with pa.memory_map(path, "rb") as src:
        try:
            return pa.ipc.open_file(src).read_all(), "file"
        except Exception:
            src.seek(0)
            return pa.ipc.open_stream(src).read_all(), "stream"


def _write_arrow_table_preserving_container(path: str, table, container: str) -> None:
    import pyarrow as pa
    tmp = path + ".tmp"
    if container == "feather":
        import pyarrow.feather as feather
        feather.write_feather(table, tmp)
    else:
        with pa.OSFile(tmp, "wb") as sink:
            writer_fn = pa.ipc.new_stream if container == "stream" else pa.ipc.new_file
            with writer_fn(sink, table.schema) as writer:
                writer.write_table(table)
    os.replace(tmp, path)


def _write_arrow_ipc_file(path: str, table) -> None:
    import pyarrow as pa
    tmp = path + ".tmp"
    with pa.OSFile(tmp, "wb") as sink:
        with pa.ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table)
    os.replace(tmp, path)


def _auto_split_arrow(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    val_tables = []
    train_rows = val_rows = 0
    for path in files:
        table, container = _read_arrow_table_with_container(path)
        n = int(table.num_rows)
        k = _split_count(n, pct)
        if k <= 0:
            train_rows += n
            continue
        train = table.slice(0, n - k)
        val = table.slice(n - k, k)
        _write_arrow_table_preserving_container(path, train, container)
        val_tables.append(val)
        train_rows += int(train.num_rows)
        val_rows += int(val.num_rows)
    _write_arrow_ipc_file(val_path, _concat_arrow_tables(val_tables))
    return {"train_rows": train_rows, "val_rows": val_rows}


def _auto_split_parquet(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    import pyarrow as pa
    import pyarrow.parquet as pq
    train_rows = val_rows = 0
    val_tmp = val_path + ".tmp"
    train_tmps: List[Tuple[str, str]] = []
    val_writer = None
    val_schema = None
    try:
        for path in files:
            pf = pq.ParquetFile(path)
            schema = pf.schema_arrow
            if val_schema is None:
                val_schema = schema
                val_writer = pq.ParquetWriter(val_tmp, schema, compression="zstd")
            elif schema != val_schema:
                raise ValueError(
                    "auto_val_split_pct requires identical Parquet schemas; "
                    f"{path!r} differs from the first shard")

            n = int(pf.metadata.num_rows)
            k = _split_count(n, pct)
            cut = n - k
            tmp = path + ".tmp"
            train_tmps.append((tmp, path))
            train_writer = pq.ParquetWriter(tmp, schema, compression="zstd")
            cursor = 0
            try:
                for batch in pf.iter_batches(batch_size=8192):
                    table = pa.Table.from_batches([batch])
                    end = cursor + int(table.num_rows)
                    train_count = max(0, min(end, cut) - cursor)
                    if train_count:
                        train_writer.write_table(table.slice(0, train_count))
                        train_rows += train_count
                    val_count = int(table.num_rows) - train_count
                    if val_count:
                        assert val_writer is not None
                        val_writer.write_table(table.slice(train_count, val_count))
                        val_rows += val_count
                    cursor = end
            finally:
                train_writer.close()
        if val_writer is None or val_rows <= 0:
            raise ValueError("auto val split produced no validation rows")
        val_writer.close()
        val_writer = None
        for tmp, path in train_tmps:
            os.replace(tmp, path)
        os.replace(val_tmp, val_path)
    except BaseException:
        if val_writer is not None:
            val_writer.close()
        for tmp, _path in train_tmps:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp)
        with contextlib.suppress(FileNotFoundError):
            os.unlink(val_tmp)
        raise
    return {"train_rows": train_rows, "val_rows": val_rows}


def _auto_split_jsonl(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    train_rows = val_rows = 0
    tmp_val = val_path + ".tmp"
    with open(tmp_val, "wb") as vf:
        for path in files:
            with open(path, "rb") as f:
                lines = f.readlines()
            n = len(lines)
            k = _split_count(n, pct)
            train_lines = lines[: n - k]
            val_lines = lines[n - k :] if k > 0 else []
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.writelines(train_lines)
            os.replace(tmp, path)
            vf.writelines(val_lines)
            train_rows += len(train_lines)
            val_rows += len(val_lines)
    os.replace(tmp_val, val_path)
    return {"train_rows": train_rows, "val_rows": val_rows}


def _auto_split_txt(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    train_bytes = val_bytes = 0
    tmp_val = val_path + ".tmp"
    with open(tmp_val, "wb") as vf:
        for path in files:
            with open(path, "rb") as f:
                data = f.read()
            n = len(data)
            if n <= 1:
                train = data
                val = b""
            else:
                k = max(1, int(round(n * (pct / 100.0))))
                k = min(k, n - 1)
                cut = n - k
                nl = data.rfind(b"\n", 0, cut)
                if nl > 0:
                    cut = nl + 1
                train, val = data[:cut], data[cut:]
            tmp = path + ".tmp"
            with open(tmp, "wb") as f:
                f.write(train)
            os.replace(tmp, path)
            vf.write(val)
            train_bytes += len(train)
            val_bytes += len(val)
    os.replace(tmp_val, val_path)
    return {"train_bytes": train_bytes, "val_bytes": val_bytes}


def _auto_split_bin(files: List[str], val_path: str, pct: float) -> Dict[str, int]:
    train_tokens = val_tokens = 0
    vals = []
    for path in files:
        arr = np.fromfile(path, dtype=np.uint16)
        n = int(arr.shape[0])
        k = _split_count(n, pct)
        train = arr[: n - k]
        val = arr[n - k :] if k > 0 else arr[:0]
        train.tofile(path + ".tmp")
        os.replace(path + ".tmp", path)
        vals.append(val)
        train_tokens += int(train.shape[0])
        val_tokens += int(val.shape[0])
    np.concatenate(vals).astype(np.uint16, copy=False).tofile(val_path)
    return {"train_tokens": train_tokens, "val_tokens": val_tokens}


def maybe_auto_val_split(cfg: Config, dataset: str) -> Optional[str]:
    """Return a validation path, optionally cutting shard tails into a new split."""
    if cfg.val_dataset:
        return str(cfg.val_dataset)
    pct = _auto_val_split_pct(cfg)
    if pct <= 0.0:
        return None
    if not (0.0 < pct < 100.0):
        raise ValueError(f"auto_val_split_pct must be in (0,100), got {pct}")
    if not os.path.isdir(dataset):
        raise ValueError("auto_val_split_pct only works when the training dataset is a directory of shards")

    fmt = str(getattr(cfg, "dataset_format", "auto") or "auto").lower()
    if is_sft_dataset(cfg):
        # SFT directories are Parquet-only even if the Hub snapshot also ships
        # a JSONL export of the same examples. Never train on both duplicates.
        files = sorted(glob.glob(os.path.join(dataset, "*.parquet")))
        inferred = "parquet"
    else:
        files = RawCorpus._resolve_files(dataset, fmt)
        inferred = fmt if fmt != "auto" else (
            RawCorpus._infer_format(files[0]) if files else "")
    if not files:
        raise ValueError(f"auto_val_split_pct found no top-level corpus files in {dataset!r} (format={fmt!r})")
    for path in files:
        if RawCorpus._infer_format(path) != inferred:
            raise ValueError("auto_val_split_pct requires one dataset format per folder; found mixed extensions")

    ext = {"arrow": ".arrow", "parquet": ".parquet", "jsonl": ".jsonl", "txt": ".txt", "bin": ".bin"}.get(inferred)
    if ext is None:
        raise ValueError(f"auto_val_split_pct unsupported dataset_format={inferred!r}")
    val_dir = os.path.join(dataset, "val")
    os.makedirs(val_dir, exist_ok=True)
    val_path = os.path.join(val_dir, "val_split" + ext)
    manifest_path = val_path + ".manifest.json"
    if os.path.exists(val_path):
        ddp_print(f"[auto-val-split] existing {val_path}; using it and NOT splitting again")
        cfg.val_dataset = val_path
        return val_path

    ddp_print(f"[auto-val-split] destructive split: pct={pct:.6g}% format={inferred} files={len(files)} -> {val_path}")
    if inferred == "arrow":
        stats = _auto_split_arrow(files, val_path, pct)
    elif inferred == "parquet":
        stats = _auto_split_parquet(files, val_path, pct)
    elif inferred == "jsonl":
        stats = _auto_split_jsonl(files, val_path, pct)
    elif inferred == "txt":
        stats = _auto_split_txt(files, val_path, pct)
    elif inferred == "bin":
        stats = _auto_split_bin(files, val_path, pct)
    else:
        raise ValueError(f"auto_val_split_pct unsupported dataset_format={inferred!r}")
    manifest = {"dataset": os.path.abspath(dataset), "format": inferred, "pct": pct,
                "files": files, "val_dataset": val_path, "stats": stats}
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    ddp_print(f"[auto-val-split] done: {stats} | manifest={manifest_path}")
    cfg.val_dataset = val_path
    return val_path


def _encode_batch_any(tok, texts: List[str]) -> List[List[int]]:
    if hasattr(tok, "encode_batch"):
        return tok.encode_batch(texts)
    return tok(texts)["input_ids"]


class ShardStream:
    _CHUNK_DOCS = 20000

    def __init__(self, path: str, cfg: Config, device: torch.device, tokenizer=None):
        self.cfg = cfg
        self.device = device
        if tokenizer is None:
            import loomformer
            tokenizer = loomformer.build_tokenizer(cfg)
        self.tok = tokenizer
        self._bos_id = _tok_special_id(self.tok, "<bos>")
        self._eos_id = _tok_special_id(self.tok, "<eos>")
        fmt_cfg = str(getattr(cfg, "dataset_format", "auto") or "auto")
        self._files = RawCorpus._resolve_files(path, fmt_cfg)
        if not self._files:
            raise ValueError(f"no corpus files found at {path!r} (format={fmt_cfg!r})")
        self._fmt = fmt_cfg if fmt_cfg != "auto" else RawCorpus._infer_format(self._files[0])
        self._text_field = getattr(cfg, "text_field", "text")
        self._rank = ddp_rank() if ddp_is_distributed() else 0
        self._world_size = ddp_world_size() if ddp_is_distributed() else 1
        row_counts = [self._file_row_count(path) for path in self._files]
        total_rows = int(sum(row_counts))
        global_start = total_rows * self._rank // self._world_size
        global_end = total_rows * (self._rank + 1) // self._world_size
        self._row_plan: List[Tuple[int, int, int]] = []
        cursor = 0
        for file_index, nrows in enumerate(row_counts):
            a = max(global_start, cursor) - cursor
            b = min(global_end, cursor + nrows) - cursor
            if a < b:
                self._row_plan.append((file_index, int(a), int(b)))
            cursor += nrows
        self._assigned_rows = global_end - global_start
        if self._assigned_rows <= 0 or not self._row_plan:
            raise ValueError(f"rank {self._rank} received no rows")
        self._content_need = int(cfg.seq_len) + 1 - (1 if self._bos_id is not None else 0)
        self._ram_queue: queue.Queue = queue.Queue(maxsize=max(1, int(cfg.prefetch_batches)))
        self._stop = threading.Event()
        self._producer_error = None
        self._gpu_batches = None
        self._gpu_pos = 0
        self._producer = threading.Thread(target=self._produce_cpu_batches, daemon=True, name=f"data-rank-{self._rank}")
        self._producer.start()
        plan = ", ".join(f"{os.path.basename(self._files[i])}[{a}:{b}]" for i, a, b in self._row_plan)
        if ddp_is_distributed():
            plans = [None] * self._world_size
            dist.all_gather_object(plans, (self._rank, self._assigned_rows, plan))
            if ddp_is_main():
                for rank, rows, desc in sorted(plans):
                    print(f"[data] rank={rank} rows={rows:,} plan={desc}", flush=True)
        else:
            print(f"[data] rank=0 rows={self._assigned_rows:,} plan={plan}", flush=True)

    def _file_row_count(self, path: str) -> int:
        if self._fmt == "parquet":
            import pyarrow.parquet as pq
            return int(pq.ParquetFile(path).metadata.num_rows)
        if self._fmt == "arrow":
            table, _ = _read_arrow_table_with_container(path)
            n = int(table.num_rows)
            del table
            return n
        if self._fmt == "jsonl":
            with open(path, "rb") as f:
                return sum(1 for line in f if line.strip())
        if self._fmt == "txt":
            return 1
        raise ValueError(f"unsupported raw dataset format {self._fmt!r}")

    def _iter_text_chunks(self, path: str, row_start: int, row_end: int):
        if self._fmt in ("parquet", "arrow"):
            if self._fmt == "parquet":
                import pyarrow.parquet as pq
                table = pq.read_table(path, columns=[self._text_field])
            else:
                table, _ = _read_arrow_table_with_container(path)
            col = table.column(self._text_field).slice(row_start, row_end - row_start)
            for start in range(0, len(col), self._CHUNK_DOCS):
                yield col.slice(start, min(self._CHUNK_DOCS, len(col) - start)).to_pylist()
            return
        if self._fmt == "jsonl":
            chunk = []
            row = 0
            with open(path, "rb") as f:
                for line in f:
                    if not line.strip():
                        continue
                    if row >= row_end:
                        break
                    if row >= row_start:
                        chunk.append(json.loads(line.decode("utf-8", errors="replace")).get(self._text_field, ""))
                        if len(chunk) >= self._CHUNK_DOCS:
                            yield chunk
                            chunk = []
                    row += 1
            if chunk:
                yield chunk
            return
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield [f.read()]

    def _produce_cpu_batches(self) -> None:
        try:
            carry = np.zeros(0, dtype=np.int64)
            eos = np.array([self._eos_id], dtype=np.int64) if self._eos_id is not None else None
            first_doc = True
            while not self._stop.is_set():
                for file_index, row_start, row_end in self._row_plan:
                    for texts in self._iter_text_chunks(self._files[file_index], row_start, row_end):
                        encoded = _encode_batch_any(self.tok, texts)
                        pieces = []
                        for ids in encoded:
                            if not first_doc and eos is not None:
                                pieces.append(eos)
                            pieces.append(np.asarray(ids, dtype=np.int64))
                            first_doc = False
                        if pieces:
                            block = np.concatenate(pieces)
                            carry = np.concatenate((carry, block)) if carry.size else block
                        batch_tokens = int(self.cfg.batch_size) * self._content_need
                        while carry.size >= batch_tokens and not self._stop.is_set():
                            block = carry[:batch_tokens].reshape(int(self.cfg.batch_size), self._content_need)
                            carry = carry[batch_tokens:]
                            batch = torch.from_numpy(block.copy())
                            if self._bos_id is not None:
                                bos = torch.full((batch.shape[0], 1), int(self._bos_id), dtype=torch.int64)
                                batch = torch.cat((bos, batch), dim=1)
                            if self.device.type == "cuda":
                                batch = batch.pin_memory()
                            self._ram_queue.put(batch)
                first_doc = True
        except BaseException as exc:
            self._producer_error = exc
            try:
                self._ram_queue.put_nowait(None)
            except queue.Full:
                pass

    def _get_cpu_batch(self) -> torch.Tensor:
        batch = self._ram_queue.get()
        if batch is None:
            raise RuntimeError("data producer failed") from self._producer_error
        return batch

    async def _load_gpu_chunk(self, count: int) -> None:
        loop = asyncio.get_running_loop()
        batches = []
        for _ in range(count):
            batches.append(await loop.run_in_executor(None, self._get_cpu_batch))
        host = torch.stack(batches)
        self._gpu_batches = host.to(self.device, non_blocking=True)
        self._gpu_pos = 0

    def _gpu_chunk_size(self) -> int:
        return min(int(self.cfg.gpu_prefetch_batches), int(self.cfg.prefetch_batches))

    async def prime(self) -> None:
        count = self._gpu_chunk_size()
        await self._load_gpu_chunk(count)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if ddp_is_distributed():
            ready = [None] * self._world_size
            dist.all_gather_object(ready, (self._rank, count))
            if ddp_is_main():
                for rank, n in sorted(ready):
                    print(f"[data] rank={rank} ready: RAM={int(self.cfg.prefetch_batches)} batches, GPU={n} batches", flush=True)
        else:
            print(f"[data] rank=0 ready: RAM={int(self.cfg.prefetch_batches)} batches, GPU={count} batches", flush=True)

    def sample_device_batch(self) -> torch.Tensor:
        batch = self._get_cpu_batch()
        return batch.to(self.device, non_blocking=True)

    def _sample_batch(self) -> torch.Tensor:
        return self.sample_device_batch()

    async def batches(self, n: int):
        chunk_size = self._gpu_chunk_size()
        yielded = 0
        while yielded < n:
            if self._gpu_batches is None or self._gpu_pos >= self._gpu_batches.shape[0]:
                await self._load_gpu_chunk(min(chunk_size, n - yielded))
            while self._gpu_pos < self._gpu_batches.shape[0] and yielded < n:
                batch = self._gpu_batches[self._gpu_pos]
                self._gpu_pos += 1
                yielded += 1
                yield batch

    def close(self) -> None:
        self._stop.set()

def is_sft_dataset(cfg: Config) -> bool:
    return str(getattr(cfg, "dataset_format", "auto") or "auto").lower() == "sft"

__all__ = ('TokenStream', 'RawCorpus', '_auto_val_split_pct', '_split_count', '_concat_arrow_tables', '_read_arrow_table_with_container', '_write_arrow_table_preserving_container', '_write_arrow_ipc_file', '_auto_split_arrow', '_auto_split_parquet', '_auto_split_jsonl', '_auto_split_txt', '_auto_split_bin', 'maybe_auto_val_split', '_encode_batch_any', 'ShardStream', 'is_sft_dataset')
