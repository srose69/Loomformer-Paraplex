"""Deterministic tokenizer fixture shared by the integration matrix and tests."""

from __future__ import annotations

from pathlib import Path
import string


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<|im_start|>",
    "<|im_end|>",
    "<think>",
    "</think>",
    "<tool_call>",
    "</tool_call>",
    "<tool_response>",
    "</tool_response>",
    "<CARRY>",
]


def build_synthetic_bpe(output: Path, vocab_size: int = 256) -> int:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    printable = string.ascii_letters + string.digits + string.punctuation + " \n\t"
    corpus = [f"complete ASCII coverage: {printable}"]
    corpus.extend(
        (
            f"document {index} alpha{index} beta{index * index} "
            f"question{index % 37} answer{index % 53} assistant system user "
            "reasoning tools packed causal sequence LoomFormer synthetic"
        )
        for index in range(2048)
    )
    corpus.append(
        "<think>reason</think> <tool_call>call</tool_call> "
        "<tool_response>result</tool_response>"
    )

    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        corpus,
        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
        ),
        length=len(corpus),
    )
    actual = tokenizer.get_vocab_size()
    if actual != vocab_size:
        raise AssertionError(
            f"synthetic BPE has {actual} entries, expected exactly {vocab_size}"
        )
    missing = [
        token for token in SPECIAL_TOKENS if tokenizer.token_to_id(token) is None
    ]
    if missing:
        raise AssertionError(f"synthetic BPE is missing special tokens: {missing}")
    tokenizer.save(str(output))
    return actual
