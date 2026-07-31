from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

class ByteTokenizer:
    vocab_size = 256

    def encode(self, s: str) -> List[int]:
        return list(s.encode("utf-8"))

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [list(s.encode("utf-8")) for s in texts]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return bytes(int(i) % 256 for i in ids).decode("utf-8", errors="replace")


class BPETokenizerWrap:
    def __init__(self, tk):
        self.tk = tk
        self.vocab_size = tk.get_vocab_size()

    def encode(self, s: str) -> List[int]:
        return self.tk.encode(s).ids

    def encode_batch(self, texts: List[str]) -> List[List[int]]:
        return [e.ids for e in self.tk.encode_batch(texts)]

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        return self.tk.decode([int(i) for i in ids], skip_special_tokens=skip_special_tokens)

    def special_id(self, token: str) -> Optional[int]:
        return self.tk.token_to_id(token)

    @staticmethod
    def load(path: str) -> "BPETokenizerWrap":
        from tokenizers import Tokenizer
        return BPETokenizerWrap(Tokenizer.from_file(path))


DEFAULT_SPECIAL_TOKENS = [
    "<pad>", "<bos>", "<eos>",
    "<|im_start|>", "<|im_end|>",                      
    "<think>", "</think>",                            
    "<tool_call>", "</tool_call>",                     
    "<tool_response>", "</tool_response>",            
    "<CARRY>",                                          
]

def _tok_special_id(tok, name: str) -> Optional[int]:
    fn = getattr(tok, "special_id", None)
    return fn(name) if fn is not None else None


class ChatTemplate:
    def __init__(self, tok, template_path: str = "chat_template.jinja"):
        import jinja2  # lazy: only chat-template users need this dependency at all
        self.tok = tok
        resolved = template_path
        if not os.path.isfile(resolved):
            # Fall back to a path next to this module, so callers don't need to be
            # run from the repo root for the default filename to resolve.
            candidate = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), template_path)
            if os.path.isfile(candidate):
                resolved = candidate
        with open(resolved, "r", encoding="utf-8") as f:
            src = f.read()
        self._tpl = jinja2.Environment().from_string(src)

        im_start = _tok_special_id(tok, "<|im_start|>")
        im_end = _tok_special_id(tok, "<|im_end|>")
        if im_start is None or im_end is None:
            raise ValueError(
                "ChatTemplate needs <|im_start|>/<|im_end|> in the tokenizer's vocab "
                "(see loomformer.DEFAULT_SPECIAL_TOKENS) -- retrain it with those "
                "special tokens if this one predates them."
            )
        self.im_start_id, self.im_end_id = im_start, im_end
        self.bos_id = _tok_special_id(tok, "<bos>")
        self.bos_token = "<bos>" if self.bos_id is not None else ""
        eos_id = _tok_special_id(tok, "<eos>")
        self.stop_ids = {i for i in (im_end, eos_id) if i is not None}
        self._assistant_header_ids = [im_start] + tok.encode("assistant\n")

    def render_text(self, messages: List[Dict], tools: Optional[List[Dict]] = None,
                     add_generation_prompt: bool = False) -> str:
        kwargs = {"messages": messages, "add_generation_prompt": add_generation_prompt,
                  "bos_token": self.bos_token}
        if tools is not None:  # presence vs absence is meaningful -- see sft.md
            kwargs["tools"] = tools
        return self._tpl.render(**kwargs)

    def render_prompt_ids(self, messages: List[Dict], tools: Optional[List[Dict]] = None) -> List[int]:
        return self.tok.encode(self.render_text(messages, tools=tools, add_generation_prompt=True))

    @staticmethod
    def _find_all(haystack: List[int], needle: List[int]) -> List[int]:
        if not needle:
            return []
        out: List[int] = []
        n, m = len(haystack), len(needle)
        i = 0
        while i <= n - m:
            if haystack[i:i + m] == needle:
                out.append(i)
                i += m
            else:
                i += 1
        return out

    def render_training_ids(self, messages: List[Dict], tools: Optional[List[Dict]] = None
                             ) -> Tuple[List[int], List[int]]:
        ids = self.tok.encode(self.render_text(messages, tools=tools, add_generation_prompt=False))
        mask = [0] * len(ids)
        for p in self._find_all(ids, self._assistant_header_ids):
            start = p + len(self._assistant_header_ids)
            q = start
            while q < len(ids) and ids[q] != self.im_end_id:
                q += 1
            end = min(q, len(ids) - 1)  # include the closing <|im_end|> itself
            for k in range(start, end + 1):
                mask[k] = 1
        return ids, mask

    def parse_tool_calls(self, text: str) -> List[Dict]:
        calls: List[Dict] = []
        i = 0
        while True:
            s = text.find("<tool_call>", i)
            if s < 0:
                break
            e = text.find("</tool_call>", s)
            if e < 0:
                break
            payload = text[s + len("<tool_call>"):e].strip()
            i = e + len("</tool_call>")
            try:
                obj = json.loads(payload)
                calls.append({"id": f"call_{len(calls)}", "type": "function",
                              "function": {"name": obj.get("name"), "arguments": obj.get("arguments")}})
            except Exception:
                continue
        return calls

__all__ = ('ByteTokenizer', 'BPETokenizerWrap', 'DEFAULT_SPECIAL_TOKENS', '_tok_special_id', 'ChatTemplate')
