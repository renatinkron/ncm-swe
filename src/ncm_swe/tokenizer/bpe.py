"""Corpus-trained BPE pre-tokenizer (stage-1 input for the VQ-VAE).

Falls back gracefully in tests; the real tokenizer is trained on
artifacts/corpus via `scripts/train_bpe.py` or inline in train_vqvae.
"""

import json
from pathlib import Path


class CorpusBPE:
    def __init__(self, tok):
        self._t = tok
        self.pad_token_id = tok.token_to_id("[PAD]")
        self.eos_token_id = tok.token_to_id("[EOS]")
        self.bos_token_id = tok.token_to_id("[BOS]")
        self.vocab_size = tok.get_vocab_size()

    @classmethod
    def train(cls, texts, vocab_size: int = 8192, out_path: str | Path | None = None):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[PAD]", "[EOS]", "[BOS]"],
            show_progress=False,
        )
        tok.train_from_iterator(texts, trainer)
        if out_path:
            Path(out_path).parent.mkdir(parents=True, exist_ok=True)
            tok.save(str(out_path))
        return cls(tok)

    @classmethod
    def load(cls, path: str | Path):
        from tokenizers import Tokenizer

        return cls(Tokenizer.from_file(str(path)))

    def encode(self, text: str) -> list[int]:
        return self._t.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self._t.encode_batch(texts)]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._t.decode(ids, skip_special_tokens=skip_special_tokens)

    def decode_batch(self, ids, skip_special_tokens: bool = False):
        return self._t.decode_batch(ids, skip_special_tokens=skip_special_tokens)

    def pad(self, ids: list[int], length: int) -> list[int]:
        ids = ids[:length]
        return ids + [self.pad_token_id] * (length - len(ids))

    def to_json(self) -> str:
        return self._t.to_str()

    def meta(self) -> dict:
        return {"vocab_size": self.vocab_size, "pad": self.pad_token_id,
                "eos": self.eos_token_id, "bos": self.bos_token_id}


def corpus_texts(corpus_dir: str | Path):
    for shard in sorted(Path(corpus_dir).glob("shard-*.jsonl")):
        for line in shard.read_text().splitlines():
            if line.strip():
                yield json.loads(line)["text"]
