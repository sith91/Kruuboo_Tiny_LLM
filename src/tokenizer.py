"""
BPE Tokenizer for Tiny LLM
----------------------------
Replaces the old character-level tokenizer with a proper Byte-Pair Encoding (BPE)
tokenizer built using HuggingFace's `tokenizers` library.

Benefits over char-level:
  - Whole words / subwords become single tokens  → much longer effective context
  - Handles unknown characters gracefully via byte-fallback
  - Same encode/decode/save/load interface as before — no other file needs changing
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Try to import the HuggingFace `tokenizers` library.
# If it isn't installed yet, we fall back to the old character-level behaviour
# so the project doesn't crash outright — just print a warning.
# ---------------------------------------------------------------------------
try:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.processors import TemplateProcessing
    _HF_AVAILABLE = True
except ImportError:                                    # pragma: no cover
    _HF_AVAILABLE = False


# ── Default BPE vocabulary size ──────────────────────────────────────────────
DEFAULT_VOCAB_SIZE = 4096   # 4 K tokens — tiny but far better than ~97 chars


class TinyTokenizer:
    """
    BPE tokenizer with the same interface as the old character-level one.

    Special tokens
    --------------
    ID 0  →  <pad>   (padding / unknown)
    ID 1  →  <eos>   (end-of-sequence — generation stop signal)
    """

    PAD_ID: int = 0
    EOS_ID: int = 1

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, _tokenizer: Optional["Tokenizer"] = None, vocab_size: int = DEFAULT_VOCAB_SIZE):
        """
        Don't call this directly.  Use:
          • TinyTokenizer.train_from_text(text)   — build a new BPE tokenizer
          • TinyTokenizer.load(path)              — restore a saved tokenizer
        """
        if not _HF_AVAILABLE:
            self._init_charfallback()
            return

        self._tokenizer: Optional[Tokenizer] = _tokenizer
        self._vocab_size: int = vocab_size if _tokenizer is None else _tokenizer.get_vocab_size()

    # ── Public API (unchanged interface) ─────────────────────────────────────

    @property
    def vocab_size(self) -> int:
        if not _HF_AVAILABLE:
            return self._char_vocab_size
        return self._vocab_size

    def encode(self, text: str) -> List[int]:
        """Convert text → list of integer token IDs."""
        if not _HF_AVAILABLE or self._tokenizer is None:
            return self._char_encode(text)
        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: List[int]) -> str:
        """Convert list of integer token IDs → text."""
        if not _HF_AVAILABLE or self._tokenizer is None:
            return self._char_decode(ids)
        # Filter pad and eos before decoding
        filtered = [i for i in ids if i not in (self.PAD_ID, self.EOS_ID)]
        return self._tokenizer.decode(filtered)

    def save(self, path: str = "tokenizer.json"):
        """Persist tokenizer to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        if not _HF_AVAILABLE or self._tokenizer is None:
            self._char_save(path)
            return
        # Save the HF tokenizer as JSON
        self._tokenizer.save(str(p))
        # Inject a small meta-block so load() can detect the format
        with open(p) as f:
            data = json.load(f)
        data["_tiny_llm_tokenizer"] = {"type": "bpe", "vocab_size": self.vocab_size}
        with open(p, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str = "tokenizer.json") -> "TinyTokenizer":
        """Restore a tokenizer from disk."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {path}")

        with open(p) as f:
            data = json.load(f)

        # ── Detect format ────────────────────────────────────────────────────
        if ("_tiny_llm_tokenizer" in data or "model" in data) and _HF_AVAILABLE:
            # New BPE format
            # Clean injected metadata to avoid HF parser issues
            data_copy = dict(data)
            data_copy.pop("_tiny_llm_tokenizer", None)
            hf_tok = Tokenizer.from_str(json.dumps(data_copy))
            obj = cls(_tokenizer=hf_tok)
            return obj

        elif "chars" in data:
            # Old character-level format — keep working without changes
            obj = cls.__new__(cls)
            obj._init_charfallback(data["chars"])
            return obj

        else:
            raise ValueError(f"Unrecognised tokenizer file format: {path}")

    # ── BPE training ─────────────────────────────────────────────────────────

    @classmethod
    def train_from_text(
        cls,
        text: str,
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        save_path: Optional[str] = None,
    ) -> "TinyTokenizer":
        """
        Train a BPE tokenizer on the given text string and return a
        ready-to-use TinyTokenizer.

        Parameters
        ----------
        text       : raw training text (can be a single long string)
        vocab_size : target vocabulary size (default 4096)
        save_path  : if given, automatically saves the tokenizer there

        Example
        -------
        >>> tok = TinyTokenizer.train_from_text("Hello world ...", vocab_size=4096)
        >>> tok.encode("Hello")
        [123, ...]
        """
        if not _HF_AVAILABLE:
            print("⚠️  `tokenizers` library not found — using character-level fallback.")
            obj = cls.__new__(cls)
            obj._init_charfallback()
            return obj

        print(f"🔨 Training BPE tokenizer  (vocab_size={vocab_size}) ...")

        # Special tokens
        special_tokens = ["<pad>", "<eos>", "<unk>", "<|im_start|>", "<|im_end|>"]

        # Build tokenizer
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=1,
            show_progress=False,
        )

        # Train from the in-memory string by writing a temp iterator
        tokenizer.train_from_iterator([text], trainer=trainer)

        # Guarantee that pad=0, eos=1
        vocab = tokenizer.get_vocab()
        assert vocab.get("<pad>", None) == 0, "Expected <pad> at ID 0"

        print(f"✅ BPE vocabulary ready — {tokenizer.get_vocab_size()} tokens")

        obj = cls(_tokenizer=tokenizer, vocab_size=tokenizer.get_vocab_size())

        if save_path:
            obj.save(save_path)
            print(f"💾 Tokenizer saved to {save_path}")

        return obj

    # ── Character-level fallback (for environments without `tokenizers`) ──────

    def _init_charfallback(self, chars: Optional[List[str]] = None):
        """Set up the old character-level data structures."""
        if chars is None:
            chars = [chr(i) for i in range(32, 127)] + ['\n', '\t']
        self._chars = chars
        self._char_vocab_size = len(chars) + 1        # +1 for padding at 0
        self._char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
        self._idx_to_char = {i + 1: ch for i, ch in enumerate(chars)}
        self._tokenizer = None
        if not hasattr(self, "_vocab_size"):
            self._vocab_size = self._char_vocab_size

    def _char_encode(self, text: str) -> List[int]:
        return [self._char_to_idx.get(ch, 0) for ch in text if ch in self._char_to_idx]

    def _char_decode(self, ids: List[int]) -> str:
        return ''.join([self._idx_to_char.get(i, '') for i in ids if i != 0])

    def _char_save(self, path: str):
        with open(path, 'w') as f:
            json.dump({
                'chars': self._chars,
                'vocab_size': self._char_vocab_size,
            }, f, indent=2)