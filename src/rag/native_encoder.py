"""
NativeSLMEncoder — Self-Contained Embedding Engine using Kruuboo-SLM-0.2
========================================================================

Extracts dense 768-dimensional sequence representations directly from the
custom 14-layer TinyLLM transformer decoder. Eliminates external dependencies
on sentence-transformers / HuggingFace models.
"""

import os
from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import torch
import yaml

from src.model import TinyLLM
from src.tokenizer import TinyTokenizer


class NativeSLMEncoder:
    """
    Self-contained RAG embedding retriever powered by Kruuboo-SLM-0.2.
    """

    def __init__(
        self,
        model_path: str = "models/tiny_llm.pt",
        tokenizer_path: str = "tokenizer.json",
        config_path: str = "config.yaml",
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load configuration
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            model_cfg = config.get("model", {})
        except Exception:
            model_cfg = {}

        self.vocab_size = model_cfg.get("vocab_size", 8192)
        self.d_model = model_cfg.get("d_model", 768)
        self.n_heads = model_cfg.get("n_heads", 12)
        self.n_layers = model_cfg.get("n_layers", 14)
        self.d_ff = model_cfg.get("d_ff", 3072)
        self.max_seq_len = model_cfg.get("max_seq_len", 1024)

        # Load Tokenizer
        if Path(tokenizer_path).exists():
            self.tokenizer = TinyTokenizer.load(tokenizer_path)
        else:
            self.tokenizer = TinyTokenizer(vocab_size=self.vocab_size)

        # Instantiate TinyLLM
        self.model = TinyLLM(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            dropout=0.0,
        )

        # Load checkpoint if available
        if Path(model_path).exists():
            try:
                state_dict = torch.load(model_path, map_location="cpu")
                self.model.load_state_dict(state_dict, strict=False)
            except Exception:
                pass

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode(
        self,
        texts: Union[str, List[str]],
        show_progress_bar: bool = False,
        batch_size: int = 16,
    ) -> np.ndarray:
        """
        Encode a single string or list of strings into normalized 768-d embedding vectors.
        """
        is_single = isinstance(texts, str)
        text_list = [texts] if is_single else texts

        if not text_list:
            return np.zeros((0, self.d_model), dtype=np.float32)

        all_embeddings = []

        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i : i + batch_size]
            batch_ids = []
            max_len = 0

            for t in batch_texts:
                tokens = self.tokenizer.encode(t if t.strip() else "<pad>")
                tokens = tokens[: self.max_seq_len]
                batch_ids.append(tokens)
                if len(tokens) > max_len:
                    max_len = len(tokens)

            max_len = max(1, max_len)
            padded = []
            for b in batch_ids:
                pad_amt = max_len - len(b)
                padded.append(b + [TinyTokenizer.PAD_ID] * pad_amt)

            x = torch.tensor(padded, dtype=torch.long, device=self.device)
            embeddings = self.model.get_embeddings(x, pad_id=TinyTokenizer.PAD_ID)
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))

        result = np.vstack(all_embeddings)
        return result[0] if is_single else result
