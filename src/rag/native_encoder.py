"""
NativeSLMEncoder — Self-Contained Embedding Engine using Kruuboo-SLM-0.2
========================================================================

Extracts dense 768-dimensional sequence representations directly from the
active ONNX Runtime session (`hidden_states` = `ln_f(h)`), eliminating the
need to load a separate 421 MB PyTorch checkpoint alongside ONNX Runtime.
"""

from pathlib import Path
from typing import List, Union, Optional
import numpy as np
import onnxruntime as ort
import yaml

from src.tokenizer import TinyTokenizer


class NativeSLMEncoder:
    """
    Self-contained RAG embedding retriever powered by Kruuboo-SLM-0.2 ONNX Runtime.
    Shares the active ONNX InferenceSession with the generation engine for zero extra RAM.
    """

    def __init__(
        self,
        model_path: str = "models/tiny_llm_quantized.onnx",
        tokenizer_path: str = "tokenizer.json",
        config_path: str = "config.yaml",
        device: Optional[str] = None,
        session: Optional[ort.InferenceSession] = None,
        tokenizer: Optional[TinyTokenizer] = None,
    ):
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
        self.head_dim = self.d_model // self.n_heads

        # Reuse or load Tokenizer
        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif Path(tokenizer_path).exists():
            self.tokenizer = TinyTokenizer.load(tokenizer_path)
        else:
            self.tokenizer = TinyTokenizer(vocab_size=self.vocab_size)

        # Reuse existing ONNX session or load ONNX model
        self.session = session
        if self.session is None:
            resolved_path = model_path
            if resolved_path.endswith(".pt") or not Path(resolved_path).exists():
                for candidate in ("models/tiny_llm_quantized.onnx", "models/tiny_llm.onnx"):
                    if Path(candidate).exists():
                        resolved_path = candidate
                        break
            if Path(resolved_path).exists() and resolved_path.endswith(".onnx"):
                self.session = ort.InferenceSession(resolved_path, providers=["CPUExecutionProvider"])

        self.has_kv_cache = False
        self.kv_cache_4d = False
        self.past_kv_names: List[str] = []
        self.has_hidden_states = False
        if self.session is not None:
            input_names = [inp.name for inp in self.session.get_inputs()]
            output_names = [out.name for out in self.session.get_outputs()]
            self.has_hidden_states = "hidden_states" in output_names
            if "past_key_0" in input_names:
                self.has_kv_cache = True
                self.kv_cache_4d = True
                self.past_kv_names = [n for n in input_names if n.startswith("past_")]
                self.n_layers = len(self.past_kv_names) // 2
                try:
                    pk0 = next(inp for inp in self.session.get_inputs() if inp.name == "past_key_0")
                    shape = pk0.shape
                    if len(shape) >= 4:
                        if isinstance(shape[1], int) and shape[1] > 0:
                            self.n_heads = shape[1]
                        if isinstance(shape[3], int) and shape[3] > 0:
                            self.head_dim = shape[3]
                except Exception:
                    pass
            elif "past_keys" in input_names and "past_values" in input_names:
                self.has_kv_cache = True
                self.kv_cache_4d = False
                try:
                    pk_meta = next(inp for inp in self.session.get_inputs() if inp.name == "past_keys")
                    shape = pk_meta.shape
                    if len(shape) >= 5:
                        if isinstance(shape[0], int) and shape[0] > 0:
                            self.n_layers = shape[0]
                        if isinstance(shape[2], int) and shape[2] > 0:
                            self.n_heads = shape[2]
                        if isinstance(shape[4], int) and shape[4] > 0:
                            self.head_dim = shape[4]
                except Exception:
                    pass

        # Lazy fallback only if an older ONNX model lacks 'hidden_states' output
        self._torch_model = None
        self._model_path = model_path
        self._device = device

    def _encode_batch_onnx(self, batch_ids: List[List[int]]) -> np.ndarray:
        """Run ONNX forward pass with static batch_size=1 (no padding waste) to extract 768d embeddings."""
        if self.kv_cache_4d:
            empty_4d = np.zeros((1, self.n_heads, 0, self.head_dim), dtype=np.float32)
            base_feed = {k_name: empty_4d for k_name in self.past_kv_names}
            vecs = []
            for tokens in batch_ids:
                seq_arr = np.array([tokens], dtype=np.int64)
                base_feed["input_ids"] = seq_arr
                features = self.session.run(["hidden_states"], base_feed)[0]  # [1, T, d_model]
                pooled = features[0].mean(axis=0)
                norm = np.linalg.norm(pooled)
                vecs.append(pooled / max(float(norm), 1e-12))
            return np.stack(vecs, axis=0).astype(np.float32)

        max_len = max(1, max(len(b) for b in batch_ids))
        padded = [b + [TinyTokenizer.PAD_ID] * (max_len - len(b)) for b in batch_ids]
        padded_ids = np.array(padded, dtype=np.int64)
        batch_size = padded_ids.shape[0]
        feed = {"input_ids": padded_ids}
        if self.has_kv_cache:
            feed["past_keys"] = np.zeros(
                (self.n_layers, batch_size, self.n_heads, 0, self.head_dim), dtype=np.float32
            )
            feed["past_values"] = np.zeros(
                (self.n_layers, batch_size, self.n_heads, 0, self.head_dim), dtype=np.float32
            )

        features = self.session.run(["hidden_states"], feed)[0]  # [B, T, d_model]
        mask = (padded_ids != TinyTokenizer.PAD_ID)[..., None].astype(np.float32)  # [B, T, 1]
        mask_sum = np.maximum(mask.sum(axis=1), 1e-9)  # [B, 1]
        pooled = (features * mask).sum(axis=1) / mask_sum  # [B, d_model]
        norms = np.linalg.norm(pooled, axis=-1, keepdims=True)
        return (pooled / np.maximum(norms, 1e-12)).astype(np.float32)

    def _encode_batch_torch_fallback(self, padded_ids: List[List[int]]) -> np.ndarray:
        """Fallback only used if an un-updated ONNX file without 'hidden_states' is loaded."""
        import torch
        from src.model import TinyLLM

        if self._torch_model is None:
            dev = self._device or ("mps" if torch.backends.mps.is_available() else "cpu")
            self._device = dev
            self._torch_model = TinyLLM(
                vocab_size=self.vocab_size,
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_layers=self.n_layers,
                d_ff=self.d_ff,
                max_seq_len=self.max_seq_len,
                dropout=0.0,
            )
            pt_path = self._model_path if self._model_path.endswith(".pt") else "models/tiny_llm.pt"
            if Path(pt_path).exists():
                state_dict = torch.load(pt_path, map_location="cpu")
                self._torch_model.load_state_dict(state_dict, strict=False)
            self._torch_model.to(self._device)
            self._torch_model.eval()

        with torch.no_grad():
            x = torch.tensor(padded_ids, dtype=torch.long, device=self._device)
            embeddings = self._torch_model.get_embeddings(x, pad_id=TinyTokenizer.PAD_ID)
            return embeddings.cpu().numpy().astype(np.float32)

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

            if self.session is not None and self.has_hidden_states:
                emb = self._encode_batch_onnx(batch_ids)
            else:
                max_len = max(1, max_len)
                padded = [b + [TinyTokenizer.PAD_ID] * (max_len - len(b)) for b in batch_ids]
                emb = self._encode_batch_torch_fallback(padded)

            all_embeddings.append(emb)

        result = np.vstack(all_embeddings)
        return result[0] if is_single else result
