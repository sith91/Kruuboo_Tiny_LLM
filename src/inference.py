"""
InferenceEngine — Proper token generation for TinyLLM
======================================================

Fixes over the original generate() loop:

  1. Top-k filtering      — keeps only the k most likely tokens before sampling,
                            eliminating random garbage from the long tail.

  2. Top-p (nucleus)      — keeps the smallest set of tokens whose cumulative
     sampling               probability ≥ p.  Combined with top-k this produces
                            coherent, varied text.

  3. Repetition penalty   — discounts logits for tokens that already appeared in
                            the context, preventing looping / repetition.

  4. Greedy at temp=0     — temperature=0 uses argmax (fully deterministic).

  5. EOS token handling   — stops on both pad (0) and EOS (1) from the BPE
                            tokenizer, not just on 0.

  6. No side-effect prints — generation is pure (no print inside loop).
                             An optional on_token callback handles streaming.

  7. Prompt stripping      — only the *new* tokens after the prompt are returned,
                             not the echoed prompt text.

  8. Sentence-boundary     — post-processing trims incomplete trailing sentences.
"""

from __future__ import annotations

import re
import numpy as np
from typing import Callable, List, Optional, Set, Tuple


# ── Special token IDs (must match TinyTokenizer) ─────────────────────────────
PAD_ID = 0
EOS_ID = 1

DEFAULT_STOP_STRINGS = [
    "<|im_end|>", "<|im_start|>",
    "\n<|im_start|>", "\n<|im_end|>",
    "\nuser\n", "\nassistant\n", "\nsystem\n",
    "\nuser:", "\nassistant:", "\nsystem:",
    "\nUser:", "\nAssistant:", "\nHuman:",
    "\nuser", "\nassistant", "\nsystem",
    "<eos>", "<pad>",
]


class InferenceEngine:
    """
    Wraps an ONNX InferenceSession and a TinyTokenizer to provide clean,
    high-quality text generation.

    Parameters
    ----------
    session    : onnxruntime.InferenceSession  (the loaded ONNX model)
    tokenizer  : TinyTokenizer
    input_name : str  (name of the ONNX input node, e.g. 'input_ids')
    max_seq_len: int  (model's context window — from RAGSystem)
    """

    def __init__(self, session, tokenizer, input_name: str, max_seq_len: int):
        self.session     = session
        self.tokenizer   = tokenizer
        self.input_name  = input_name
        self.max_seq_len = max_seq_len
        self._init_stop_tokens()

        # KV Cache detection & metadata extraction (supports 4D per-layer or 5D stacked)
        self.has_kv_cache = False
        self.kv_cache_4d = False
        self.n_layers = 14
        self.n_heads = 12
        self.head_dim = 64
        self.past_kv_names: List[str] = []
        self.present_kv_names: List[str] = []

        if self.session is not None:
            input_names = [inp.name for inp in self.session.get_inputs()]
            if "past_key_0" in input_names:
                self.has_kv_cache = True
                self.kv_cache_4d = True
                self.past_kv_names = [n for n in input_names if n.startswith("past_")]
                self.n_layers = len(self.past_kv_names) // 2
                self.present_kv_names = [
                    f"present_{t}_{i}" for i in range(self.n_layers) for t in ("key", "value")
                ]
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

    def _init_stop_tokens(self) -> None:
        """Precomputes single-token stop IDs and multi-token stop sequences to avoid O(N^2) decoding."""
        self.stop_token_ids: Set[int] = {PAD_ID, EOS_ID}
        self.stop_token_seqs: List[Tuple[int, ...]] = []
        self.sentence_boundary_ids: Set[int] = set()

        if self.tokenizer is not None:
            # Explicitly add special tokens if available
            for special in ["<|im_end|>", "<|im_start|>", "<eos>", "<pad>"]:
                try:
                    enc = self.tokenizer.encode(special)
                    if len(enc) == 1:
                        self.stop_token_ids.add(enc[0])
                except Exception:
                    pass

            for s in DEFAULT_STOP_STRINGS:
                try:
                    tokens = self.tokenizer.encode(s)
                    if not tokens:
                        continue
                    if len(tokens) == 1:
                        self.stop_token_ids.add(tokens[0])
                    else:
                        seq = tuple(tokens)
                        if seq not in self.stop_token_seqs:
                            self.stop_token_seqs.append(seq)
                except Exception:
                    pass

            # Sort sequences by length descending so longer matching sequences trigger first
            self.stop_token_seqs.sort(key=len, reverse=True)

            # Precompute sentence boundary token IDs (e.g., ., !, ?, \n, :) for trailing checks
            for punct in [".", "!", "?", "\n", ".\n", "!\n", "?\n", ":", ":\n"]:
                try:
                    enc = self.tokenizer.encode(punct)
                    if enc:
                        self.sentence_boundary_ids.update(enc)
                except Exception:
                    pass

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens:    int   = 120,
        temperature:       float = 0.65,
        min_p:             float = 0.05,
        top_k:             int   = 0,
        top_p:             float = 1.0,
        repetition_penalty:float = 1.15,
        repetition_window: int   = 50,
        stop_sequences:    Optional[List[str]] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate text that continues *prompt* with Min-P sampling and contextual temperature.

        Parameters
        ----------
        prompt             : The text the model will continue.
        max_new_tokens     : Maximum tokens to generate (not counting the prompt).
        temperature        : 0 = greedy, 0.2 = factual/remedies, 0.65 = natural conversational.
        min_p              : Dynamic cutoff threshold relative to top token (p_max * min_p).
        top_k              : Optional top-k filter (0 = disabled in favor of Min-P).
        top_p              : Optional nucleus probability threshold (1.0 = disabled).
        repetition_penalty : Values > 1 penalise repeated tokens (1 = off).
        repetition_window  : Rolling token window for penalty (defaults to 50 generated tokens).
        stop_sequences     : Optional custom stop strings (defaults to DEFAULT_STOP_STRINGS).
        on_token           : Optional callback for streaming tokens.

        Returns
        -------
        str : Only the *newly generated* text (prompt is stripped).
        """
        if self.session is None:
            return "[No model loaded — run src/train.py then scripts/export_onnx.py]"

        # ── Setup active stop tokens & sequences ───────────────────────────
        active_stop_strings = stop_sequences if stop_sequences is not None else DEFAULT_STOP_STRINGS
        if stop_sequences is not None:
            active_stop_token_ids = set(self.stop_token_ids)
            active_stop_token_seqs = list(self.stop_token_seqs)
            for s in stop_sequences:
                enc = self.tokenizer.encode(s)
                if len(enc) == 1:
                    active_stop_token_ids.add(enc[0])
                elif len(enc) > 1:
                    seq = tuple(enc)
                    if seq not in active_stop_token_seqs:
                        active_stop_token_seqs.append(seq)
            active_stop_token_seqs.sort(key=len, reverse=True)
        else:
            active_stop_token_ids = self.stop_token_ids
            active_stop_token_seqs = self.stop_token_seqs

        # ── Encode prompt ──────────────────────────────────────────────────
        prompt_ids = self.tokenizer.encode(prompt)

        # Keep the trailing window if prompt is longer than context
        max_prompt = self.max_seq_len - 1          # leave room for at least one new token
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        current_ids: List[int] = list(prompt_ids)
        generated_ids: List[int] = []

        if self.has_kv_cache:
            # ── KV Cache Accelerated Generation (O(1) decode per step) ─────────
            input_arr = np.array([prompt_ids], dtype=np.int64)
            if self.kv_cache_4d:
                req_outputs = ["logits"] + self.present_kv_names
                empty_4d = np.zeros((1, self.n_heads, 0, self.head_dim), dtype=np.float32)
                feed = {"input_ids": input_arr}
                for k_name in self.past_kv_names:
                    feed[k_name] = empty_4d
                outputs = self.session.run(req_outputs, feed)
                logits = outputs[0][0, -1, :].astype(np.float64)
                past_kv_list = outputs[1:]
            else:
                empty_k = np.zeros((self.n_layers, 1, self.n_heads, 0, self.head_dim), dtype=np.float32)
                empty_v = np.zeros((self.n_layers, 1, self.n_heads, 0, self.head_dim), dtype=np.float32)
                outputs = self.session.run(
                    ["logits", "present_keys", "present_values"],
                    {
                        "input_ids": input_arr,
                        "past_keys": empty_k,
                        "past_values": empty_v,
                    },
                )
                logits = outputs[0][0, -1, :].astype(np.float64)
                past_k = outputs[1]
                past_v = outputs[2]

            for step in range(max_new_tokens):
                # Repetition penalty (applied strictly to generated tokens within the rolling window)
                if repetition_penalty != 1.0 and generated_ids:
                    penalty_tokens = generated_ids[-repetition_window:] if repetition_window > 0 else generated_ids
                    logits = _apply_repetition_penalty(logits, penalty_tokens, repetition_penalty)

                # Temperature scaling  (temperature=0 → greedy)
                if temperature == 0:
                    next_token = int(np.argmax(logits))
                else:
                    logits = logits / max(temperature, 1e-8)
                    if min_p > 0.0:
                        logits = _min_p_filter(logits, min_p)
                    if top_k > 0:
                        logits = _top_k_filter(logits, top_k)
                    if top_p < 1.0:
                        logits = _top_p_filter(logits, top_p)
                    probs = _softmax(logits)
                    next_token = int(np.random.choice(len(probs), p=probs))

                # Stop check on single token IDs in O(1)
                if next_token in active_stop_token_ids:
                    break

                generated_ids.append(next_token)

                # Check multi-token stop sequences in token space (O(1) tail slice)
                should_stop = False
                for seq in active_stop_token_seqs:
                    seq_len = len(seq)
                    if len(generated_ids) >= seq_len and tuple(generated_ids[-seq_len:]) == seq:
                        should_stop = True
                        del generated_ids[-seq_len:]
                        break
                if should_stop:
                    break

                # Fallback check at sentence boundaries
                if next_token in self.sentence_boundary_ids:
                    tail_tokens = generated_ids[-min(len(generated_ids), 16):]
                    tail_text = self.tokenizer.decode(tail_tokens)
                    for stop_seq in active_stop_strings:
                        if stop_seq in tail_text:
                            should_stop = True
                            break
                    if should_stop:
                        break

                # Streaming callback
                if on_token is not None:
                    token_text = self.tokenizer.decode([next_token])
                    on_token(token_text)

                # Stop if context window limit is reached
                cur_cache_len = past_kv_list[0].shape[2] if self.kv_cache_4d else past_k.shape[3]
                if cur_cache_len >= self.max_seq_len:
                    break

                if step == max_new_tokens - 1:
                    break

                # 2. Decode step: evaluate single new token using cached past KV
                input_arr = np.array([[next_token]], dtype=np.int64)
                if self.kv_cache_4d:
                    feed = {"input_ids": input_arr}
                    for idx, k_name in enumerate(self.past_kv_names):
                        feed[k_name] = past_kv_list[idx]
                    outputs = self.session.run(req_outputs, feed)
                    logits = outputs[0][0, -1, :].astype(np.float64)
                    past_kv_list = outputs[1:]
                else:
                    outputs = self.session.run(
                        ["logits", "present_keys", "present_values"],
                        {
                            "input_ids": input_arr,
                            "past_keys": past_k,
                            "past_values": past_v,
                        },
                    )
                    logits = outputs[0][0, -1, :].astype(np.float64)
                    past_k = outputs[1]
                    past_v = outputs[2]

        else:
            # ── Fallback Full-Sequence Autoregressive Generation ────────────────
            for _ in range(max_new_tokens):
                input_arr = np.array([current_ids], dtype=np.int64)
                outputs   = self.session.run(None, {self.input_name: input_arr})
                logits    = outputs[0][0, -1, :].astype(np.float64)   # [vocab_size]

                # 1. Repetition penalty
                if repetition_penalty != 1.0 and generated_ids:
                    penalty_tokens = generated_ids[-repetition_window:] if repetition_window > 0 else generated_ids
                    logits = _apply_repetition_penalty(logits, penalty_tokens, repetition_penalty)

                # 2. Temperature scaling
                if temperature == 0:
                    next_token = int(np.argmax(logits))
                else:
                    logits = logits / max(temperature, 1e-8)
                    if min_p > 0.0:
                        logits = _min_p_filter(logits, min_p)
                    if top_k > 0:
                        logits = _top_k_filter(logits, top_k)
                    if top_p < 1.0:
                        logits = _top_p_filter(logits, top_p)
                    probs      = _softmax(logits)
                    next_token = int(np.random.choice(len(probs), p=probs))

                # 3. Stop check on single token IDs
                if next_token in active_stop_token_ids:
                    break

                generated_ids.append(next_token)
                current_ids.append(next_token)

                if len(current_ids) > self.max_seq_len:
                    current_ids = current_ids[-self.max_seq_len:]

                should_stop = False
                for seq in active_stop_token_seqs:
                    seq_len = len(seq)
                    if len(generated_ids) >= seq_len and tuple(generated_ids[-seq_len:]) == seq:
                        should_stop = True
                        del generated_ids[-seq_len:]
                        break
                if should_stop:
                    break

                if next_token in self.sentence_boundary_ids:
                    tail_tokens = generated_ids[-min(len(generated_ids), 16):]
                    tail_text = self.tokenizer.decode(tail_tokens)
                    for stop_seq in active_stop_strings:
                        if stop_seq in tail_text:
                            should_stop = True
                            break
                    if should_stop:
                        break

                if on_token is not None:
                    token_text = self.tokenizer.decode([next_token])
                    on_token(token_text)

        # ── Decode and post-process once upon completion ───────────────────
        raw_text = self.tokenizer.decode(generated_ids)
        # Strip out any trailing stop sequence characters that might remain
        for stop_seq in active_stop_strings:
            if stop_seq in raw_text:
                raw_text = raw_text.split(stop_seq)[0]

        return _postprocess(raw_text)


# ── Sampling helpers ─────────────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    e = np.exp(logits - np.max(logits))
    return e / e.sum()


def _apply_repetition_penalty(
    logits: np.ndarray,
    context_ids: List[int],
    penalty: float,
) -> np.ndarray:
    """
    Divide logits of tokens that already appear in *context_ids* by *penalty*.
    For positive logits this lowers their probability; for negative logits it
    raises them (we want to suppress, so we always divide the absolute value and
    keep the sign).
    """
    if penalty == 1.0 or not context_ids:
        return logits
    logits = logits.copy()
    seen = set(context_ids)
    for tok in seen:
        if 0 <= tok < len(logits):
            if logits[tok] > 0:
                logits[tok] /= penalty
            else:
                logits[tok] *= penalty
    return logits


def _min_p_filter(logits: np.ndarray, min_p: float = 0.05) -> np.ndarray:
    """
    Min-P sampling filter:
    Calculates dynamic threshold = max_prob * min_p.
    Tokens with probability below this dynamic threshold are masked out.
    """
    if min_p <= 0.0 or min_p >= 1.0:
        return logits
    logits = logits.copy()
    probs = _softmax(logits)
    p_max = float(np.max(probs))
    threshold = p_max * min_p
    logits[probs < threshold] = -1e10
    return logits


def _top_k_filter(logits: np.ndarray, k: int) -> np.ndarray:
    """Zero out all logits except the top-k."""
    if k <= 0 or k >= len(logits):
        return logits
    logits = logits.copy()
    threshold = np.partition(logits, -k)[-k]
    logits[logits < threshold] = -1e10
    return logits


def _top_p_filter(logits: np.ndarray, p: float) -> np.ndarray:
    """
    Nucleus (top-p) sampling: keep the smallest set of tokens whose
    cumulative softmax probability is ≥ p, zero out the rest.
    """
    if p >= 1.0:
        return logits
    logits   = logits.copy()
    probs    = _softmax(logits)
    sorted_i = np.argsort(probs)[::-1]           # descending
    cum_prob = np.cumsum(probs[sorted_i])
    # First index where cumulative sum exceeds p
    cutoff   = np.searchsorted(cum_prob, p) + 1  # include the token that crosses p
    remove   = sorted_i[cutoff:]
    logits[remove] = -1e10
    return logits


# ── Post-processing ──────────────────────────────────────────────────────────

def _postprocess(text: str) -> str:
    """
    Clean up raw generated text:
      - Strip leading/trailing whitespace
      - Remove duplicate whitespace
      - Trim to the last complete sentence if text ends mid-sentence
      - Remove degenerate output (all punctuation / very short)
    """
    text = text.strip()
    if not text:
        return text

    # Collapse multiple spaces / newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Trim to last sentence boundary (. ! ?)
    # Only trim if the text doesn't already end with punctuation
    if len(text) > 20 and text[-1] not in ".!?\"'":
        match = re.search(r"[.!?][^.!?]*$", text)
        if match and match.start() > len(text) // 2:
            # There is a sentence end in the second half — trim after it
            text = text[: match.start() + 1]

    return text.strip()
