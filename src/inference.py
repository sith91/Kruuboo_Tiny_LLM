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
from typing import Callable, List, Optional


# ── Special token IDs (must match TinyTokenizer) ─────────────────────────────
PAD_ID = 0
EOS_ID = 1


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
        on_token           : Optional callback for streaming tokens.

        Returns
        -------
        str : Only the *newly generated* text (prompt is stripped).
        """
        if self.session is None:
            return "[No model loaded — run src/train.py then scripts/export_onnx.py]"

        # ── Encode prompt ──────────────────────────────────────────────────
        prompt_ids = self.tokenizer.encode(prompt)

        # Keep the trailing window if prompt is longer than context
        max_prompt = self.max_seq_len - 1          # leave room for at least one new token
        if len(prompt_ids) > max_prompt:
            prompt_ids = prompt_ids[-max_prompt:]

        current_ids: List[int] = list(prompt_ids)
        generated_ids: List[int] = []

        # ── Token-by-token generation ──────────────────────────────────────
        for _ in range(max_new_tokens):
            input_arr = np.array([current_ids], dtype=np.int64)
            outputs   = self.session.run(None, {self.input_name: input_arr})
            logits    = outputs[0][0, -1, :].astype(np.float64)   # [vocab_size]

            # 1. Repetition penalty
            logits = _apply_repetition_penalty(logits, current_ids, repetition_penalty)

            # 2. Temperature scaling  (temperature=0 → greedy)
            if temperature == 0:
                next_token = int(np.argmax(logits))
            else:
                logits = logits / max(temperature, 1e-8)

                # 3. Min-P dynamic filtering (eliminates noise dynamically scaled to top token)
                if min_p > 0.0:
                    logits = _min_p_filter(logits, min_p)

                # 4. Top-k filtering (if enabled)
                if top_k > 0:
                    logits = _top_k_filter(logits, top_k)

                # 5. Top-p (nucleus) filtering (if enabled)
                if top_p < 1.0:
                    logits = _top_p_filter(logits, top_p)

                # 6. Sample
                probs      = _softmax(logits)
                next_token = int(np.random.choice(len(probs), p=probs))

            # 7. Stop on EOS / PAD
            is_bpe = getattr(self.tokenizer, "_tokenizer", None) is not None
            if next_token == PAD_ID or (is_bpe and next_token == EOS_ID):
                break

            generated_ids.append(next_token)
            current_ids.append(next_token)

            # Slide window
            if len(current_ids) > self.max_seq_len:
                current_ids = current_ids[-self.max_seq_len:]

            # Early break if stop sequence appears in generated text so far
            decoded_so_far = self.tokenizer.decode(generated_ids)
            stop_sequences = [
                "<|im_end|>", "<|im_start|>",
                "\n<|im_start|>", "\n<|im_end|>",
                "\nuser\n", "\nassistant\n", "\nsystem\n",
                "\nuser:", "\nassistant:", "\nsystem:",
                "\nUser:", "\nAssistant:", "\nHuman:",
                "\nuser", "\nassistant", "\nsystem"
            ]
            should_stop = False
            for stop_seq in stop_sequences:
                if stop_seq in decoded_so_far:
                    should_stop = True
                    break
            if should_stop:
                break

            # Streaming callback
            if on_token is not None:
                token_text = self.tokenizer.decode([next_token])
                on_token(token_text)

        # ── Decode and post-process ────────────────────────────────────────
        raw_text = self.tokenizer.decode(generated_ids)
        # Strip out any trailing stop sequence characters
        for stop_seq in stop_sequences:
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
    if penalty == 1.0:
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
