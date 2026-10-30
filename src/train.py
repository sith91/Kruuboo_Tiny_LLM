"""Training script for Tiny LLM — upgraded with high-performance ChatML SFT, target masking, and dynamic batching."""

import math
import os
import time
import json
import re
import argparse
from typing import List, Optional, Dict, Tuple, Any, Callable
from pathlib import Path
import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
import yaml

from src.model import TinyLLM
from src.tokenizer import TinyTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters & Configuration
# ─────────────────────────────────────────────────────────────────────────────

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

D_MODEL      = config["model"]["d_model"]
N_HEADS      = config["model"]["n_heads"]
N_LAYERS     = config["model"]["n_layers"]
D_FF         = config["model"]["d_ff"]
BLOCK_SIZE   = config["model"]["max_seq_len"]
VOCAB_SIZE   = config["model"]["vocab_size"]
DROPOUT      = config["model"]["dropout"]

# Training settings
BATCH_SIZE   = config["training"]["batch_size"]
MAX_ITERS    = config["training"]["max_iters"]
WARMUP_ITERS = config["training"]["warmup_iters"]
LR_MAX       = float(config["training"]["learning_rate_max"])
LR_MIN       = float(config["training"]["learning_rate_min"])
GRAD_CLIP    = config["training"]["grad_clip"]
VAL_RATIO    = config["training"]["val_ratio"]
EVAL_EVERY   = 50    # evaluate on validation set every N steps


# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int, max_iters: int = MAX_ITERS, warmup_iters: int = WARMUP_ITERS, lr_max: float = LR_MAX, lr_min: float = LR_MIN) -> float:
    """Cosine decay with linear warm-up."""
    if step < warmup_iters:
        return lr_max * (step + 1) / max(1, warmup_iters)
    # Cosine decay from warmup_iters → max_iters
    progress = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return lr_min + (lr_max - lr_min) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# Chat Sample & Data Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

class ChatSample:
    """Represents a tokenized conversation or document with target-only loss mask."""
    __slots__ = ("input_ids", "labels", "length")
    def __init__(self, input_ids: List[int], labels: List[int]):
        self.input_ids = input_ids
        self.labels = labels
        self.length = len(input_ids)


def prepare_chat_samples(raw_samples: List[str], tokenizer: TinyTokenizer, max_seq_len: int = BLOCK_SIZE) -> List[ChatSample]:
    """
    Fast tokenization and target masking:
    - Non-assistant turns (system/user) are masked to -100 so no loss is computed on prompts.
    - Assistant header '<|im_start|>assistant\n' is masked to -100.
    - Assistant reply content and '<|im_end|>' are kept as target token IDs for loss calculation.
    """
    turn_pattern = re.compile(r'(<\|im_start\|>(?:system|user|assistant)\n.*?<\|im_end\|>\n?)', re.DOTALL)
    assistant_hdr_ids = tokenizer.encode("<|im_start|>assistant\n")
    hdr_len = len(assistant_hdr_ids)

    chat_samples = []

    for text in raw_samples:
        if not text.strip():
            continue

        turns = turn_pattern.findall(text)
        if not turns:
            token_ids = tokenizer.encode(text)
            if not token_ids:
                continue
            if len(token_ids) <= max_seq_len:
                chat_samples.append(ChatSample(token_ids, list(token_ids)))
            else:
                for start in range(0, len(token_ids) - max_seq_len + 1, max_seq_len // 2):
                    end = start + max_seq_len
                    sub_ids = token_ids[start:end]
                    chat_samples.append(ChatSample(sub_ids, list(sub_ids)))
            continue

        all_ids: List[int] = []
        all_labels: List[int] = []

        for turn in turns:
            turn_ids = tokenizer.encode(turn)
            if not turn_ids:
                continue
            if turn.startswith("<|im_start|>assistant\n"):
                hl = min(hdr_len, len(turn_ids))
                labels = [-100] * hl + turn_ids[hl:]
            else:
                labels = [-100] * len(turn_ids)
            
            all_ids.extend(turn_ids)
            all_labels.extend(labels)

        if not all_ids:
            continue

        # Ensure at least one token is not masked to -100
        if not any(lbl != -100 for lbl in all_labels):
            all_labels = list(all_ids)

        if len(all_ids) <= max_seq_len:
            chat_samples.append(ChatSample(all_ids, all_labels))
        else:
            # Chunk long dialogues cleanly
            step = max_seq_len // 2
            for start in range(0, len(all_ids) - max_seq_len + 1, step):
                end = start + max_seq_len
                sub_ids = all_ids[start:end]
                sub_labels = all_labels[start:end]
                if any(lbl != -100 for lbl in sub_labels):
                    chat_samples.append(ChatSample(sub_ids, sub_labels))

    return chat_samples


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, model: TinyLLM, tokenizer: TinyTokenizer, device: str = "cpu", lr_max: float = LR_MAX, lr_min: float = LR_MIN):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device
        self.lr_max    = lr_max
        self.lr_min    = lr_min

        # AdamW with decoupled weight decay (skip bias / LayerNorm params)
        decay_params   = [p for n, p in model.named_parameters()
                          if p.requires_grad and p.dim() >= 2]
        no_decay_params= [p for n, p in model.named_parameters()
                          if p.requires_grad and p.dim() < 2]
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params,    "weight_decay": 0.1},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.lr_max, betas=(0.9, 0.95), eps=1e-8,
        )

    # ── Dynamic Batch Sampler ─────────────────────────────────────────────────

    def get_batch(self, samples: List[ChatSample], batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample a batch of conversation sequences with dynamic length padding."""
        if not samples:
            dummy_x = torch.zeros((batch_size, 16), dtype=torch.long, device=self.device)
            dummy_y = torch.zeros((batch_size, 16), dtype=torch.long, device=self.device)
            return dummy_x, dummy_y

        indices = torch.randint(len(samples), (batch_size,))
        batch_samples = [samples[int(i)] for i in indices]

        # Determine dynamic sequence length for this batch to save compute & VRAM
        max_batch_len = max(s.length for s in batch_samples)
        batch_len = min(256, BLOCK_SIZE, max(16, max_batch_len))

        batch_x = []
        batch_y = []

        for s in batch_samples:
            ids = s.input_ids[:batch_len]
            labs = s.labels[:batch_len]

            pad_len = batch_len - len(ids)
            if pad_len > 0:
                x_seq = ids + [TinyTokenizer.PAD_ID] * pad_len
                y_seq = labs + [-100] * pad_len
            else:
                x_seq = ids
                y_seq = labs

            batch_x.append(torch.tensor(x_seq, dtype=torch.long))
            batch_y.append(torch.tensor(y_seq, dtype=torch.long))

        x_stacked = torch.stack(batch_x)
        y_stacked = torch.stack(batch_y)

        # Fallback if any sequence row has all targets masked
        for b in range(batch_size):
            if (y_stacked[b, 1:] != -100).sum() == 0:
                y_stacked[b, 1:] = x_stacked[b, 1:]

        x = x_stacked[:, :-1].to(self.device)
        y = y_stacked[:, 1:].to(self.device)
        return x, y

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, val_samples: List[ChatSample], eval_batches: int = 10, micro_batch_size: int = 2) -> float:
        """Estimate validation loss without gradient computation."""
        self.model.eval()
        losses = []
        for _ in range(eval_batches):
            x, y = self.get_batch(val_samples, micro_batch_size)
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            if not torch.isnan(loss):
                losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(1, len(losses))

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(
        self,
        raw_samples: List[str],
        max_iters: int = MAX_ITERS,
        batch_size: int = BATCH_SIZE,
        micro_batch_size: int = 2,
        eval_every: int = EVAL_EVERY,
        on_step: Optional[Callable] = None,
        stop_event: Optional[Any] = None,
        pause_event: Optional[Any] = None,
    ) -> TinyLLM:
        """
        Full SFT training loop with target-only loss masking, gradient accumulation,
        dynamic sequence padding, and real-time step callback hooks.
        """
        print("\n⚙️  Preparing and tokenizing dataset...")
        t0 = time.time()
        samples = prepare_chat_samples(raw_samples, self.tokenizer, BLOCK_SIZE)
        print(f"✅ Prepared {len(samples):,} tokenized sequences in {time.time()-t0:.2f}s")

        split = int(len(samples) * (1 - VAL_RATIO))
        train_samples = samples[:split] if split > 0 else samples
        val_samples   = samples[split:] if split > 0 else samples

        print(f"📊 Train set: {len(train_samples):,} | Validation set: {len(val_samples):,}")

        accum_steps = max(1, batch_size // micro_batch_size)
        print(f"🚀 Training config: max_iters={max_iters}, effective_batch_size={batch_size} (micro_batch={micro_batch_size}, accum={accum_steps})")

        best_val_loss = float("inf")
        lock_path = "/tmp/kruboo_chat_active.lock"

        self.model.train()
        start_time = time.time()
        tokens_processed = 0

        for step in range(max_iters):
            if stop_event is not None and stop_event.is_set():
                print("🛑 Stop event detected. Halting training loop gracefully.")
                break

            # Check for active chat lock or pause event
            if os.path.exists(lock_path) or (pause_event is not None and pause_event.is_set()):
                print("\n⏸️  Training paused. Waiting...")
                while os.path.exists(lock_path) or (pause_event is not None and pause_event.is_set()):
                    if stop_event is not None and stop_event.is_set():
                        break
                    time.sleep(0.5)
                print("▶️  Resuming training...")

            # LR update
            lr = get_lr(step, max_iters=max_iters, lr_max=self.lr_max, lr_min=self.lr_min)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # Gradient accumulation step
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(accum_steps):
                x, y = self.get_batch(train_samples, micro_batch_size)
                tokens_processed += x.numel()
                logits = self.model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
                loss_scaled = loss / accum_steps
                loss_scaled.backward()
                step_loss += loss.item() / accum_steps

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()

            # Logging & evaluation
            is_eval = (step % eval_every == 0 or step == max_iters - 1)
            val_loss = step_loss
            is_best = False

            if is_eval:
                val_loss = self.evaluate(val_samples, eval_batches=6, micro_batch_size=micro_batch_size)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    is_best = True
                    # Save best checkpoint
                    Path("models").mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), "models/tiny_llm.pt")

                elapsed = max(0.1, time.time() - start_time)
                tok_per_sec = tokens_processed / elapsed
                marker = " 🌟 best" if is_best else ""

                print(f"  step {step:>5}/{max_iters} | "
                      f"train_loss: {step_loss:.4f} | "
                      f"val_loss: {val_loss:.4f} | "
                      f"lr: {lr:.2e} | "
                      f"speed: {tok_per_sec:,.0f} tok/s{marker}")

            elapsed = max(0.1, time.time() - start_time)
            tok_per_sec = tokens_processed / elapsed

            if on_step is not None:
                on_step(
                    step=step + 1,
                    total_steps=max_iters,
                    train_loss=step_loss,
                    val_loss=val_loss,
                    lr=lr,
                    speed=tok_per_sec,
                    tokens=tokens_processed,
                    elapsed=elapsed,
                    is_best=is_best,
                )

        print(f"\n🏁 Training finished in {time.time()-start_time:.1f}s. Best val_loss: {best_val_loss:.4f}")
        return self.model


# ─────────────────────────────────────────────────────────────────────────────
# Corpus Loading Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_foundational_chatml_samples() -> List[str]:
    """
    Build clean multi-turn conversational, identity-memory, and RAG-grounded ChatML samples
    matching the exact system prompt format used by RAGSystem.ask at inference time.
    """
    sys_base = "You are Kruuboo SLM, a helpful, knowledgeable, and empathetic AI assistant. The user is in a neutral, calm mood. Respond clearly, helpfully, and naturally."
    samples: List[str] = []

    # 1. Core conversational & greeting turns
    core_pairs = [
        ("Hi, how are you today?", "Hello! I am doing great, thank you for asking. How can I help you today?"),
        ("Hi", "Hi there! How can I help you today?"),
        ("Hello!", "Hello! How can I assist you today?"),
        ("Hey, how is it going?", "Everything is going well! What would you like to chat about or work on today?"),
        ("Good morning!", "Good morning! I hope you are having a wonderful day. How can I help you?"),
        ("Who are you?", "I am Kruuboo SLM, a local Small Language Model assistant with semantic RAG memory."),
        ("What is your name?", "My name is Kruuboo SLM. How can I assist you today?"),
        ("What can you do?", "I can hold multi-turn conversations, remember context from our chat, and answer questions across medical remedies, legal agreements, survival techniques, science, and general knowledge."),
        ("Thank you so much!", "You are very welcome! Let me know if there is anything else I can help you with."),
        ("I feel a bit stressed today.", "I am sorry to hear you are feeling stressed. Take a slow, deep breath. Would you like to talk about what is on your mind or take things one step at a time?"),
        ("What is 2 + 2?", "2 + 2 is 4."),
        ("What is 10 * 5?", "10 * 5 is 50."),
        ("Calculate 25 + 75", "25 + 75 is 100."),
        ("What is machine learning?", "Machine learning is a branch of artificial intelligence where computer systems learn patterns from data to make predictions or decisions without being explicitly programmed."),
    ]
    for q, a in core_pairs:
        samples.append(
            f"<|im_start|>system\n{sys_base}<|im_end|>\n"
            f"<|im_start|>user\n{q}<|im_end|>\n"
            f"<|im_start|>assistant\n{a}<|im_end|>\n"
        )

    # 2. Multi-turn name & context memory dialogues (single turn + 2-turn follow-up)
    names = ["Sithija", "Alex", "Maya", "Liam", "Sophia", "Noah", "Elena", "David"]
    for name in names:
        samples.append(
            f"<|im_start|>system\n{sys_base}<|im_end|>\n"
            f"<|im_start|>user\nMy name is {name}<|im_end|>\n"
            f"<|im_start|>assistant\nNice to meet you, {name}! How can I help you today?<|im_end|>\n"
        )
        samples.append(
            f"<|im_start|>system\n{sys_base}<|im_end|>\n"
            f"<|im_start|>user\nMy name is {name}<|im_end|>\n"
            f"<|im_start|>assistant\nNice to meet you, {name}! How can I help you today?<|im_end|>\n"
            f"<|im_start|>user\nWhat is my name?<|im_end|>\n"
            f"<|im_start|>assistant\nYour name is {name}.<|im_end|>\n"
        )

    return samples * 4


def load_training_corpus(custom_file: Optional[str] = None) -> List[str]:
    """Load and format clean ChatML dialogues and RAG-grounded Q&A pairs."""
    custom_texts: List[str] = []

    if custom_file:
        p = Path(custom_file)
        if p.exists():
            print(f"📖 Loading custom file: {p}")
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "<|im_start|>" in content:
                    raw_dlgs = content.split("<|im_start|>system")
                    custom_texts.extend(["<|im_start|>system" + d for d in raw_dlgs if d.strip()])
                else:
                    custom_texts.extend([c for c in content.split("\n\n") if len(c.strip()) > 30])
        else:
            print(f"❌ Custom file not found: {custom_file}")
        return custom_texts

    sys_base = "You are Kruuboo SLM, a helpful, knowledgeable, and empathetic AI assistant. The user is in a neutral, calm mood. Respond clearly, helpfully, and naturally."

    # 1. Foundational multi-turn conversational & memory samples
    custom_texts.extend(_build_foundational_chatml_samples())

    # 2. Structured domain JSON datasets (both direct Q&A and RAG-grounded synthesis)
    td_dir = Path("training_data")
    if td_dir.exists():
        for json_file in sorted(td_dir.glob("*.json")):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for item in data:
                    q, a, ctx = None, None, None
                    if "question" in item and "advice" in item:
                        q = item["question"]
                        a = item["advice"]
                        ctx = a
                    elif "ailment" in item and "remedy" in item:
                        ailment = item["ailment"]
                        q = f"What is the remedy for {ailment.lower()}?"
                        a = item["remedy"]
                        ctx = f"Home Remedy for {ailment}: {a}"
                    elif "topic" in item and "instructions" in item:
                        topic = item["topic"]
                        q = f"How to {topic.lower()} in survival?"
                        a = item["instructions"]
                        ctx = f"Survival Guide - {topic} ({item.get('situation', '')}): {a}"

                    if q and a:
                        direct = (
                            f"<|im_start|>system\n{sys_base}<|im_end|>\n"
                            f"<|im_start|>user\n{q}<|im_end|>\n"
                            f"<|im_start|>assistant\n{a}<|im_end|>\n"
                        )
                        grounded = (
                            f"<|im_start|>system\n{sys_base}\n\n"
                            f"Relevant Knowledge Context:\n• {ctx}\n\n"
                            f"Answer the user's question accurately using the context above. Keep explanations clear and conversational.<|im_end|>\n"
                            f"<|im_start|>user\n{q}<|im_end|>\n"
                            f"<|im_start|>assistant\n{a}<|im_end|>\n"
                        )
                        custom_texts.extend([direct, grounded] * 4)
            except Exception as e:
                print(f"⚠️ Error reading JSON {json_file}: {e}")

        # 3. Curated concise ChatML dialogues from conversational_dialogues.txt & alpaca_chatml.txt
        for fname, limit in [("conversational_dialogues.txt", 180), ("alpaca_chatml.txt", 180)]:
            txt_file = td_dir / fname
            if not txt_file.exists():
                continue
            try:
                with open(txt_file, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                if "<|im_start|>system" in content:
                    raw_dlgs = content.split("<|im_start|>system")
                    dlgs = [
                        "<|im_start|>system" + d.replace("You are a helpful assistant.", sys_base)
                        for d in raw_dlgs if d.strip() and 60 < len(d) < 450
                    ]
                    custom_texts.extend(dlgs[:limit])
            except Exception as e:
                print(f"⚠️ Error reading TXT {txt_file}: {e}")

    # 4. Knowledge Base RAG-grounded Q&A pairs
    kb_file = Path("knowledge_base/knowledge.json")
    if kb_file.exists():
        try:
            with open(kb_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            for doc in data[:120]:
                if isinstance(doc, dict) and doc.get("text"):
                    txt = doc["text"].strip()
                    if ":" in txt and len(txt) < 500:
                        head, body = txt.split(":", 1)
                        if 3 <= len(head.strip()) <= 60 and len(body.strip()) >= 15:
                            q = f"What is {head.strip().lower()}?"
                            a = body.strip()
                            grounded = (
                                f"<|im_start|>system\n{sys_base}\n\n"
                                f"Relevant Knowledge Context:\n• {txt}\n\n"
                                f"Answer the user's question accurately using the context above. Keep explanations clear and conversational.<|im_end|>\n"
                                f"<|im_start|>user\n{q}<|im_end|>\n"
                                f"<|im_start|>assistant\n{a}<|im_end|>\n"
                            )
                            custom_texts.append(grounded)
        except Exception as e:
            print(f"⚠️ Error reading KB {kb_file}: {e}")

    import random
    random.seed(42)
    random.shuffle(custom_texts)
    return custom_texts


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def train_model(
    iters: Optional[int] = None,
    batch_size: Optional[int] = None,
    lr: Optional[float] = None,
    custom_file: Optional[str] = None,
    export_onnx: bool = True,
):
    """Build tokenizer if needed, instantiate model, run training, and export ONNX."""
    print("=" * 70)
    print("🎯 TINY LLM (SLM-0.2) — NEURAL MODEL TRAINING")
    print("=" * 70)

    max_iters = iters or MAX_ITERS
    effective_batch_size = batch_size or BATCH_SIZE
    lr_max = lr or LR_MAX

    # 1. Load corpus
    training_samples = load_training_corpus(custom_file)
    if not training_samples:
        print("ℹ️ No custom datasets found. Falling back to default baseline corpus...")
        sample_text = (
            "<|im_start|>system\nYou are Kruuboo-SLM, a helpful, intelligent, and knowledgeable assistant.<|im_end|>\n"
            "<|im_start|>user\nWhat is Tiny LLM?<|im_end|>\n"
            "<|im_start|>assistant\nTiny LLM is an efficient transformer language model designed for fast local edge inference.<|im_end|>\n"
        ) * 50
        training_samples = [sample_text]

    print(f"📚 Loaded {len(training_samples):,} training conversations/samples")

    # 2. Tokenizer
    if Path("tokenizer.json").exists():
        print("📦 Loading existing tokenizer from tokenizer.json...")
        tokenizer = TinyTokenizer.load("tokenizer.json")
    else:
        sample_text_for_tokenizer = "\n\n".join(training_samples[:500])
        tokenizer = TinyTokenizer.train_from_text(
            text=sample_text_for_tokenizer,
            vocab_size=VOCAB_SIZE,
            save_path="tokenizer.json",
        )
    print(f"✅ Tokenizer ready (vocab_size={tokenizer.vocab_size})")

    # 3. Device selection
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"🖥️  Compute Device: {device.upper()}")

    # 4. Model instantiation
    model = TinyLLM(
        vocab_size   = tokenizer.vocab_size,
        d_model      = D_MODEL,
        n_heads      = N_HEADS,
        n_layers     = N_LAYERS,
        d_ff         = D_FF,
        max_seq_len  = BLOCK_SIZE,
        dropout      = DROPOUT,
    )

    ckpt_path = Path("models/tiny_llm.pt")
    if ckpt_path.exists():
        try:
            print(f"💾 Loading pre-trained weights from {ckpt_path}...")
            state_dict = torch.load(str(ckpt_path), map_location="cpu")
            model_dict = model.state_dict()
            matched_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            if len(matched_dict) == len(model_dict):
                model.load_state_dict(matched_dict)
                print("✅ All weights matched and loaded successfully!")
            elif matched_dict:
                model.load_state_dict(matched_dict, strict=False)
                print(f"⚠️ Loaded {len(matched_dict)} matching layers; {len(model_dict)-len(matched_dict)} layers initialized.")
            else:
                print("ℹ️ Checkpoint architecture mismatch. Initializing weights from scratch.")
        except Exception as e:
            print(f"⚠️ Checkpoint load warning: {e}. Starting fresh.")

    print(f"🧠 Model Architecture: {model.num_parameters():,} parameters ({model.num_parameters()/1e6:.1f} M)")
    print(f"   d_model={D_MODEL}, n_heads={N_HEADS}, n_layers={N_LAYERS}, d_ff={D_FF}, max_seq_len={BLOCK_SIZE}")

    # Full batch size on MPS/CUDA for high throughput on right-sized model
    micro_batch_size = effective_batch_size if device in ("mps", "cuda") else 4

    # 5. Train
    trainer = Trainer(model, tokenizer, device=device, lr_max=lr_max, lr_min=LR_MIN)
    trained_model = trainer.train(
        training_samples,
        max_iters=max_iters,
        batch_size=effective_batch_size,
        micro_batch_size=micro_batch_size,
        eval_every=EVAL_EVERY,
    )

    # 6. Save final checkpoint
    Path("models").mkdir(parents=True, exist_ok=True)
    save_path = "models/tiny_llm.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"💾 Trained model saved to {save_path}")

    # 7. Export to ONNX & Quantized ONNX and refresh knowledge base embeddings
    if export_onnx:
        print("\n" + "=" * 70)
        print("📦 EXPORTING MODEL TO ONNX & INT8 QUANTIZATION")
        print("=" * 70)
        try:
            from scripts.export_onnx import export_to_onnx
            onnx_path, quant_path = export_to_onnx(
                model_path=save_path,
                tokenizer_path="tokenizer.json",
                output_path="models/tiny_llm.onnx",
            )
            print(f"🎉 Model exported successfully!\n   ONNX: {onnx_path}\n   INT8: {quant_path}")

            # Refresh semantic knowledge base embeddings with the newly trained model
            kb_path = Path("knowledge_base/knowledge.json")
            if kb_path.exists():
                print("🔄 Re-indexing knowledge_base/knowledge.json with trained native embeddings...")
                from src.rag.native_encoder import NativeSLMEncoder
                encoder = NativeSLMEncoder(model_path=quant_path, tokenizer_path="tokenizer.json")
                with open(kb_path, "r", encoding="utf-8") as f:
                    kb_docs = json.load(f)
                texts = [d.get("text", "") for d in kb_docs]
                if texts:
                    embs = encoder.encode(texts, batch_size=32)
                    for doc_item, emb in zip(kb_docs, embs):
                        if "metadata" not in doc_item or not isinstance(doc_item["metadata"], dict):
                            doc_item["metadata"] = {}
                        doc_item["metadata"]["embedding"] = emb.tolist()
                    with open(kb_path, "w", encoding="utf-8") as f:
                        json.dump(kb_docs, f, indent=2)
                    print(f"✅ Re-indexed {len(kb_docs)} knowledge documents ({embs.shape[1]}d vectors).")
        except Exception as e:
            print(f"⚠️ ONNX export warning: {e}")

    return trained_model, tokenizer


def run_training_session(
    monitor: Any,
    max_iters: int = 100,
    dataset_source: str = "knowledge_base",
    custom_data: Optional[List[str]] = None,
    probe_prompt: str = "How do I treat a mild burn at home?",
    stop_event: Optional[Any] = None,
    pause_event: Optional[Any] = None,
):
    """
    Programmatic training session coordinator designed for real-time monitoring and web visualizer.
    """
    monitor.log_message(f"Preparing dataset from source: '{dataset_source}'...")
    
    # 1. Load dataset based on source
    if custom_data and len(custom_data) > 0:
        raw_samples = custom_data
    elif dataset_source == "conversational":
        raw_samples = load_training_corpus("training_data/conversational_dialogues.txt")
    elif dataset_source == "survival":
        raw_samples = load_training_corpus("training_data/US_Army_Survival_Manual_FM21-76.txt")
    elif dataset_source == "all":
        raw_samples = load_training_corpus()
    else:  # knowledge_base (default)
        raw_samples = load_training_corpus()

    if not raw_samples:
        monitor.log_message("⚠️ No training samples found. Creating fallback conversational dataset...")
        raw_samples = [
            "<|im_start|>system\nYou are a helpful medical and domain specialist.<|im_end|>\n<|im_start|>user\nHow do I treat a mild burn at home?<|im_end|>\n<|im_start|>assistant\nCool the burn with cool running water for 10-15 minutes. Apply aloe vera and cover with a sterile non-stick bandage. Never apply ice or butter.<|im_end|>\n",
            "<|im_start|>system\nYou are a legal advisor.<|im_end|>\n<|im_start|>user\nWhat makes a contract valid?<|im_end|>\n<|im_start|>assistant\nA valid contract requires offer, acceptance, consideration, legal capacity, and lawful purpose.<|im_end|>\n"
        ]

    monitor.log_message(f"Dataset ready with {len(raw_samples):,} documents/dialogues.")

    # 2. Setup Device
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    monitor.log_message(f"Using compute acceleration: {device.upper()}")

    # 3. Tokenizer & Model
    tokenizer = TinyTokenizer.load("tokenizer.json")
    model = TinyLLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=BLOCK_SIZE,
        dropout=DROPOUT,
    )

    ckpt_path = Path("models/tiny_llm.pt")
    if ckpt_path.exists():
        try:
            state_dict = torch.load(str(ckpt_path), map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            monitor.log_message("✅ Loaded previous model weights from models/tiny_llm.pt")
        except Exception as e:
            monitor.log_message(f"⚠️ Could not load checkpoint ({e}), initializing fresh.")

    trainer = Trainer(model, tokenizer, device=device, lr_max=LR_MAX, lr_min=LR_MIN)
    monitor.set_active_model(model, tokenizer, device=device)

    # Initial probe at Step 0 (direct in-memory argument, 0 disk I/O)
    monitor.log_message(f"Probing initial model response for '{probe_prompt}'...")
    monitor.probe_model(probe_prompt, model=model, tokenizer=tokenizer, device=device)

    # Callback wrapper
    def step_hook(step, total_steps, train_loss, val_loss, lr, speed, tokens, elapsed, is_best):
        monitor.record_step(
            step=step,
            total_steps=total_steps,
            train_loss=train_loss,
            val_loss=val_loss,
            lr=lr,
            speed=speed,
            tokens=tokens,
            elapsed=elapsed,
            is_best=is_best,
        )
        if step % 25 == 0 and step < total_steps:
            monitor.probe_model(probe_prompt, model=model, tokenizer=tokenizer, device=device)

    # Run train
    micro_batch = 2 if device in ("mps", "cpu") else 4
    try:
        trainer.train(
            raw_samples=raw_samples,
            max_iters=max_iters,
            batch_size=BATCH_SIZE,
            micro_batch_size=micro_batch,
            eval_every=min(25, max(5, max_iters // 4)),
            on_step=step_hook,
            stop_event=stop_event,
            pause_event=pause_event,
        )
    finally:
        monitor.clear_active_model()

    # Final probe
    monitor.probe_model(probe_prompt, model=model, tokenizer=tokenizer, device=device)

    # Save checkpoint
    Path("models").mkdir(parents=True, exist_ok=True)
    save_path = "models/tiny_llm.pt"
    torch.save(model.state_dict(), save_path)
    monitor.log_message(f"💾 Checkpoint saved to {save_path}")

    # Re-export ONNX asynchronously/quietly if needed
    try:
        from scripts.export_onnx import export_to_onnx
        monitor.log_message("📦 Updating ONNX runtime export...")
        export_to_onnx(
            model_path=save_path,
            tokenizer_path="tokenizer.json",
            output_path="models/tiny_llm.onnx",
        )
        monitor.log_message("✅ ONNX & INT8 models updated.")
    except Exception as e:
        monitor.log_message(f"⚠️ ONNX export note: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tiny LLM (SLM-0.2)")
    parser.add_argument("--iters", type=int, default=None, help="Total training iterations")
    parser.add_argument("--batch-size", type=int, default=None, help="Effective batch size")
    parser.add_argument("--lr", type=float, default=None, help="Peak learning rate")
    parser.add_argument("--file", type=str, default=None, help="Custom text file to train on")
    parser.add_argument("--no-onnx", action="store_true", help="Skip ONNX export")
    args, _ = parser.parse_known_args()

    train_model(
        iters=args.iters,
        batch_size=args.batch_size,
        lr=args.lr,
        custom_file=args.file,
        export_onnx=not args.no_onnx,
    )