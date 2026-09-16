"""Training script for Tiny LLM — upgraded to match new model architecture."""

import math
from typing import List, Optional, Dict, Tuple
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.append('.')

from src.model import TinyLLM
from src.tokenizer import TinyTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

# Model dimensions and training parameters — loaded dynamically from config.yaml
import yaml
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
EVAL_EVERY   = 100    # evaluate on validation set every N steps



# ─────────────────────────────────────────────────────────────────────────────
# Learning rate schedule: linear warmup → cosine decay
# ─────────────────────────────────────────────────────────────────────────────

def get_lr(step: int) -> float:
    """Cosine decay with linear warm-up."""
    if step < WARMUP_ITERS:
        return LR_MAX * (step + 1) / WARMUP_ITERS
    # Cosine decay from WARMUP_ITERS → MAX_ITERS
    progress = (step - WARMUP_ITERS) / max(1, MAX_ITERS - WARMUP_ITERS)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return LR_MIN + (LR_MAX - LR_MIN) * cosine


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class ChatSample:
    """Represents a tokenized conversation or document with target-only loss mask."""
    def __init__(self, input_ids: List[int], labels: List[int]):
        self.input_ids = input_ids
        self.labels = labels


def prepare_chat_samples(raw_samples: List[str], tokenizer: TinyTokenizer, max_seq_len: int) -> List[ChatSample]:
    """Tokenize and mask prompt tokens so loss is computed only on assistant replies."""
    import re
    chat_samples = []
    
    for text in raw_samples:
        if not text.strip():
            continue
        token_ids = tokenizer.encode(text)
        if not token_ids:
            continue

        # If text contains ChatML assistant turns, mask out system and user prompt tokens
        if "<|im_start|>assistant" in text:
            labels = [-100] * len(token_ids)
            for m in re.finditer(r'<\|im_start\|>assistant\n(.*?)(?:<\|im_end\|>|$)', text, flags=re.DOTALL):
                prefix_text = text[:m.start()]
                assistant_full_text = text[:m.end()]
                
                start_tok_idx = len(tokenizer.encode(prefix_text))
                end_tok_idx = len(tokenizer.encode(assistant_full_text))
                
                for idx in range(start_tok_idx, min(end_tok_idx, len(token_ids))):
                    labels[idx] = token_ids[idx]
        else:
            labels = list(token_ids)

        if len(token_ids) <= max_seq_len:
            chat_samples.append(ChatSample(token_ids, labels))
        else:
            # Chunk longer documents
            step = max(max_seq_len // 2, 64)
            for start in range(0, len(token_ids) - max_seq_len + 1, step):
                end = start + max_seq_len
                chat_samples.append(ChatSample(token_ids[start:end], labels[start:end]))

    return chat_samples


class Trainer:
    def __init__(self, model: TinyLLM, tokenizer: TinyTokenizer, device: str = "cpu"):
        self.model     = model.to(device)
        self.tokenizer = tokenizer
        self.device    = device
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
            lr=LR_MAX, betas=(0.9, 0.95), eps=1e-8,
        )

    # ── Batch sampler ─────────────────────────────────────────────────────────

    def get_batch(self, samples: List[ChatSample], batch_size: int) -> tuple:
        """Sample a batch of conversation sequences with target masking and padding."""
        if not samples:
            dummy_x = torch.zeros((batch_size, BLOCK_SIZE - 1), dtype=torch.long, device=self.device)
            dummy_y = torch.zeros((batch_size, BLOCK_SIZE - 1), dtype=torch.long, device=self.device)
            return dummy_x, dummy_y

        indices = torch.randint(len(samples), (batch_size,))
        batch_x = []
        batch_y = []

        for idx in indices:
            s = samples[int(idx)]
            ids = s.input_ids
            labs = s.labels

            if len(ids) < BLOCK_SIZE:
                pad_len = BLOCK_SIZE - len(ids)
                x_seq = ids + [TinyTokenizer.PAD_ID] * pad_len
                y_seq = labs + [-100] * pad_len
            else:
                x_seq = ids[:BLOCK_SIZE]
                y_seq = labs[:BLOCK_SIZE]

            batch_x.append(torch.tensor(x_seq, dtype=torch.long))
            batch_y.append(torch.tensor(y_seq, dtype=torch.long))

        x_stacked = torch.stack(batch_x)
        y_stacked = torch.stack(batch_y)

        # Fallback if an entire row is masked to -100
        for b in range(batch_size):
            if (y_stacked[b, 1:] != -100).sum() == 0:
                y_stacked[b, 1:] = x_stacked[b, 1:]

        x = x_stacked[:, :-1].to(self.device)
        y = y_stacked[:, 1:].to(self.device)
        return x, y

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, val_samples: List[ChatSample], eval_batches: int = 10) -> float:
        """Estimate validation loss (no gradient)."""
        self.model.eval()
        losses = []
        for _ in range(eval_batches):
            x, y = self.get_batch(val_samples, BATCH_SIZE)
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / max(1, len(losses))

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, raw_samples: List[str]) -> TinyLLM:
        """
        Full SFT training loop with target-only loss masking:
          - Train / validation split
          - Cosine LR schedule with linear warmup
          - Gradient clipping
          - Periodic validation loss reporting
        """
        samples = prepare_chat_samples(raw_samples, self.tokenizer, BLOCK_SIZE)
        
        split = int(len(samples) * (1 - VAL_RATIO))
        train_samples = samples[:split] if split > 0 else samples
        val_samples   = samples[split:] if split > 0 else samples
        
        print(f"📊 Dataset: {len(samples):,} processed conversation samples "
              f"(train {len(train_samples):,} | val {len(val_samples):,})")

        best_val_loss = float("inf")

        import os
        import time

        for step in range(MAX_ITERS):
            # ── Check for active chat lock (pause training if user is chatting) ────
            lock_path = "/tmp/kruboo_chat_active.lock"
            if os.path.exists(lock_path):
                print(f"\n⏸️  User is chatting! Suspending training to free up GPU resources...")
                while os.path.exists(lock_path):
                    time.sleep(1.0)
                print("▶️  User idle. Resuming training...")

            # ── LR update ────────────────────────────────────────────────────
            lr = get_lr(step)
            for group in self.optimizer.param_groups:
                group["lr"] = lr

            # ── Forward / backward ───────────────────────────────────────────
            x, y   = self.get_batch(train_samples, BATCH_SIZE)
            logits = self.model(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-100)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()

            # ── Logging / evaluation ─────────────────────────────────────────
            if step % EVAL_EVERY == 0 or step == MAX_ITERS - 1:
                val_loss = self.evaluate(val_samples)
                marker   = " ✅ best" if val_loss < best_val_loss else ""
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                print(f"  step {step:>5}/{MAX_ITERS}  "
                      f"train_loss={loss.item():.4f}  "
                      f"val_loss={val_loss:.4f}  "
                      f"lr={lr:.2e}{marker}")

        print(f"\n🏁 Training complete. Best val_loss: {best_val_loss:.4f}")
        return self.model


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def train_model():
    """Build BPE tokenizer, instantiate upgraded model, run training."""
    import argparse
    import json
    
    print("=" * 60)
    print("TINY LLM — UPGRADED TRAINING")
    print("=" * 60)

    # ── Parse CLI arguments if run directly ───────────────────────────────────
    parser = argparse.ArgumentParser(description="Train Tiny LLM on custom datasets")
    parser.add_argument("--file", type=str, default=None, help="Path to a custom plain text file to train on")
    args, unknown = parser.parse_known_args()

    # ── Training corpus loading ────────────────────────────────────────────────
    custom_texts = []
    
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            print(f"📖 Loading custom file: {file_path}")
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                custom_texts.append(f.read())
        else:
            print(f"❌ File not found: {args.file}")
    else:
        # Scan knowledge_base directory
        kb_dirs = [Path("knowledge_base")]
        for kb_dir in kb_dirs:
            if kb_dir.exists():
                kb_files = list(kb_dir.glob("**/knowledge.json"))
                for kb_file in kb_files:
                    if kb_file.exists():
                        print(f"📂 Loading data from knowledge base: {kb_file}")
                        try:
                            with open(kb_file, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                for doc in data:
                                    if isinstance(doc, dict) and doc.get("text"):
                                        custom_texts.append(doc["text"])
                        except Exception as e:
                            print(f"⚠️ Error reading {kb_file}: {e}")

        # Scan training_data directory for text datasets
        td_dir = Path("training_data")
        if td_dir.exists():
            td_files = list(td_dir.glob("*.txt"))
            for td_file in td_files:
                if td_file.exists():
                    # Exclude non-conversational raw text corpora to keep assistant focused
                    if td_file.name in ["wikitext2.txt", "tinystories.txt", "US_Army_Survival_Manual_FM21-76.txt"]:
                        print(f"⏭️  Skipping raw text corpus: {td_file.name}")
                        continue
                    print(f"📂 Loading training dataset text: {td_file}")
                    try:
                        with open(td_file, "r", encoding="utf-8", errors="ignore") as f:
                            text_content = f.read()
                            if text_content.strip():
                                custom_texts.append(text_content)
                    except Exception as e:
                        print(f"⚠️ Error reading training dataset {td_file}: {e}")

            # Scan and format JSON datasets
            json_files = list(td_dir.glob("*.json"))
            for json_file in json_files:
                if json_file.exists():
                    print(f"📂 Loading and formatting training dataset JSON: {json_file}")
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            for item in data:
                                if "formatted_chatml" in item:
                                    custom_texts.append(item["formatted_chatml"])
                                elif "question" in item and "advice" in item:
                                    q = item["question"]
                                    a = item["advice"]
                                    formatted = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
                                    custom_texts.append(formatted)
                                elif "ailment" in item and "remedy" in item:
                                    ailment = item["ailment"]
                                    remedy = item["remedy"]
                                    formatted = f"<|im_start|>user\nWhat is the remedy for {ailment}?<|im_end|>\n<|im_start|>assistant\n{remedy}<|im_end|>\n"
                                    custom_texts.append(formatted)
                    except Exception as e:
                        print(f"⚠️ Error reading training dataset {json_file}: {e}")

    # Combine custom texts or fallback
    if custom_texts:
        training_samples = custom_texts
        print(f"📈 Loaded {len(training_samples):,} training conversations/samples")
        sample_text_for_tokenizer = "\n\n".join(custom_texts[:500])
    else:
        print("ℹ️ No custom datasets found. Falling back to default sample corpus...")
        sample_text_for_tokenizer = (
            "The quick brown fox jumps over the lazy dog. "
            "Python is a high-level, general-purpose programming language known for "
            "its readability and broad standard library. "
            "Machine learning is a subset of artificial intelligence that allows "
            "computers to learn from data without being explicitly programmed. "
            "Transformers are powerful neural network architectures that rely on "
            "self-attention mechanisms to process sequential data efficiently. "
            "Retrieval-Augmented Generation (RAG) combines a language model with a "
            "knowledge retrieval system so the model can reference external documents. "
            "ONNX is an open format for machine learning models that enables "
            "interoperability between different frameworks and runtimes. "
            "Natural language processing is the field of AI concerned with the "
            "interaction between computers and human language. "
            "Deep learning uses multi-layer neural networks to automatically learn "
            "hierarchical representations from raw data. "
            "Attention mechanisms allow models to focus on relevant parts of the "
            "input sequence when producing each output token. "
        ) * 80  # repeat to give BPE enough material
        training_samples = [sample_text_for_tokenizer]

    # ── BPE tokenizer ─────────────────────────────────────────────────────────
    if Path("tokenizer.json").exists():
        print("📦 Loading existing tokenizer from tokenizer.json...")
        tokenizer = TinyTokenizer.load("tokenizer.json")
    else:
        tokenizer = TinyTokenizer.train_from_text(
            text=sample_text_for_tokenizer,
            vocab_size=VOCAB_SIZE,
            save_path="tokenizer.json",
        )
    print(f"✅ Tokenizer ready  (vocab_size={tokenizer.vocab_size})")

    # ── Device selection ──────────────────────────────────────────────────────
    device = (
        "cuda"  if torch.cuda.is_available()  else
        "mps"   if torch.backends.mps.is_available() else
        "cpu"
    )
    print(f"🖥️  Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = TinyLLM(
        vocab_size   = tokenizer.vocab_size,
        d_model      = D_MODEL,
        n_heads      = N_HEADS,
        n_layers     = N_LAYERS,
        d_ff         = D_FF,
        max_seq_len  = BLOCK_SIZE,
        dropout      = DROPOUT,
    )
    if Path("models/tiny_llm.pt").exists():
        try:
            print("💾 Checking pre-trained model weights from models/tiny_llm.pt...")
            state_dict = torch.load("models/tiny_llm.pt", map_location="cpu")
            model_dict = model.state_dict()
            matched_dict = {}
            mismatched_keys = []
            
            for k, v in state_dict.items():
                if k in model_dict and model_dict[k].shape == v.shape:
                    matched_dict[k] = v
                else:
                    mismatched_keys.append(k)
                    
            if len(matched_dict) == len(model_dict):
                model.load_state_dict(matched_dict)
                print("✅ All pre-trained weights loaded successfully!")
            elif matched_dict:
                model.load_state_dict(matched_dict, strict=False)
                print(f"⚠️ Model dimensions changed. Loaded {len(matched_dict)} compatible layers; newly initialized {len(mismatched_keys)} scaled layers.")
            else:
                print(f"ℹ️ Model scaled up to new dimensions ({D_MODEL}d / {N_LAYERS}L / {D_FF}ff). Initializing new weights from scratch.")
        except Exception as e:
            print(f"⚠️ Could not load existing checkpoint: {e}. Training from scratch.")
            
    print(f"🧠 Parameters: {model.num_parameters():,}"
          f"  ({model.num_parameters()/1e6:.2f} M)")
    print(f"   d_model={D_MODEL}  n_heads={N_HEADS}  "
          f"n_layers={N_LAYERS}  d_ff={D_FF}  seq_len={BLOCK_SIZE}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer      = Trainer(model, tokenizer, device=device)
    trained_model= trainer.train(training_samples)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    save_path = "models/tiny_llm.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")

    return trained_model, tokenizer


if __name__ == "__main__":
    train_model()