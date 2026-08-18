"""Training script for Tiny LLM — upgraded to match new model architecture."""

import math
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

    def get_batch(self, data: torch.Tensor, batch_size: int) -> tuple:
        """Sample a random batch of (input, target) pairs."""
        ix = torch.randint(len(data) - BLOCK_SIZE, (batch_size,))
        x  = torch.stack([data[i    : i + BLOCK_SIZE    ] for i in ix])
        y  = torch.stack([data[i + 1: i + BLOCK_SIZE + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    # ── Evaluation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, val_data: torch.Tensor, eval_batches: int = 10) -> float:
        """Estimate validation loss (no gradient)."""
        self.model.eval()
        losses = []
        for _ in range(eval_batches):
            x, y = self.get_batch(val_data, BATCH_SIZE)
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self, text: str) -> TinyLLM:
        """
        Full training loop with:
          - Train / validation split
          - Cosine LR schedule with linear warmup
          - Gradient clipping
          - Periodic validation loss reporting
        """
        # Tokenise
        token_ids = self.tokenizer.encode(text)
        data      = torch.tensor(token_ids, dtype=torch.long)

        # Train / val split
        split    = int(len(data) * (1 - VAL_RATIO))
        train_data = data[:split]
        val_data   = data[split:]
        print(f"📊 Dataset: {len(data):,} tokens  "
              f"(train {len(train_data):,}  |  val {len(val_data):,})")

        if len(train_data) <= BLOCK_SIZE:
            print("⚠️  Training data too small — results will be poor. "
                  "Replace sample_text with a real corpus.")

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
            x, y   = self.get_batch(train_data, BATCH_SIZE)
            logits = self.model(x)
            loss   = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
            self.optimizer.step()

            # ── Logging / evaluation ─────────────────────────────────────────
            if step % EVAL_EVERY == 0 or step == MAX_ITERS - 1:
                val_loss = self.evaluate(val_data)
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
                                if not isinstance(item, dict):
                                    continue
                                if "question" in item and "advice" in item:
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
        combined_text = "\n\n".join(custom_texts)
        if len(combined_text) < 2000:
            print(f"⚠️ Custom training text is too short ({len(combined_text)} chars). Repeating it to reach baseline length...")
            repeats = (10000 // len(combined_text)) + 1
            sample_text = combined_text * repeats
        else:
            sample_text = combined_text
        print(f"📈 Loaded custom training corpus: {len(combined_text):,} characters (expanded to {len(sample_text):,} characters for training)")
    else:
        print("ℹ️ No custom datasets found. Falling back to default sample corpus...")
        sample_text = (
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

    # ── BPE tokenizer ─────────────────────────────────────────────────────────
    if Path("tokenizer.json").exists():
        print("📦 Loading existing tokenizer from tokenizer.json...")
        tokenizer = TinyTokenizer.load("tokenizer.json")
    else:
        tokenizer = TinyTokenizer.train_from_text(
            text=sample_text,
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
            print("💾 Loading pre-trained model weights from models/tiny_llm.pt...")
            state_dict = torch.load("models/tiny_llm.pt", map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            print("✅ Weights loaded successfully!")
        except Exception as e:
            print(f"⚠️ Failed to load pre-trained weights: {e}. Training from scratch.")
            
    print(f"🧠 Parameters: {model.num_parameters():,}"
          f"  ({model.num_parameters()/1e6:.2f} M)")
    print(f"   d_model={D_MODEL}  n_heads={N_HEADS}  "
          f"n_layers={N_LAYERS}  d_ff={D_FF}  seq_len={BLOCK_SIZE}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer      = Trainer(model, tokenizer, device=device)
    trained_model= trainer.train(sample_text)

    # ── Save ──────────────────────────────────────────────────────────────────
    Path("models").mkdir(exist_ok=True)
    save_path = "models/tiny_llm.pt"
    torch.save(trained_model.state_dict(), save_path)
    print(f"💾 Model saved to {save_path}")

    return trained_model, tokenizer


if __name__ == "__main__":
    train_model()