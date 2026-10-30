"""
Multi-Profile On-Demand Trainer & Exporter (`train/all_profiles.py`)
====================================================================

Trains and exports any profile(s) registered in `config/profiles.yaml`.

Usage:
  python train/all_profiles.py --profiles test6L
  python train/all_profiles.py --profiles micro small
  python train/all_profiles.py --all-flagged
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model import TinyLLM
from src.profiles import ModelProfile, ProfileRegistry
from src.tokenizer import TinyTokenizer
from src.train import Trainer, load_training_corpus
from scripts.export_onnx import export_to_onnx


def _prepare_profile_tokenizer(profile: ModelProfile, training_samples: List[str]) -> TinyTokenizer:
    """Ensure a tokenizer matching `profile.vocab_size` exists at `profile.tokenizer_path`."""
    tok_dest = Path(profile.tokenizer_path)
    tok_dest.parent.mkdir(parents=True, exist_ok=True)

    if tok_dest.exists():
        tok = TinyTokenizer.load(str(tok_dest))
        if tok.vocab_size == profile.vocab_size:
            return tok

    root_tok_path = Path("tokenizer.json")
    if root_tok_path.exists():
        root_tok = TinyTokenizer.load(str(root_tok_path))
        if root_tok.vocab_size == profile.vocab_size:
            shutil.copy2(root_tok_path, tok_dest)
            return root_tok

    # Train a dedicated BPE tokenizer for custom vocab_size (e.g., 4096)
    print(f"📦 Building BPE tokenizer for vocab_size={profile.vocab_size} → {tok_dest} ...")
    sample_text = "\n\n".join(training_samples[:400])
    return TinyTokenizer.train_from_text(
        text=sample_text,
        vocab_size=profile.vocab_size,
        save_path=str(tok_dest),
    )


def train_single_profile(
    profile: ModelProfile,
    iters: int = 200,
    batch_size: int = 8,
    lr: float = 6e-4,
    device: Optional[str] = None,
) -> tuple:
    """
    Train and export a single `ModelProfile` to `checkpoints/<name>/` and `artifacts/<name>_int8.onnx`.
    """
    print("\n" + "=" * 78)
    print(f"🚀 TRAINING PROFILE: '{profile.name}' ({profile.targets['estimated_params_m']:.2f}M Params)")
    print(
        f"   Dims: n_layers={profile.n_layers}, d_model={profile.d_model}, "
        f"n_heads={profile.n_heads}, n_kv_heads={profile.n_kv_heads}, "
        f"d_ff={profile.d_ff}, max_seq_len={profile.max_seq_len}, vocab_size={profile.vocab_size}"
    )
    print("=" * 78)

    # 1. Load training corpus
    training_samples = load_training_corpus()

    # 2. Prepare tokenizer matching profile.vocab_size
    tokenizer = _prepare_profile_tokenizer(profile, training_samples)

    # 3. Select compute device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"🖥️  Compute Device: {device.upper()}")

    # 4. Instantiate TinyLLM with profile's build-time dimensions
    model = TinyLLM(
        vocab_size=profile.vocab_size,
        d_model=profile.d_model,
        n_heads=profile.n_heads,
        n_kv_heads=profile.n_kv_heads,
        n_layers=profile.n_layers,
        d_ff=profile.d_ff,
        max_seq_len=profile.max_seq_len,
        dropout=profile.dropout,
    )

    ckpt_dir = Path(profile.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_file = ckpt_dir / "tiny_llm.pt"

    if ckpt_file.exists():
        try:
            state_dict = torch.load(str(ckpt_file), map_location="cpu")
            model_dict = model.state_dict()
            matched = {
                k: v for k, v in state_dict.items()
                if k in model_dict and model_dict[k].shape == v.shape
            }
            if len(matched) == len(model_dict):
                model.load_state_dict(matched)
                print(f"💾 Resumed matching checkpoint from {ckpt_file}")
        except Exception as e:
            print(f"ℹ️ Starting fresh weights for '{profile.name}': {e}")

    # 5. Train
    micro_batch = batch_size if device in ("mps", "cuda") else min(4, batch_size)
    eval_every = max(10, min(50, iters // 4))
    trainer = Trainer(model, tokenizer, device=device, lr_max=lr, lr_min=3e-5)
    trained_model = trainer.train(
        training_samples,
        max_iters=iters,
        batch_size=batch_size,
        micro_batch_size=micro_batch,
        eval_every=eval_every,
    )

    # 6. Save profile checkpoint
    torch.save(trained_model.state_dict(), str(ckpt_file))
    print(f"💾 Saved checkpoint to {ckpt_file}")

    # 7. Export ONNX and INT8 quantized ONNX to artifacts/
    onnx_path, quant_path = export_to_onnx(
        model_path=str(ckpt_file),
        tokenizer_path=str(profile.tokenizer_path),
        output_path=str(profile.onnx_path),
        quantized_path=str(profile.artifact_path),
        d_model=profile.d_model,
        n_heads=profile.n_heads,
        n_kv_heads=profile.n_kv_heads,
        n_layers=profile.n_layers,
        d_ff=profile.d_ff,
        max_seq_len=profile.max_seq_len,
        vocab_size=profile.vocab_size,
    )

    print(f"✅ Profile '{profile.name}' ready!")
    print(f"   • Checkpoint : {ckpt_file}")
    print(f"   • INT8 ONNX  : {quant_path}")
    print(f"   • Run command: python main.py --profile {profile.name}")
    return str(ckpt_file), str(quant_path)


def main():
    parser = argparse.ArgumentParser(
        description="Train and export one or more profiles from config/profiles.yaml"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--profiles",
        nargs="+",
        metavar="NAME",
        help="One or more profile names from config/profiles.yaml to train",
    )
    group.add_argument(
        "--all-flagged",
        action="store_true",
        help="Train all profiles in config/profiles.yaml with flagged: true",
    )
    parser.add_argument("--iters", type=int, default=200, help="Training iterations per profile (default: 200)")
    parser.add_argument("--batch-size", type=int, default=8, help="Effective batch size (default: 8)")
    parser.add_argument("--lr", type=float, default=6e-4, help="Peak learning rate (default: 6e-4)")
    parser.add_argument("--device", type=str, default=None, help="Override compute device (cpu, mps, cuda)")
    parser.add_argument("--registry", type=str, default="config/profiles.yaml", help="Path to profiles.yaml")

    args = parser.parse_args()
    registry = ProfileRegistry(args.registry)

    if args.all_flagged:
        targets = registry.get_flagged()
        if not targets:
            print("ℹ️ No profiles have `flagged: true` in config/profiles.yaml.")
            return
    else:
        targets = [registry.get(name) for name in args.profiles]

    print(f"📋 Selected {len(targets)} profile(s) to train: {[p.name for p in targets]}")
    for prof in targets:
        train_single_profile(
            profile=prof,
            iters=args.iters,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )


if __name__ == "__main__":
    main()

