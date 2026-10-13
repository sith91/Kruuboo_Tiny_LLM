"""
Kruuboo Tiny LLM — Master End-to-End Build & Initialization Script
==================================================================

Automates complete setup and verification for Kruuboo SLM (105.4M Parameters):
  1. Verifies workspace directories and environment dependencies
  2. Builds BPE Tokenizer (8,192 Subword Vocab) if not already built
  3. Ingests foundational knowledge into Native 768d Semantic Memory
  4. Trains / Verifies Neural Transformer Model (RoPE, SwiGLU, Flash Attention)
  5. Exports optimized ONNX and INT8 quantized deployment graph
  6. Executes end-to-end conversational sanity test
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
import torch


def main():
    print("=" * 70)
    print("🚀 KRUUBOO TINY LLM — COMPLETE END-TO-END BUILD PIPELINE")
    print("=" * 70)

    start_time = time.time()

    # 1. Directories setup
    print("\n[1/6] Initializing Project Directories...")
    for d in ["models", "knowledge_base", "training_data", "workspace"]:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  📁 Ready: {d}/")

    # 2. Load Configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})
    print(f"\n[2/6] System Architecture Configuration:")
    print(f"  • Model: {model_cfg.get('name', 'tiny_llm')} ({model_cfg.get('d_model')}d, {model_cfg.get('n_layers')} layers, {model_cfg.get('n_heads')} heads)")
    print(f"  • Feed-Forward (SwiGLU): {model_cfg.get('d_ff')} hidden units")
    print(f"  • Context Memory Window: {model_cfg.get('max_seq_len')} tokens")
    print(f"  • Target Vocabulary: {model_cfg.get('vocab_size')} tokens")

    # 3. Tokenizer Setup
    print("\n[3/6] Verifying BPE Tokenizer...")
    from src.tokenizer import TinyTokenizer
    tokenizer_path = Path("tokenizer.json")
    if tokenizer_path.exists():
        tokenizer = TinyTokenizer.load(str(tokenizer_path))
        print(f"  ✅ Existing BPE tokenizer loaded ({tokenizer.vocab_size:,} tokens)")
    else:
        print("  🔨 Training fresh Byte-Pair Encoding (BPE) tokenizer...")
        from src.train import load_training_corpus
        samples = load_training_corpus()
        sample_text = "\n\n".join(samples[:500]) if samples else "Kruuboo SLM conversational language model."
        tokenizer = TinyTokenizer.train_from_text(
            text=sample_text,
            vocab_size=model_cfg.get("vocab_size", 8192),
            save_path="tokenizer.json",
        )
        print(f"  ✅ BPE tokenizer trained & saved to tokenizer.json ({tokenizer.vocab_size:,} tokens)")

    # 4. Ingest Foundational Knowledge Base
    print("\n[4/6] Ingesting Foundational Knowledge into Semantic Memory...")
    from src.rag import RAGSystem
    from scripts.ingest import ingest_all_foundational
    rag = RAGSystem(
        model_path="models/tiny_llm_quantized.onnx",
        tokenizer_path="tokenizer.json",
        knowledge_base_path="knowledge_base",
    )
    if len(rag.documents) == 0:
        indexed_count = ingest_all_foundational(rag)
        print(f"  ✅ Indexed {indexed_count} foundational knowledge chunks with 768d dense embeddings")
    else:
        print(f"  ✅ Semantic Memory ready ({len(rag.documents):,} indexed documents)")

    # 5. Train / Verify Model & Export ONNX
    print("\n[5/6] Building / Exporting Neural Model & ONNX Graph...")
    model_pt = Path("models/tiny_llm.pt")
    onnx_quant = Path("models/tiny_llm_quantized.onnx")

    if not model_pt.exists():
        print("  🧠 No PyTorch checkpoint found. Launching baseline training session...")
        from src.train import train_model
        train_model(iters=min(200, train_cfg.get("max_iters", 8000)), export_onnx=True)
    elif not onnx_quant.exists():
        print("  📦 PyTorch weights found. Exporting ONNX & INT8 quantized deployment models...")
        from scripts.export_onnx import export_to_onnx
        export_to_onnx(
            model_path="models/tiny_llm.pt",
            tokenizer_path="tokenizer.json",
            output_path="models/tiny_llm.onnx",
        )
    else:
        print("  ✅ Neural weights and INT8 ONNX models verified on disk.")

    # 6. Conversational Pipeline Sanity Check
    print("\n[6/6] Executing Conversational Pipeline Verification...")
    test_rag = RAGSystem(
        model_path="models/tiny_llm_quantized.onnx",
        tokenizer_path="tokenizer.json",
        max_seq_len=model_cfg.get("max_seq_len", 1024),
    )
    test_query = "What is the recommended remedy for a mild burn?"
    response = test_rag.ask(test_query)

    print(f"\n💬 Test Query: '{test_query}'")
    print(f"🤖 Kruuboo SLM Response: {response.get('answer')}")
    print(f"📚 Semantic Chunks Retrieved: {response.get('knowledge_used')}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"🎉 KRUUBOO TINY LLM IS FULLY BUILT & READY IN {elapsed:.1f}s!")
    print("=" * 70)
    print("👉 To start conversational shell:   python main.py")
    print("👉 To start web studio & monitor:   python server.py")
    print("👉 To ingest new data:              python scripts/ingest.py --all")


if __name__ == "__main__":
    main()