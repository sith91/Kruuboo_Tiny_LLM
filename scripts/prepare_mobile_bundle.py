import json
import shutil
from pathlib import Path

def prepare_bundle():
    print("📦 Preparing Native Mobile Bundle...")
    
    # 1. Create mobile assets directory
    bundle_path = Path("mobile_assets")
    bundle_path.mkdir(exist_ok=True)
    
    # 2. Copy the Quantized ONNX Model (The Brain)
    model_src = Path("models/tiny_llm_quantized.onnx")
    if model_src.exists():
        shutil.copy(model_src, bundle_path / "model.onnx")
        print("  ✅ Model bundled")
    else:
        print("  ❌ Model not found! Run create_everything.py first.")
        return

    # 3. Copy the Tokenizer
    tokenizer_src = Path("tokenizer.json")
    if tokenizer_src.exists():
        shutil.copy(tokenizer_src, bundle_path / "tokenizer.json")
        print("  ✅ Tokenizer bundled")

    # 4. Consolidate Knowledge Base
    print("  🔍 Bundling semantic knowledge base...")
    kb_file = Path("knowledge_base/knowledge.json")
    if kb_file.exists():
        shutil.copy(kb_file, bundle_path / "knowledge_base.json")
        with open(kb_file, "r", encoding="utf-8") as f:
            docs = json.load(f)
        print(f"  ✅ Knowledge base bundled ({len(docs)} documents)")
    else:
        print("  ⚠️ No knowledge base found at knowledge_base/knowledge.json")

    # 5. Export Mobile Manifest & Config
    cfg_file = Path("config.yaml")
    if cfg_file.exists():
        shutil.copy(cfg_file, bundle_path / "config.yaml")
    
    manifest = {
        "model_name": "Kruuboo-SLM-0.2",
        "parameters": "105.4M",
        "quantization": "INT8",
        "d_model": 768,
        "n_heads": 12,
        "n_layers": 14,
        "d_ff": 3072,
        "max_seq_len": 1024,
        "vocab_size": 8192,
        "special_tokens": {
            "pad": "<pad>",
            "bos": "<|im_start|>",
            "eos": "<|im_end|>"
        }
    }
    with open(bundle_path / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print("  ✅ Mobile metadata manifest written")

    print(f"\n🎉 DONE! Mobile bundle ready in '{bundle_path}'.")

if __name__ == "__main__":
    prepare_bundle()
