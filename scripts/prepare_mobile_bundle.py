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

    # 4. Consolidate Domain Knowledge
    print("  🔍 Gathering domain knowledge...")
    kb_root = Path("knowledge_bases")
    mobile_kb = {}
    
    if kb_root.exists():
        for domain_dir in kb_root.iterdir():
            if domain_dir.is_dir():
                kb_file = domain_dir / "knowledge.json"
                if kb_file.exists():
                    with open(kb_file, 'r') as f:
                        mobile_kb[domain_dir.name] = json.load(f)
        
        with open(bundle_path / "knowledge_base.json", 'w') as f:
            json.dump(mobile_kb, f)
        print(f"  ✅ Knowledge bundled ({len(mobile_kb)} domains)")

    print(f"\n🎉 DONE! Copy the '{bundle_path}' folder into your mobile app project.")

if __name__ == "__main__":
    prepare_bundle()
