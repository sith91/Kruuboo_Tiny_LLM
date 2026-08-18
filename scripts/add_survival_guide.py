import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGSystem

# Config
MODEL_MAX_SEQ_LEN = 128
MODEL_PATH = "models/tiny_llm_quantized.onnx"
TOKENIZER_PATH = "tokenizer.json"

def main():
    print("=" * 60)
    print("INGESTING SURVIVAL GUIDE DATASET")
    print("=" * 60)

    # 1. Initialize RAG
    print("🤖 Initializing RAG System...")
    rag = RAGSystem(MODEL_PATH, TOKENIZER_PATH, max_seq_len=MODEL_MAX_SEQ_LEN)

    # 2. Read JSON
    json_path = Path("training_data/survival_guide.json")
    if not json_path.exists():
        print(f"❌ Dataset file not found at {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Ingest
    print("\n✍️ Loading survival items into the 'survival' domain...")
    count = 0
    for item in data:
        topic = item["topic"]
        sit = item["situation"]
        inst = item["instructions"]
        text = f"Survival Scenario: {sit}. Solution for {topic}: {inst}"
        
        # Add to survival domain
        rag.add_domain_knowledge("survival", text, f"SurvivalGuide:{topic}")
        print(f" ✅ Ingested guide for: {topic}")
        count += 1

    print(f"\n🎉 Successfully added {count} survival guides to the 'survival' domain!")
    print("You can switch to the 'survival' domain in the shell:")
    print("  /mode survival")
    print("And ask questions, like: 'how to start a fire'")

if __name__ == "__main__":
    main()
