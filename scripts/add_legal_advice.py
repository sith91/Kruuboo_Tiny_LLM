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
    print("INGESTING LEGAL ADVICE DATASET")
    print("=" * 60)

    # 1. Initialize RAG
    print("🤖 Initializing RAG System...")
    rag = RAGSystem(MODEL_PATH, TOKENIZER_PATH, max_seq_len=MODEL_MAX_SEQ_LEN)

    # 2. Read JSON
    json_path = Path("training_data/legal_advice.json")
    if not json_path.exists():
        print(f"❌ Dataset file not found at {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 3. Ingest
    print("\n✍️ Loading legal advice items into the 'legal' domain...")
    count = 0
    for item in data:
        topic = item["topic"]
        q = item["question"]
        adv = item["advice"]
        text = f"Legal Advice on {topic}: Question: {q} Advice: {adv}"
        
        # Add to legal domain
        rag.add_domain_knowledge("legal", text, f"LegalAdvice:{topic}")
        print(f" ✅ Ingested advice for: {topic}")
        count += 1

    print(f"\n🎉 Successfully added {count} legal advice documents to the 'legal' domain!")
    print("You can switch to the 'legal' domain in the shell:")
    print("  /mode legal")
    print("And ask questions, like: 'What is a NDA?'")

if __name__ == "__main__":
    main()
