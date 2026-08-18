import sys
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
    print("INGESTING US ARMY SURVIVAL MANUAL (FM 21-76)")
    print("=" * 60)

    # 1. Initialize RAG
    print("🤖 Initializing RAG System...")
    rag = RAGSystem(MODEL_PATH, TOKENIZER_PATH, max_seq_len=MODEL_MAX_SEQ_LEN)

    # 2. Read Text Manual
    manual_path = Path("training_data/US_Army_Survival_Manual_FM21-76.txt")
    if not manual_path.exists():
        print(f"❌ Manual file not found at {manual_path}")
        return

    print("📖 Reading and chunking manual...")
    with open(manual_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    # Split by double newlines to keep paragraphs together, or fallback to fixed size
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0
    
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        # If adding this paragraph exceeds 400 chars, flush current chunk
        if current_len + len(p) > 400:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [p]
            current_len = len(p)
        else:
            current_chunk.append(p)
            current_len += len(p) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    print(f"Loaded {len(chunks)} text chunks.")

    # 3. Ingest first 1000 chunks to keep RAG size and performance balanced
    print("\n✍️ Loading chunks into RAG 'survival' domain...")
    count = 0
    limit = 1000
    for i, chunk in enumerate(chunks[:limit]):
        # Add to survival domain
        rag.add_domain_knowledge("survival", chunk, f"US_Army_Survival_Manual:Part_{i}")
        count += 1
        if count % 200 == 0:
            print(f"   Ingested {count} chunks...")

    print(f"\n🎉 Successfully added {count} chunks from the survival manual to the 'survival' domain!")

if __name__ == "__main__":
    main()
