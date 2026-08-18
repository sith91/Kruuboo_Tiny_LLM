import sys
import os
import requests
import csv
import io
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
    print("INGESTING SINHALA NEWS DATASET")
    print("=" * 60)

    # 1. Initialize RAG System
    print("🤖 Initializing RAG System...")
    rag = RAGSystem(MODEL_PATH, TOKENIZER_PATH, max_seq_len=MODEL_MAX_SEQ_LEN)

    # 2. Download Dataset
    csv_url = "https://huggingface.co/datasets/NLPC-UOM/Sinhala-News-Category-classification/resolve/main/sinhala-news-categories.csv"
    print(f"📥 Downloading Sinhala News CSV from: {csv_url} ...")
    
    r = requests.get(csv_url)
    if r.status_code != 200:
        print(f"❌ Failed to download dataset: {r.status_code}")
        return

    # 3. Parse CSV
    print(" Parsing CSV data...")
    f = io.StringIO(r.text)
    reader = csv.reader(f)
    header = next(reader)
    
    rows = list(reader)
    print(f"✅ Loaded {len(rows)} raw comments.")

    # 4. Ingest into sinhala_news domain
    # We ingest a subset of 500 items to avoid bloating the tiny local model DB
    subset_size = 500
    print(f" Ingesting first {subset_size} items into the 'sinhala_news' domain...")
    
    count = 0
    for i, row in enumerate(rows[:subset_size]):
        if len(row) >= 2:
            comment = row[0].strip()
            label = row[1].strip()
            if comment:
                # Add to domain knowledge base
                rag.add_domain_knowledge("sinhala_news", comment, f"SinhalaNews_Label_{label}")
                count += 1
                if count % 100 == 0:
                    print(f"   Processed {count}/{subset_size} items...")

    print(f"✅ Ingestion complete! Successfully ingested {count} news items.")
    print("You can now switch to the 'sinhala_news' domain in your shell:")
    print("  /mode sinhala_news")
    print("And ask questions in Sinhala!")

if __name__ == "__main__":
    main()
