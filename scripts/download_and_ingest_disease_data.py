import sys
import csv
import io
import requests
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
    print("DOWNLOADING & INGESTING MEDICAL DISEASE DATASET")
    print("=" * 60)

    # 1. Initialize RAG
    print("🤖 Initializing RAG System...")
    rag = RAGSystem(MODEL_PATH, TOKENIZER_PATH, max_seq_len=MODEL_MAX_SEQ_LEN)

    # 2. Download Description & Precaution CSVs
    desc_url = "https://raw.githubusercontent.com/bharatdappili/chatbot/master/symptom_Description.csv"
    prec_url = "https://raw.githubusercontent.com/bharatdappili/chatbot/master/symptom_precaution.csv"

    print("📥 Downloading disease descriptions...")
    r_desc = requests.get(desc_url)
    if r_desc.status_code != 200:
        print("❌ Failed to download descriptions dataset")
        return

    print("📥 Downloading disease precautions...")
    r_prec = requests.get(prec_url)
    if r_prec.status_code != 200:
        print("❌ Failed to download precautions dataset")
        return

    # 3. Parse Descriptions
    descriptions = {}
    f_desc = io.StringIO(r_desc.text)
    reader_desc = csv.reader(f_desc)
    for row in reader_desc:
        if len(row) >= 2:
            disease = row[0].strip()
            desc = row[1].strip()
            descriptions[disease] = desc

    # 4. Parse Precautions
    precautions = {}
    f_prec = io.StringIO(r_prec.text)
    reader_prec = csv.reader(f_prec)
    for row in reader_prec:
        if len(row) >= 2:
            disease = row[0].strip()
            precs = [p.strip() for p in row[1:] if p.strip()]
            precautions[disease] = precs

    # 5. Ingest joined data
    print("\n✍️ Ingesting diseases dataset into RAG 'medical' domain...")
    count = 0
    for disease, desc in descriptions.items():
        precs = precautions.get(disease, [])
        precs_str = ", ".join(precs) if precs else "None specified"
        
        # Format the RAG document content nicely
        text = f"Disease: {disease}. Description: {desc}. Precautions & Prevention: {precs_str}."
        
        # Add to medical domain
        rag.add_domain_knowledge("medical", text, f"DiseaseDataset:{disease}")
        print(f" ✅ Ingested info for: {disease}")
        count += 1

    print(f"\n🎉 Successfully added {count} diseases with descriptions and precautions to the 'medical' domain!")
    print("You can query them in the shell after it restarts, for example: 'what is Malaria' or 'precautions for Drug Reaction'")

if __name__ == "__main__":
    main()
