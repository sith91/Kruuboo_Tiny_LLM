"""
Unified Ingestion Pipeline for Kruuboo Tiny LLM & Semantic RAG
==============================================================

Replaces all fragmented one-off ingestion scripts with a single, robust tool:
- Automatically parses .json, .csv, .txt, and .md files
- Supports single files, directories, batch processing, and direct text input
- Automatically chunks, formats, and indexes into global semantic memory
- Uses Native 768d dense SLM embeddings
"""

import sys
import os
import re
import csv
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag import RAGSystem


def parse_json_file(file_path: Path) -> List[Tuple[str, str, Dict]]:
    """Extracts text items, sources, and metadata from JSON files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    items = []
    fname = file_path.name

    if isinstance(data, list):
        for idx, entry in enumerate(data):
            if isinstance(entry, dict):
                # Check for home remedy / ailment pattern
                if "ailment" in entry and "remedy" in entry:
                    ailment = entry["ailment"]
                    remedy = entry["remedy"]
                    text = f"Home Remedy for {ailment}: {remedy}"
                    items.append((text, f"{fname}:{ailment}", {"category": "home_remedies", "topic": ailment}))
                # Check for legal advice pattern
                elif "topic" in entry and "question" in entry and "advice" in entry:
                    topic = entry["topic"]
                    q = entry["question"]
                    adv = entry["advice"]
                    text = f"Legal Advice on {topic}: Question: {q} Advice: {adv}"
                    items.append((text, f"{fname}:{topic}", {"category": "legal", "topic": topic}))
                # Check for survival guide pattern
                elif "situation" in entry and "instructions" in entry:
                    topic = entry.get("topic", f"Scenario_{idx}")
                    sit = entry["situation"]
                    inst = entry["instructions"]
                    text = f"Survival Scenario: {sit}. Solution for {topic}: {inst}"
                    items.append((text, f"{fname}:{topic}", {"category": "survival", "topic": topic}))
                # Check for Q&A pattern
                elif "question" in entry and "answer" in entry:
                    q = entry["question"]
                    a = entry["answer"]
                    text = f"Question: {q}\nAnswer: {a}"
                    items.append((text, f"{fname}:item_{idx}", {"question": q}))
                # Check for generic text field
                elif "text" in entry:
                    t = entry["text"].strip()
                    if t:
                        src = entry.get("source", f"{fname}:item_{idx}")
                        items.append((t, src, entry.get("metadata", {})))
                # Generic dictionary
                else:
                    pairs = [f"{k}: {v}" for k, v in entry.items() if v]
                    if pairs:
                        items.append(("\n".join(pairs), f"{fname}:item_{idx}", {}))
            elif isinstance(entry, str) and entry.strip():
                items.append((entry.strip(), f"{fname}:item_{idx}", {}))

    elif isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (str, int, float)):
                items.append((f"{k}: {v}", f"{fname}:{k}", {}))
            elif isinstance(v, dict):
                sub_pairs = [f"{sk}: {sv}" for sk, sv in v.items() if sv]
                items.append((f"{k} details: " + "; ".join(sub_pairs), f"{fname}:{k}", {}))
            elif isinstance(v, list):
                for sub_idx, sub_item in enumerate(v):
                    items.append((f"{k} [{sub_idx}]: {sub_item}", f"{fname}:{k}_{sub_idx}", {}))

    return items


def parse_csv_file(file_path: Path) -> List[Tuple[str, str, Dict]]:
    """Extracts formatted text rows from CSV files."""
    items = []
    fname = file_path.name

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []

        clean_headers = [h.strip() for h in header]

        # Disease descriptions special handling
        if "symptom_description" in fname.lower():
            for idx, row in enumerate(reader):
                if len(row) >= 2:
                    disease = row[0].strip()
                    desc = row[1].strip()
                    if disease and desc:
                        items.append((f"Disease: {disease}. Description: {desc}.", f"{fname}:{disease}", {"disease": disease}))
            return items

        # Disease precautions special handling
        if "symptom_precaution" in fname.lower():
            for idx, row in enumerate(reader):
                if len(row) >= 2:
                    disease = row[0].strip()
                    precs = [p.strip() for p in row[1:] if p.strip()]
                    if disease and precs:
                        items.append((f"Disease: {disease}. Precautions: {', '.join(precs)}.", f"{fname}:{disease}", {"disease": disease}))
            return items

        # Generic CSV rows
        for idx, row in enumerate(reader):
            if not any(row):
                continue
            row_parts = []
            for h, val in zip(clean_headers, row):
                v = val.strip()
                if v:
                    row_parts.append(f"{h}: {v}")
            if row_parts:
                items.append((" | ".join(row_parts), f"{fname}:row_{idx+1}", {}))

    return items


def parse_text_file(file_path: Path, max_chunk_size: int = 800) -> List[Tuple[str, str, Dict]]:
    """Extracts paragraph/section chunks from plain text or markdown files."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    fname = file_path.name
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [content.strip()] if content.strip() else []

    items = []
    current_chunk = []
    current_len = 0

    for idx, p in enumerate(paragraphs):
        p_len = len(p)
        if current_len + p_len > max_chunk_size and current_chunk:
            combined = "\n\n".join(current_chunk)
            items.append((combined, f"{fname}:chunk_{len(items)+1}", {}))
            current_chunk = [p]
            current_len = p_len
        else:
            current_chunk.append(p)
            current_len += p_len + 2

    if current_chunk:
        combined = "\n\n".join(current_chunk)
        items.append((combined, f"{fname}:chunk_{len(items)+1}", {}))

    return items


def ingest_file(file_path: Path, rag: RAGSystem, max_items: Optional[int] = None) -> int:
    """Ingests a single file into the RAG system."""
    suffix = file_path.suffix.lower()
    if suffix == ".json":
        items = parse_json_file(file_path)
    elif suffix == ".csv":
        items = parse_csv_file(file_path)
    elif suffix in (".txt", ".md", ".log"):
        items = parse_text_file(file_path)
    else:
        print(f"⏭️ Skipping unsupported format: {file_path}")
        return 0

    if max_items:
        items = items[:max_items]

    print(f"📖 Ingesting '{file_path.name}' ({len(items)} items)...")
    for text, source, meta in items:
        rag.add_text(text, source=source, metadata=meta)

    print(f"✅ Ingested {len(items)} items from {file_path.name}")
    return len(items)


def ingest_all_foundational(rag: RAGSystem) -> int:
    """Ingests all available datasets in training_data into the unified semantic memory."""
    td_dir = Path("training_data")
    if not td_dir.exists():
        print("❌ training_data directory not found.")
        return 0

    total = 0
    # 1. JSON files
    for jf in sorted(td_dir.glob("*.json")):
        total += ingest_file(jf, rag)

    # 2. CSV files (descriptions and precautions)
    for cf in sorted(td_dir.glob("*.csv")):
        if "ProblemSolution" not in cf.name:  # skip massive raw code datasets for RAG
            total += ingest_file(cf, rag)

    # 3. Selected foundational text manuals
    for tf in ["US_Army_Survival_Manual_FM21-76.txt"]:
        p = td_dir / tf
        if p.exists():
            total += ingest_file(p, rag, max_items=250)

    return total


def main():
    parser = argparse.ArgumentParser(description="Unified Ingestion Engine for Kruuboo Tiny LLM & RAG")
    parser.add_argument("path", nargs="?", default=None, help="File or directory path to ingest")
    parser.add_argument("--all", action="store_true", help="Ingest all foundational datasets in training_data/")
    parser.add_argument("--text", type=str, default=None, help="Direct text string to ingest")
    parser.add_argument("--source", type=str, default="CLI Direct", help="Source attribution for direct text")
    parser.add_argument("--max-items", type=int, default=None, help="Limit number of items to ingest from file")
    args = parser.parse_args()

    print("=" * 65)
    print("🧠 KRUUBOO SLM — UNIFIED KNOWLEDGE INGESTION ENGINE")
    print("=" * 65)

    rag = RAGSystem(
        model_path="models/tiny_llm_quantized.onnx",
        tokenizer_path="tokenizer.json",
        knowledge_base_path="knowledge_base",
    )

    if args.text:
        rag.add_text(args.text, source=args.source)
        print(f"✅ Successfully ingested direct text input (source: '{args.source}')")
        return

    if args.all:
        print("🚀 Ingesting all foundational datasets from training_data/...")
        count = ingest_all_foundational(rag)
        print(f"\n🎉 Finished batch ingestion: {count} total knowledge chunks indexed.")
        stats = rag.get_stats()
        print(f"📚 Total Global Semantic Memory: {stats['total_documents']:,} documents")
        return

    if args.path:
        p = Path(args.path)
        if not p.exists():
            print(f"❌ Error: Path not found: {p}")
            sys.exit(1)

        if p.is_file():
            count = ingest_file(p, rag, max_items=args.max_items)
        elif p.is_dir():
            count = 0
            for f in sorted(p.iterdir()):
                if f.is_file() and f.suffix.lower() in (".json", ".csv", ".txt", ".md"):
                    count += ingest_file(f, rag, max_items=args.max_items)
        print(f"\n🎉 Finished ingestion: {count} knowledge chunks indexed into global memory.")
        stats = rag.get_stats()
        print(f"📚 Total Global Semantic Memory: {stats['total_documents']:,} documents")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
