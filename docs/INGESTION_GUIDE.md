# 📥 Kruuboo Tiny LLM (SLM-0.2) — Unified Knowledge Ingestion Guide

The `scripts/ingest.py` utility provides a unified command-line pipeline for parsing, chunking, embedding, and indexing documents into Kruuboo's **Native 768d Semantic RAG Memory** (`knowledge_base/knowledge.json`).

This single script replaces all legacy, one-off ingestion tools (`add_home_remedies.py`, `add_legal_advice.py`, `build_knowledge_base.py`, etc.) with a standardized, high-performance ingestion engine.

---

## 🚀 Quick Usage Examples

### 1. Ingest a Single File
```bash
python3 scripts/ingest.py training_data/home_remedies.json
```

### 2. Ingest an Entire Directory (Recursive)
```bash
python3 scripts/ingest.py training_data/
```

### 3. Ingest Direct Text via CLI
```bash
python3 scripts/ingest.py \
  --text "Photosynthesis converts carbon dioxide and water into glucose and oxygen in the presence of sunlight." \
  --source "Biology101"
```

### 4. Batch Ingest Selected Files
```bash
python3 scripts/ingest.py --batch \
  training_data/home_remedies.json \
  training_data/legal_advice.json \
  training_data/survival_guide.json
```

### 5. Dry-Run Verification (Preview Chunks Without Modifying Index)
```bash
python3 scripts/ingest.py training_data/symptom_Description.csv --dry-run
```

---

## 📊 Supported File Formats & Auto-Detection Schemas

`scripts/ingest.py` inspects the structure of ingested files and applies domain-specific parsing:

### 1. JSON Files (`.json`)
The parser recognizes standard structures automatically:
- **Home Remedy Pattern**:
  ```json
  [
    { "ailment": "Fever", "remedy": "Drink plenty of fluids, rest, and use a cool damp cloth on the forehead." }
  ]
  ```
  *Indexed as*: `Home Remedy for Fever: Drink plenty of fluids...`

- **Legal Advice Pattern**:
  ```json
  [
    { "topic": "Non-Disclosure Agreements", "question": "What is an NDA?", "advice": "An NDA is a contract that binds parties to secrecy..." }
  ]
  ```
  *Indexed as*: `Legal Advice on Non-Disclosure Agreements: Question: What is an NDA? Advice: ...`

- **Survival Guide Pattern**:
  ```json
  [
    { "topic": "Wilderness Water Purification", "situation": "Dirty stream water", "instructions": "Boil vigorously for at least 1 minute..." }
  ]
  ```

- **Q&A Datasets**:
  ```json
  [
    { "question": "What is Python?", "answer": "Python is a high-level interpreted programming language..." }
  ]
  ```

- **Generic Text & Key-Value Dictionaries**:
  ```json
  [
    { "text": "Raw document paragraph...", "source": "CustomDoc", "metadata": { "category": "science" } }
  ]
  ```

### 2. CSV Files (`.csv`)
- Header row is automatically detected.
- Each row is rendered into a key-value semantic block:
  ```text
  Disease: Fungal infection
  Description: Fungal infection is a common skin condition caused by fungus...
  ```

### 3. Text & Markdown (`.txt`, `.md`)
- Large documents (such as manuals or books) are split using sentence-aware paragraph chunking.
- Default settings: `chunk_size = 800` characters, `chunk_overlap = 80` characters.

---

## ⚙️ CLI Parameter Reference

```text
usage: ingest.py [-h] [--text TEXT] [--source SOURCE] [--batch BATCH [BATCH ...]]
                 [--chunk-size CHUNK_SIZE] [--chunk-overlap CHUNK_OVERLAP]
                 [--output OUTPUT] [--dry-run]
                 [input_path]

positional arguments:
  input_path             Path to file (.json, .csv, .txt, .md) or directory to ingest

options:
  -h, --help             Show this help message and exit
  --text TEXT            Direct text content to ingest into global semantic memory
  --source SOURCE        Custom source name when using --text (default: Direct_CLI_Input)
  --batch [FILES ...]    List of specific files to ingest sequentially
  --chunk-size INT       Character length per chunk for text documents (default: 800)
  --chunk-overlap INT    Character overlap between consecutive chunks (default: 80)
  --output OUTPUT        Path to target knowledge JSON (default: knowledge_base/knowledge.json)
  --dry-run              Parse files and print chunks without computing embeddings or writing to disk
```

---

## 🧠 Embedding & Deduplication Pipeline

During ingestion:
1. **Cleaning**: Whitespace and formatting noise are normalized.
2. **Chunking**: Text exceeding `chunk_size` is split at natural paragraph or sentence boundaries.
3. **Deduplication**: Content identical to an existing document text in `knowledge.json` is automatically skipped to prevent memory bloat.
4. **Native Embedding**: Every new chunk is fed to `NativeSLMEncoder`, generating a dense 768-dimensional unit vector using Kruuboo's 14-layer transformer decoder.
5. **Persistence**: The updated index and embeddings are saved atomically to `knowledge_base/knowledge.json`.
