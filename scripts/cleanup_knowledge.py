import json
from pathlib import Path

docs_path = Path("knowledge_base/knowledge.json")
if not docs_path.exists():
    print("❌ knowledge.json not found.")
    exit(1)

with open(docs_path, "r", encoding="utf-8") as f:
    docs = json.load(f)

print(f"Total documents before cleanup: {len(docs)}")

# Filter out contaminated Wikipedia entries
filtered_docs = []
for doc in docs:
    source = doc.get("source", "")
    if any(pattern in source for pattern in ["Last Summer", "What to Do", "How Do You Know"]):
        print(f"🗑️  Removing: {source}")
        continue
    filtered_docs.append(doc)

print(f"Total documents after cleanup: {len(filtered_docs)}")

with open(docs_path, "w", encoding="utf-8") as f:
    json.dump(filtered_docs, f, indent=4)

print("✅ Cleaned knowledge.json successfully.")
