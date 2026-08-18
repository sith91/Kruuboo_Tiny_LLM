import sys
import os
import requests
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def download_and_format_conversations():
    print("=" * 60)
    print("DOWNLOADING & FORMATTING CONVERSATIONAL DIALOGUE DATASET")
    print("=" * 60)

    url = "https://raw.githubusercontent.com/icip-cas/ChatAlpaca/main/data/chatalpaca-10k.json"
    dest_path = Path("training_data/conversational_dialogues.txt")
    dest_path.parent.mkdir(exist_ok=True)

    print(f"📥 Downloading ChatAlpaca from {url} ...")
    r = requests.get(url)
    if r.status_code != 200:
        print(f"❌ Failed to download ChatAlpaca: {r.status_code}")
        return

    # The file contains one JSON object per line (JSON lines) or a single JSON list.
    # In the preview we saw they are separate JSON lines (each line is a conversation object).
    lines = r.text.splitlines()
    print(f"✅ Loaded {len(lines)} raw multi-turn conversation threads.")

    # Convert the first 500 multi-turn threads into ChatML format for training
    subset_size = min(500, len(lines))
    print(f" Formatting first {subset_size} threads into ChatML style...")
    
    chatml_dialogs = []
    for line in lines[:subset_size]:
        line = line.strip()
        if not line:
            continue
        try:
            conv_data = json.loads(line)
            turns = conv_data.get("conversations", [])
            
            # Format the multi-turn dialogue into ChatML blocks
            dialog_block = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for turn in turns:
                role = "user" if turn.get("from") == "human" else "assistant"
                val = turn.get("value", "").strip()
                dialog_block += f"<|im_start|>{role}\n{val}<|im_end|>\n"
            
            chatml_dialogs.append(dialog_block)
        except Exception as e:
            continue

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(chatml_dialogs))

    size_mb = dest_path.stat().st_size / (1024 * 1024)
    print(f"✅ Saved formatted multi-turn dialogues to {dest_path} ({size_mb:.2f} MB)")

if __name__ == "__main__":
    download_and_format_conversations()
