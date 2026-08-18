import os
import urllib.request
from pathlib import Path

def download_datasets():
    training_data_dir = Path("training_data")
    training_data_dir.mkdir(exist_ok=True)
    
    # 1. WikiText-2 download
    wikitext_url = "https://raw.githubusercontent.com/pytorch/examples/master/word_language_model/data/wikitext-2/train.txt"
    wikitext_path = training_data_dir / "wikitext2.txt"
    print(f"📥 Downloading WikiText-2 from {wikitext_url} ...")
    try:
        urllib.request.urlretrieve(wikitext_url, wikitext_path)
        size_mb = wikitext_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved {wikitext_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed to download WikiText-2: {e}")

    # 2. TinyStories stream download (10 MB subset)
    tinystories_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories-train.txt"
    tinystories_path = training_data_dir / "tinystories.txt"
    print(f"📥 Streaming TinyStories subset (10 MB limit) from {tinystories_url} ...")
    try:
        limit_bytes = 10 * 1024 * 1024  # 10 MB
        downloaded = 0
        
        req = urllib.request.Request(
            tinystories_url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        
        with urllib.request.urlopen(req) as response:
            with open(tinystories_path, 'wb') as f:
                while downloaded < limit_bytes:
                    chunk = response.read(1024 * 64)  # 64 KB chunks
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
        size_mb = tinystories_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved TinyStories subset to {tinystories_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed to download TinyStories: {e}")

    # 3. Alpaca ChatML download and format (approx 4,000 instruction subset)
    alpaca_url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    alpaca_path = training_data_dir / "alpaca_chatml.txt"
    print(f"📥 Downloading Alpaca from {alpaca_url} ...")
    try:
        import json
        req = urllib.request.Request(
            alpaca_url, 
            headers={'User-Agent': 'Mozilla/5.0'}
        )
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode('utf-8'))
            
        # Format the first 4,000 dialogue instructions in ChatML style
        chatml_lines = []
        for item in data[:4000]:
            inst = item.get("instruction", "")
            inp = item.get("input", "")
            out = item.get("output", "")
            
            user_msg = f"{inst}\n{inp}".strip() if inp else inst
            chatml_block = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{out}<|im_end|>\n"
            )
            chatml_lines.append(chatml_block)
            
        with open(alpaca_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chatml_lines))
            
        size_mb = alpaca_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved Alpaca ChatML subset to {alpaca_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed to download/format Alpaca: {e}")

    # 4. Medical datasets download
    print("📥 Downloading disease symptom descriptions and precautions...")
    try:
        desc_url = "https://raw.githubusercontent.com/bharatdappili/chatbot/master/symptom_Description.csv"
        prec_url = "https://raw.githubusercontent.com/bharatdappili/chatbot/master/symptom_precaution.csv"
        
        desc_path = training_data_dir / "symptom_Description.csv"
        prec_path = training_data_dir / "symptom_precaution.csv"
        
        urllib.request.urlretrieve(desc_url, desc_path)
        urllib.request.urlretrieve(prec_url, prec_path)
        
        print(f"✅ Saved disease descriptions to {desc_path}")
        print(f"✅ Saved disease precautions to {prec_path}")
    except Exception as e:
        print(f"❌ Failed to download medical datasets: {e}")

    # 5. ChatAlpaca conversational dataset download
    print("📥 Downloading ChatAlpaca multi-turn conversational dataset...")
    try:
        import json
        chatalpaca_url = "https://raw.githubusercontent.com/icip-cas/ChatAlpaca/main/data/chatalpaca-10k.json"
        chatalpaca_path = training_data_dir / "conversational_dialogues.txt"
        
        req = urllib.request.Request(chatalpaca_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            lines = response.read().decode('utf-8').splitlines()
            
        chatml_dialogs = []
        subset_size = min(500, len(lines))
        for line in lines[:subset_size]:
            line = line.strip()
            if not line:
                continue
            conv_data = json.loads(line)
            turns = conv_data.get("conversations", [])
            
            dialog_block = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            for turn in turns:
                role = "user" if turn.get("from") == "human" else "assistant"
                val = turn.get("value", "").strip()
                dialog_block += f"<|im_start|>{role}\n{val}<|im_end|>\n"
            chatml_dialogs.append(dialog_block)
            
        with open(chatalpaca_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chatml_dialogs))
            
        size_mb = chatalpaca_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved formatted multi-turn dialogues to {chatalpaca_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed to download/format ChatAlpaca: {e}")

    # 6. US Army Survival Manual (FM 21-76) download
    print("📥 Downloading US Army Survival Manual (FM 21-76)...")
    try:
        fm_url = "https://archive.org/download/FM2176NEWU.S.ArmySurvivalHandbook/FM%2021-76%20NEW%20U.S.%20Army%20Survival%20Handbook%20_djvu.txt"
        fm_path = training_data_dir / "US_Army_Survival_Manual_FM21-76.txt"
        
        req = urllib.request.Request(fm_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            text = response.read().decode('utf-8', errors='ignore')
            
        with open(fm_path, "w", encoding="utf-8") as f:
            f.write(text)
            
        size_mb = fm_path.stat().st_size / (1024 * 1024)
        print(f"✅ Saved US Army Survival Manual to {fm_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ Failed to download US Army Survival Manual: {e}")

if __name__ == "__main__":
    download_datasets()
