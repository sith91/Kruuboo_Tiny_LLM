import json
from datasets import load_dataset
from pathlib import Path

def download_and_format():
    print("📥 Loading dataset from Hugging Face: aifeifei798/DPO_Pairs-Roleplay-NSFW...")
    dataset = load_dataset("aifeifei798/DPO_Pairs-Roleplay-NSFW", split="train")
    
    output_path = Path("training_data/roleplay_nsfw.txt")
    output_path.parent.mkdir(exist_ok=True)
    
    print(f"📦 Formatting {len(dataset)} examples into ChatML dialogues...")
    
    formatted_dialogues = []
    boilerplate = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n"
    
    for item in dataset:
        prompt = item.get("prompt", "")
        chosen = item.get("chosen", "")
        
        # Clean boilerplate header to save context space
        clean_prompt = prompt.replace(boilerplate, "").strip()
        
        # Structure into standard ChatML turns
        turn = (
            f"<|im_start|>user\n{clean_prompt}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{chosen.strip()}\n<|im_end|>\n"
        )
        formatted_dialogues.append(turn)
        
    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(formatted_dialogues))
        
    print(f"✅ Success! Saved {len(formatted_dialogues)} roleplay dialogues to {output_path}")

if __name__ == "__main__":
    download_and_format()
