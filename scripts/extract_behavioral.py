import zipfile
from pathlib import Path
import re

def extract_and_format():
    zip_path = Path("/Users/sithija/Downloads/archive(1).zip")
    if not zip_path.exists():
        # Fallback to standard path if moved
        zip_path = Path("training_data/archive(1).zip")
        if not zip_path.exists():
            print("❌ archive(1).zip not found in Downloads or training_data.")
            return

    output_path = Path("training_data/behavioral_analysis.txt")
    output_path.parent.mkdir(exist_ok=True)
    
    print("📂 Opening ZIP archive and scanning response files...")
    
    formatted_data = []
    max_files = 1500  # Extract 1500 files for a high-quality subset
    count = 0
    
    with zipfile.ZipFile(zip_path, "r") as z:
        # Find all files inside the responses/ folder
        response_files = [f for f in z.namelist() if f.startswith("responses/") and f.endswith(".txt")]
        print(f"📋 Found {len(response_files)} files in total. Extracting {max_files}...")
        
        for file_name in response_files[:max_files]:
            try:
                # Read content
                text = z.read(file_name).decode("utf-8", errors="ignore").strip()
                if not text:
                    continue
                
                # Extract input traits if listed in the text
                # (usually at the start of Step 2 or top of file)
                traits_match = re.findall(r"-\s+\*\*([a-zA-Z\-/]+):\s*(\d+)\*\*", text)
                if traits_match:
                    traits_str = ", ".join([f"{k}: {v}" for k, v in traits_match])
                    user_prompt = f"Perform a research-based behavioral analysis and evaluation for a subject with the following traits: {traits_str}."
                else:
                    user_prompt = "Perform a detailed behavioral profile analysis and research-based validation of emotional and psychological traits."
                
                # Format into ChatML turns
                turn = (
                    f"<|im_start|>user\n{user_prompt}\n<|im_end|>\n"
                    f"<|im_start|>assistant\n{text}\n<|im_end|>\n"
                )
                formatted_data.append(turn)
                count += 1
                
                if count % 200 == 0:
                    print(f"⚡ Formatted {count}/{max_files} files...")
                    
            except Exception as e:
                continue

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("".join(formatted_data))
        
    print(f"✅ Success! Extracted and saved {count} behavioral analyses to {output_path}")

if __name__ == "__main__":
    extract_and_format()
