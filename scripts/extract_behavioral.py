import os
import re
import zipfile
import argparse
from pathlib import Path
from typing import Optional


def find_archive(custom_path: Optional[str] = None) -> Optional[Path]:
    """Locate the target ZIP archive across provided path or standard locations."""
    if custom_path:
        p = Path(custom_path).expanduser().resolve()
        if p.exists() and p.is_file():
            return p
        print(f"⚠️ Specified path not found: {custom_path}")

    # Standard candidate search locations
    candidates = [
        Path("training_data/archive(1).zip"),
        Path("training_data/archive.zip"),
        Path("data/archive(1).zip"),
        Path("data/archive.zip"),
        Path("archive(1).zip"),
        Path("archive.zip"),
        Path.home() / "Downloads" / "archive(1).zip",
        Path.home() / "Downloads" / "archive.zip",
    ]

    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand.resolve()

    return None


def extract_and_format(zip_path: Optional[str] = None, output_path: str = "training_data/behavioral_analysis.txt", max_files: int = 1500):
    archive_file = find_archive(zip_path)
    if not archive_file:
        print("❌ Could not find archive ZIP file.")
        print("💡 Place 'archive.zip' or 'archive(1).zip' in 'training_data/' or specify via --zip <path>")
        return

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print(f"📂 Opening ZIP archive: {archive_file}")
    print(f"🎯 Target output: {out_p}")

    formatted_data = []
    count = 0

    with zipfile.ZipFile(archive_file, "r") as z:
        # Find all text files inside the responses/ folder
        response_files = [f for f in z.namelist() if f.startswith("responses/") and f.endswith(".txt")]
        if not response_files:
            # Fallback: scan any .txt files in the archive
            response_files = [f for f in z.namelist() if f.endswith(".txt") and not f.startswith("__MACOSX")]

        print(f"📋 Found {len(response_files):,} candidate files in archive. Extracting up to {max_files:,}...")

        for file_name in response_files[:max_files]:
            try:
                # Read content
                text = z.read(file_name).decode("utf-8", errors="ignore").strip()
                if not text:
                    continue

                # Extract input traits if listed in the text
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

                if count % 250 == 0:
                    print(f"⚡ Formatted {count}/{min(max_files, len(response_files))} files...")

            except Exception:
                continue

    # Write to file
    with open(out_p, "w", encoding="utf-8") as f:
        f.write("".join(formatted_data))

    print(f"✅ Success! Extracted and saved {count:,} behavioral analyses to {out_p}")


def main():
    parser = argparse.ArgumentParser(description="Extract and format behavioral analysis dataset from ZIP archive")
    parser.add_argument("--zip", "-z", type=str, default=None, help="Path to archive.zip or archive(1).zip")
    parser.add_argument("--output", "-o", type=str, default="training_data/behavioral_analysis.txt", help="Output text file path")
    parser.add_argument("--max-files", "-m", type=int, default=1500, help="Maximum number of response files to extract")
    args = parser.parse_args()

    extract_and_format(zip_path=args.zip, output_path=args.output, max_files=args.max_files)


if __name__ == "__main__":
    main()
