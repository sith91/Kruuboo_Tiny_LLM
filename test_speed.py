
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.rag import RAGSystem

import yaml

with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)
MODEL_MAX_SEQ_LEN = cfg.get("model", {}).get("max_seq_len", 1024)

def test():
    print("Initializing...")
    rag = RAGSystem("models/tiny_llm_quantized.onnx", "tokenizer.json",
                    max_seq_len=MODEL_MAX_SEQ_LEN)
    print("Asking...")
    response = rag.ask("What is machine learning?")
    print("Response:", response.get("answer"))

if __name__ == "__main__":
    test()
