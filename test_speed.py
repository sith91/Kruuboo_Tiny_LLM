
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from src.rag import RAGSystem

# Must match config.yaml model.max_seq_len and src/train.py BLOCK_SIZE
MODEL_MAX_SEQ_LEN = 128

def test():
    print("Initializing...")
    rag = RAGSystem("models/tiny_llm_quantized.onnx", "tokenizer.json",
                    max_seq_len=MODEL_MAX_SEQ_LEN)
    print("Asking...")
    # Small max_tokens for speed
    response = rag.ask_with_domain("What is ML?", auto_detect=False)
    print("Raw response:", response)

if __name__ == "__main__":
    test()
