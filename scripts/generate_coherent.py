import argparse
import json
import sys
import traceback
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag import RAGSystem


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No prompt provided. Usage: python scripts/generate_coherent.py '<prompt>' [max_tokens]"}))
        return

    # Handle standard positional CLI or flags
    parser = argparse.ArgumentParser(description="Generate coherent text using Kruuboo-SLM Native ONNX Runtime")
    parser.add_argument("prompt", type=str, help="Input prompt or question")
    parser.add_argument("max_tokens", type=int, nargs="?", default=50, help="Maximum new tokens to generate")
    parser.add_argument("--model", type=str, default="models/tiny_llm_quantized.onnx", help="Path to ONNX model")
    parser.add_argument("--tokenizer", type=str, default="tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--raw", action="store_true", help="Generate directly without RAG retrieval synthesis")
    parser.add_argument("--temperature", type=float, default=0.65, help="Sampling temperature")
    parser.add_argument("--min-p", type=float, default=0.05, help="Min-p sampling threshold")

    args = parser.parse_args()

    try:
        rag = RAGSystem(
            model_path=args.model,
            tokenizer_path=args.tokenizer,
        )

        if args.raw:
            answer = rag.generate(
                args.prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                min_p=args.min_p
            )
            print(json.dumps({"answer": answer.strip()}))
        else:
            result = rag.ask(args.prompt)
            print(json.dumps({
                "answer": result.get("answer", "").strip(),
                "knowledge_used": result.get("knowledge_used", 0),
                "retrieved_context": result.get("retrieved_context", []),
            }))
    except Exception as e:
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))


if __name__ == "__main__":
    main()

