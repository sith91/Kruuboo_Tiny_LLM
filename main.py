"""
Main CLI — Conversational Assistant Shell for Kruuboo Tiny LLM
===============================================================

Interactive terminal shell powered by Kruuboo-SLM-0.1 (105.4M Parameters)
with unified semantic RAG and sliding dialogue memory.
"""

import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag import RAGSystem

# Load config
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_MAX_SEQ_LEN = config["model"]["max_seq_len"]


class ConversationalCLI:
    """Streamlined interactive conversational shell."""

    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
        self.history = []

    def show_header(self):
        print("\n" + "=" * 70)
        print("🧠 KRUUBOO TINY LLM (SLM-0.1) — CONVERSATIONAL SHELL")
        print("   105.4M Transformer Decoder with Native 768d Semantic RAG")
        print("=" * 70)
        stats = self.rag.get_stats()
        print(f"📚 Global Memory: {stats['total_documents']:,} indexed knowledge documents")
        print("Type your message naturally or use commands like /help or /quit.\n")

    def show_help(self):
        print("\n" + "-" * 70)
        print("COMMANDS:")
        print("  /research [topic] - Scrape the web and ingest knowledge on-the-fly")
        print("  /add [text]       - Ingest custom text into global semantic memory")
        print("  /clear            - Clear dialogue context history")
        print("  /info             - Show system architecture & memory stats")
        print("  /help             - Show this help menu")
        print("  /quit             - Exit")
        print("-" * 70)

    def run(self):
        self.show_header()
        self.show_help()

        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()

                    if cmd in ("/quit", "/exit"):
                        print("👋 Goodbye!")
                        break

                    elif cmd == "/help":
                        self.show_help()
                        continue

                    elif cmd == "/clear":
                        self.history = []
                        print("🧹 Conversation memory cleared.")
                        continue

                    elif cmd == "/info":
                        stats = self.rag.get_stats()
                        print("\n📊 SYSTEM METRICS:")
                        print(f"  • Architecture: 105.4M Parameters (14 Layers, 12 Heads, RoPE, SwiGLU)")
                        print(f"  • Context Memory: {self.rag.max_seq_len} Tokens")
                        print(f"  • Semantic Memory: {stats['total_documents']} Documents")
                        continue

                    elif cmd == "/research" and len(parts) > 1:
                        topic = parts[1]
                        self.rag.research_and_learn(topic)
                        continue

                    elif cmd == "/add" and len(parts) > 1:
                        text = parts[1]
                        self.rag.add_text(text, source="CLI Input")
                        print("✅ Added to global semantic memory.")
                        continue

                # Process conversational message
                print("\n🤖 Kruuboo SLM: ", end="", flush=True)
                response = self.rag.ask(user_input, history=self.history)
                answer = response.get("answer", "")
                print(answer)

                # Update multi-turn history
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": answer})

                if response.get("knowledge_used", 0) > 0:
                    print(f"\n   [📚 Grounded on {response['knowledge_used']} retrieved semantic memory chunks]")

            except KeyboardInterrupt:
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    rag = RAGSystem(
        model_path="models/tiny_llm_quantized.onnx",
        tokenizer_path="tokenizer.json",
        max_seq_len=MODEL_MAX_SEQ_LEN
    )
    cli = ConversationalCLI(rag)
    cli.run()


if __name__ == "__main__":
    main()