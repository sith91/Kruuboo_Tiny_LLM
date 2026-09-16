"""Main entry point with domain support"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag import RAGSystem
from src.tokenizer import TinyTokenizer

# Must match config.yaml model.max_seq_len and src/train.py BLOCK_SIZE
MODEL_MAX_SEQ_LEN = 128


class DomainAwareUI:
    """Interactive UI with domain awareness"""
    
    def __init__(self, rag_system: RAGSystem):
        self.rag = rag_system
        self.auto_detect = True
        self.history = []
    
    def show_header(self):
        """Show application header"""
        print("\n" + "=" * 70)
        print("🎯 TINY LLM WITH DOMAIN-SPECIFIC RAG")
        print("=" * 70)
        
        # Show current domain
        current = self.rag.get_current_domain()
        if current:
            print(f"\n📌 CURRENT MODE: {current.config.name}")
            print(f"   {current.config.description[:60]}...")
        
        # Show available domains
        print("\n📋 AVAILABLE DOMAINS:")
        domains = self.rag.get_available_domains()
        for domain in domains:
            print(f"   • {domain['name']} - {domain['description'][:40]}...")
    
    def show_help(self):
        """Show help menu"""
        print("\n" + "-" * 70)
        print("COMMANDS:")
        print("  /domains              - Show all available domains")
        print("  /mode [domain_id]     - Switch domain (e.g., /mode teaching_abc)")
        print("  /current              - Show current domain info")
        print("  /auto                 - Toggle auto-domain detection")
        print("  /add [text]           - Add knowledge to current domain")
        print("  /research [topic]     - Search the web and learn about a topic")
        print("  /info                 - Show domain statistics")
        print("  /engine [mode]        - Switch generation engine (smart_rag | toy_llm | hybrid | rstar)")
        print("  /help                 - Show this help")
        print("  /quit                 - Exit")
        print("-" * 70)
    
    def run(self):
        """Run the interactive session"""
        self.show_header()
        self.show_help()
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()
                    
                    if cmd == '/quit' or cmd == '/exit':
                        print("👋 Goodbye!")
                        break
                    
                    elif cmd == '/help':
                        self.show_help()
                    
                    elif cmd == '/domains':
                        print("\n📋 AVAILABLE DOMAINS:")
                        for domain in self.rag.get_available_domains():
                            print(f"\n  🔹 {domain['name']} (id: {domain['id']})")
                            print(f"     {domain['description']}")
                            print(f"     Style: {domain['response_style']}")
                            print(f"     Knowledge: {domain['knowledge_count']} items")
                            if domain.get('example_questions'):
                                print(f"     Example: {domain['example_questions'][0]}")
                    
                    elif cmd == '/current':
                        current = self.rag.get_current_domain()
                        if current:
                            info = current.get_info()
                            print(f"\n📌 CURRENT DOMAIN: {info['name']}")
                            print(f"   Description: {info['description']}")
                            print(f"   Response style: {info['response_style']}")
                            print(f"   Knowledge items: {info['knowledge_count']}")
                    
                    elif cmd == '/mode' and len(parts) > 1:
                        domain_id = parts[1]
                        if self.rag.switch_domain(domain_id):
                            current = self.rag.get_current_domain()
                            print(f"\n✅ Switched to: {current.config.name}")
                            print(f"   {current.config.description}")
                        else:
                            print(f"❌ Domain '{domain_id}' not found")
                            print("Available domains:", 
                                  [d['id'] for d in self.rag.get_available_domains()])
                    
                    elif cmd == '/auto':
                        self.auto_detect = not self.auto_detect
                        print(f"\n🔄 Auto-domain detection: {'ON' if self.auto_detect else 'OFF'}")
                    
                    elif cmd == '/info':
                        stats = self.rag.get_stats() if hasattr(self.rag, 'get_stats') else {}
                        print("\n📊 SYSTEM STATISTICS:")
                        print(f"   Total documents: {stats.get('total_documents', 0)}")
                        print(f"   Active domain: {self.rag.current_domain_id if hasattr(self.rag, 'current_domain_id') else 'N/A'}")
                        print(f"   Auto-detect: {'ON' if self.auto_detect else 'OFF'}")
                    
                    elif cmd == '/add' and len(parts) > 1:
                        text = parts[1]
                        source = input("   Source name: ").strip()
                        if self.rag.add_domain_knowledge(self.rag.current_domain_id, text, source):
                            print("✅ Knowledge added successfully!")
                        else:
                            print("❌ Failed to add knowledge")
                    
                    elif cmd == '/research' and len(parts) > 1:
                        topic = parts[1]
                        self.rag.research_and_learn(topic)
                    
                    elif cmd == '/engine' and len(parts) > 1:
                        mode = parts[1].strip().lower()
                        if mode in ["smart_rag", "toy_llm", "hybrid", "rstar"]:
                            self.rag.response_mode = mode
                            self.rag.use_pretrained = False
                            print(f"\n✅ Switched generation engine to: {mode} (Mutual Reasoning)")
                        else:
                            print("\n❌ Invalid mode. Choose from: smart_rag, toy_llm, hybrid, rstar")
                    
                    else:
                        print(f"Unknown command: {cmd}. Type /help for available commands.")
                    
                    continue
                
                # Process regular question
                print("🤖 Thinking...")
                
                import os
                LOCK_PATH = "/tmp/kruboo_chat_active.lock"
                
                def lock_training():
                    try:
                        with open(LOCK_PATH, "w") as f:
                            f.write("active")
                    except:
                        pass
                
                def unlock_training():
                    try:
                        if os.path.exists(LOCK_PATH):
                            os.remove(LOCK_PATH)
                    except:
                        pass

                lock_training()
                try:
                    if hasattr(self.rag, 'ask_with_domain'):
                        response = self.rag.ask_with_domain(
                            user_input, 
                            auto_detect=self.auto_detect,
                            history=self.history
                        )
                        
                        answer = response.get('answer', '')
                        if answer:
                            print(f"\n🤖 {answer}")
                            self.history.append({"role": "user", "content": user_input})
                            self.history.append({"role": "assistant", "content": answer})
                        
                        if response.get('knowledge_used', 0) > 0:
                            print(f"📚 Used {response['knowledge_used']} knowledge sources")
                    else:
                        # Fallback to regular ask
                        answer = self.rag.ask(user_input, history=self.history)
                        print(f"\n🤖 {answer}")
                        self.history.append({"role": "user", "content": user_input})
                        self.history.append({"role": "assistant", "content": answer})
                finally:
                    unlock_training()
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    """Main entry point"""
    print("=" * 60)
    print("TINY LLM WITH DYNAMIC DOMAIN SPECIALIZATION")
    print("=" * 60)
    
    # Check if model exists
    model_path = "models/tiny_llm_quantized.onnx"
    tokenizer_path = "tokenizer.json"
    
    if not Path(model_path).exists():
        print("\n⚠️  Model not found! Creating a test instance...")
        print("   For full functionality, train the model first:")
        print("   1. python src/train.py")
        print("   2. python scripts/export_onnx.py")
        
        # Create a simple fallback
        from src.tokenizer import TinyTokenizer
        tokenizer = TinyTokenizer()
        tokenizer.save(tokenizer_path)

        # Create RAG system without model (fallback mode)
        rag = RAGSystem(model_path, tokenizer_path, max_seq_len=MODEL_MAX_SEQ_LEN)
        rag.enable_domains = True
        
        # Manually set up domains for demo
        from domains.domain_registry import init_domains
        rag.domain_registry = init_domains("knowledge_bases")
        rag.current_domain_id = "general"
        rag.auto_detect_domains = True
        rag.enable_domains = True
        
    else:
        # Normal initialization
        rag = RAGSystem(model_path, tokenizer_path, max_seq_len=MODEL_MAX_SEQ_LEN)
        rag.enable_domains = True
    
    # Run UI
    ui = DomainAwareUI(rag)
    ui.run()


if __name__ == "__main__":
    main()