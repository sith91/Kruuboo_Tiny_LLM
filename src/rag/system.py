import json
import time
import logging
import re
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional

import yaml
import onnxruntime as ort

from .document import Document
from .domain import Domain, get_default_domains, load_domains_from_yaml
from .greetings import (
    GREETINGS_MAP,
    extract_user_name_from_history,
    check_naming_request,
    check_bot_name_query,
    check_user_name_query
)
from .search import hybrid_search

from ..tokenizer import TinyTokenizer
from ..inference import InferenceEngine
from ..web_researcher import WebResearcher

# Load global configuration
try:
    with open("config.yaml", "r") as f:
        _config = yaml.safe_load(f)
except Exception:
    _config = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGSystem:
    """Complete RAG system with FAISS and Tiny LLM"""

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        knowledge_base_path: str = "knowledge_base",
        max_seq_len: int = None,
    ):
        # Load LLM
        logger.info(f"Loading ONNX model from {model_path}")
        try:
            self.session = ort.InferenceSession(model_path)
            # Find max sequence length from input shape
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            
            # Read shape info from ONNX graph to make max_seq_len single source of truth
            shape = input_meta.shape
            if max_seq_len is not None:
                self.max_seq_len = max_seq_len
                logger.info(f"max_seq_len set to {max_seq_len} (source: argument)")
            elif len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 0:
                self.max_seq_len = shape[1]
                logger.info(f"max_seq_len set to {self.max_seq_len} (source: ONNX model)")
            else:
                self.max_seq_len = 128
                logger.warning(f"Could not read sequence length from ONNX graph. Falling back to default: {self.max_seq_len}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}. Running in text-only mode.")
            self.session = None
            self.input_name = None
            self.max_seq_len = 128

        self.tokenizer = TinyTokenizer.load(tokenizer_path)

        # Build the InferenceEngine
        self._engine = InferenceEngine(
            session    = self.session,
            tokenizer  = self.tokenizer,
            input_name = self.input_name,
            max_seq_len= self.max_seq_len,
        )

        # Initialize Global Knowledge Base
        self.embedding_model_name = embedding_model
        self._encoder = None
        self.kb_path = knowledge_base_path
        self.documents = []
        self._load_knowledge_base()
        
        # Online research support
        self.researcher = WebResearcher()
        self.research_enabled = True
        
        # Response mode setting (toy_llm, smart_rag, hf_llm, hybrid)
        self.response_mode = "hybrid"
        
        # Pre-import in main thread to avoid macOS thread crashes during requests
        self.use_pretrained = False
        self._pretrained_generator = None
            
        # Domain Management
        self.assistant_name = "Assistant"
        self.enable_domains = True
        self.current_domain_id = "general"
        self.domains_registry = get_default_domains()
        self._load_domains_config()

        logger.info("RAG System initialized")
        
    @property
    def encoder(self):
        if self._encoder is None:
            logger.info(f"Loading embedding model: {self.embedding_model_name}...")
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(self.embedding_model_name)
        return self._encoder

    def _load_domains_config(self):
        """Load domain configurations dynamically from domains.yaml"""
        self.domains_registry = load_domains_from_yaml(self.domains_registry)
    
    def _load_knowledge_base(self):
        """Load existing knowledge base from disk"""
        docs_path = Path(f"{self.kb_path}/knowledge.json")
        if docs_path.exists():
            try:
                with open(docs_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.documents = []
                for item in data:
                    self.documents.append(Document(
                        text=item.get("text", ""),
                        source=item.get("source", "manual"),
                        timestamp=item.get("timestamp", time.time()),
                        metadata=item.get("metadata", {})
                    ))
                logger.info(f"Loaded knowledge base with {len(self.documents)} documents")
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
                self.documents = []
        else:
            self.documents = []
            
    def add_text(self, text: str, source: str = "manual", metadata: Dict = None):
        """Add text to knowledge base"""
        chunk_size = _config.get("rag", {}).get("chunk_size", 500)
        overlap = _config.get("rag", {}).get("chunk_overlap", 50)
        chunks = []
        
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        # Add each chunk
        for i, chunk in enumerate(chunks):
            try:
                emb = self.encoder.encode(chunk).tolist()
            except Exception as e:
                logger.error(f"Failed to generate embedding for chunk: {e}")
                emb = None
                
            doc = Document(
                text=chunk,
                source=source,
                metadata={
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'embedding': emb,
                    **(metadata or {})
                }
            )
            self.documents.append(doc)
        
        logger.info(f"Added {len(chunks)} chunks from {source}")
        self.save()
        
    def add_from_file(self, file_path: str, source: str = None):
        """Add text from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        source = source or file_path
        self.add_text(text, source=source)
        
    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Search for relevant documents using hybrid search (Keyword + Vector similarity)"""
        results, needs_save = hybrid_search(query, self.documents, self.encoder, k)
        if needs_save:
            self.save()
        return results

    def generate(
        self,
        prompt: str,
        max_tokens: int = 120,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.90,
        repetition_penalty: float = 1.15,
        stream: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Generate text using the InferenceEngine or extractive match."""
        if self.session is None:
            return "[No model loaded. Run src/train.py then scripts/export_onnx.py]"

        mode = getattr(self, "response_mode", "smart_rag")

        # ── Mode 1: Smart RAG (Extractive Sentence Matcher) ───────────────────
        if mode == "smart_rag":
            try:
                question = ""
                context_str = ""
                if "<|im_start|>user" in prompt:
                    parts = prompt.split("<|im_start|>user\n")
                    user_content = parts[-1].split("<|im_end|>")[0].strip()
                    if "Context: " in user_content:
                        c_parts = user_content.split("Context: ")
                        if len(c_parts) > 1:
                            q_parts = c_parts[1].split("\n\nQuestion: ")
                            context_str = q_parts[0].strip()
                            if len(q_parts) > 1:
                                question = q_parts[1].strip()
                    else:
                        question = user_content
                elif "Question: " in prompt:
                    parts = prompt.split("Question: ")
                    context_str = parts[0].replace("Context:", "").strip()
                    question = parts[1].split("\n\nAnswer:")[0].strip()
                
                if question and context_str:
                    import re
                    raw_sents = re.split(r'(?<=[.!?])\s+', context_str)
                    sentences = [s.strip() for s in raw_sents if s.strip()]
                    
                    if sentences:
                        question_words = set(re.findall(r'\w+', question.lower()))
                        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'to', 'with', 'about', 'how', 'why', 'where', 'when', 'who', 'does', 'do', 'did', 'can', 'could', 'would', 'should'}
                        keywords = question_words - stopwords
                        if not keywords:
                            keywords = question_words
                            
                        best_sent = sentences[0]
                        max_score = -1
                        for sent in sentences:
                            sent_words = set(re.findall(r'\w+', sent.lower()))
                            score = len(keywords & sent_words)
                            if score > max_score:
                                max_score = score
                                best_sent = sent
                        
                        answer = best_sent
                        if on_token:
                            on_token(answer)
                        elif stream:
                            print(answer)
                        return answer
            except Exception as e:
                logger.error(f"Extractive RAG failed: {e}")

        # ── Mode 2: Pre-trained HuggingFace LLM (Removed to have only Tiny LLM) ────────
        elif mode == "hf_llm":
            return "[HuggingFace engine has been removed. Switch to toy_llm or hybrid.]"

        # Build a streaming callback that prints to stdout (if requested)
        def _print_token(t: str):
            print(t, end="", flush=True)

        callback = None
        if on_token is not None:
            callback = on_token
        elif stream:
            callback = _print_token

        answer = self._engine.generate(
            prompt,
            max_new_tokens    = max_tokens,
            temperature       = temperature,
            top_k             = top_k,
            top_p             = top_p,
            repetition_penalty= repetition_penalty,
            on_token          = callback,
        )

        if stream or on_token:
            print()

        return answer

    def ask(self, question: str, use_rag: bool = True, k: int = 3, max_tokens: int = 120, history: List[Dict] = None) -> str:
        """Ask a question with optional RAG context and conversational history."""
        clean_q = question.strip().lower().strip("?.!,")
        
        # Check greetings map first
        if clean_q in GREETINGS_MAP:
            return GREETINGS_MAP[clean_q]

        # Conversational memory name extraction
        if check_user_name_query(clean_q):
            name = extract_user_name_from_history(history)
            if name != "my friend":
                return f"Your name is {name}."
            return "I don't know your name yet! You can tell me by saying 'My name is [your name]'."

        system_msg = "You are a helpful assistant."
        if use_rag:
            retrieved = self.search(question, k=k)
            if retrieved:
                context_parts = [doc.text.strip() for doc, _ in retrieved]
                context = "\n".join(context_parts)
                user_msg = f"Context: {context}\n\nQuestion: {question}"
                logger.info(f"Using RAG with {len(retrieved)} documents")
            else:
                user_msg = question
                logger.warning("No relevant documents found, using base model")
        else:
            user_msg = question

        # Format prompt based on the active engine mode
        engine_mode = getattr(self, "response_mode", "smart_rag")
        if engine_mode == "hf_llm" or getattr(self, "use_pretrained", False):
            history_block = ""
            if history:
                for turn in history[-6:]:
                    role = turn.get("role")
                    content = turn.get("content", "").strip()
                    if role == "user":
                        history_block += f"User: {content}\n"
                    elif role == "assistant":
                        history_block += f"Assistant: {content}\n"
            prompt = (
                f"System: {system_msg}\n\n"
                f"{history_block}"
                f"User: {user_msg}\n"
                f"Assistant: "
            )
        else:
            history_block = ""
            if history:
                for turn in history[-6:]:
                    role = turn.get("role")
                    content = turn.get("content", "").strip()
                    if role == "user":
                        history_block += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        history_block += f"<|im_start|>assistant\n{content}<|im_end|>\n"
            prompt = (
                f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                f"{history_block}"
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        return self.generate(prompt, max_tokens=max_tokens)

    def research_and_learn(self, query: str) -> bool:
        """Search the web for a query and add it to the current knowledge base"""
        if not self.research_enabled:
            return False
            
        print(f"🌐 Researching '{query}' online...")
        findings = self.researcher.research_topic(query)
        
        if not findings:
            print("❌ No information found online.")
            return False
            
        for finding in findings:
            self.add_text(finding['text'], source=finding['source'])
        
        print(f"✅ Learned {len(findings)} new items from the web!")
        return True

    def save(self):
        """Save knowledge base to disk"""
        Path(self.kb_path).mkdir(parents=True, exist_ok=True)
        docs_path = Path(f"{self.kb_path}/knowledge.json")
        try:
            serialized = [
                {
                    "text": doc.text,
                    "source": doc.source,
                    "timestamp": doc.timestamp,
                    "metadata": doc.metadata
                }
                for doc in self.documents
            ]
            with open(docs_path, 'w', encoding='utf-8') as f:
                json.dump(serialized, f, indent=2)
            logger.info(f"Saved knowledge base to {docs_path}")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            'total_documents': len(self.documents),
            'knowledge_base_path': self.kb_path
        }

    def get_current_domain(self) -> Optional[Domain]:
        dom = self.domains_registry.get(self.current_domain_id)
        if dom:
            count = sum(1 for doc in self.documents if doc.metadata.get('domain') == dom.id)
            if dom.id == "general":
                count = sum(1 for doc in self.documents if not doc.metadata.get('domain') or doc.metadata.get('domain') == 'general')
            
            # Setup dynamically returned get_info function
            def get_info_with_count():
                info = {
                    'id': dom.id,
                    'name': dom.name,
                    'description': dom.description,
                    'response_style': dom.response_style,
                    'knowledge_count': count,
                    'example_questions': dom.example_questions
                }
                return info
            dom.get_info = get_info_with_count
        return dom

    def get_available_domains(self) -> List[Dict]:
        res = []
        for dom_id, dom in self.domains_registry.items():
            count = sum(1 for doc in self.documents if doc.metadata.get('domain') == dom_id)
            if dom_id == "general":
                count = sum(1 for doc in self.documents if not doc.metadata.get('domain') or doc.metadata.get('domain') == 'general')
            info = {
                'id': dom.id,
                'name': dom.name,
                'description': dom.description,
                'response_style': dom.response_style,
                'knowledge_count': count,
                'example_questions': dom.example_questions
            }
            res.append(info)
        return res

    def switch_domain(self, domain_id: str) -> bool:
        if domain_id in self.domains_registry:
            self.current_domain_id = domain_id
            return True
        return False

    def add_domain_knowledge(self, domain_id: str, text: str, source: str) -> bool:
        if domain_id in self.domains_registry:
            self.add_text(text, source=source, metadata={'domain': domain_id})
            return True
        return False

    def ask_with_domain(self, question: str, auto_detect: bool = True, history: List[Dict] = None) -> Dict:
        clean_q = question.strip().lower().strip("?.!,")
        
        # Check greetings map first
        if clean_q in GREETINGS_MAP:
            return {
                'answer': GREETINGS_MAP[clean_q],
                'domain_used': self.current_domain_id,
                'domain_name': self.domains_registry[self.current_domain_id].name,
                'knowledge_used': 0,
                'temperature_used': 0.7
            }

        # Check if user is naming the assistant
        bot_name = check_naming_request(clean_q)
        if bot_name:
            self.assistant_name = bot_name
            user_name = extract_user_name_from_history(history)
            user_suffix = f", {user_name}" if user_name != "my friend" else ""
            return {
                'answer': f"Great! From now on, you can call me {bot_name}{user_suffix}. How can I assist you today?",
                'domain_used': self.current_domain_id,
                'domain_name': self.domains_registry[self.current_domain_id].name,
                'knowledge_used': 0,
                'temperature_used': 0.7
            }

        # Check if user is asking for assistant's name
        if check_bot_name_query(clean_q):
            ast_name = getattr(self, "assistant_name", "Assistant")
            return {
                'answer': f"My name is {ast_name}. How can I assist you today?",
                'domain_used': self.current_domain_id,
                'domain_name': self.domains_registry[self.current_domain_id].name,
                'knowledge_used': 0,
                'temperature_used': 0.7
            }

        # Check if user is asking for their own name
        if check_user_name_query(clean_q):
            name = extract_user_name_from_history(history)
            if name != "my friend":
                answer = f"Your name is {name}."
            else:
                answer = "I don't know your name yet! You can tell me by saying 'My name is [your name]'."
            return {
                'answer': answer,
                'domain_used': self.current_domain_id,
                'domain_name': self.domains_registry[self.current_domain_id].name,
                'knowledge_used': 0,
                'temperature_used': 0.7
            }

        # Auto detect domain from keywords and update state
        if auto_detect:
            best_domain_id = None
            max_matches = 0
            best_priority = 9999
            
            from .search import STOPWORDS
            for dom_id, dom in self.domains_registry.items():
                if dom_id == "general":
                    continue
                
                matches = 0
                for kw in dom.keywords:
                    if kw.lower() in STOPWORDS:
                        continue
                    # Match whole words or phrases with boundary check
                    if re.search(r'\b' + re.escape(kw.lower()) + r'\b', clean_q):
                        matches += 1
                        
                if matches > 0:
                    if matches > max_matches:
                        max_matches = matches
                        best_domain_id = dom_id
                        best_priority = getattr(dom, 'priority', 99)
                    elif matches == max_matches:
                        curr_priority = getattr(dom, 'priority', 99)
                        if curr_priority < best_priority:
                            best_domain_id = dom_id
                            best_priority = curr_priority
            
            if best_domain_id:
                self.current_domain_id = best_domain_id

        dom = self.domains_registry[self.current_domain_id]
        
        # Search & filter knowledge by domain
        k = 3
        retrieved_with_scores = []
        if dom.id == "general":
            retrieved_with_scores = self.search(question, k=k)
        else:
            all_retrieved = self.search(question, k=15)
            retrieved_with_scores = [item for item in all_retrieved if item[0].metadata.get('domain') == dom.id][:k]
            
        retrieved = [doc for doc, _ in retrieved_with_scores]

        system_msg = f"You are a helpful assistant. Response style: {dom.response_style}."
        if getattr(dom, 'require_disclaimer', False) or dom.id == 'medical':
            system_msg += " Disclaimer: This is for informational purposes only. Consult a professional/doctor for medical advice."

        # Update and inject emotional mood instruction
        try:
            from src.feeling import update_mood, get_mood_instruction
            self.current_mood = update_mood(question)
            mood_instruction = get_mood_instruction(self.current_mood)
            system_msg += f" {mood_instruction}"
        except Exception as e:
            print(f"⚠️ Error updating mood: {e}")
            self.current_mood = None

        # Format retrieval context only if we have a strong match (score >= 0.60)
        has_strong_context = retrieved_with_scores and retrieved_with_scores[0][1] >= 0.60
        if retrieved and has_strong_context:
            context = "\n".join([doc.text for doc in retrieved])
            user_msg = f"Context: {context}\n\nQuestion: {question}"
        else:
            user_msg = question

        # Choose temperature
        temp = 0.7
        if dom.id in ['medical', 'legal']:
            temp = 0.2

        # Format prompt based on response engine mode
        engine_mode = getattr(self, "response_mode", "hybrid")
        
        # Decide actual mode for hybrid engine
        actual_mode = engine_mode
        if engine_mode == "hybrid":
            # If we have a strong database match, use extractive RAG. Otherwise generate.
            if retrieved_with_scores and retrieved_with_scores[0][1] >= 0.70:
                actual_mode = "smart_rag"
            else:
                # Prefer our own fine-tuned custom engine (toy_llm) for generation
                actual_mode = "toy_llm"
        
        if actual_mode == "smart_rag" and retrieved:
            answer = retrieved[0].text
        else:
            if actual_mode == "hf_llm" or getattr(self, "use_pretrained", False):
                history_block = ""
                if history:
                    for turn in history[-6:]:
                        role = turn.get("role")
                        content = turn.get("content", "").strip()
                        if role == "user":
                            history_block += f"User: {content}\n"
                        elif role == "assistant":
                            history_block += f"Assistant: {content}\n"
                prompt = (
                    f"System: {system_msg}\n\n"
                    f"{history_block}"
                    f"User: {user_msg}\n"
                    f"Assistant: "
                )
            else:
                history_block = ""
                if history:
                    for turn in history[-6:]:
                        role = turn.get("role")
                        content = turn.get("content", "").strip()
                        if role == "user":
                            history_block += f"<|im_start|>user\n{content}<|im_end|>\n"
                        elif role == "assistant":
                            history_block += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                prompt = (
                    f"<|im_start|>system\n{system_msg}<|im_end|>\n"
                    f"{history_block}"
                    f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            answer = self.generate(prompt, temperature=temp, stream=False)

        return {
            'answer': answer,
            'domain_used': dom.id,
            'domain_name': dom.name,
            'knowledge_used': len(retrieved),
            'temperature_used': temp,
            'mood': self.current_mood.to_dict() if getattr(self, "current_mood", None) else None
        }
