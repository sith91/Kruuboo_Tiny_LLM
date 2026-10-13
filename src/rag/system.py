import json
import time
import logging
import re
import ast
import operator
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional

import yaml
import onnxruntime as ort

from .document import Document
from .greetings import (
    GREETINGS_MAP,
    extract_user_name_from_history,
    check_user_intro,
    check_naming_request,
    check_bot_name_query,
    check_user_name_query
)
from .search import hybrid_search
from .native_encoder import NativeSLMEncoder

SAFE_MATH_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_math_node(node):
    if isinstance(node, ast.Num):
        return node.n
    elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    elif isinstance(node, ast.BinOp) and type(node.op) in SAFE_MATH_OPS:
        left = _eval_math_node(node.left)
        right = _eval_math_node(node.right)
        return SAFE_MATH_OPS[type(node.op)](left, right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in SAFE_MATH_OPS:
        operand = _eval_math_node(node.operand)
        return SAFE_MATH_OPS[type(node.op)](operand)
    raise ValueError("Unsafe or non-arithmetic node")


def solve_arithmetic_query(text: str) -> Optional[str]:
    """Safely solve direct arithmetic questions (e.g. '1+1', 'what is 25 * 4', 'calculate 100/5')."""
    clean = text.lower().strip("?.! \t\n")
    prefixes = [
        r"^i\s+need\s+to\s+know\s+what\s+is\s+",
        r"^can\s+you\s+tell\s+me\s+what\s+is\s+",
        r"^tell\s+me\s+what\s+is\s+",
        r"^what\s+is\s+",
        r"^whats\s+",
        r"^calculate\s+",
        r"^solve\s+",
        r"^how\s+much\s+is\s+",
        r"^evaluate\s+",
        r"^what\s+does\s+",
        r"^equal\s+"
    ]
    for pat in prefixes:
        clean = re.sub(pat, "", clean).strip()

    clean = clean.replace("equals", "").replace("equal", "").strip("?.! \t\n")
    expr = clean.replace("^", "**").replace("x", "*").replace("×", "*").replace("÷", "/")

    if re.fullmatch(r"[\d\s\+\-\*\/\%\(\)\.]+", expr) and any(c in expr for c in "+-*/%"):
        try:
            tree = ast.parse(expr, mode="eval")
            val = _eval_math_node(tree.body)
            if isinstance(val, float) and val.is_integer():
                val = int(val)
            return f"{clean} is {val}."
        except Exception:
            return None
    return None

from ..tokenizer import TinyTokenizer
from ..inference import InferenceEngine
from ..web_researcher import WebResearcher
from ..rstar import RStarEngine

# Load global configuration
try:
    with open("config.yaml", "r") as f:
        _config = yaml.safe_load(f)
except Exception:
    _config = {}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGSystem:
    """
    Unified Conversational Cognitive Pipeline for Kruuboo Tiny LLM.
    Combines native 768d semantic memory retrieval, sliding dialogue context,
    rStar self-play reasoning, and ChatML neural autoregressive generation into one unit.
    """

    def __init__(
        self,
        model_path: str = "models/tiny_llm_quantized.onnx",
        tokenizer_path: str = "tokenizer.json",
        embedding_model: str = "native",
        knowledge_base_path: str = "knowledge_base",
        max_seq_len: int = None,
    ):
        # 1. Load ONNX LLM
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        logger.info(f"Loading ONNX model from {model_path}")
        try:
            self.session = ort.InferenceSession(model_path)
            input_meta = self.session.get_inputs()[0]
            self.input_name = input_meta.name
            shape = input_meta.shape
            if max_seq_len is not None:
                self.max_seq_len = max_seq_len
            elif len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 0:
                self.max_seq_len = shape[1]
            else:
                self.max_seq_len = 1024
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}. Running in text-only mode.")
            self.session = None
            self.input_name = None
            self.max_seq_len = 1024

        # 2. Tokenizer & Inference Engine
        self.tokenizer = TinyTokenizer.load(tokenizer_path)
        self._engine = InferenceEngine(
            session=self.session,
            tokenizer=self.tokenizer,
            input_name=self.input_name,
            max_seq_len=self.max_seq_len,
        )

        # 3. Native SLM Semantic Retriever
        self.embedding_model_name = embedding_model
        self._encoder = None
        self.kb_path = knowledge_base_path
        self.documents: List[Document] = []
        self._load_knowledge_base()

        # 4. Online Researcher & Persona State
        self.researcher = WebResearcher()
        self.research_enabled = True
        self.assistant_name = "Kruuboo SLM"
        self.user_name = None
        self.current_mood = None

        logger.info(f"Unified Conversational SLM initialized (Context: {self.max_seq_len} tokens, Knowledge: {len(self.documents)} docs)")

    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """
        Hot-reload updated ONNX model weights and refresh the inference engine & native encoder.
        """
        target_path = model_path or self.model_path
        logger.info(f"🔄 Hot-reloading ONNX model session from: {target_path}")
        try:
            new_session = ort.InferenceSession(target_path)
            input_meta = new_session.get_inputs()[0]
            self.input_name = input_meta.name
            shape = input_meta.shape
            if len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 0:
                self.max_seq_len = shape[1]

            self.session = new_session
            self.model_path = target_path

            # Re-create InferenceEngine
            self._engine = InferenceEngine(
                session=self.session,
                tokenizer=self.tokenizer,
                input_name=self.input_name,
                max_seq_len=self.max_seq_len,
            )

            # Invalidate encoder cache to load new PyTorch weights if native
            self._encoder = None
            logger.info("✅ Model hot-reload successful. New weights active in memory.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to hot-reload model from {target_path}: {e}")
            return False

    @property
    def encoder(self):
        if self._encoder is None:
            if self.embedding_model_name in ["native", "tiny_llm", "kruuboo", "slm"]:
                logger.info("Initializing Kruuboo-SLM-0.2 Native SLM Encoder (768d)...")
                self._encoder = NativeSLMEncoder(
                    model_path="models/tiny_llm.pt",
                    tokenizer_path="tokenizer.json"
                )
            else:
                try:
                    logger.info(f"Loading embedding model: {self.embedding_model_name}...")
                    from sentence_transformers import SentenceTransformer
                    self._encoder = SentenceTransformer(self.embedding_model_name)
                except Exception as e:
                    logger.warning(f"Fallback to NativeSLMEncoder: {e}")
                    self._encoder = NativeSLMEncoder()
        return self._encoder

    def _load_knowledge_base(self):
        """Load global knowledge documents into unified memory."""
        docs_path = Path(f"{self.kb_path}/knowledge.json")
        if docs_path.exists():
            try:
                with open(docs_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.documents = [
                    Document(
                        text=item.get("text", ""),
                        source=item.get("source", "manual"),
                        timestamp=item.get("timestamp", time.time()),
                        metadata=item.get("metadata", {})
                    )
                    for item in data if item.get("text", "").strip()
                ]
                logger.info(f"Loaded {len(self.documents)} global knowledge documents")
            except Exception as e:
                logger.error(f"Failed to load knowledge base: {e}")
                self.documents = []
        else:
            self.documents = []

    def add_text(self, text: str, source: str = "manual", metadata: Dict = None):
        """Add text chunk into the unified knowledge store."""
        chunk_size = _config.get("rag", {}).get("chunk_size", 800)
        overlap = _config.get("rag", {}).get("chunk_overlap", 80)
        chunks = []

        for i in range(0, len(text), max(1, chunk_size - overlap)):
            chunk = text[i : i + chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())

        for i, chunk in enumerate(chunks):
            try:
                emb = self.encoder.encode(chunk, show_progress_bar=False).tolist()
            except Exception:
                emb = None

            doc = Document(
                text=chunk,
                source=source,
                metadata={
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "embedding": emb,
                    **(metadata or {})
                }
            )
            self.documents.append(doc)

        logger.info(f"Added {len(chunks)} chunks from {source}")
        self.save()

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Semantic hybrid search across the unified knowledge store."""
        results, needs_save = hybrid_search(query, self.documents, self.encoder, k)
        if needs_save:
            self.save()
        return results

    def _resolve_query_context(self, question: str, history: Optional[List[Dict]]) -> str:
        """Resolve pronouns or follow-up references using recent dialogue history."""
        if not history or len(history) == 0:
            return question

        clean_q = question.strip().lower()
        pronouns = ["it", "its", "that", "this", "them", "these", "those", "how to treat it", "what causes it", "cure"]
        words = set(re.findall(r"\b\w+\b", clean_q))
        has_ambiguous = any(p in words or p in clean_q for p in pronouns) or len(words) <= 4

        if has_ambiguous:
            for turn in reversed(history):
                if turn.get("role") == "user":
                    prev_text = turn.get("content", "").strip()
                    if len(prev_text) > 4 and not prev_text.startswith("/"):
                        return f"{question} {prev_text}"
        return question

    def _build_budgeted_prompt(
        self,
        system_msg: str,
        user_msg: str,
        history: Optional[List[Dict]],
        max_prompt_budget: int
    ) -> str:
        """Assemble multi-turn ChatML prompt with sliding turn-by-turn dialogue window."""
        system_turn = f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        user_turn = f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n"

        base_tokens = len(self.tokenizer.encode(system_turn + user_turn))
        if base_tokens >= max_prompt_budget or not history:
            return system_turn + user_turn

        allowed_turns = []
        current_tokens = base_tokens

        # Slide backwards through history turn-by-turn preserving most recent context
        for turn in reversed(history[-8:]):
            role = turn.get("role", "user")
            content = turn.get("content", "").strip()
            if not content:
                continue
            turn_str = f"<|im_start|>{role}\n{content}<|im_end|>\n"
            turn_tokens = len(self.tokenizer.encode(turn_str))

            if current_tokens + turn_tokens <= max_prompt_budget:
                allowed_turns.insert(0, turn_str)
                current_tokens += turn_tokens
            else:
                # Reached token budget limit, stop adding older turns
                break

        return system_turn + "".join(allowed_turns) + user_turn

    def generate(
        self,
        prompt: str,
        max_tokens: int = 120,
        temperature: float = 0.65,
        min_p: float = 0.05,
        top_k: int = 0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.15,
        stream: bool = False,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Autoregressive neural generation with Min-P sampling."""
        if self.session is None:
            return "[No model loaded. Run src/train.py then scripts/export_onnx.py]"

        def _print_token(t: str):
            print(t, end="", flush=True)

        callback = on_token if on_token is not None else (_print_token if stream else None)

        return self._engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            min_p=min_p,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            on_token=callback,
        )

    def ask(self, question: str, history: Optional[List[Dict]] = None) -> Dict:
        """
        Unified Conversational Chat Pipeline.
        Handles conversational chit-chat, identity resolution, semantic RAG retrieval,
        and neural generation across all topics without rigid domain switches.
        """
        clean_q = re.sub(r"[^\w\s]", "", question).lower()
        clean_q = " ".join(clean_q.split()).strip()

        # 1. User Introduction detection (prioritized)
        intro_name = check_user_intro(clean_q)
        if intro_name:
            self.user_name = intro_name
            return {
                "answer": f"Nice to meet you, {intro_name}! How can I help you today?",
                "knowledge_used": 0,
                "retrieved_context": [],
                "temperature_used": 0.7,
            }

        # 2. Exact or Word-bounded Greetings & Chit-Chat matching
        if clean_q in GREETINGS_MAP:
            return {
                "answer": GREETINGS_MAP[clean_q],
                "knowledge_used": 0,
                "retrieved_context": [],
                "temperature_used": 0.7,
            }
        for g_key, g_ans in GREETINGS_MAP.items():
            if re.search(rf"\b{re.escape(g_key)}\b", clean_q):
                if len(clean_q.split()) <= len(g_key.split()) + 3:
                    return {
                        "answer": g_ans,
                        "knowledge_used": 0,
                        "retrieved_context": [],
                        "temperature_used": 0.7,
                    }

        # 3. Assistant / User Name queries
        if check_bot_name_query(clean_q):
            return {
                "answer": f"My name is {self.assistant_name}. How can I assist you today?",
                "knowledge_used": 0,
                "retrieved_context": [],
                "temperature_used": 0.7,
            }

        if check_user_name_query(clean_q):
            name = getattr(self, "user_name", None) or extract_user_name_from_history(history)
            if name and name != "my friend":
                ans = f"Your name is {name}."
            else:
                ans = "I don't know your name yet! You can tell me by saying 'My name is [your name]'."
            return {
                "answer": ans,
                "knowledge_used": 0,
                "retrieved_context": [],
                "temperature_used": 0.7,
            }

        # 4. Direct Arithmetic & Math Expression Resolution
        math_solution = solve_arithmetic_query(question)
        if math_solution:
            return {
                "answer": math_solution,
                "knowledge_used": 0,
                "retrieved_context": [],
                "temperature_used": 0.0,
            }

        # 5. Semantic Knowledge Retrieval (Native 768d Hybrid RAG)
        search_query = self._resolve_query_context(question, history)
        retrieved_with_scores = self.search(search_query, k=3)
        seen_texts = set()
        retrieved = []
        for doc, score in retrieved_with_scores:
            clean_doc_text = doc.text.strip()
            if score >= 0.40 and clean_doc_text not in seen_texts:
                seen_texts.add(clean_doc_text)
                retrieved.append(doc)

        # 6. System persona & emotional valence
        system_msg = "You are Kruuboo SLM, a helpful, knowledgeable, and empathetic AI assistant."
        try:
            from src.feeling import update_mood, get_mood_instruction
            self.current_mood = update_mood(question)
            mood_instruction = get_mood_instruction(self.current_mood)
            system_msg += f" {mood_instruction}"
        except Exception:
            self.current_mood = None

        # 6. Response Synthesis (Neural Generation with Grounded Context)
        max_budget = max(64, self.max_seq_len - 140)

        if retrieved:
            # Grounded Neural Synthesis: Condition the model on retrieved facts
            context_text = "\n".join([f"• {d.text}" for d in retrieved[:3]])
            grounded_system = (
                f"{system_msg}\n\n"
                f"Relevant Knowledge Context:\n{context_text}\n\n"
                "Answer the user's question accurately using the context above. Keep explanations clear and conversational."
            )
            temp = 0.35
            prompt = self._build_budgeted_prompt(grounded_system, question, history, max_budget)
            answer = self.generate(prompt, max_tokens=140, temperature=temp, min_p=0.05)

            # Fallback to direct fact if neural output is degenerate or empty
            if not answer or len(answer.strip().split()) < 3:
                answer = retrieved[0].text
        else:
            # Natural Open-Ended Generation with Min-P sampling
            temp = 0.65
            prompt = self._build_budgeted_prompt(system_msg, question, history, max_budget)
            answer = self.generate(prompt, max_tokens=120, temperature=temp, min_p=0.05)

        # 7. Anti-Repetition Guardrail
        words = answer.split()
        if len(words) > 8:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.35 or words.count("given") >= 3 or words.count("the") > len(words) * 0.38:
                if retrieved:
                    answer = retrieved[0].text
                else:
                    answer = "I am Kruuboo Tiny LLM. I can help answer questions across medical, legal, survival, math, and conversational topics. What would you like to know?"

        return {
            "answer": answer,
            "knowledge_used": len(retrieved),
            "retrieved_context": [d.text for d in retrieved],
            "temperature_used": temp,
            "mood": self.current_mood.to_dict() if getattr(self, "current_mood", None) else None,
        }

    def research_and_learn(self, query: str) -> bool:
        """Online web research and on-the-fly ingestion."""
        if not self.research_enabled:
            return False
        print(f"🌐 Researching '{query}' online...")
        findings = self.researcher.research_topic(query)
        if not findings:
            print("❌ No information found online.")
            return False
        for finding in findings:
            self.add_text(finding["text"], source=finding["source"])
        print(f"✅ Learned {len(findings)} new items from the web!")
        return True

    def save(self):
        """Save global knowledge base to disk."""
        Path(self.kb_path).mkdir(parents=True, exist_ok=True)
        docs_path = Path(f"{self.kb_path}/knowledge.json")
        try:
            serialized = [
                {
                    "text": doc.text,
                    "source": doc.source,
                    "timestamp": doc.timestamp,
                    "metadata": doc.metadata,
                }
                for doc in self.documents
            ]
            with open(docs_path, "w", encoding="utf-8") as f:
                json.dump(serialized, f, indent=2)
            logger.info(f"Saved {len(self.documents)} knowledge documents to {docs_path}")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def get_stats(self) -> Dict:
        return {
            "total_documents": len(self.documents),
            "knowledge_base_path": self.kb_path,
        }
