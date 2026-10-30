import json
import time
import logging
import re
from pathlib import Path
from typing import Callable, List, Dict, Tuple, Optional

import yaml
import onnxruntime as ort

from .document import Document
from .search import hybrid_search, FAISSVectorIndex
from .native_encoder import NativeSLMEncoder
from .db_connector import DatabaseKnowledgeSource
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
        db_uris: Optional[List[str]] = None,
    ):
        # 1. Load ONNX LLM
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        logger.info(f"Loading ONNX model from {model_path}")
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
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
        self.vector_index = FAISSVectorIndex()
        if self.documents:
            self.vector_index.build(self.documents)

        # 4. Online Researcher & Persona State
        self.researcher = WebResearcher()
        self.research_enabled = _config.get("web_research", {}).get("enabled", True)
        self.auto_web_fallback = _config.get("web_research", {}).get("auto_fallback", True)
        self.assistant_name = "Kruuboo SLM"
        self.user_name = None
        self.current_mood = None

        # 5. rStar Mutual Reasoning Engine
        self.rstar = RStarEngine(self._engine, self.tokenizer, max_seq_len=min(512, self.max_seq_len))

        # 6. Live Database Knowledge Connectors (SQLite / PostgreSQL / MySQL / MongoDB)
        self.db_sources: Dict[str, DatabaseKnowledgeSource] = {}
        configured_dbs = _config.get("database_sources", [])
        if isinstance(configured_dbs, list):
            for entry in configured_dbs:
                if isinstance(entry, str):
                    self.connect_db(entry)
                elif isinstance(entry, dict) and entry.get("uri"):
                    self.connect_db(
                        uri=entry["uri"],
                        name=entry.get("name"),
                        tables=entry.get("tables"),
                    )
        if db_uris:
            for uri in db_uris:
                self.connect_db(uri)

        logger.info(
            f"Unified Conversational SLM initialized (Context: {self.max_seq_len} tokens, "
            f"Knowledge: {len(self.documents)} docs, Live DBs: {len(self.db_sources)})"
        )

    def reload_model(self, model_path: Optional[str] = None) -> bool:
        """
        Hot-reload updated ONNX model weights and refresh the inference engine & native encoder.
        """
        target_path = model_path or self.model_path
        logger.info(f"🔄 Hot-reloading ONNX model session from: {target_path}")
        try:
            sess_opts = ort.SessionOptions()
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            new_session = ort.InferenceSession(
                target_path,
                sess_options=sess_opts,
                providers=["CPUExecutionProvider"],
            )
            input_meta = new_session.get_inputs()[0]
            self.input_name = input_meta.name
            shape = input_meta.shape
            if len(shape) > 1 and isinstance(shape[1], int) and shape[1] > 0:
                self.max_seq_len = shape[1]

            self.session = new_session
            self.model_path = target_path

            # Re-create InferenceEngine & RStarEngine
            self._engine = InferenceEngine(
                session=self.session,
                tokenizer=self.tokenizer,
                input_name=self.input_name,
                max_seq_len=self.max_seq_len,
            )
            self.rstar = RStarEngine(self._engine, self.tokenizer, max_seq_len=min(512, self.max_seq_len))

            # Invalidate encoder cache and refresh knowledge base vector index
            self._encoder = None
            self._load_knowledge_base()
            if hasattr(self, "vector_index") and self.documents:
                emb0 = self.documents[0].metadata.get("embedding")
                if emb0:
                    self.vector_index.dimension = len(emb0)
                self.vector_index.build(self.documents, force_rebuild=True)
            logger.info("✅ Model hot-reload successful. New weights active in memory.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to hot-reload model from {target_path}: {e}")
            return False

    @property
    def encoder(self):
        if self._encoder is None:
            if self.embedding_model_name in ["native", "tiny_llm", "kruuboo", "slm"]:
                logger.info("Initializing Kruuboo-SLM-0.2 Native SLM Encoder (768d ONNX)...")
                self._encoder = NativeSLMEncoder(
                    model_path=self.model_path,
                    tokenizer_path=self.tokenizer_path,
                    session=self.session,
                    tokenizer=self.tokenizer,
                )
            else:
                try:
                    logger.info(f"Loading embedding model: {self.embedding_model_name}...")
                    from sentence_transformers import SentenceTransformer
                    self._encoder = SentenceTransformer(self.embedding_model_name)
                except Exception as e:
                    logger.warning(f"Fallback to NativeSLMEncoder: {e}")
                    self._encoder = NativeSLMEncoder(
                        model_path=self.model_path,
                        tokenizer_path=self.tokenizer_path,
                        session=self.session,
                        tokenizer=self.tokenizer,
                    )
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
        if hasattr(self, "vector_index"):
            self.vector_index.build(self.documents, force_rebuild=True)
        self.save()

    def connect_db(
        self,
        uri: str,
        name: Optional[str] = None,
        tables: Optional[List[str]] = None,
        custom_query: Optional[str] = None,
    ) -> DatabaseKnowledgeSource:
        """Connect a live read-only database (SQLite, PostgreSQL, MySQL, MongoDB) to RAG retrieval."""
        source = DatabaseKnowledgeSource(
            uri=uri,
            name=name,
            tables=tables,
            custom_query=custom_query,
        )
        self.db_sources[source.name] = source
        return source

    def disconnect_db(self, name: str) -> bool:
        """Disconnect a live database source by name."""
        if name in self.db_sources:
            del self.db_sources[name]
            return True
        return False

    def list_databases(self) -> List[Dict]:
        """Return metadata and discovered schema for all connected live databases."""
        return [src.to_info_dict() for src in self.db_sources.values()]

    def search(self, query: str, k: int = 3) -> List[Tuple[Document, float]]:
        """Semantic hybrid search across the unified knowledge store + live connected databases."""
        idx = getattr(self, "vector_index", None)
        results, needs_save = hybrid_search(query, self.documents, self.encoder, k, vector_index=idx)
        if needs_save:
            self.save()

        # Query all connected live databases in parallel/sequence and merge results
        if getattr(self, "db_sources", None):
            db_hits: List[Tuple[Document, float]] = []
            for db_src in self.db_sources.values():
                db_hits.extend(db_src.search(query, k=k))
            if db_hits:
                combined = list(results) + db_hits
                combined.sort(key=lambda x: x[1], reverse=True)
                return combined[: max(k, 4)]

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

        # Only inject prior turns when the current message references prior dialogue context
        q_lower = user_msg.lower().strip()
        if q_lower.startswith(("my name is", "call me", "i am ", "i'm ")):
            return system_turn + user_turn
        q_words = set(re.findall(r"\b\w+\b", q_lower))
        context_triggers = {
            "my", "name", "who", "remember", "it", "its", "that", "this",
            "them", "these", "those", "more", "else", "again", "previous", "said"
        }
        if not (q_words & context_triggers):
            return system_turn + user_turn

        allowed_turns = []
        current_tokens = base_tokens

        # Slide backwards through recent history (up to last 2 turns / 1 exchange)
        for turn in reversed(history[-2:]):
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
        repetition_window: int = 50,
        stop_sequences: Optional[List[str]] = None,
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
            repetition_window=repetition_window,
            stop_sequences=stop_sequences,
            on_token=callback,
        )

    def ask(
        self,
        question: str,
        history: Optional[List[Dict]] = None,
        auto_fallback: Optional[bool] = None,
        use_reasoning: bool = False,
        on_reasoning_step: Optional[Callable[[str], None]] = None,
    ) -> Dict:
        """
        Unified Conversational Chat Pipeline.
        Routes all queries directly through semantic RAG retrieval and the neural SLM
        without hardcoded regex, greeting dictionary, or calculator interceptors.
        """
        # 1. Semantic Knowledge Retrieval (Native 768d Hybrid RAG)
        search_query = self._resolve_query_context(question, history)
        retrieved_with_scores = self.search(search_query, k=3)
        seen_texts = set()
        retrieved = []
        for doc, score in retrieved_with_scores:
            clean_doc_text = doc.text.strip()
            if score >= 0.40 and clean_doc_text not in seen_texts:
                seen_texts.add(clean_doc_text)
                retrieved.append(doc)

        # 1.1 Automated Multi-Provider Web Research Fallback (Wolfram / DDG / Wiki / ArXiv)
        web_research_used = False
        should_fallback = auto_fallback if auto_fallback is not None else getattr(self, "auto_web_fallback", True)
        if not retrieved and self.research_enabled and should_fallback:
            cleaned_topic = self.researcher.clean_query(search_query)
            if len(cleaned_topic) >= 3 and len(cleaned_topic.split()) >= 2:
                logger.info(f"🌐 Low semantic memory confidence for '{search_query}'. Triggering automated web fallback...")
                try:
                    findings = self.researcher.research_topic(search_query, max_results=2)
                    if findings:
                        web_research_used = True
                        logger.info(f"✅ Retrieved {len(findings)} web research items. Ingesting into semantic memory...")
                        for finding in findings:
                            self.add_text(
                                finding["text"],
                                source=finding["source"],
                                metadata={"url": finding.get("url", "")}
                            )
                        fresh_results = self.search(search_query, k=3)
                        for doc, score in fresh_results:
                            clean_doc_text = doc.text.strip()
                            if score >= 0.35 and clean_doc_text not in seen_texts:
                                seen_texts.add(clean_doc_text)
                                retrieved.append(doc)

                        if not retrieved:
                            for finding in findings:
                                doc_obj = Document(
                                    text=finding["text"],
                                    source=finding["source"],
                                    metadata={"url": finding.get("url", "")}
                                )
                                retrieved.append(doc_obj)
                except Exception as web_err:
                    logger.debug(f"Automated web fallback skipped: {web_err}")

        # 2. System persona & emotional valence
        system_msg = "You are Kruuboo SLM, a helpful, knowledgeable, and empathetic AI assistant."
        try:
            from src.feeling import update_mood, get_mood_instruction
            self.current_mood = update_mood(question)
            mood_instruction = get_mood_instruction(self.current_mood)
            system_msg += f" {mood_instruction}"
        except Exception:
            self.current_mood = None

        # 3. Response Synthesis (rStar Mutual Reasoning OR Neural Generation)
        max_budget = max(64, self.max_seq_len - 140)
        reasoning_info = None

        if use_reasoning:
            context_text = "\n".join([f"• {d.text}" for d in retrieved[:3]]) if retrieved else ""
            traj = self.rstar.reason_and_verify(
                question=question,
                context=context_text,
                num_candidates=3,
                on_step=on_reasoning_step,
            )
            answer = traj.final_answer
            temp = 0.2
            reasoning_info = {
                "score": round(traj.score, 3),
                "steps": [
                    {
                        "type": s.step_type,
                        "thought": s.thought,
                        "confidence": round(s.confidence, 3)
                    }
                    for s in traj.steps
                ]
            }
        elif retrieved:
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
        else:
            # Natural Open-Ended Generation with Min-P sampling
            temp = 0.65
            prompt = self._build_budgeted_prompt(system_msg, question, history, max_budget)
            answer = self.generate(prompt, max_tokens=120, temperature=temp, min_p=0.05)

        return {
            "answer": answer,
            "knowledge_used": len(retrieved),
            "retrieved_context": [d.text for d in retrieved],
            "temperature_used": temp,
            "mood": self.current_mood.to_dict() if getattr(self, "current_mood", None) else None,
            "web_research_used": web_research_used,
            "reasoning": reasoning_info,
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
            "connected_databases": [src.to_info_dict() for src in getattr(self, "db_sources", {}).values()],
        }
