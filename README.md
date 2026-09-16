# 🎯 Tiny LLM with Domain-Specific RAG & rStar Self-Play

A lightweight, modular, and privacy-first AI platform combining a custom-trained **105.4M High-Capacity Transformer Decoder** (~100 MB INT8 footprint with **RoPE**, **SwiGLU**, **1024 Context Memory**, and **Flash Attention**) with a **Domain-Specific Retrieval-Augmented Generation (RAG)** pipeline and an **rStar-style Mutual Reasoning Self-Play Engine**.

This project demonstrates how small-scale, locally trained neural models can achieve high accuracy, multi-turn conversational fluency, and complex logical reasoning on consumer hardware without cloud dependencies.

---

## 🌟 Key Highlights

- **Hyper-Local & 100MB Footprint**: Runs a custom-built ~105.4M parameter Transformer Decoder locally on CPU or Apple Silicon (Metal/MPS) with a sleek ~100MB quantized runtime footprint.
- **Extended Context Memory**: Features a **1,024 token context window** for long multi-turn dialogues and rich document synthesis.
- **Modern Transformer Architecture**: Uses **Rotary Position Embeddings (RoPE)**, **SwiGLU Gated Feed-Forward Networks**, **Pre-LayerNorm**, and **Flash Attention**.
- **rStar-Style Mutual Reasoning & Self-Play**: Employs a dual-role self-play architecture (Proposer + Verifier) to generate multi-step reasoning chains, fact-check against RAG context, and prune hallucinations.
- **SFT ChatML Training with Target-Only Loss Masking**: Computes Cross-Entropy loss exclusively on assistant response tokens, masking out prompt boilerplate.
- **Conversational Memory & Coreference Resolution**: Dynamic sliding window that protects system instructions and resolves follow-up queries (e.g., *"How do I treat it?"* $\rightarrow$ linked to preceding ailment).
- **Domain-Specific Semantic Routing**: Automatically classifies queries into isolated domain knowledge bases (Medical, Legal, Survival, Math, General) to eliminate context contamination.
- **Autonomous Web Researcher**: Built-in scraper and live embedder that crawls external sources to dynamically patch knowledge gaps on demand.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    UserQuery([User Query]) --> Coref[Contextual Coreference Resolver]
    Coref --> Router{Auto-Domain Router}
    Router -->|Keyword & Semantic Match| ActiveDomain[Selected Domain KB]
    
    ActiveDomain --> Embed[Sentence-Transformers Embeddings]
    Embed --> FAISS[FAISS Vector Store]
    FAISS --> Context[Retrieved Context Chunks]
    
    Context --> EngineSwitch{Inference Engine}
    
    EngineSwitch -->|rstar| RStar[rStar Mutual Reasoning Self-Play Engine]
    EngineSwitch -->|hybrid| Hybrid[Hybrid RAG + Neural Generator]
    EngineSwitch -->|toy_llm| Generator[Custom Transformer Decoder Engine]
    EngineSwitch -->|smart_rag| Extractive[Extractive Knowledge Matcher]
    
    subgraph Custom Neural Engine (17M Params)
        Generator --> Tokenizer[BPE Tokenizer (8K Vocab)]
        Tokenizer --> Model[Transformer: RoPE + SwiGLU + FlashAttn]
        Model --> Output[Final Answer]
    end
    
    RStar --> Output
    Hybrid --> Output
    Extractive --> Output
    
    WebDaemon[Web Researcher / Scraper] -.->|Live Ingest| FAISS
```

---

## 🛡️ Core Engine Layers

### 1. Neural Transformer (`src/model.py`)
- **RoPE (Rotary Position Embeddings)**: Dynamically rotates query/key vectors in complex space for superior sequence generalization and no position-table overflow.
- **SwiGLU FFN**: Replaces standard GELU MLPs with Swish-gated linear projections ($W_2(\text{SiLU}(W_1 x) \odot W_3 x)$) for higher representational efficiency.
- **Attention**: PyTorch 2.0+ fused scaled dot-product attention with manual causal mask fallback.
- **Weight Tying**: Shared embeddings and output projection weights for parameter efficiency.

### 2. rStar Mutual Reasoning Self-Play (`src/rstar.py` & `scripts/rstar_self_play.py`)
- **Proposer (Generator)**: Breaks queries into sub-problems and proposes candidate reasoning trajectories at diverse temperatures.
- **Mutual Verifier (Discriminator)**: Evaluates factual overlap and consistency against retrieved RAG context.
- **Best-of-N Tree Search**: Prunes flawed logic and selects the highest-confidence reasoning path.
- **Synthetic Self-Play Data Generation**: Generates verified `<thought>...</thought>` traces in ChatML format to train the model in step-by-step reasoning.

### 3. SFT Training & Loss Masking (`src/train.py`)
- **Target-Only Loss Masking**: Sets labels for user and system prompt tokens to `-100` (`ignore_index`), focusing 100% of gradient updates on generating high-quality assistant responses.
- **Cosine Learning Rate Decay with Linear Warmup**: Decoupled weight decay (AdamW) with gradient clipping.

### 4. Domain-Specific RAG (`src/rag/`)
- Dynamic domain configurations loaded via `domains.yaml`.
- Hybrid search combining keyword matching and `all-MiniLM-L6-v2` dense vector embeddings.
- Automatic disclaimer injection for sensitive domains (e.g., Medical, Legal).

---

## 📂 Directory Structure

```text
tiny_llm_rag_project/
├── config.yaml                     # Model dimensions, RAG, and training hyperparameters
├── domains.yaml                    # Domain routing rules, keywords, and priority settings
├── requirements.txt                # Project dependencies
├── main.py                         # Interactive domain-aware CLI
├── server.py                       # FastAPI server with responsive chat web UI
│
├── src/
│   ├── model.py                    # Custom Transformer Decoder (RoPE + SwiGLU)
│   ├── tokenizer.py                # Byte-Pair Encoding (BPE) subword tokenizer
│   ├── inference.py                # Sampling, top-k/p filtering, stop-token management
│   ├── rstar.py                    # rStar mutual reasoning self-play engine
│   ├── train.py                    # SFT training pipeline with target-only loss masking
│   ├── feeling.py                  # Sentiment analysis and mood tracking
│   ├── web_researcher.py           # Real-time web search and knowledge scraper
│   └── rag/
│       ├── system.py               # Main RAG coordinator and memory budget manager
│       ├── search.py               # Hybrid keyword + FAISS vector search
│       ├── domain.py               # Domain registry and rule matcher
│       └── greetings.py            # Greeting, name, and identity management
│
├── scripts/
│   ├── rstar_self_play.py          # Generate synthetic reasoning datasets via self-play
│   ├── export_onnx.py              # Export PyTorch weights to optimized ONNX format
│   ├── download_datasets.py        # Dataset downloader and formatter
│   └── build_knowledge_base.py     # Knowledge base builder
│
├── training_data/                  # Raw and synthetic SFT datasets (.json, .txt)
└── knowledge_base/                 # Domain document chunks and vector indexes
```

---

## 🛠️ Getting Started

### 1. Installation

Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive CLI

Launch the domain-aware interactive shell:
```bash
python3 main.py
```

#### Available Slash Commands:
* `/engine [rstar|hybrid|toy_llm|smart_rag]` — Switch inference mode.
* `/domains` — List all registered knowledge domains.
* `/mode [domain_id]` — Manually switch active domain (e.g., `/mode medical`, `/mode survival`).
* `/auto` — Toggle automatic domain detection on/off.
* `/research [topic]` — Scrape the web and ingest knowledge on-the-fly.
* `/info` — View system and knowledge base statistics.
* `/quit` — Exit session.

### 3. Run the Web Application

Start the local server and web UI:
```bash
python3 server.py
```
Open `http://localhost:8000` in your browser for the responsive desktop and mobile chat interface.

### 4. Train the Model & Run Self-Play

1. **Generate rStar Self-Play Reasoning Data**:
   ```bash
   python3 scripts/rstar_self_play.py
   ```

2. **Train the Transformer with SFT Loss Masking**:
   ```bash
   python3 src/train.py
   ```

3. **Export to ONNX (Optional for Edge Deployment)**:
   ```bash
   python3 scripts/export_onnx.py
   ```

---

## ⚡ Inference Engines Summary

| Engine Mode | Description | Best For |
| :--- | :--- | :--- |
| **`rstar`** | Multi-path mutual reasoning self-play with grounding verification and answer synthesis. | Complex questions, multi-step deduction, factual problem-solving. |
| **`hybrid`** | Blends high-confidence RAG context retrieval with custom neural generation. | General balanced assistant use. |
| **`toy_llm`** | Direct token-by-token generation from the custom 17M Transformer. | Creative exploration, raw LLM testing. |
| **`smart_rag`** | Exact extractive snippet matching from domain vector stores. | Fact lookup, strict regulatory compliance. |
