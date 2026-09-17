# 🎯 Kruuboo Tiny LLM (SLM-0.1) with Domain-Specific RAG & rStar Self-Play

A lightweight, high-capacity, and privacy-first Small Language Model (SLM) platform combining a custom-trained **105.4M Parameter Transformer Decoder** (~100 MB INT8 footprint with **RoPE**, **SwiGLU**, **1,024 Context Memory**, and **Flash Attention**) with a **Domain-Specific Retrieval-Augmented Generation (RAG)** pipeline and an **rStar-style Mutual Reasoning Self-Play Engine**.

This project demonstrates how small-scale, locally trained neural models can achieve high accuracy, multi-turn conversational fluency, and robust multi-step logical deduction on consumer hardware (CPU / Apple Silicon Metal MPS / NVIDIA CUDA) without cloud dependencies.

---

## 🌟 Key Highlights

- **105.4M Parameter High-Capacity SLM**: Built from scratch with an optimized 14-layer transformer architecture, offering an ideal balance between reasoning capacity and edge runtime efficiency (~100 MB INT8 / ~210 MB FP16).
- **Extended 1,024-Token Context Memory**: Extended context memory window supporting long multi-turn dialogues, multi-document synthesis, and reasoning chains.
- **State-of-the-Art Transformer Building Blocks**:
  - **Rotary Position Embeddings (RoPE)** for length extrapolation without position table limits.
  - **SwiGLU Gated Feed-Forward Networks** for superior representational density.
  - **Flash Attention** via PyTorch 2.0+ `scaled_dot_product_attention` fused kernels.
  - **Pre-LayerNorm** and **GPT-2 style scaled residual initialization**.
  - **Embedding & LM Head Weight Tying** for parameter efficiency.
- **rStar Mutual Reasoning & Self-Play**: Dual-role engine (Proposer + Verifier) that generates diverse reasoning trajectories (`<thought>...</thought>`), fact-checks each step against retrieved RAG evidence, and prunes hallucinations.
- **SFT ChatML Training with Target-Only Loss Masking**: Computes Cross-Entropy loss exclusively on assistant response tokens (masking system instructions and user queries to `-100`).
- **Conversational Memory & Contextual Coreference Resolution**: Dynamic sliding memory buffer that preserves system instructions and resolves follow-up queries (e.g., *"How do I treat it?"* $\rightarrow$ linked to preceding ailment).
- **Domain-Specific Semantic Routing**: Automatically classifies queries into isolated domain knowledge bases (Medical/Home Remedies, Legal, Survival, Math, Sinhala News, General) via `domains.yaml` to eliminate context contamination.
- **Affective Sentiment & Mood Tracking**: Real-time emotional valence tracking and persistent state updates (`mood_state.json`) for empathetic persona modulation.
- **Autonomous Web Researcher**: Built-in scraper and live embedder that searches external sources to dynamically patch knowledge gaps on demand.

---

## 📊 Model Architecture Specifications

| Hyperparameter / Feature | Specification | Architectural Rationale |
| :--- | :--- | :--- |
| **Model Name** | `Kruuboo-SLM-0.1` (`tiny_llm`) | Custom decoder-only autoregressive transformer |
| **Total Parameters** | **105,426,432 (~105.4M)** | High-capacity small language model |
| **Layers ($N_{\text{layers}}$)** | **14 Transformer Blocks** | Deeper feature representation and hierarchical reasoning |
| **Hidden Dim ($d_{\text{model}}$)** | **768** | Standard high-capacity representation size |
| **Attention Heads ($N_{\text{heads}}$)**| **12 Heads** | Head dimension $d_{\text{head}} = 64$ ($768 / 12$) |
| **FFN Intermediate Dim ($d_{\text{ff}}$)**| **3072** ($\approx \frac{2}{3} d_{\text{ff}} = 2048$ SwiGLU) | Swish-gated linear projections ($W_1, W_2, W_3$) |
| **Context Window ($T_{\text{max}}$)** | **1,024 Tokens** | Supports long multi-turn context and RAG chunks |
| **Vocabulary Size ($V$)** | **8,192 Subwords** | Byte-Pair Encoding (BPE) with ChatML special tokens |
| **Positional Encoding** | **RoPE (Rotary Position Embeddings)** | Frequency base $\theta = 10,000$; supports dynamic sequence lengths |
| **Attention Mechanism** | **Flash Attention (Fused SDPA)** | $O(N)$ memory execution on PyTorch 2.0+ with causal masking |
| **Activation Function** | **SwiGLU** | $W_2(\text{SiLU}(W_1 x) \odot W_3 x)$ |
| **Normalization** | **Pre-LayerNorm** | Applied before attention and FFN sub-layers ($\epsilon = 10^{-5}$) |
| **Weight Sharing** | **Tied Embeddings** | `token_embedding.weight` $\equiv$ `lm_head.weight` |
| **Weight Initialization** | **GPT-2 Scaled Residual** | Residual projections scaled by $\frac{0.02}{\sqrt{2 \cdot N_{\text{layers}}}}$ |
| **Memory Footprint** | **~105 MB (INT8) / ~210 MB (FP16)** | Instant edge inference on CPU, Metal (MPS), and CUDA |

---

## 🏗️ System Architecture

```mermaid
graph TD
    UserQuery([User Query / Chat Message]) --> Mood[Sentiment & Affective Analyzer]
    UserQuery --> Coref[Contextual Coreference Resolver]
    Coref --> Router{Dynamic Domain Router}
    
    Router -->|Rules & Keyword Match| ActiveDomain[Selected Domain KB]
    Router -->|Low Confidence| Fallback[General Knowledge Base]
    
    ActiveDomain --> Embed[Dense Embedding Model: all-MiniLM-L6-v2]
    Fallback --> Embed
    Embed --> FAISS[FAISS Vector Store / Index]
    FAISS --> Context[Retrieved Grounded Context Chunks]
    
    Context --> EngineSwitch{Inference Engine Selector}
    
    EngineSwitch -->|rstar| RStar[rStar Mutual Reasoning Engine]
    EngineSwitch -->|hybrid| Hybrid[Hybrid RAG + Neural Generator]
    EngineSwitch -->|toy_llm| Generator[Custom 105.4M Transformer Decoder]
    EngineSwitch -->|smart_rag| Extractive[Extractive Knowledge Matcher]
    
    subgraph Custom Neural Engine (105.4M Parameters)
        Generator --> Tokenizer[BPE Tokenizer (8K Vocab + ChatML)]
        Tokenizer --> RoPE[RoPE Rotary Embeddings]
        RoPE --> Blocks[14x Blocks: FlashAttn + SwiGLU FFN]
        Blocks --> Norm[Final LayerNorm & Tied LM Head]
        Norm --> Output[Generated Logits & Sampling]
    end
    
    RStar --> Output
    Hybrid --> Output
    Extractive --> Output
    
    WebDaemon[Autonomous Web Researcher / Scraper] -.->|Live Ingest| FAISS
    Mood -.->|Update Mood State| Workspace[(mood_state.json)]
```

---

## 🛡️ Core Engine Components

### 1. Neural Transformer (`src/model.py`)
- **Rotary Position Embeddings (RoPE)**: Dynamically rotates query and key representations in complex vector sub-spaces, maintaining relative distance awareness without fixed learned position tables.
- **SwiGLU Gated Feed-Forward**: Employs dual-gate linear projections modulated by the SiLU (Swish) activation function to maximize non-linear expressiveness.
- **PyTorch 2.0+ Flash Attention**: Utilizes kernel-level memory fused dot-product attention for lower peak VRAM and faster forward passes.
- **Tied Input-Output Weights**: Shared embedding tensor reduces parameter overhead and aligns token input representations directly with logit projections.

### 2. rStar Mutual Reasoning Self-Play (`src/rstar.py` & `scripts/rstar_self_play.py`)
- **Proposer (Generator)**: Decomposes complex queries into atomic deductive steps, generating reasoning trajectories enclosed in `<thought>...</thought>` blocks.
- **Mutual Verifier (Discriminator)**: Performs fact-checking and consistency validation against retrieved RAG ground truth.
- **Best-of-N Candidate Search**: Explores multiple reasoning trajectories at varying temperatures ($T \in [0.6, 1.0]$) and selects the highest-confidence path.
- **Synthetic Self-Play Generation**: Synthesizes verified multi-turn ChatML datasets to fine-tune the model for autonomous step-by-step reasoning.

### 3. SFT Training & Loss Masking (`src/train.py`)
- **Target-Only Loss Masking**: Sets labels for system prompts and user query turns to `-100` (`ignore_index`), directing 100% of gradient descent updates to assistant outputs.
- **Cosine Learning Rate Decay with Linear Warmup**: Peak learning rate $\eta_{\text{max}} = 3\times 10^{-4}$ decaying smoothly to $\eta_{\text{min}} = 1\times 10^{-5}$ with AdamW decoupled weight decay ($0.1$) and gradient clipping ($1.0$).

### 4. Domain-Specific Semantic RAG (`src/rag/`)
- **Dynamic Domain Registry (`domains.yaml`)**: Domain-specific isolation for Medical, Legal, Survival, Mathematics, Teaching ABC, Product Pricing, and Sinhala News.
- **Hybrid Search**: Combines BM25 keyword matching with dense `sentence-transformers/all-MiniLM-L6-v2` embeddings in FAISS.
- **Safety Disclaimers**: Automatic rule-based injection of regulatory disclaimers for sensitive domains (e.g., medical advice, legal questions).

---

## 📂 Directory Structure

```text
tiny_llm_rag_project/
├── config.yaml                     # Model hyperparams (105.4M, 14 layers, 1024 seq_len), RAG & training settings
├── domains.yaml                    # Domain routing rules, keyword triggers, and auto-detect thresholds
├── requirements.txt                # Project dependencies
├── main.py                         # Interactive domain-aware CLI with slash commands
├── server.py                       # FastAPI backend & responsive chat Web UI
│
├── src/
│   ├── model.py                    # 105.4M Transformer Decoder (RoPE, SwiGLU, FlashAttn, Pre-LN)
│   ├── tokenizer.py                # BPE Subword Tokenizer (8,192 vocab, ChatML special tokens)
│   ├── inference.py                # Autoregressive generation, temperature, top-k/top-p sampling
│   ├── rstar.py                    # rStar mutual reasoning self-play engine
│   ├── train.py                    # SFT training pipeline with target-only loss masking
│   ├── feeling.py                  # Sentiment analysis, affective state & mood tracker
│   ├── web_researcher.py           # Real-time web scraping and dynamic knowledge ingestion
│   └── rag/
│       ├── system.py               # RAG coordinator, sliding memory window, and context budget
│       ├── search.py               # Hybrid keyword + FAISS dense vector search
│       ├── domain.py               # Domain registry and automatic rule matcher
│       └── greetings.py            # Greeting, name, and persona management
│
├── scripts/
│   ├── rstar_self_play.py          # Synthetic reasoning dataset generator via self-play
│   ├── export_onnx.py              # Export PyTorch model to optimized ONNX format
│   ├── download_datasets.py        # Dataset downloader and formatter
│   └── build_knowledge_base.py     # Knowledge base builder and FAISS indexer
│
├── training_data/                  # SFT training datasets (.json, .txt)
├── knowledge_base/                 # Domain document chunks, raw knowledge, and vector indexes
└── workspace/                      # Dynamic runtime state (e.g., mood_state.json)
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

#### Available CLI Commands:
* `/engine [rstar|hybrid|toy_llm|smart_rag]` — Switch active inference engine.
* `/domains` — List all registered knowledge domains.
* `/mode [domain_id]` — Manually switch active domain (e.g., `/mode medical`, `/mode survival`).
* `/auto` — Toggle automatic domain detection on/off.
* `/research [topic]` — Scrape the web and ingest knowledge on-the-fly.
* `/info` — View system, model architecture, and knowledge base statistics.
* `/quit` — Exit the session.

### 3. Run the Web Application

Start the local server and web UI:
```bash
python3 server.py
```
Open `http://localhost:8000` in your browser for the responsive desktop and mobile chat interface.

### 4. Training & Self-Play Pipeline

1. **Generate rStar Self-Play Reasoning Data**:
   ```bash
   python3 scripts/rstar_self_play.py
   ```

2. **Train the 105.4M Transformer with SFT Loss Masking**:
   ```bash
   python3 src/train.py
   ```

3. **Export to ONNX (Optional for Edge / C++ Deployment)**:
   ```bash
   python3 scripts/export_onnx.py
   ```

---

## ⚡ Inference Engines Summary

| Engine Mode | Description | Best For |
| :--- | :--- | :--- |
| **`rstar`** | Multi-path mutual reasoning self-play with grounding verification and answer synthesis. | Complex questions, multi-step deduction, factual problem-solving. |
| **`hybrid`** | Blends high-confidence RAG context retrieval with custom neural generation. | General balanced conversational assistant use. |
| **`toy_llm`** | Direct token-by-token generation from the custom 105.4M Transformer. | Creative exploration, raw LLM testing. |
| **`smart_rag`** | Exact extractive snippet matching from domain vector stores. | Fact lookup, strict regulatory compliance. |

