# 🎯 Kruuboo Tiny LLM (SLM-0.1) — Unified Conversational SLM & Native RAG

A high-capacity, privacy-first Small Language Model (SLM) platform combining a custom-trained **105.4M Parameter Transformer Decoder** (~100 MB INT8 footprint with **RoPE**, **SwiGLU**, **1,024 Context Memory**, and **Flash Attention**) with a **Native 768d Semantic RAG Memory**, **Min-P Sampling**, and an interactive **Live Learning Visualizer & Training Studio**.

Kruuboo-SLM-0.1 operates as a **single unified conversational agent** — responding naturally to general dialogue, personal introductions, medical remedies, legal advice, and survival scenarios without rigid domain gating or external third-party dependencies.

---

## 🌟 Key Architectural Highlights

- **105.4M Parameter High-Capacity SLM**: Built from scratch with an optimized 14-layer transformer architecture, offering an ideal balance between reasoning capacity and edge runtime efficiency (~100 MB INT8 / ~210 MB FP16).
- **Unified Conversational Chat (Zero Domain Silos)**: Replaced brittle rule-based domain selectors with unified **ChatML** multi-turn dialogue memory. The model conversationally answers questions across all domains using its weights and global semantic memory.
- **Native 768d Semantic RAG Encoder**: Self-contained RAG retrieval powered directly by the 14-layer transformer decoder (`ln_f(h)`). Eliminates external dependencies on Hugging Face / `sentence-transformers` and unifies the embedding and vocabulary space ($V=8,192, d_{\text{model}}=768$).
- **Modern Min-P Sampling & Contextual Temperature**:
  - **Min-P Sampling** ($p_{\text{base}} = 0.05$): Dynamically scales token pruning relative to top-token confidence ($p \ge p_{\text{max}} \times 0.05$), eliminating low-probability hallucinations while allowing rich conversational vocabulary.
  - **Dynamic Contextual Temperature**: Automatically scales between $T = 0.2$ for factual retrieval and $T = 0.65$ for natural conversational flow.
- **Live Learning Monitor & Visualizer (Web UI)**: Interactive telemetry dashboard on `http://localhost:8000` with real-time Chart.js loss/perplexity curves, token throughput speed (tok/s), cosine LR schedule tracking, on-the-go fine-tuning controls, and weight probing cards.
- **State-of-the-Art Transformer Building Blocks**:
  - **Rotary Position Embeddings (RoPE)** for length extrapolation without position table limits.
  - **SwiGLU Gated Feed-Forward Networks** for superior representational density.
  - **Flash Attention** via PyTorch 2.0+ `scaled_dot_product_attention` fused kernels.
  - **Pre-LayerNorm** and **GPT-2 style scaled residual initialization**.
  - **Tied Input-Output Weights** for parameter efficiency.
- **SFT ChatML Training with Target-Only Loss Masking**: Masks system instructions and user queries (`labels = -100`), calculating loss exclusively on assistant response tokens.
- **Affective Sentiment & Mood Tracking**: Real-time emotional valence tracking and persistent state updates (`workspace/mood_state.json`) for empathetic persona modulation.

---

## 📊 Model Architecture Specifications

| Hyperparameter / Feature | Specification | Architectural Rationale |
| :--- | :--- | :--- |
| **Model Name** | `Kruuboo-SLM-0.1` (`tiny_llm`) | Custom decoder-only autoregressive transformer |
| **Total Parameters** | **105,426,432 (~105.4M)** | High-capacity small language model |
| **Layers ($N_{\text{layers}}$)** | **14 Transformer Blocks** | Deep hierarchical reasoning and dense representation |
| **Hidden Dim ($d_{\text{model}}$)** | **768** | High-capacity representation size |
| **Attention Heads ($N_{\text{heads}}$)**| **12 Heads** | Head dimension $d_{\text{head}} = 64$ ($768 / 12$) |
| **FFN Intermediate Dim ($d_{\text{ff}}$)**| **3072** ($\approx \frac{2}{3} d_{\text{ff}} = 2048$ SwiGLU) | Swish-gated linear projections ($W_1, W_2, W_3$) |
| **Context Window ($T_{\text{max}}$)** | **1,024 Tokens** | Extended multi-turn context and RAG memory window |
| **Vocabulary Size ($V$)** | **8,192 Subwords** | Byte-Pair Encoding (BPE) with ChatML special tokens |
| **Positional Encoding** | **RoPE (Rotary Position Embeddings)** | Frequency base $\theta = 10,000$; supports dynamic sequence lengths |
| **Attention Mechanism** | **Flash Attention (Fused SDPA)** | $O(N)$ memory execution on PyTorch 2.0+ with causal masking |
| **Activation Function** | **SwiGLU** | $W_2(\text{SiLU}(W_1 x) \odot W_3 x)$ |
| **Normalization** | **Pre-LayerNorm** | Applied before attention and FFN sub-layers ($\epsilon = 10^{-5}$) |
| **Weight Sharing** | **Tied Embeddings** | `token_embedding.weight` $\equiv$ `lm_head.weight` |
| **Weight Initialization** | **GPT-2 Scaled Residual** | Residual projections scaled by $\frac{0.02}{\sqrt{2 \cdot N_{\text{layers}}}}$ |
| **Sampling Technique** | **Min-P Sampling ($p_{\text{base}} = 0.05$)** | Confidence-adaptive thresholding preventing tail hallucinations |
| **Memory Footprint** | **~100 MB (INT8) / ~210 MB (FP16)** | Instant edge inference on CPU, Apple Silicon Metal (MPS), and CUDA |

---

## 🏗️ Unified System Architecture

```mermaid
graph TD
    UserQuery([User Input / Message]) --> Mood[Sentiment & Emotional Valence Tracker]
    UserQuery --> ContextResolver[Sliding Context & Coreference Resolver]
    
    ContextResolver --> NativeEncoder[Native SLM 768d Encoder: src/rag/native_encoder.py]
    NativeEncoder --> HybridSearch[Hybrid Semantic Search: 65% Lexical + 35% 768d Cosine]
    
    HybridSearch --> GlobalKB[(Global Semantic Memory: 758+ Docs)]
    GlobalKB --> RetrievedContext[Retrieved Grounded Evidence]
    
    RetrievedContext --> ChatMLBuilder[ChatML Multi-Turn Prompt Builder]
    UserQuery --> ChatMLBuilder
    
    subgraph Kruuboo-SLM-0.1 Neural Engine (105.4M Parameters)
        ChatMLBuilder --> Tokenizer[BPE Tokenizer: 8,192 Vocab + ChatML]
        Tokenizer --> RoPE[RoPE Rotary Embeddings]
        RoPE --> Blocks[14x Transformer Blocks: FlashAttn + SwiGLU]
        Blocks --> Norm[Final LayerNorm & Tied LM Head]
        Norm --> MinP[Min-P Sampler: p_base = 0.05, Contextual Temp]
    end
    
    MinP --> AssistantReply([Conversational Grounded Response])
    
    subgraph Live Learning Studio & Telemetry
        LiveMonitor[Web UI Live Visualizer] <-->|SSE Stream: /api/training/events| Trainer[SFT Trainer: Target-Only Loss Masking]
        Trainer -.->|Weight Updates| Blocks
    end
```

---

## 🛡️ Core System Components

### 1. Neural Transformer (`src/model.py`)
- **RoPE & SwiGLU**: Implements rotary position embedding calculation and SwiGLU gating for fast convergence and high expressiveness.
- **Native Embedding Extraction**: Exposes `forward_features()` and `get_embeddings()` using post-LayerNorm hidden states with mean-pooling and L2 normalization for self-contained RAG retrieval.
- **Flash Attention**: Accelerated scaled dot-product attention kernels natively running on Apple Silicon (MPS), NVIDIA CUDA, and vectorized CPU.

### 2. Unified Native RAG & Hybrid Search (`src/rag/`)
- **`NativeSLMEncoder` (`src/rag/native_encoder.py`)**: Self-contained embedding retriever running on the same weights and 8,192 vocabulary as the language model.
- **`hybrid_search` (`src/rag/search.py`)**: Vectorized cosine similarity computation combined with robust lexical matching, preventing semantic noise from dominating factual matches.
- **Sliding Multi-Turn Context**: Preserves dialogue turns and resolves conversational follow-ups (e.g. *"What is my name?"* or *"How to treat it?"*).

### 3. Min-P Inference Engine (`src/inference.py`)
- Implements autoregressive generation with dynamic **Min-P sampling** ($p_{\text{base}} = 0.05$), repetition penalties, and real-time token streaming callbacks.

### 4. SFT Training & Loss Masking (`src/train.py`)
- **Target-Only Loss Masking**: Sets system/user token labels to `-100`, focusing optimization strictly on assistant answers.
- **Cosine Schedule with Linear Warmup**: Peak learning rate $\eta_{\text{max}} = 3\times 10^{-4}$ decaying to $\eta_{\text{min}} = 1\times 10^{-5}$ with AdamW decoupled weight decay ($0.1$).

### 5. Live Learning Studio & Web UI (`server.py`)
- **Conversational Chat Tab**: Clean, glassmorphic ChatGPT/Claude-style chat with markdown rendering, streaming responses, and source transparency.
- **Live Learning Monitor Tab**: Real-time SSE telemetry charts displaying training/validation loss, perplexity ($e^{\text{loss}}$), throughput (tok/s), and real-time probe generation cards.

---

## 📂 Directory Structure

```text
tiny_llm_rag_project/
├── config.yaml                     # Model hyperparams (105.4M, 14 layers, 1024 seq_len), RAG & training settings
├── requirements.txt                # Python dependencies
├── main.py                         # Unified conversational terminal CLI
├── server.py                       # FastAPI server & Web UI (Chat + Live Learning Monitor)
│
├── src/
│   ├── model.py                    # 105.4M Transformer Decoder (RoPE, SwiGLU, FlashAttn, Native Embeddings)
│   ├── tokenizer.py                # BPE Subword Tokenizer (8,192 vocab, ChatML special tokens)
│   ├── inference.py                # Autoregressive generation with Min-P sampling and contextual temperature
│   ├── rstar.py                    # rStar mutual reasoning self-play engine
│   ├── train.py                    # SFT training pipeline with target-only loss masking and step callbacks
│   ├── feeling.py                  # Sentiment analysis, affective state & mood tracker
│   ├── web_researcher.py           # Real-time web scraping and dynamic knowledge ingestion
│   └── rag/
│       ├── system.py               # Unified conversational RAG coordinator and dialogue manager
│       ├── native_encoder.py       # Self-contained 768d SLM dense vector encoder
│       ├── search.py               # Hybrid lexical + 768d vector cosine search
│       ├── greetings.py            # Conversational greetings, persona, and name memory resolution
│       └── document.py             # Document data structures and serialization
│
├── scripts/
│   ├── rstar_self_play.py          # Synthetic reasoning dataset generator via self-play
│   ├── export_onnx.py              # Export PyTorch model to optimized ONNX format
│   ├── download_datasets.py        # Dataset downloader and formatter
│   └── build_knowledge_base.py     # Global semantic knowledge base indexer
│
├── training_data/                  # SFT training datasets (.json, .txt)
├── knowledge_base/                 # Global semantic documents and metadata (knowledge.json)
└── workspace/                      # Dynamic runtime telemetry and mood state
```

---

## 🛠️ Getting Started

### 1. Installation

Ensure Python 3.9+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Run the Web Application & Live Learning Studio

Start the local server and web UI:
```bash
python3 server.py
```
Open **`http://localhost:8000`** in your browser:
- **Conversational Chat**: Interactive multi-turn chat with grounded semantic memory.
- **Live Learning Monitor**: Watch real-time loss and perplexity curves, monitor token speed, and launch on-the-go fine-tuning.

### 3. Run the Interactive CLI

Launch the conversational terminal shell:
```bash
python3 main.py
```

#### Available CLI Commands:
* `/clear` — Clear current conversation context memory.
* `/research [topic]` — Scrape the web and ingest knowledge on-the-fly.
* `/add [text]` — Ingest custom text directly into global semantic memory.
* `/info` — View system architecture and memory statistics.
* `/help` — Show help menu.
* `/quit` — Exit the session.

### 4. Training & Model Export

1. **Train the 105.4M Transformer with SFT ChatML**:
   ```bash
   python3 src/train.py
   ```

2. **Generate Synthetic rStar Reasoning Data**:
   ```bash
   python3 scripts/rstar_self_play.py
   ```

3. **Export to INT8 ONNX for Fast Edge Deployment**:
   ```bash
   python3 scripts/export_onnx.py
   ```

---

## 🧪 Verified Capabilities

| Query Scenario | Example Input | Expected Output Behavior |
| :--- | :--- | :--- |
| **Casual Conversation** | `"Hi, how are you today?"` | Empathetic, fluent natural dialogue without domain switching. |
| **Identity & Memory** | `"My name is Sithija"` $\rightarrow$ `"What is my name?"` | Multi-turn contextual memory recall (*"Your name is Sithija"*). |
| **Medical Remedies** | `"What is the remedy for fever?"` | Grounded clinical advice retrieved from global semantic memory. |
| **Legal Agreements** | `"What is an NDA and what must it contain?"` | Grounded contract definitions and structural requirements. |
| **Survival Wilderness** | `"How to build a shelter in survival?"` | Actionable lean-to shelter construction guidelines. |
