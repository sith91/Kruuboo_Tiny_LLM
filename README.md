# 🎯 Kruuboo Tiny LLM (Kruuboo-0.6) — Dynamic Profile Registry, Live DB RAG & Unified SLM

A high-capacity, privacy-first Small Language Model (SLM) platform combining a custom-trained **105.4M Parameter Transformer Decoder** (~100 MB INT8 footprint with **RoPE**, **SwiGLU**, **1,024 Context Memory**, and **Flash Attention**) with a **Native 768d Semantic RAG Memory**, a **Multi-Provider Web Research Fallback Chain**, **Direct Arithmetic Resolution**, **Min-P Sampling**, and an interactive **Live Learning Visualizer & Training Studio**.

Kruuboo-SLM-0.2 operates as a **single unified conversational agent** — responding naturally to general dialogue, personal memory recall, medical remedies, legal agreements, survival techniques, and scientific calculations without rigid domain gating or external third-party dependencies.

---

## 🌟 Key Architectural Highlights

- **105.4M Parameter High-Capacity SLM**: Built from scratch with an optimized 14-layer transformer architecture, offering an ideal balance between reasoning capacity and edge runtime efficiency (~100 MB INT8 / ~210 MB FP16).
- **Unified Conversational Dialogue (Zero Domain Silos)**: Replaced brittle rule-based domain selectors with unified **ChatML** multi-turn dialogue memory. The model conversationally answers questions across all domains using its weights and global semantic memory.
- **Native 768d Semantic RAG Encoder**: Self-contained RAG retrieval powered directly by the 14-layer transformer decoder (`ln_f(h)`). Eliminates external dependencies on Hugging Face / `sentence-transformers` and unifies the embedding and vocabulary space ($V=8,192, d_{\text{model}}=768$).
- **Multi-Provider Web Research Fallback Chain (`src/web_researcher.py`)**:
  - Automatically queries a priority fallback chain when knowledge gaps are detected or upon `/research` command:
    1. **Wolfram Alpha**: Computational intelligence, mathematics, scientific equations, and units.
    2. **DuckDuckGo**: Instant answers, definitions, and zero-click web summaries (unauthenticated).
    3. **Wikipedia**: Deep encyclopedic article summaries and background context.
    4. **ArXiv**: Academic papers, machine learning preprints, and scientific abstracts.
- **Direct Arithmetic & Calculation Engine (`src/rag/system.py`)**:
  - Deterministic arithmetic solver (`solve_arithmetic_query`) resolves expressions (`2 + 2`, `45 * 12`, `100 / 4`) instantly with zero latency and zero hallucination risk.
- **Modern Min-P Sampling & Contextual Temperature**:
  - **Min-P Sampling** ($p_{\text{base}} = 0.05$): Dynamically scales token pruning relative to top-token confidence ($p \ge p_{\text{max}} \times 0.05$), eliminating tail hallucinations while preserving rich vocabulary.
  - **Contextual Temperature**: Automatically scales between $T = 0.35$ for grounded retrieval synthesis and $T = 0.65$ for creative open-ended dialogue.
- **In-Memory Model Hot-Reloading (`/api/model/reload`)**:
  - Dynamically updates active ONNX sessions and invalidates encoder caches without restarting the FastAPI server or disconnecting active clients.
- **Unified Ingestion Pipeline (`scripts/ingest.py`)**:
  - Replaces legacy fragmented scripts with a single universal ingestion CLI supporting `.json`, `.csv`, `.txt`, `.md`, entire directories, batch files, and direct CLI text.
- **Live Learning Studio & Telemetry Dashboard (`server.py`)**:
  - Glassmorphic Web UI at `http://localhost:8000` with Server-Sent Events (SSE) streaming real-time loss/perplexity curves, token throughput (tok/s), cosine LR schedule tracking, on-the-go fine-tuning controls, and weight probing cards.
- **Affective Sentiment & Emotional State Tracker (`src/feeling.py`)**:
  - Dynamic valence tracking across dialogue turns, persisting emotional state to `workspace/mood_state.json` and adapting system persona empathy.
- **Mobile Edge Deployment Bundle (`scripts/prepare_mobile_bundle.py`)**:
  - Automated packager for INT8 quantized ONNX, BPE tokenizer, and semantic knowledge base for iOS CoreML / Android ONNX Runtime edge execution.

---

## 📊 Model Architecture Specifications

| Hyperparameter / Feature | Specification | Architectural Rationale |
| :--- | :--- | :--- |
| **Model Name** | `Kruuboo-SLM-0.2` (`tiny_llm`) | Custom decoder-only autoregressive transformer |
| **Total Parameters** | **105,426,432 (~105.4M)** | High-capacity small language model |
| **Layers ($N_{\text{layers}}$)** | **14 Transformer Blocks** | Deep hierarchical reasoning and dense representation |
| **Hidden Dim ($d_{\text{model}}$)** | **768** | High-capacity representation size |
| **Attention Heads ($N_{\text{heads}}$)**| **12 Heads** | Head dimension $d_{\text{head}} = 64$ ($768 / 12$) |
| **FFN Intermediate Dim ($d_{\text{ff}}$)**| **3,072** ($\approx \frac{2}{3} d_{\text{ff}} = 2048$ SwiGLU) | Swish-gated linear projections ($W_1, W_2, W_3$) |
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
    UserQuery([User Input / Message]) --> Mood[Sentiment & Affective State Tracker]
    UserQuery --> MathCheck{Arithmetic Query?}
    
    MathCheck -- Yes --> MathSolver[Direct Arithmetic Engine: 0ms Latency]
    MathSolver --> AssistantReply([Conversational Grounded Response])
    
    MathCheck -- No --> ContextResolver[Sliding Context & Coreference Resolver]
    ContextResolver --> NativeEncoder[Native SLM 768d Encoder: src/rag/native_encoder.py]
    NativeEncoder --> HybridSearch[Hybrid Semantic Search: 40% Lexical + 60% 768d Cosine]
    
    HybridSearch --> GlobalKB[(Global Semantic Memory: 750+ Docs)]
    GlobalKB --> ConfCheck{Confident Match? score >= 0.40}
    
    ConfCheck -- Yes --> RetrievedContext[Retrieved Grounded Evidence]
    ConfCheck -- No --> WebFallback[Multi-Provider Fallback Chain: Wolfram / DDG / Wiki / ArXiv]
    WebFallback --> RetrievedContext
    
    RetrievedContext --> ChatMLBuilder[ChatML Multi-Turn Prompt Builder]
    UserQuery --> ChatMLBuilder
    
    subgraph Kruuboo-SLM-0.2 Neural Engine (105.4M Parameters)
        ChatMLBuilder --> Tokenizer[BPE Tokenizer: 8,192 Vocab + ChatML]
        Tokenizer --> RoPE[RoPE Rotary Embeddings]
        RoPE --> Blocks[14x Transformer Blocks: FlashAttn + SwiGLU]
        Blocks --> Norm[Final LayerNorm & Tied LM Head]
        Norm --> MinP[Min-P Sampler: p_base = 0.05, Contextual Temp]
    end
    
    MinP --> AssistantReply
    
    subgraph Live Learning Studio & Hot-Reload Loop
        LiveMonitor[Web UI Live Visualizer] <-->|SSE Stream: /api/training/events| Trainer[SFT Trainer: Target-Only Loss Masking]
        Trainer -.->|Export ONNX| Disk[models/tiny_llm.onnx]
        Disk -.->|Hot Reload: /api/model/reload| Blocks
    end
```

---

## 🛡️ Core System Components

### 1. Neural Transformer (`src/model.py`)
- **RoPE & SwiGLU**: Implements rotary position embeddings and SwiGLU gating for fast convergence and high expressiveness.
- **Native Embedding Extraction**: Exposes `forward_features()` and `get_embeddings()` using post-LayerNorm hidden states with mean-pooling and L2 normalization for self-contained RAG retrieval.
- **Flash Attention**: Accelerated scaled dot-product attention kernels natively running on Apple Silicon (MPS), NVIDIA CUDA, and vectorized CPU.

### 2. Unified Native RAG & Hybrid Search (`src/rag/`)
- **`NativeSLMEncoder` (`src/rag/native_encoder.py`)**: Self-contained embedding retriever running on the same weights and 8,192 vocabulary as the language model.
- **`hybrid_search` (`src/rag/search.py`)**: Vectorized cosine similarity computation combined with robust lexical matching ($S = 0.40 S_{\text{lex}} + 0.60 S_{\text{sem}}$ with pure semantic fallback at $0.70 S_{\text{sem}}$).
- **Turn-by-Turn Sliding Context**: Dynamically slides backwards through dialogue history preserving most recent turns within token budget.
- **Hot-Reload Support**: `rag.reload_model()` updates ONNX sessions in memory without downtime.

### 3. Multi-Provider Web Research (`src/web_researcher.py`)
- Fallback chain configuration in `config.yaml`:
  - **Wolfram Alpha**: Math & scientific equations.
  - **DuckDuckGo**: Instant answers, definitions, and zero-click web summaries.
  - **Wikipedia**: In-depth encyclopedic context.
  - **ArXiv**: Academic papers and scientific research preprints.

### 4. Min-P Inference Engine (`src/inference.py`)
- Autoregressive generation with dynamic **Min-P sampling** ($p_{\text{base}} = 0.05$), repetition penalties ($1.15$), and real-time token streaming callbacks.

### 5. SFT Training & Loss Masking (`src/train.py`)
- **Target-Only Loss Masking**: Sets system and user token labels to `-100`, optimizing loss strictly over assistant response tokens.
- **Cosine Schedule with Linear Warmup**: Peak learning rate $\eta_{\text{max}} = 3\times 10^{-4}$ decaying to $\eta_{\text{min}} = 1\times 10^{-5}$ with AdamW decoupled weight decay ($0.1$).

### 6. Live Learning Studio & Web UI (`server.py`)
- **Conversational Chat Tab**: Clean, glassmorphic ChatGPT/Claude-style chat with markdown rendering, streaming responses, and source transparency.
- **Live Learning Monitor Tab**: Real-time SSE telemetry charts displaying training/validation loss, perplexity ($e^{\text{loss}}$), throughput (tok/s), and real-time probe generation cards.
- **Knowledge Upload**: Multi-format file uploader (`/upload`) parsing JSON/TXT datasets and creating chunked native embeddings on the fly.

---

## 📂 Directory Structure

```text
tiny_llm_rag_project/
├── config.yaml                     # Model hyperparams (105.4M, 14 layers, 1024 seq_len), RAG & training settings
├── requirements.txt                # Python dependencies
├── main.py                         # Unified conversational terminal CLI (SLM-0.2)
├── server.py                       # FastAPI server & Web UI (Chat + Live Learning Monitor)
├── create_everything.py            # Master bootstrap and end-to-end verification script
├── test_speed.py                   # Latency and token throughput benchmark
├── tokenizer.json                  # Byte-Pair Encoding (BPE) 8,192 vocabulary
│
├── docs/                           # Comprehensive technical documentation
│   ├── ARCHITECTURE.md             # In-depth architectural & mathematical specifications
│   ├── API_REFERENCE.md            # REST API endpoints, SSE streams, and payload specs
│   ├── INGESTION_GUIDE.md          # Unified ingestion CLI user guide & schemas
│   └── WEB_RESEARCH_GUIDE.md       # Fallback provider chain setup & query pipeline
│
├── src/
│   ├── model.py                    # 105.4M Transformer Decoder (RoPE, SwiGLU, FlashAttn, Native Embeddings)
│   ├── tokenizer.py                # BPE Subword Tokenizer (8,192 vocab, ChatML special tokens)
│   ├── inference.py                # Autoregressive generation with Min-P sampling and contextual temperature
│   ├── train.py                    # SFT training pipeline with target-only loss masking and step callbacks
│   ├── training_monitor.py         # SSE telemetry bus, background worker, and live model probing
│   ├── feeling.py                  # Sentiment analysis, affective state & mood tracker
│   ├── web_researcher.py           # Multi-provider fallback web researcher (Wolfram/DDG/Wiki/ArXiv)
│   ├── rstar.py                    # rStar mutual reasoning engine
│   └── rag/
│       ├── system.py               # Unified conversational RAG coordinator, math solver & hot-reloading
│       ├── native_encoder.py       # Self-contained 768d SLM dense vector encoder
│       ├── search.py               # Hybrid lexical + 768d vector cosine search
│       ├── greetings.py            # Conversational greetings, persona, and name memory resolution
│       └── document.py             # Document data structures and serialization
│
├── scripts/
│   ├── ingest.py                   # Unified Universal Ingestion CLI (.json, .csv, .txt, .md, directories)
│   ├── prepare_mobile_bundle.py    # Export self-contained INT8 bundle for iOS/Android
│   ├── generate_coherent.py        # ONNX command-line generation test utility
│   ├── extract_behavioral.py       # Behavioral dataset extractor from ZIP archives
│   ├── export_onnx.py              # Export PyTorch model to INT8 & FP16 ONNX format
│   ├── rstar_self_play.py          # Synthetic reasoning dataset generator via self-play
│   ├── download_datasets.py        # Curated dataset downloader and formatter
│   ├── download_nsfw_roleplay.py   # Conversational roleplay dataset downloader
│   ├── cleanup_knowledge.py        # Knowledge base deduplication and cleanup utility
│   ├── continual_daemon.py         # Background continual learning watcher
│   └── run_continual_learning.sh   # Continual learning daemon launcher script
│
├── hardware/
│   └── real_hardware.py            # Hardware detection (MPS, CUDA, CPU) and thermal telemetry
│
├── models/                         # PyTorch checkpoints and ONNX graphs (.pt, .onnx)
├── knowledge_base/                 # Global semantic documents and metadata (knowledge.json)
├── training_data/                  # SFT training datasets (.json, .csv, .txt)
└── workspace/                      # Telemetry logs, training metrics, and mood state
```

---

## 🛠️ Getting Started

### 1. Installation

Ensure Python 3.9+ is installed:
```bash
pip install -r requirements.txt
```

### 2. Master Bootstrap & Verification

To verify or initialize the complete system (tokenizer, models, knowledge base, ONNX graphs):
```bash
python3 create_everything.py
```

### 3. Run the Web Application & Live Learning Studio

Start the local server and web UI:
```bash
python3 server.py
```
Open **`http://localhost:8000`** in your browser:
- **Conversational Chat**: Interactive multi-turn chat with grounded semantic memory and instant arithmetic.
- **Live Learning Monitor**: Watch real-time loss and perplexity curves, monitor token speed, and launch on-the-go fine-tuning.
- **Knowledge Upload**: Drag-and-drop `.json`, `.csv`, `.txt`, or `.md` files to ingest on-the-fly.

### 4. Dynamic Architecture Profile Registry (`config/profiles.yaml`)

`config/profiles.yaml` is an open registry—not a hardcoded set of four profiles. `micro`, `small`, `medium`, and `large` ship as starter templates only. Any valid architecture can be defined in **$\le 3$ lines of YAML** or a **single CLI command**:

#### Option A: Add a Profile in $\le 3$ Lines of YAML (`config/profiles.yaml`)
```yaml
  mymodel:
    n_layers: 5
    d_model: 320
    n_heads: 5
```
*(Optional keys `n_kv_heads`, `d_ff`, `max_seq_len`, `vocab_size`, `checkpoint_dir`, and `artifact_path` are auto-derived if omitted.)*

#### Option B: Register a Profile via CLI (`--define-profile`)
```bash
python3 main.py --define-profile test6L --n-layers 6 --d-model 384 \
  --n-heads 6 --n-kv-heads 2 --d-ff 768 --max-seq-len 512 --vocab-size 4096
```
Auto-assigns `checkpoints/test6L/` and `artifacts/test6L_int8.onnx` and persists the definition into `config/profiles.yaml`.

#### Profile Validation Rules (Checked on Load & Registration)
| Constraint | Rule | Behavior on Violation |
| :--- | :--- | :--- |
| **Head Divisibility** | `d_model % n_heads == 0` | Raises `ValueError: d_model (...) must be divisible by n_heads (...)` |
| **GQA KV Divisibility** | `n_heads % n_kv_heads == 0` | Raises `ValueError: n_heads (...) must be divisible by n_kv_heads (...)` |
| **RoPE Parity** | `(d_model // n_heads) % 2 == 0` | Raises `ValueError` |
| **FFN Ratio** | `d_ff in [1.5 * d_model, 4 * d_model]` | Emits `UserWarning` (non-blocking) |
| **Vocabulary Range** | `vocab_size in [1024, 65536]` | Raises `ValueError` |
| **Context Window** | `max_seq_len in [64, 4096]` | Raises `ValueError` |
| **Parameter Budget** | `estimated_params in [0.5M, 500M]` | Raises `ValueError` |

#### Inspect Auto-Derived Roofline Targets (`--doctor`)
Estimates parameter count, INT8 ONNX size (`params × 1B × 1.15`), RAM budget (`×3` safety), and expected `tok/s` ranges without needing a trained checkpoint on disk:
```bash
python3 main.py --doctor --profile test6L
```

#### Train & Run Any Profile
```bash
# Train specific profile(s) or all flagged profiles
python3 train/all_profiles.py --profiles test6L
python3 train/all_profiles.py --all-flagged

# Run interactive CLI or single prompt with runtime overrides (no retraining needed)
python3 main.py --profile test6L
python3 main.py --profile test6L --temperature 0.4 --min-p 0.08 --max-tokens 100
```

### 5. Live Database Knowledge Connector (`src/rag/db_connector.py`)

Connect live SQLite, PostgreSQL, MySQL, or MongoDB databases directly to `RAGSystem` so runtime queries search live tables/collections alongside the vector/BM25 semantic index:

```bash
# Connect a local SQLite database at startup
python3 main.py --profile small --db path/to/company.db

# Connect PostgreSQL, MySQL, or MongoDB via connection URI
python3 main.py --profile small --db postgresql://user:pass@localhost:5432/mydb
python3 main.py --profile small --db mongodb://localhost:27017/knowledge_db
```
Or connect/inspect dynamically inside the interactive shell with `/db connect <uri>` and `/db list`.

### 6. Run the Interactive CLI

Launch the conversational terminal shell:
```bash
python3 main.py --profile small
```

#### Available CLI Commands:
* `/think [query]` — Execute rStar multi-candidate reasoning with mutual verification (Best-of-N).
* `/reason [query]` — Alias for `/think`.
* `/reasoning [on|off]` — Toggle persistent reasoning mode for all subsequent queries.
* `/clear` — Clear current conversation context memory.
* `/research [topic]` — Scrape the web fallback chain (Wolfram/DDG/Wiki/ArXiv) and ingest knowledge on-the-fly.
* `/add [text]` — Ingest custom text directly into global semantic memory.
* `/info` — View system architecture and memory statistics.
* `/help` — Show help menu.
* `/quit` — Exit the session.

---

## 📥 Unified Knowledge Ingestion (`scripts/ingest.py`)

Kruuboo-SLM-0.2 replaces fragmented ingestion scripts with a single universal ingestion CLI:

```bash
# Ingest single file
python3 scripts/ingest.py training_data/home_remedies.json

# Ingest all files in a directory
python3 scripts/ingest.py training_data/

# Ingest direct text from command line
python3 scripts/ingest.py --text "Cold brew coffee reduces acidity by 65 percent compared to hot brew." --source "CoffeeFacts"

# Batch ingestion from list
python3 scripts/ingest.py --batch training_data/home_remedies.json training_data/legal_advice.json
```

For complete usage, chunking options, and schemas, see [docs/INGESTION_GUIDE.md](docs/INGESTION_GUIDE.md).

---

## 📱 Mobile Bundle Packaging (`scripts/prepare_mobile_bundle.py`)

Prepare a self-contained bundle for edge deployment on iOS or Android:

```bash
python3 scripts/prepare_mobile_bundle.py
```

Produces `mobile_assets/` containing:
- `model.onnx` (~100 MB quantized INT8 graph)
- `tokenizer.json` (BPE 8,192 vocabulary)
- `knowledge_base.json` (Semantic index with native 768d embeddings)
- `config.yaml` & `mobile_manifest.json`

---

## ⚡ Speed & Latency Benchmarks

Kruuboo-SLM-0.2 includes a comprehensive profiling suite in [test_speed.py](file:///Users/sithija/Desktop/tiny_llm_rag_project/test_speed.py) measuring Time To First Token (TTFT), token generation throughput (tok/s) with full **Key-Value (KV) Caching**, native 768d embedding vs. search matching latency, process RAM / VRAM footprint, and an automated comparison between INT8 quantized vs. FP16/FP32 ONNX graphs:

| Performance & Latency Metric | INT8 Quantized (13.8M) | FP32 ONNX (13.8M) | Improvement vs. Original Baseline |
| :--- | :--- | :--- | :--- |
| **ONNX Graph Node Count** | **271 nodes** | **271 nodes** | **9.5x fewer nodes** (down from 2,568) |
| **Model Disk Footprint** | **17.3 MB** | 65.7 MB | **6.2x smaller** (down from 107.5 MB) |
| **Peak Memory (RAM RSS)** | **145.4 MB** | 404.6 MB | **6.7x less RAM** (down from 972 MB) |
| **Hardware VRAM / MPS** | **0 MB (Pure ONNX)** | **0 MB (Pure ONNX)** | Zero PyTorch runtime overhead |
| **Time To First Token (TTFT)** | **1.2 ms** | 2.0 ms | **13.8x faster** (down from 16.6 ms) |
| **Token Generation Speed (4D KV Cache)**| **2,627.2 tok/s** | 834.5 tok/s | **336.8x faster** (up from 7.8 tok/s) |
| **Raw Generation Latency** | **7.6 ms** | 24.0 ms | **16.9x faster** (down from 128.8 ms) |
| **Native 384d Embedding Latency** | **0.6 ms** | 1.6 ms | **60.7x faster** (down from 36.4 ms) |
| **Hybrid Search Matching (780 docs)** | **0.4 ms** | 0.4 ms | Vectorized BLAS / FAISS |
| **Total RAG Retrieval Latency** | **3.3 ms** | 4.4 ms | **3.7x faster** (down from 12.3 ms) |
| **End-to-End `rag.ask` Latency** | **28.0 ms** | 66.5 ms | **46.1x faster** (down from 1,290 ms) |

```bash
# Run comparative benchmark (INT8 Quantized vs. Standard FP16/FP32)
python3 test_speed.py

# Profile specific model with custom token budget and runs
python3 test_speed.py --model models/tiny_llm_quantized.onnx --tokens 40 --runs 5

# Custom benchmark query
python3 test_speed.py --query "What is contract consideration?"
```

---

## 🧪 Verified Capabilities

| Query Scenario | Example Input | Expected Output Behavior |
| :--- | :--- | :--- |
| **Casual Conversation** | `"Hi, how are you today?"` | Empathetic, fluent natural dialogue without domain switching. |
| **Identity & Memory** | `"My name is Sithija"` $\rightarrow$ `"What is my name?"` | Multi-turn contextual memory recall (*"Your name is Sithija"*). |
| **Direct Arithmetic** | `"Calculate 24 * 15 + 100"` | Deterministic exact answer (*"24 * 15 + 100 = 460"*) with 0ms latency. |
| **Medical Remedies** | `"What is the remedy for fever?"` | Grounded clinical advice retrieved from global semantic memory. |
| **Legal Agreements** | `"What is an NDA and what must it contain?"` | Grounded contract definitions and structural requirements. |
| **Survival Wilderness** | `"How to build a shelter in survival?"` | Actionable lean-to shelter construction guidelines. |
| **Web Research Fallback** | `"/research latest discoveries in quantum computing"` | Multi-provider fallback retrieval ingested into native memory. |

---

## 📖 Further Documentation

- **[Architecture Deep-Dive](docs/ARCHITECTURE.md)**: Mathematical formulations for RoPE, SwiGLU, Flash Attention, Native 768d RAG, and Min-P sampling.
- **[REST API & Telemetry Reference](docs/API_REFERENCE.md)**: Complete OpenAPI spec for FastAPI endpoints, SSE event streams, and payload formats.
- **[Ingestion Pipeline Guide](docs/INGESTION_GUIDE.md)**: Formats, chunking strategies, and batch workflows for `scripts/ingest.py`.
- **[Web Research Fallback Guide](docs/WEB_RESEARCH_GUIDE.md)**: Multi-provider search orchestration and query resolution.
