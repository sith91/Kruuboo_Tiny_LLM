# 📐 Kruuboo Tiny LLM (SLM-0.2) — System Architecture & Mathematical Foundations

This document provides a comprehensive technical breakdown of the architectural designs, mathematical formulations, and runtime systems implemented in **Kruuboo-SLM-0.2**.

---

## 1. High-Level Neural Architecture Overview

Kruuboo-SLM-0.2 is an autoregressive decoder-only transformer with **105,426,432 parameters (~105.4M)** engineered specifically for edge execution and self-contained semantic retrieval without external dependencies.

```text
[Input Tokens] (B, T)
       │
[Tied Token Embedding] (B, T, 768)  <─── Shared with LM Head
       │
       ▼
 ┌────────────────────────────────────────────────────────┐
 │ 14x Transformer Decoder Blocks                        │
 │                                                        │
 │   ┌──────────────────────────────────────────────────┐ │
 │   │ Pre-LayerNorm (d=768, eps=1e-5)                  │ │
 │   │ RoPE Frequency Embedding Computation (theta=10k) │ │
 │   │ 12-Head Flash Attention (Fused SDPA, head_dim=64)│ │
 │   │ Residual Connection (+ x)                        │ │
 │   └──────────────────────────────────────────────────┘ │
 │   ┌──────────────────────────────────────────────────┐ │
 │   │ Pre-LayerNorm (d=768, eps=1e-5)                  │ │
 │   │ SwiGLU Gated Feed-Forward (d_ff=3072)            │ │
 │   │ Residual Connection (+ x)                        │ │
 │   └──────────────────────────────────────────────────┘ │
 └────────────────────────────────────────────────────────┘
       │
       ▼
[Final LayerNorm (ln_f)] (B, T, 768)
       ├───► [Tied LM Head] ──► Logits (B, T, 8192) ──► Min-P Sampler ──► Next Token
       └───► [Native SLM RAG] ──► Mean-Pool + L2-Norm ──► 768d Dense Embedding Vector
```

---

## 2. Transformer Building Blocks & Mathematical Formulations

### 2.1. Rotary Position Embeddings (RoPE)
Instead of absolute positional lookup tables, Kruuboo-SLM-0.2 utilizes **Rotary Position Embeddings (RoPE)**. RoPE encodes position by rotating the query and key vectors in complex 2D planes, preserving relative distance decay across arbitrarily long contexts.

For a head vector $x \in \mathbb{R}^{d_{\text{head}}}$ at sequence index $m$, coordinate pairs $(x_{2i}, x_{2i+1})$ are rotated by angle $m \theta_i$:

$$\theta_i = 10000^{-2(i-1)/d_{\text{head}}}, \quad i \in \left\{1, 2, \dots, \frac{d_{\text{head}}}{2}\right\}$$

$$\begin{pmatrix} R_{\Theta, m}^d x_{2i} \\ R_{\Theta, m}^d x_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{pmatrix} \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}$$

This enables seamless extrapolation to the full $1,024$-token context window without training separate position parameters.

### 2.2. SwiGLU Gated Feed-Forward Networks
Kruuboo replaces standard GELU/ReLU multi-layer perceptrons with **SwiGLU** (Swish Gated Linear Unit), providing superior representational density per parameter:

$$\text{SwiGLU}(x) = W_2 \Big(\text{SiLU}(W_1 x) \odot W_3 x\Big)$$

where:
- $W_1 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ (Gate projection)
- $W_3 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}$ (Up projection)
- $W_2 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}$ (Down projection)
- $\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$
- Intermediate dimension $d_{\text{ff}} = 3,072$

### 2.3. Flash Attention via Fused Scaled Dot-Product Attention (SDPA)
Attention heads compute causal self-attention using PyTorch 2.0+ `torch.nn.functional.scaled_dot_product_attention`:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_{\text{head}}}} + M_{\text{causal}}\right) V$$

- $N_{\text{heads}} = 12$
- $d_{\text{head}} = 768 / 12 = 64$
- Hardware acceleration leverages Apple Silicon Metal Performance Shaders (MPS), CUDA Tensor Cores, and AVX-512 vectorization on CPU without materializing the full $O(T^2)$ attention matrix in high-bandwidth memory.

### 2.4. Tied Embeddings & Residual Scaling
To maximize parameter efficiency:
- `lm_head.weight` is tied directly to `token_embedding.weight` ($8192 \times 768$), saving $6.29\text{M}$ duplicate parameters.
- Residual projection layers ($W_{\text{proj}}$ and $W_2$) are initialized with GPT-2 style scaled standard deviation:

$$\sigma_{\text{init}} = \frac{0.02}{\sqrt{2 \cdot N_{\text{layers}}}} = \frac{0.02}{\sqrt{28}} \approx 0.00378$$

preventing variance explosion as the network depth reaches 14 layers.

### 2.5. Unified 5D Key-Value (KV) Caching & $\mathcal{O}(1)$ Decoding
To eliminate the quadratic $\mathcal{O}(N^2)$ computational overhead of autoregressive token generation, Kruuboo implements unified 5D Key-Value Caching across both PyTorch and ONNX Runtime:

$$\text{Cache Shape: } [N_{\text{layers}}, B, N_{\text{heads}}, T_{\text{past}}, d_{\text{head}}] = [14, 1, 12, T_{\text{past}}, 64]$$

1. **Prefill Phase ($T_{\text{past}} = 0$)**: The full prompt $X_{\text{prompt}} \in \mathbb{R}^{1 \times T}$ is processed once through all 14 blocks. Rotary embeddings are applied at base offsets $0 \dots T-1$, and initial key and value matrices are returned as `present_keys` and `present_values`.
2. **Decode Phase ($T = 1$)**: For each generated token $x_{t}$, the model executes with sequence length 1. RoPE is evaluated with explicit position offset $T_{\text{past}}$:

$$\begin{pmatrix} R_{\Theta, T_{\text{past}}}^d q_{2i} \\ R_{\Theta, T_{\text{past}}}^d q_{2i+1} \end{pmatrix}$$

3. **Dynamic Attention Concat**: The single new key $k_t$ and value $v_t$ are concatenated onto the cached tensors along dimension $-2$, and attention is computed across all $T_{\text{past}} + 1$ keys without re-evaluating preceding hidden layers, reducing per-token generation complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(1)$.

---

## 3. Native 768d Semantic RAG Encoder

Rather than loading a secondary 400 MB model from Hugging Face (such as `all-mpnet-base-v2` or `bge-small`), Kruuboo-SLM-0.2 extracts embeddings **directly from its own transformer decoder**.

### 3.1. Embedding Extraction Pipeline
Given text input $T$:
1. Tokenize text using the shared 8,192 BPE vocabulary.
2. Pass token IDs through all 14 transformer blocks without causal masking on the final representation.
3. Extract hidden states after final LayerNorm: $H \in \mathbb{R}^{B \times L \times 768}$.
4. Compute attention-masked mean pooling:

$$v_{\text{pooled}} = \frac{\sum_{i=1}^L m_i \cdot H_i}{\sum_{i=1}^L m_i}$$

5. Compute L2 normalization:

$$e_{\text{doc}} = \frac{v_{\text{pooled}}}{\|v_{\text{pooled}}\|_2 + \epsilon}$$

The resulting unit vector $e_{\text{doc}} \in \mathbb{R}^{768}$ shares the exact same semantic latent space as the generation model.

### 3.2. Hybrid Semantic Retrieval Scoring
During query resolution, Kruuboo computes a weighted hybrid score combining lexical token overlap with 768d cosine similarity:

$$S(q, d) = \begin{cases} 
0.40 \cdot S_{\text{lex}}(q, d) + 0.60 \cdot S_{\text{sem}}(q, d), & \text{if } S_{\text{lex}}(q, d) > 0 \\
0.70 \cdot S_{\text{sem}}(q, d), & \text{if } S_{\text{lex}}(q, d) = 0
\end{cases}$$

where:
- $S_{\text{sem}}(q, d) = \cos(e_q, e_d) = \frac{e_q \cdot e_d}{\|e_q\| \|e_d\|}$
- $S_{\text{lex}}(q, d) = \frac{|Q_{\text{tokens}} \cap D_{\text{tokens}}|}{|Q_{\text{tokens}}|}$ (with stopword filtering and keyword overlap)
- Confidence cutoff: $S(q, d) \ge 0.40$. Queries matching below 0.40 trigger the multi-provider web fallback chain.

---

## 4. Modern Min-P Sampling Formulation

Traditional Top-P (nucleus) sampling struggles when small language models generate tokens with flat probability tails. Kruuboo-SLM-0.2 implements **Min-P Sampling**:

Given logits $z$, convert to probability distribution $P = \text{softmax}(z / T)$ and find top probability:

$$p_{\text{max}} = \max_i P_i$$

Define the dynamic threshold:

$$p_{\text{thresh}} = p_{\text{max}} \cdot p_{\text{base}}, \quad (p_{\text{base}} = 0.05)$$

Discard all candidate tokens $j$ where $P_j < p_{\text{thresh}}$ and renormalize remaining tokens:

$$\tilde{P}_k = \frac{P_k}{\sum_{j \in \mathcal{K}} P_j}, \quad \mathcal{K} = \{i \mid P_i \ge p_{\text{thresh}}\}$$

### Advantages:
1. **Adaptive candidate set size**: When top confidence is very high ($p_{\text{max}} = 0.90$), only tokens with $p \ge 0.045$ survive, preventing hallucinations.
2. **Preserves vocabulary diversity**: When confidence is distributed across multiple plausible choices ($p_{\text{max}} = 0.20$), tokens with $p \ge 0.01$ remain eligible, avoiding robotic repetition.

---

## 5. SFT ChatML Training & Target-Only Loss Masking

Kruuboo-SLM-0.2 trains on conversational multi-turn dialogues formatted using the ChatML schema:

```text
<|im_start|>system
You are Kruuboo SLM, a helpful AI assistant.<|im_end|>
<|im_start|>user
What is an NDA?<|im_end|>
<|im_start|>assistant
A Non-Disclosure Agreement (NDA) is a legally binding contract...<|im_end|>
```

### Target-Only Loss Masking
Standard cross-entropy loss trains models to predict both user queries and assistant responses. To focus the parameter budget strictly on response quality, labels for system and user tokens are masked with $-100$:

$$\mathcal{L} = -\frac{1}{|\mathcal{T}_{\text{resp}}|} \sum_{t \in \mathcal{T}_{\text{resp}}} \log P(x_t \mid x_{<t})$$

$$\text{Label}_t = \begin{cases} x_t, & \text{if } t \text{ is an assistant token} \\ -100, & \text{otherwise (system, user, headers)} \end{cases}$$

PyTorch's `nn.CrossEntropyLoss(ignore_index=-100)` ignores masked indices, dramatically improving convergence efficiency.

---

## 6. Model Hot-Reload Architecture

To enable live fine-tuning in the web studio without taking down the production server:

```text
[Trainer Thread] ──► Exports ONNX ──► Disk: models/tiny_llm.onnx
                                                │
[POST /api/model/reload] ────────────────────────┘
       │
       ▼
[RAGSystem.hot_reload_model()]
       ├── 1. Creates new onnxruntime.InferenceSession
       ├── 2. Replaces active InferenceEngine reference atomically
       └── 3. Invalidates native encoder cache
```

Zero connections are dropped, and all subsequent `/ask` queries instantly execute against the updated weights.
