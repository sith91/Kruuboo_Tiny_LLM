"""
Kruuboo Tiny LLM — Comprehensive Inference, RAG & Latency Benchmark Suite
========================================================================

Profiles real-time performance of Kruuboo-SLM-0.2:
  1. Time To First Token (TTFT in milliseconds)
  2. End-to-End Latency and Generation Throughput (tokens/sec)
  3. RAG Embedding Retrieval Latency (native 768d encoder vs. search matching)
  4. Peak Memory Footprint (Process RAM RSS & Hardware VRAM / MPS)
  5. Summary breakdown comparing ONNX FP16/FP32 vs INT8 quantized execution
"""

import os
import sys
import time
import re
import argparse
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root in python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml
import psutil
import numpy as np
import onnxruntime as ort

from src.rag import RAGSystem
from src.rag.search import hybrid_search
from src.tokenizer import TinyTokenizer

# Suppress warnings for clean benchmark output
import warnings
warnings.filterwarnings("ignore")


def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) in Megabytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_vram_mb() -> Tuple[Optional[float], str]:
    """
    Returns GPU / Unified VRAM usage if CUDA or Apple Silicon MPS is available.
    Returns (memory_in_mb, device_type_string).
    """
    try:
        import torch
        if torch.cuda.is_available():
            vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
            return vram, f"CUDA ({device_name})"
        if hasattr(torch, "mps") and torch.backends.mps.is_available() and hasattr(torch.mps, "current_allocated_memory"):
            vram = torch.mps.current_allocated_memory() / (1024 * 1024)
            return vram, "Apple Silicon (MPS Unified)"
    except Exception:
        pass
    return None, "CPU-only (Host RAM)"


def format_ms(val: float) -> str:
    return f"{val:,.1f} ms"


def format_tok_s(val: float) -> str:
    return f"{val:,.1f} tok/s"


def format_mb(val: float) -> str:
    return f"{val:,.1f} MB"


class SpeedBenchmark:
    """Benchmarking coordinator for Kruuboo SLM ONNX models and RAG pipeline."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f) or {}
        self.max_seq_len = self.cfg.get("model", {}).get("max_seq_len", 1024)
        self.tokenizer_path = self.cfg.get("storage", {}).get("tokenizer_path", "tokenizer.json")

    def profile_model(
        self,
        model_path: str,
        query: str = "What is machine learning?",
        max_tokens: int = 35,
        runs: int = 3,
        run_rag: bool = True,
    ) -> Dict:
        """Profiles a single ONNX model across TTFT, generation speed, retrieval, and memory."""
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"Model not found at: {model_path}")

        file_size_mb = p.stat().st_size / (1024 * 1024)
        mem_before = get_process_memory_mb()

        # 1. Initialize RAG system with target model
        init_t0 = time.perf_counter()
        rag = RAGSystem(
            model_path=model_path,
            tokenizer_path=self.tokenizer_path,
            max_seq_len=self.max_seq_len,
        )
        init_latency_ms = (time.perf_counter() - init_t0) * 1000
        mem_loaded = get_process_memory_mb()

        # Warmup run to prime ONNX session, PyTorch MPS shader compilation, and CPU caches
        rag.generate(query, max_tokens=10)
        if run_rag and len(rag.documents) > 0:
            _ = rag.encoder.encode(query, show_progress_bar=False)

        # 2. Benchmark Raw Autoregressive Generation (without RAG overhead)
        ttft_measurements: List[float] = []
        gen_throughput_measurements: List[float] = []
        gen_latency_measurements: List[float] = []
        tokens_generated_counts: List[int] = []

        raw_prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        for _ in range(runs):
            first_token_time: Optional[float] = None

            def token_callback(_t: str):
                nonlocal first_token_time
                if first_token_time is None:
                    first_token_time = time.perf_counter()

            t_start = time.perf_counter()
            gen_text = rag.generate(
                raw_prompt,
                max_tokens=max_tokens,
                temperature=0.0,  # greedy for deterministic timing
                on_token=token_callback,
            )
            t_end = time.perf_counter()

            # Calculate TTFT
            if first_token_time is not None:
                ttft_ms = (first_token_time - t_start) * 1000
            else:
                ttft_ms = (t_end - t_start) * 1000
            ttft_measurements.append(ttft_ms)

            gen_latency_ms = (t_end - t_start) * 1000
            gen_latency_measurements.append(gen_latency_ms)

            # Count generated tokens
            gen_tokens = len(rag.tokenizer.encode(gen_text)) if gen_text else 1
            tokens_generated_counts.append(gen_tokens)

            gen_time_s = max(1e-6, t_end - t_start)
            tok_per_sec = gen_tokens / gen_time_s
            gen_throughput_measurements.append(tok_per_sec)

        # 3. Benchmark RAG Retrieval Components (Native 768d Encoder vs Search Matching)
        emb_latencies_ms: List[float] = []
        search_matching_latencies_ms: List[float] = []
        total_search_latencies_ms: List[float] = []
        ask_latencies_ms: List[float] = []

        if run_rag and len(rag.documents) > 0:
            for _ in range(runs):
                # A. Native 768d Vector Encoding
                t0_emb = time.perf_counter()
                query_emb = np.array(rag.encoder.encode(query, show_progress_bar=False))
                if query_emb.ndim > 1:
                    query_emb = query_emb.squeeze(0)
                t1_emb = time.perf_counter()
                emb_latencies_ms.append((t1_emb - t0_emb) * 1000)

                # B. Pure Accelerated Search Matching across indexed chunks (FAISS IndexFlatIP + Pre-cached Tokens)
                t0_match = time.perf_counter()
                raw_tokens = set(re.findall(r'\b[a-zA-Z0-9_\-]+\b', query.lower()))
                from src.rag.search import STOPWORDS
                keywords = {w for w in (raw_tokens - STOPWORDS) if not (w.isdigit() and len(w) <= 2) and len(w) > 1}
                vec_scores = rag.vector_index.compute_similarities(query_emb)
                num_kw = len(keywords)
                scored = []
                for i, doc in enumerate(rag.vector_index.documents):
                    doc_words = getattr(doc, '_words', None)
                    if doc_words is None:
                        doc_words = set(re.findall(r'\w+', doc.text.lower()))
                        doc._words = doc_words
                    overlap = len(keywords & doc_words) if num_kw > 0 else 0
                    kw_score = (overlap / num_kw) if num_kw > 0 else 0.0
                    raw_v = float(vec_scores[i]) if i < len(vec_scores) else 0.0
                    normalized_vec = max(0.0, min(1.0, raw_v))
                    h_score = (0.40 * kw_score + 0.60 * normalized_vec) if kw_score > 0 else (0.70 * normalized_vec)
                    if h_score >= 0.30:
                        scored.append((doc, h_score))
                scored.sort(key=lambda x: x[1], reverse=True)
                t1_match = time.perf_counter()
                search_matching_latencies_ms.append((t1_match - t0_match) * 1000)

                # C. Total RAG Retrieval (Encoding + Search Combined)
                t0_search = time.perf_counter()
                _ = rag.search(query, k=3)
                t1_search = time.perf_counter()
                total_search_latencies_ms.append((t1_search - t0_search) * 1000)

                # D. Full End-to-End rag.ask Pipeline (Retrieval + Prompt Assembly + Generation)
                t0_ask = time.perf_counter()
                _ = rag.ask(query, auto_fallback=False)
                t1_ask = time.perf_counter()
                ask_latencies_ms.append((t1_ask - t0_ask) * 1000)

        mem_peak = get_process_memory_mb()
        vram_mb, vram_type = get_vram_mb()

        return {
            "model_path": model_path,
            "model_name": Path(model_path).name,
            "file_size_mb": file_size_mb,
            "init_latency_ms": init_latency_ms,
            "mem_baseline_mb": mem_before,
            "mem_loaded_mb": mem_loaded,
            "mem_peak_mb": mem_peak,
            "mem_delta_mb": mem_peak - mem_before,
            "vram_mb": vram_mb,
            "vram_type": vram_type,
            "ttft_ms_mean": float(np.mean(ttft_measurements)),
            "ttft_ms_std": float(np.std(ttft_measurements)),
            "tok_s_mean": float(np.mean(gen_throughput_measurements)),
            "tok_s_std": float(np.std(gen_throughput_measurements)),
            "gen_latency_ms_mean": float(np.mean(gen_latency_measurements)),
            "tokens_generated_mean": int(np.mean(tokens_generated_counts)),
            "emb_latency_ms_mean": float(np.mean(emb_latencies_ms)) if emb_latencies_ms else 0.0,
            "search_matching_latency_ms_mean": float(np.mean(search_matching_latencies_ms)) if search_matching_latencies_ms else 0.0,
            "total_search_latency_ms_mean": float(np.mean(total_search_latencies_ms)) if total_search_latencies_ms else 0.0,
            "ask_latency_ms_mean": float(np.mean(ask_latencies_ms)) if ask_latencies_ms else 0.0,
            "doc_count": len(rag.documents),
        }

    def print_comparison_table(self, res_quant: Dict, res_fp: Optional[Dict] = None):
        """Renders an aligned comparison table between quantized and unquantized models."""
        print("\n" + "=" * 94)
        print("⚡ KRUUBOO TINY LLM (SLM-0.2) — INFERENCE & RAG BENCHMARK REPORT")
        print("=" * 94)
        print(f"🖥️  System Platform  : {platform.system()} {platform.machine()} ({platform.processor() or 'Apple Silicon'})")
        print(f"🐍 Python Runtime   : {platform.python_version()} | ONNX Runtime: {ort.__version__}")
        print(f"🧠 Model Architecture: 105.4M Parameters (14 Layers, 768d, 12 Heads, RoPE, SwiGLU)")
        print(f"📚 Knowledge Base   : {res_quant.get('doc_count', 0):,} Indexed Semantic Chunks")
        print("-" * 94)

        has_both = res_fp is not None

        if has_both:
            headers = ["Performance & Latency Metric", "INT8 Quantized", "FP16 / FP32 ONNX", "Improvement / Delta"]
            row_fmt = "{:<36} {:<16} {:<18} {:<18}"
        else:
            headers = ["Performance & Latency Metric", "Measurement Value"]
            row_fmt = "{:<48} {:<30}"

        print(row_fmt.format(*headers))
        print("-" * 94)

        rows = []

        # Model File Size
        if has_both:
            ratio = res_fp["file_size_mb"] / max(1e-3, res_quant["file_size_mb"])
            rows.append(("Model Disk Footprint", format_mb(res_quant["file_size_mb"]), format_mb(res_fp["file_size_mb"]), f"{ratio:.1f}x smaller"))
        else:
            rows.append(("Model Disk Footprint", format_mb(res_quant["file_size_mb"])))

        # Process Peak RAM (RSS)
        if has_both:
            mem_ratio = res_fp["mem_peak_mb"] / max(1e-3, res_quant["mem_peak_mb"])
            rows.append(("Peak Memory (RAM RSS)", format_mb(res_quant["mem_peak_mb"]), format_mb(res_fp["mem_peak_mb"]), f"{mem_ratio:.1f}x less RAM"))
        else:
            rows.append(("Peak Memory (RAM RSS)", format_mb(res_quant["mem_peak_mb"])))

        # Hardware VRAM / Device Memory
        vram_val = res_quant.get("vram_mb")
        vram_type = res_quant.get("vram_type", "Unified Memory")
        vram_str = f"{format_mb(vram_val)} ({vram_type})" if vram_val is not None and vram_val > 0 else f"{vram_type}"
        if has_both:
            fp_vram = res_fp.get("vram_mb")
            fp_vram_str = f"{format_mb(fp_vram)} ({vram_type})" if fp_vram is not None and fp_vram > 0 else f"{vram_type}"
            rows.append(("Hardware VRAM / MPS", vram_str, fp_vram_str, "Shared / Unified"))
        else:
            rows.append(("Hardware VRAM / MPS", vram_str))

        # Time To First Token (TTFT)
        if has_both:
            speedup_ttft = res_fp["ttft_ms_mean"] / max(1e-3, res_quant["ttft_ms_mean"])
            rows.append(("Time To First Token (TTFT)", format_ms(res_quant["ttft_ms_mean"]), format_ms(res_fp["ttft_ms_mean"]), f"{speedup_ttft:.1f}x faster"))
        else:
            rows.append(("Time To First Token (TTFT)", format_ms(res_quant["ttft_ms_mean"])))

        # Token Generation Speed
        if has_both:
            speedup_tok = res_quant["tok_s_mean"] / max(1e-3, res_fp["tok_s_mean"])
            rows.append(("Token Generation Speed", format_tok_s(res_quant["tok_s_mean"]), format_tok_s(res_fp["tok_s_mean"]), f"{speedup_tok:.1f}x faster"))
        else:
            rows.append(("Token Generation Speed", format_tok_s(res_quant["tok_s_mean"])))

        # Raw Generation Latency
        if has_both:
            speedup_gen = res_fp["gen_latency_ms_mean"] / max(1e-3, res_quant["gen_latency_ms_mean"])
            rows.append(("Raw Generation Latency", format_ms(res_quant["gen_latency_ms_mean"]), format_ms(res_fp["gen_latency_ms_mean"]), f"{speedup_gen:.1f}x faster"))
        else:
            rows.append(("Raw Generation Latency", format_ms(res_quant["gen_latency_ms_mean"])))

        # Dense 768d Embedding Latency (Query Vector Encoding)
        if res_quant.get("emb_latency_ms_mean", 0) > 0:
            if has_both:
                rows.append(("Native 768d Embedding Latency", format_ms(res_quant["emb_latency_ms_mean"]), format_ms(res_fp["emb_latency_ms_mean"]), "Identical (PyTorch)"))
            else:
                rows.append(("Native 768d Embedding Latency", format_ms(res_quant["emb_latency_ms_mean"])))

        # Hybrid Search Matching Latency (Cosine + BM25 across docs)
        if res_quant.get("search_matching_latency_ms_mean", 0) > 0:
            doc_cnt = res_quant.get('doc_count', 0)
            label = f"Search Matching Latency ({doc_cnt} docs)"
            if has_both:
                rows.append((label, format_ms(res_quant["search_matching_latency_ms_mean"]), format_ms(res_fp["search_matching_latency_ms_mean"]), "-"))
            else:
                rows.append((label, format_ms(res_quant["search_matching_latency_ms_mean"])))

        # Total RAG Retrieval Latency (Encoder + Search)
        if res_quant.get("total_search_latency_ms_mean", 0) > 0:
            if has_both:
                rows.append(("Total RAG Retrieval Latency", format_ms(res_quant["total_search_latency_ms_mean"]), format_ms(res_fp["total_search_latency_ms_mean"]), "Encoder + Search"))
            else:
                rows.append(("Total RAG Retrieval Latency", format_ms(res_quant["total_search_latency_ms_mean"])))

        # End-to-End RAG Latency (rag.ask)
        if res_quant.get("ask_latency_ms_mean", 0) > 0:
            if has_both:
                speedup_ask = res_fp["ask_latency_ms_mean"] / max(1e-3, res_quant["ask_latency_ms_mean"])
                rows.append(("End-to-End rag.ask Latency", format_ms(res_quant["ask_latency_ms_mean"]), format_ms(res_fp["ask_latency_ms_mean"]), f"{speedup_ask:.1f}x faster"))
            else:
                rows.append(("End-to-End rag.ask Latency", format_ms(res_quant["ask_latency_ms_mean"])))

        for r in rows:
            print(row_fmt.format(*r))

        print("=" * 94)

        # Print Executive Summary Insights
        if has_both:
            print("\n📌 EXECUTIVE BENCHMARK INSIGHTS:")
            print(f"  • Edge Optimization : INT8 dynamic quantization yields a {res_fp['file_size_mb'] / res_quant['file_size_mb']:.1f}x disk reduction ({res_quant['file_size_mb']:.1f} MB vs {res_fp['file_size_mb']:.1f} MB).")
            print(f"  • RAM Footprint     : INT8 consumes {res_fp['mem_peak_mb'] / res_quant['mem_peak_mb']:.1f}x less peak RSS ({res_quant['mem_peak_mb']:.1f} MB vs {res_fp['mem_peak_mb']:.1f} MB).")
            print(f"  • Responsiveness    : INT8 achieves {speedup_ttft:.1f}x faster Time To First Token ({format_ms(res_quant['ttft_ms_mean'])} vs {format_ms(res_fp['ttft_ms_mean'])}).")
            print(f"  • Generation Speed  : INT8 delivers {speedup_tok:.1f}x higher token throughput ({format_tok_s(res_quant['tok_s_mean'])} vs {format_tok_s(res_fp['tok_s_mean'])}).")
            if res_quant.get("emb_latency_ms_mean", 0) > 0 and res_quant.get("search_matching_latency_ms_mean", 0) > 0:
                sum_retrieval = res_quant['emb_latency_ms_mean'] + res_quant['search_matching_latency_ms_mean']
                emb_pct = (res_quant['emb_latency_ms_mean'] / sum_retrieval) * 100
                search_pct = (res_quant['search_matching_latency_ms_mean'] / sum_retrieval) * 100
                print(f"  • Retrieval Split   : Native 768d encoding accounts for {emb_pct:.1f}% ({format_ms(res_quant['emb_latency_ms_mean'])}), while hybrid search matching accounts for {search_pct:.1f}% ({format_ms(res_quant['search_matching_latency_ms_mean'])}).")
            print("=" * 94 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Kruuboo SLM-0.2 Inference & Latency Benchmark")
    parser.add_argument("--model", type=str, default=None, help="Path to specific ONNX model (optional)")
    parser.add_argument("--query", type=str, default="What is machine learning?", help="Benchmark evaluation query")
    parser.add_argument("--tokens", type=int, default=30, help="Number of tokens to generate")
    parser.add_argument("--runs", type=int, default=3, help="Benchmark iterations for statistical stability")
    parser.add_argument("--no-rag", action="store_true", help="Skip RAG retrieval benchmarks")
    args = parser.parse_args()

    bench = SpeedBenchmark()

    quant_path = "models/tiny_llm_quantized.onnx"
    fp_path = "models/tiny_llm.onnx"

    if args.model:
        print(f"📊 Profiling target model: {args.model}")
        res = bench.profile_model(
            model_path=args.model,
            query=args.query,
            max_tokens=args.tokens,
            runs=args.runs,
            run_rag=not args.no_rag,
        )
        bench.print_comparison_table(res)
    else:
        # Default: auto-detect both INT8 and FP models and render comparative report
        has_quant = Path(quant_path).exists()
        has_fp = Path(fp_path).exists()

        if has_quant and has_fp:
            print(f"🚀 Profiling INT8 Quantized Model ({quant_path})...")
            res_quant = bench.profile_model(
                model_path=quant_path,
                query=args.query,
                max_tokens=args.tokens,
                runs=args.runs,
                run_rag=not args.no_rag,
            )

            print(f"🚀 Profiling Standard FP Model ({fp_path})...")
            res_fp = bench.profile_model(
                model_path=fp_path,
                query=args.query,
                max_tokens=args.tokens,
                runs=args.runs,
                run_rag=not args.no_rag,
            )

            bench.print_comparison_table(res_quant, res_fp)

        elif has_quant:
            res_quant = bench.profile_model(quant_path, query=args.query, max_tokens=args.tokens, runs=args.runs, run_rag=not args.no_rag)
            bench.print_comparison_table(res_quant)
        elif has_fp:
            res_fp = bench.profile_model(fp_path, query=args.query, max_tokens=args.tokens, runs=args.runs, run_rag=not args.no_rag)
            bench.print_comparison_table(res_fp)
        else:
            print("❌ No ONNX models found in models/. Run 'python3 scripts/export_onnx.py' first.")


if __name__ == "__main__":
    main()
