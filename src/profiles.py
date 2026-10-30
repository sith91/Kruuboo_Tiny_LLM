"""
Dynamic Architecture Profile Registry & Roofline Estimator (`src/profiles.py`)
==============================================================================

Manages open-ended model profiles from `config/profiles.yaml`:
  1. Open Registry: Any valid transformer architecture can be defined in YAML or via `--define-profile`.
  2. Strict Build-Time Validation: Validates head divisibility, GQA `n_kv_heads` divisibility,
     RoPE even head dimension, `vocab_size`, `max_seq_len`, `d_ff` warnings, and parameter bounds.
  3. Build-Time vs. Runtime Split: Separates immutable checkpoint architecture dimensions from
     per-invocation sampling/RAG runtime flags.
  4. Roofline Auto-Derived Targets (`doctor()`): Analytically computes exact parameter counts,
     INT8/FP32 ONNX sizes, RAM safety budgets, and expected token/sec throughput ranges without
     requiring a trained checkpoint on disk.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml


PROFILES_YAML_PATH = Path("config/profiles.yaml")

BUILD_TIME_KEYS = {
    "n_layers",
    "d_model",
    "n_heads",
    "n_kv_heads",
    "d_ff",
    "max_seq_len",
    "vocab_size",
    "dropout",
    "checkpoint_dir",
    "onnx_path",
    "artifact_path",
    "tokenizer_path",
    "flagged",
    "description",
}

RUNTIME_KEYS = {
    "temperature",
    "min_p",
    "top_k",
    "top_p",
    "max_tokens",
    "repetition_penalty",
    "use_rag",
    "rag_k",
    "reasoning",
}

DEFAULT_RUNTIME = {
    "temperature": 0.65,
    "min_p": 0.05,
    "top_k": 0,
    "top_p": 1.0,
    "max_tokens": 120,
    "repetition_penalty": 1.15,
    "use_rag": True,
    "rag_k": 3,
    "reasoning": False,
}


def estimate_parameters(
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    d_ff: int,
    vocab_size: int,
) -> int:
    """
    Calculate the exact trainable parameter count of TinyLLM for the given dimensions:
      - Tied token embedding / lm_head: vocab_size * d_model
      - Per block (n_layers):
          * ln1 (weight + bias): 2 * d_model
          * attn.qkv (no bias, GQA-aware): d_model * ((n_heads + 2 * n_kv_heads) * head_dim)
          * attn.proj (no bias): d_model * d_model
          * ln2 (weight + bias): 2 * d_model
          * SwiGLU FFN (w1, w2, w3, no bias): 3 * d_model * swiglu_hidden
      - Final ln_f (weight + bias): 2 * d_model
    """
    head_dim = d_model // max(1, n_heads)
    swiglu_hidden = int(d_ff * 2 / 3)
    swiglu_hidden = ((swiglu_hidden + 63) // 64) * 64

    embedding_params = vocab_size * d_model
    qkv_params = d_model * ((n_heads + 2 * n_kv_heads) * head_dim)
    proj_params = d_model * d_model
    ffn_params = 3 * d_model * swiglu_hidden
    ln_block_params = 4 * d_model  # ln1 + ln2

    per_layer_params = qkv_params + proj_params + ffn_params + ln_block_params
    final_ln_params = 2 * d_model

    return embedding_params + n_layers * per_layer_params + final_ln_params


def derive_targets(
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    d_ff: int,
    max_seq_len: int,
    vocab_size: int,
) -> Dict[str, Any]:
    """
    Auto-derive parameter count, ONNX INT8/FP32 sizes, RAM budget, and roofline tok/s ranges.
    Formula:
      - INT8 ONNX size ≈ params * 1 byte * 1.15 overhead
      - Peak RAM budget ≈ INT8 size * 3 safety factor
      - Roofline tok/s derived from effective memory bandwidth & per-token compute/byte load
    """
    params = estimate_parameters(n_layers, d_model, n_heads, n_kv_heads, d_ff, vocab_size)
    int8_bytes = params * 1.0 * 1.15
    fp32_bytes = params * 4.0 * 1.03

    int8_mb = int8_bytes / (1024 * 1024)
    fp32_mb = fp32_bytes / (1024 * 1024)
    ram_mb = int8_mb * 3.0

    # Simple memory-bandwidth roofline model for single-stream autoregressive decoding:
    # Effective SIMD/NEON CPU bandwidth on modern laptops/edge ~ 18 GB/s (low) to 42 GB/s (high)
    # adjusted for KV cache and layer dispatch overhead.
    dispatch_overhead_ms = 0.035 * n_layers
    bw_low_bytes_per_ms = 16.0 * 1e6   # 16 GB/s conservative
    bw_high_bytes_per_ms = 40.0 * 1e6  # 40 GB/s fast SIMD / Apple Silicon

    ms_per_tok_int8_low = (int8_bytes / bw_high_bytes_per_ms) + dispatch_overhead_ms
    ms_per_tok_int8_high = (int8_bytes / bw_low_bytes_per_ms) + dispatch_overhead_ms * 1.6

    tok_s_int8_min = round(1000.0 / max(0.05, ms_per_tok_int8_high), 1)
    tok_s_int8_max = round(1000.0 / max(0.05, ms_per_tok_int8_low), 1)

    ms_per_tok_fp32_low = (fp32_bytes / bw_high_bytes_per_ms) + dispatch_overhead_ms
    ms_per_tok_fp32_high = (fp32_bytes / bw_low_bytes_per_ms) + dispatch_overhead_ms * 1.6

    tok_s_fp32_min = round(1000.0 / max(0.1, ms_per_tok_fp32_high), 1)
    tok_s_fp32_max = round(1000.0 / max(0.1, ms_per_tok_fp32_low), 1)

    return {
        "estimated_params": params,
        "estimated_params_m": round(params / 1e6, 2),
        "onnx_int8_mb": round(int8_mb, 2),
        "onnx_fp32_mb": round(fp32_mb, 2),
        "ram_budget_mb": round(ram_mb, 2),
        "tok_s_int8_range": (tok_s_int8_min, tok_s_int8_max),
        "tok_s_fp32_range": (tok_s_fp32_min, tok_s_fp32_max),
    }


def validate_profile_dims(
    name: str,
    n_layers: int,
    d_model: int,
    n_heads: int,
    n_kv_heads: int,
    d_ff: int,
    max_seq_len: int,
    vocab_size: int,
) -> Dict[str, Any]:
    """
    Validate profile dimensions against all architectural constraints.
    Raises ValueError naming the exact constraint violated.
    Emits a UserWarning (non-blocking) if d_ff is outside [1.5 * d_model, 4 * d_model].
    """
    if n_layers < 1:
        raise ValueError(f"n_layers ({n_layers}) must be >= 1")
    if d_model < 16 or n_heads < 1:
        raise ValueError(f"d_model ({d_model}) must be >= 16 and n_heads ({n_heads}) must be >= 1")
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    head_dim = d_model // n_heads
    if head_dim % 2 != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be even for RoPE: d_model ({d_model}) / n_heads ({n_heads}) = {head_dim}"
        )

    if n_kv_heads < 1 or n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

    if not (1024 <= vocab_size <= 65536):
        raise ValueError(f"vocab_size ({vocab_size}) must be in [1024, 65536]")

    if not (64 <= max_seq_len <= 4096):
        raise ValueError(f"max_seq_len ({max_seq_len}) must be in [64, 4096]")

    min_dff = 1.5 * d_model
    max_dff = 4.0 * d_model
    if not (min_dff <= d_ff <= max_dff):
        warnings.warn(
            f"Profile '{name}': d_ff ({d_ff}) is outside recommended range "
            f"[{int(min_dff)}, {int(max_dff)}] ([1.5×d_model, 4×d_model]). Proceeding anyway.",
            UserWarning,
            stacklevel=2,
        )

    targets = derive_targets(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
    )
    est_params = targets["estimated_params"]
    if not (500_000 <= est_params <= 500_000_000):
        raise ValueError(
            f"estimated params ({est_params:,} / {est_params/1e6:.2f}M) must be in [0.5M, 500M]"
        )

    return targets


@dataclass
class ModelProfile:
    """Validated model profile combining immutable build-time specs, auto-derived paths, and runtime defaults."""
    name: str
    n_layers: int
    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    max_seq_len: int
    vocab_size: int
    dropout: float = 0.05
    description: str = ""
    flagged: bool = False
    checkpoint_dir: str = ""
    onnx_path: str = ""
    artifact_path: str = ""
    tokenizer_path: str = ""
    runtime: Dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_RUNTIME))
    targets: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.checkpoint_dir:
            self.checkpoint_dir = f"checkpoints/{self.name}"
        if not self.onnx_path:
            self.onnx_path = f"artifacts/{self.name}.onnx"
        if not self.artifact_path:
            self.artifact_path = f"artifacts/{self.name}_int8.onnx"
        if not self.tokenizer_path:
            self.tokenizer_path = f"{self.checkpoint_dir}/tokenizer.json"

        self.targets = validate_profile_dims(
            name=self.name,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            vocab_size=self.vocab_size,
        )

    @property
    def checkpoint_file(self) -> str:
        return str(Path(self.checkpoint_dir) / "tiny_llm.pt")

    def resolve_tokenizer_path(self) -> str:
        if Path(self.tokenizer_path).exists():
            return self.tokenizer_path
        return "tokenizer.json"

    def resolve_onnx_artifact(self) -> str:
        """Return the profile's INT8 ONNX path, or FP32 ONNX path if INT8 is absent."""
        if Path(self.artifact_path).exists():
            return self.artifact_path
        if Path(self.onnx_path).exists():
            return self.onnx_path
        return self.artifact_path

    def with_runtime_overrides(self, **overrides) -> "ModelProfile":
        """Apply per-invocation runtime overrides without mutating build-time architecture."""
        merged_rt = dict(self.runtime)
        for k, v in overrides.items():
            if v is not None:
                if k in BUILD_TIME_KEYS:
                    raise ValueError(
                        f"Cannot override build-time key '{k}' at runtime. "
                        f"Use --define-profile to create a new profile and train it."
                    )
                merged_rt[k] = v
        clone = ModelProfile(
            name=self.name,
            n_layers=self.n_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_ff=self.d_ff,
            max_seq_len=self.max_seq_len,
            vocab_size=self.vocab_size,
            dropout=self.dropout,
            description=self.description,
            flagged=self.flagged,
            checkpoint_dir=self.checkpoint_dir,
            onnx_path=self.onnx_path,
            artifact_path=self.artifact_path,
            tokenizer_path=self.tokenizer_path,
            runtime=merged_rt,
        )
        return clone

    def to_yaml_dict(self) -> Dict[str, Any]:
        """Serialize profile definition for persistence in config/profiles.yaml."""
        d: Dict[str, Any] = {
            "description": self.description or f"Custom {self.targets['estimated_params_m']}M profile ({self.n_layers}L, {self.d_model}d, {self.n_heads}H/{self.n_kv_heads}KV)",
            "n_layers": int(self.n_layers),
            "d_model": int(self.d_model),
            "n_heads": int(self.n_heads),
            "n_kv_heads": int(self.n_kv_heads),
            "d_ff": int(self.d_ff),
            "max_seq_len": int(self.max_seq_len),
            "vocab_size": int(self.vocab_size),
            "checkpoint_dir": self.checkpoint_dir,
            "artifact_path": self.artifact_path,
            "flagged": bool(self.flagged),
        }
        return d


class ProfileRegistry:
    """Loads, validates, queries, and persists model profiles in `config/profiles.yaml`."""

    def __init__(self, registry_path: Union[str, Path] = PROFILES_YAML_PATH):
        self.registry_path = Path(registry_path)
        self.default_profile: str = "small"
        self.runtime_defaults: Dict[str, Any] = dict(DEFAULT_RUNTIME)
        self.profiles: Dict[str, ModelProfile] = {}
        self.load()

    def load(self) -> None:
        if not self.registry_path.exists():
            self.profiles = {}
            return

        with open(self.registry_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self.default_profile = raw.get("default_profile", "small")
        rt_defs = raw.get("runtime_defaults", {})
        self.runtime_defaults = {**DEFAULT_RUNTIME, **rt_defs}

        raw_profiles = raw.get("profiles", {})
        loaded: Dict[str, ModelProfile] = {}
        for name, cfg in raw_profiles.items():
            if not isinstance(cfg, dict):
                continue
            loaded[name] = self._parse_profile(name, cfg)
        self.profiles = loaded

    def _parse_profile(self, name: str, cfg: Dict[str, Any]) -> ModelProfile:
        n_layers = int(cfg["n_layers"])
        d_model = int(cfg["d_model"])
        n_heads = int(cfg["n_heads"])
        n_kv_heads = int(cfg.get("n_kv_heads", n_heads))
        d_ff = int(cfg.get("d_ff", 2 * d_model))
        max_seq_len = int(cfg.get("max_seq_len", 512))
        vocab_size = int(cfg.get("vocab_size", 8192))
        dropout = float(cfg.get("dropout", 0.05))

        rt = dict(self.runtime_defaults)
        if isinstance(cfg.get("runtime"), dict):
            rt.update(cfg["runtime"])
        for rk in RUNTIME_KEYS:
            if rk in cfg:
                rt[rk] = cfg[rk]

        return ModelProfile(
            name=name,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            d_ff=d_ff,
            max_seq_len=max_seq_len,
            vocab_size=vocab_size,
            dropout=dropout,
            description=cfg.get("description", ""),
            flagged=bool(cfg.get("flagged", False)),
            checkpoint_dir=cfg.get("checkpoint_dir", f"checkpoints/{name}"),
            onnx_path=cfg.get("onnx_path", f"artifacts/{name}.onnx"),
            artifact_path=cfg.get("artifact_path", f"artifacts/{name}_int8.onnx"),
            tokenizer_path=cfg.get("tokenizer_path", f"checkpoints/{name}/tokenizer.json"),
            runtime=rt,
        )

    def get(self, name: Optional[str] = None) -> ModelProfile:
        target = name or self.default_profile
        if target not in self.profiles:
            available = ", ".join(sorted(self.profiles.keys()))
            raise KeyError(
                f"Profile '{target}' not found in {self.registry_path}. "
                f"Available profiles: [{available}]. "
                f"Define it in {self.registry_path} or run: python main.py --define-profile {target} ..."
            )
        return self.profiles[target]

    def list_names(self) -> List[str]:
        return list(self.profiles.keys())

    def get_flagged(self) -> List[ModelProfile]:
        return [p for p in self.profiles.values() if p.flagged]

    def register_profile(
        self,
        name: str,
        n_layers: int,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        d_ff: Optional[int] = None,
        max_seq_len: int = 512,
        vocab_size: int = 8192,
        description: str = "",
        flagged: bool = True,
        persist: bool = True,
    ) -> ModelProfile:
        """
        Validate and register a new or updated profile in the registry.
        Persists to `config/profiles.yaml` when `persist=True`.
        """
        resolved_kv = n_kv_heads if n_kv_heads is not None else n_heads
        resolved_dff = d_ff if d_ff is not None else (2 * d_model)

        profile = ModelProfile(
            name=name,
            n_layers=int(n_layers),
            d_model=int(d_model),
            n_heads=int(n_heads),
            n_kv_heads=int(resolved_kv),
            d_ff=int(resolved_dff),
            max_seq_len=int(max_seq_len),
            vocab_size=int(vocab_size),
            description=description,
            flagged=flagged,
            checkpoint_dir=f"checkpoints/{name}",
            onnx_path=f"artifacts/{name}.onnx",
            artifact_path=f"artifacts/{name}_int8.onnx",
            tokenizer_path=f"checkpoints/{name}/tokenizer.json",
            runtime=dict(self.runtime_defaults),
        )
        self.profiles[name] = profile

        if persist:
            self.save()

        return profile

    def save(self) -> None:
        """Persist current registry back to `config/profiles.yaml`."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        raw: Dict[str, Any] = {}
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
            except Exception:
                raw = {}

        raw["default_profile"] = self.default_profile
        raw["runtime_defaults"] = self.runtime_defaults
        if "profiles" not in raw or not isinstance(raw["profiles"], dict):
            raw["profiles"] = {}

        for name, prof in self.profiles.items():
            raw["profiles"][name] = prof.to_yaml_dict()

        with open(self.registry_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(raw, f, sort_keys=False, default_flow_style=False)


def doctor(profile_name: Optional[str] = None, registry_path: Union[str, Path] = PROFILES_YAML_PATH) -> Dict[str, Any]:
    """
    Inspect a profile (or all profiles) and print auto-derived targets:
      - Estimated parameters
      - Estimated ONNX INT8 & FP32 sizes
      - Estimated RAM budget (×3 safety)
      - Expected tok/s roofline ranges
      - Artifact & checkpoint status on disk (without requiring them to exist)
    """
    reg = ProfileRegistry(registry_path)
    target_profiles = [reg.get(profile_name)] if profile_name else list(reg.profiles.values())

    print("\n" + "=" * 78)
    print("🩺 KRUUBOO SLM — PROFILE DOCTOR & ROOFLINE ESTIMATOR")
    print("=" * 78)

    reports = {}
    for p in target_profiles:
        t = p.targets
        ckpt_exists = Path(p.checkpoint_file).exists()
        int8_exists = Path(p.artifact_path).exists()
        actual_int8_mb = (
            round(Path(p.artifact_path).stat().st_size / (1024 * 1024), 2)
            if int8_exists else None
        )

        head_dim = p.d_model // p.n_heads
        gqa_ratio = f"{p.n_heads}:{p.n_kv_heads} (GQA {p.n_heads // p.n_kv_heads}x)" if p.n_kv_heads < p.n_heads else f"{p.n_heads}:{p.n_kv_heads} (MHA)"

        print(f"\n📌 Profile: '{p.name}' — {p.description or 'Custom Architecture'}")
        print("-" * 78)
        print(f"  • Build-Time Dims    : n_layers={p.n_layers}, d_model={p.d_model}, d_ff={p.d_ff}")
        print(f"  • Attention Heads    : n_heads={p.n_heads}, n_kv_heads={p.n_kv_heads} [{gqa_ratio}, head_dim={head_dim}]")
        print(f"  • Context & Vocab    : max_seq_len={p.max_seq_len}, vocab_size={p.vocab_size:,}")
        print(f"  • Estimated Params   : {t['estimated_params']:,} ({t['estimated_params_m']:.2f} M)")
        print(f"  • Est. ONNX INT8 Size: {t['onnx_int8_mb']:.2f} MB  (params × 1B × 1.15 overhead)"
              + (f"  [Actual on disk: {actual_int8_mb:.2f} MB]" if actual_int8_mb is not None else ""))
        print(f"  • Est. ONNX FP32 Size: {t['onnx_fp32_mb']:.2f} MB")
        print(f"  • Est. Peak RAM (×3) : {t['ram_budget_mb']:.2f} MB")
        print(f"  • Expected Throughput: {t['tok_s_int8_range'][0]:,.0f} – {t['tok_s_int8_range'][1]:,.0f} tok/s (INT8) | "
              f"{t['tok_s_fp32_range'][0]:,.0f} – {t['tok_s_fp32_range'][1]:,.0f} tok/s (FP32)")
        print(f"  • Paths              : {p.checkpoint_dir}/ | {p.artifact_path}")
        print(f"  • Disk Status        : Checkpoint={'✅ READY' if ckpt_exists else '⏳ NOT TRAINED'} | "
              f"INT8 ONNX={'✅ READY' if int8_exists else '⏳ NOT EXPORTED'}")

        reports[p.name] = {
            "profile": p.name,
            "dims": {
                "n_layers": p.n_layers,
                "d_model": p.d_model,
                "n_heads": p.n_heads,
                "n_kv_heads": p.n_kv_heads,
                "d_ff": p.d_ff,
                "max_seq_len": p.max_seq_len,
                "vocab_size": p.vocab_size,
            },
            **t,
            "checkpoint_ready": ckpt_exists,
            "artifact_ready": int8_exists,
        }

    print("\n" + "=" * 78)
    return reports

