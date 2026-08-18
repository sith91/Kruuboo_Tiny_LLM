"""Export trained TinyLLM to ONNX — synced with upgraded model architecture."""

import torch
import onnx
import onnxruntime as ort
from pathlib import Path
import sys
sys.path.append('.')

from src.model import TinyLLM
from src.tokenizer import TinyTokenizer

# ── Model dimensions — loaded dynamically from config.yaml ───────────────────
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

D_MODEL     = config["model"]["d_model"]
N_HEADS     = config["model"]["n_heads"]
N_LAYERS    = config["model"]["n_layers"]
D_FF        = config["model"]["d_ff"]
MAX_SEQ_LEN = config["model"]["max_seq_len"]



def export_to_onnx(
    model_path:     str = "models/tiny_llm.pt",
    tokenizer_path: str = "tokenizer.json",
    output_path:    str = "models/tiny_llm.onnx",
) -> tuple:
    """
    Load trained weights, export to ONNX, verify, and produce an INT8-quantised
    variant for mobile/edge deployment.

    Returns
    -------
    (onnx_path, quantized_path)
    """

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print("📦 Loading tokenizer ...")
    tokenizer = TinyTokenizer.load(tokenizer_path)
    vocab_size = tokenizer.vocab_size
    print(f"   vocab_size = {vocab_size}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("🧠 Building model ...")
    model = TinyLLM(
        vocab_size   = vocab_size,
        d_model      = D_MODEL,
        n_heads      = N_HEADS,
        n_layers     = N_LAYERS,
        d_ff         = D_FF,
        max_seq_len  = MAX_SEQ_LEN,
    )
    print(f"   Parameters: {model.num_parameters():,}  ({model.num_parameters()/1e6:.2f} M)")

    # ── Load weights ──────────────────────────────────────────────────────────
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Model weights not found at '{model_path}'.\n"
            "Run:  python3 src/train.py"
        )
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print(f"   Weights loaded from {model_path}")

    # ── Dummy input for tracing ───────────────────────────────────────────────
    # Use a short sequence so the export is fast; dynamic_axes handles the rest.
    dummy_seq_len = 16
    dummy_input   = torch.randint(0, vocab_size, (1, dummy_seq_len))

    # ── Export ────────────────────────────────────────────────────────────────
    print(f"\n⚙️  Exporting to ONNX  →  {output_path} ...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params        = True,
        opset_version        = 17,          # 17 supports scaled_dot_product_attn
        do_constant_folding  = True,
        input_names          = ["input_ids"],
        output_names         = ["logits"],
        dynamic_axes         = {
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits":    {0: "batch_size", 1: "seq_len"},
        },
    )
    print(f"✅ ONNX model exported")

    # ── Verify ────────────────────────────────────────────────────────────────
    print("🔍 Verifying ONNX model ...")
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model check passed")

    # Quick runtime check
    sess  = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])
    import numpy as np
    out   = sess.run(None, {"input_ids": dummy_input.numpy().astype(np.int64)})
    assert out[0].shape == (1, dummy_seq_len, vocab_size), "Shape mismatch!"
    print(f"✅ Runtime output shape: {out[0].shape}")

    # ── INT8 quantisation (for mobile / edge) ─────────────────────────────────
    print("\n📉 Quantising to INT8 ...")
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_path = output_path.replace(".onnx", "_quantized.onnx")
    quantize_dynamic(output_path, quantized_path, weight_type=QuantType.QInt8)

    orig_kb = Path(output_path).stat().st_size   / 1024
    quant_kb= Path(quantized_path).stat().st_size/ 1024
    print(f"   Original  : {orig_kb:>8.1f} KB")
    print(f"   Quantised : {quant_kb:>8.1f} KB  "
          f"({orig_kb/quant_kb:.1f}× smaller)")
    print(f"✅ Quantised model saved to {quantized_path}")

    return output_path, quantized_path


if __name__ == "__main__":
    onnx_path, quant_path = export_to_onnx()
    print(f"\n🎉 Done!\n   ONNX      : {onnx_path}\n   Quantised : {quant_path}")