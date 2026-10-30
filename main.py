"""
Main CLI — Profile-Aware Conversational Shell & Profile Manager for Kruuboo SLM
===============================================================================

Supports:
  1. Dynamic Profile Selection:
       python main.py --profile small
  2. Defining & Registering Custom Architecture Profiles:
       python main.py --define-profile test6L --n-layers 6 --d-model 384 \\
         --n-heads 6 --n-kv-heads 2 --d-ff 768 --max-seq-len 512 --vocab-size 4096
  3. Roofline Doctor & Target Estimator (no checkpoint needed on disk):
       python main.py --doctor --profile test6L
  4. Per-Invocation Runtime Overrides:
       python main.py --profile small --temperature 0.4 --min-p 0.08 --max-tokens 100
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.profiles import ModelProfile, ProfileRegistry, doctor
from src.rag import RAGSystem


class ConversationalCLI:
    """Streamlined interactive conversational shell bound to an active ModelProfile."""

    def __init__(self, rag_system: RAGSystem, profile: Optional[ModelProfile] = None):
        self.rag = rag_system
        self.profile = profile
        self.history = []
        self.reasoning_mode = bool(profile.runtime.get("reasoning", False)) if profile else False

    def show_header(self):
        print("\n" + "=" * 70)
        if self.profile:
            p = self.profile
            params_m = p.targets["estimated_params_m"]
            print(f"🧠 KRUUBOO TINY LLM — PROFILE: '{p.name}' ({params_m:.2f}M Params)")
            print(
                f"   Arch: {p.n_layers}L | {p.d_model}d | {p.n_heads}H ({p.n_kv_heads} KV) | "
                f"d_ff={p.d_ff} | ctx={p.max_seq_len} | vocab={p.vocab_size}"
            )
            print(f"   Artifact: {p.resolve_onnx_artifact()}")
        else:
            print("🧠 KRUUBOO TINY LLM (SLM-0.2) — CONVERSATIONAL SHELL")
        print("=" * 70)
        stats = self.rag.get_stats()
        dbs = stats.get("connected_databases", [])
        db_str = f" | Live DBs: {len(dbs)}" if dbs else ""
        print(f"📚 Global Memory: {stats['total_documents']:,} indexed knowledge documents{db_str}")
        print("Type your message naturally or use commands like /help or /quit.\n")

    def show_help(self):
        print("\n" + "-" * 70)
        print("COMMANDS:")
        print("  /think [query]      - Deep rStar reasoning trajectory (Best-of-N self-play)")
        print("  /reason [query]     - Alias for /think")
        print("  /reasoning [on|off] - Toggle persistent reasoning mode for all messages")
        print("  /research [topic]   - Scrape the web and ingest knowledge on-the-fly")
        print("  /add [text]         - Ingest custom text into global semantic memory")
        print("  /db [connect|list]  - Connect or inspect live SQL/MongoDB knowledge sources")
        print("  /clear              - Clear dialogue context history")
        print("  /info               - Show active profile architecture & memory stats")
        print("  /help               - Show this help menu")
        print("  /quit               - Exit")
        print("-" * 70)

    def run(self):
        self.show_header()
        self.show_help()

        while True:
            try:
                mode_badge = " [🧠 Reasoning]" if self.reasoning_mode else ""
                user_input = input(f"\n💬 You{mode_badge}: ").strip()
                if not user_input:
                    continue

                # Handle slash commands
                if user_input.startswith("/"):
                    parts = user_input.split(maxsplit=1)
                    cmd = parts[0].lower()

                    if cmd in ("/quit", "/exit"):
                        print("👋 Goodbye!")
                        break

                    elif cmd == "/help":
                        self.show_help()
                        continue

                    elif cmd == "/clear":
                        self.history = []
                        print("🧹 Conversation memory cleared.")
                        continue

                    elif cmd == "/info":
                        stats = self.rag.get_stats()
                        print("\n📊 SYSTEM METRICS:")
                        if self.profile:
                            p = self.profile
                            print(
                                f"  • Profile     : {p.name} ({p.targets['estimated_params_m']:.2f}M Parameters)"
                            )
                            print(
                                f"  • Architecture: {p.n_layers} Layers, {p.n_heads} Heads "
                                f"({p.n_kv_heads} KV Heads), d_model={p.d_model}, d_ff={p.d_ff}"
                            )
                        print(f"  • Context Memory : {self.rag.max_seq_len} Tokens")
                        print(f"  • Semantic Memory: {stats['total_documents']} Documents")
                        print(f"  • Reasoning Mode : {'ENABLED 🧠' if self.reasoning_mode else 'DISABLED'}")
                        continue

                    elif cmd in ("/think", "/reason") and len(parts) > 1:
                        target_query = parts[1]
                        print("\n🤔 [rStar Mutual Reasoning Active]")
                        response = self.rag.ask(
                            target_query,
                            history=self.history,
                            use_reasoning=True,
                            on_reasoning_step=lambda step_msg: print(f"  {step_msg}"),
                        )
                        self._render_response(target_query, response)
                        continue

                    elif cmd == "/reasoning":
                        if len(parts) > 1:
                            sub = parts[1].strip().lower()
                            if sub in ("on", "true", "1", "enable"):
                                self.reasoning_mode = True
                            elif sub in ("off", "false", "0", "disable"):
                                self.reasoning_mode = False
                        else:
                            self.reasoning_mode = not self.reasoning_mode
                        state = "ON 🧠" if self.reasoning_mode else "OFF"
                        print(f"⚙️ Persistent reasoning mode: {state}")
                        continue

                    elif cmd == "/research" and len(parts) > 1:
                        topic = parts[1]
                        self.rag.research_and_learn(topic)
                        continue

                    elif cmd == "/add" and len(parts) > 1:
                        text = parts[1]
                        self.rag.add_text(text, source="CLI Input")
                        print("✅ Added to global semantic memory.")
                        continue

                    elif cmd == "/db":
                        sub_parts = parts[1].split(maxsplit=1) if len(parts) > 1 else ["list"]
                        sub_cmd = sub_parts[0].lower()
                        if sub_cmd == "connect" and len(sub_parts) > 1:
                            uri = sub_parts[1].strip()
                            src = self.rag.connect_db(uri)
                            print(
                                f"✅ Connected live DB '{src.name}' ({src.engine_type}) — "
                                f"Tables/Collections: {list(src.schema.keys())}"
                            )
                        elif sub_cmd == "disconnect" and len(sub_parts) > 1:
                            name = sub_parts[1].strip()
                            if self.rag.disconnect_db(name):
                                print(f"🔌 Disconnected live DB '{name}'.")
                            else:
                                print(f"❌ Database '{name}' not found.")
                        else:
                            dbs = self.rag.list_databases()
                            if not dbs:
                                print("ℹ️ No live databases connected. Use: /db connect <sqlite_path_or_db_uri>")
                            else:
                                print("\n🗄️ CONNECTED LIVE DATABASES:")
                                for d in dbs:
                                    print(f"  • {d['name']} ({d['engine']}) -> {d['uri']}")
                                    for tbl, cols in d["tables"].items():
                                        print(f"      - {tbl}: {', '.join(cols)}")
                        continue

                # Process conversational message
                print("\n🤖 Kruuboo SLM: ", end="", flush=True)
                response = self.rag.ask(
                    user_input,
                    history=self.history,
                    use_reasoning=self.reasoning_mode,
                    on_reasoning_step=lambda step_msg: print(f"\n  {step_msg}") if self.reasoning_mode else None,
                )
                self._render_response(user_input, response)

            except (KeyboardInterrupt, EOFError):
                print("\n👋 Exiting...")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

    def _render_response(self, user_msg: str, response: Dict):
        answer = response.get("answer", "")
        print(answer)

        # Update multi-turn history
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": answer})

        if response.get("reasoning"):
            r_info = response["reasoning"]
            print(f"\n   🧠 [rStar Trajectory Verified — Score: {r_info['score']:.2f}]")
            for step in r_info.get("steps", []):
                stype = step["type"].capitalize()
                print(f"      • {stype} (confidence {step['confidence']}): {step['thought']}")

        if response.get("knowledge_used", 0) > 0:
            if response.get("web_research_used"):
                print(
                    f"\n   [🌐 Grounded via Multi-Provider Web Research Fallback ({response['knowledge_used']} findings)]"
                )
            else:
                print(
                    f"\n   [📚 Grounded on {response['knowledge_used']} retrieved semantic memory chunks]"
                )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kruuboo Tiny LLM — Dynamic Profile Registry, Doctor & Conversational CLI"
    )

    # Profile selection & management
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Profile name to load from config/profiles.yaml (default: registry default_profile)",
    )
    parser.add_argument(
        "--define-profile",
        type=str,
        metavar="NAME",
        default=None,
        help="Validate and register a new profile in config/profiles.yaml",
    )
    parser.add_argument(
        "--doctor",
        action="store_true",
        help="Run roofline estimator & profile health check (does not require checkpoint on disk)",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all profiles currently registered in config/profiles.yaml",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default="config/profiles.yaml",
        help="Path to profile registry YAML (default: config/profiles.yaml)",
    )

    # Build-time architecture flags (used with --define-profile)
    parser.add_argument("--n-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--d-model", type=int, default=None, help="Hidden representation dimension")
    parser.add_argument("--n-heads", type=int, default=None, help="Number of query attention heads")
    parser.add_argument("--n-kv-heads", type=int, default=None, help="Number of key/value heads (GQA/MQA)")
    parser.add_argument("--d-ff", type=int, default=None, help="Feed-forward intermediate dimension")
    parser.add_argument("--max-seq-len", type=int, default=None, help="Maximum context sequence length")
    parser.add_argument("--vocab-size", type=int, default=None, help="BPE vocabulary size")
    parser.add_argument("--description", type=str, default="", help="Optional description for --define-profile")
    parser.add_argument(
        "--train-now",
        action="store_true",
        help="Immediately start training after --define-profile registration",
    )

    # Runtime flags (overridable per invocation without retraining)
    parser.add_argument("--temperature", type=float, default=None, help="Sampling temperature override")
    parser.add_argument("--min-p", type=float, default=None, help="Min-P sampling threshold override")
    parser.add_argument("--top-k", type=int, default=None, help="Top-K sampling override")
    parser.add_argument("--top-p", type=float, default=None, help="Top-P nucleus sampling override")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max new tokens to generate")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty override")
    parser.add_argument("--no-rag", action="store_true", help="Disable RAG retrieval for this session")
    parser.add_argument("--rag-k", type=int, default=None, help="Number of RAG chunks to retrieve")
    parser.add_argument("--reasoning", action="store_true", help="Enable rStar reasoning mode")
    parser.add_argument(
        "--db",
        action="append",
        default=[],
        metavar="URI",
        help="Connect live database(s) as RAG knowledge sources (SQLite path, postgresql://, mysql://, mongodb://)",
    )
    parser.add_argument("--prompt", type=str, default=None, help="Run a single prompt and exit")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # 1. Handle --define-profile
    if args.define_profile:
        missing = [
            flag for flag, val in [
                ("--n-layers", args.n_layers),
                ("--d-model", args.d_model),
                ("--n-heads", args.n_heads),
            ]
            if val is None
        ]
        if missing:
            raise ValueError(
                f"--define-profile '{args.define_profile}' requires {', '.join(missing)}"
            )

        registry = ProfileRegistry(args.registry)
        profile = registry.register_profile(
            name=args.define_profile,
            n_layers=args.n_layers,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_kv_heads=args.n_kv_heads if args.n_kv_heads is not None else args.n_heads,
            d_ff=args.d_ff if args.d_ff is not None else (2 * args.d_model),
            max_seq_len=args.max_seq_len if args.max_seq_len is not None else 512,
            vocab_size=args.vocab_size if args.vocab_size is not None else 8192,
            description=args.description,
            flagged=True,
            persist=True,
        )
        t = profile.targets
        print(f"✅ Registered profile '{profile.name}' in {args.registry}")
        print(f"   • Architecture  : {profile.n_layers}L, d_model={profile.d_model}, "
              f"n_heads={profile.n_heads}, n_kv_heads={profile.n_kv_heads}, "
              f"d_ff={profile.d_ff}, max_seq_len={profile.max_seq_len}, vocab_size={profile.vocab_size}")
        print(f"   • Est. Params   : {t['estimated_params']:,} ({t['estimated_params_m']:.2f} M)")
        print(f"   • Est. INT8 Size: {t['onnx_int8_mb']:.2f} MB | Est. RAM: {t['ram_budget_mb']:.2f} MB")
        print(f"   • Auto Paths    : {profile.checkpoint_dir}/ | {profile.artifact_path}")
        print(f"\n👉 To train and export this profile, run:\n   python train/all_profiles.py --profiles {profile.name}")

        if args.train_now:
            from train.all_profiles import train_single_profile
            train_single_profile(profile)
        sys.exit(0)

    # Guard against passing build-time flags without --define-profile
    build_flags_passed = [
        flag for flag, val in [
            ("--n-layers", args.n_layers),
            ("--d-model", args.d_model),
            ("--n-heads", args.n_heads),
            ("--n-kv-heads", args.n_kv_heads),
            ("--d-ff", args.d_ff),
            ("--max-seq-len", args.max_seq_len),
            ("--vocab-size", args.vocab_size),
        ]
        if val is not None
    ]
    if build_flags_passed:
        raise ValueError(
            f"Build-time architecture flags ({', '.join(build_flags_passed)}) cannot be overridden at runtime. "
            f"Use `--define-profile <name>` to register a new profile first."
        )

    # 2. Handle --doctor
    if args.doctor:
        doctor(profile_name=args.profile, registry_path=args.registry)
        sys.exit(0)

    # 3. Handle --list-profiles
    registry = ProfileRegistry(args.registry)
    if args.list_profiles:
        print(f"\n📋 Registered Profiles in {args.registry} (default: {registry.default_profile}):")
        for name in registry.list_names():
            p = registry.get(name)
            t = p.targets
            ready = "✅" if Path(p.artifact_path).exists() else "⏳"
            print(
                f"  {ready} {name:12s} | {t['estimated_params_m']:6.2f}M params | "
                f"{p.n_layers:2d}L {p.d_model:4d}d {p.n_heads:2d}H/{p.n_kv_heads:2d}KV | "
                f"INT8 ~{t['onnx_int8_mb']:5.1f} MB | {p.artifact_path}"
            )
        sys.exit(0)

    # 4. Load active profile & apply per-invocation runtime overrides
    base_profile = registry.get(args.profile)
    active_profile = base_profile.with_runtime_overrides(
        temperature=args.temperature,
        min_p=args.min_p,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
        use_rag=(False if args.no_rag else None),
        rag_k=args.rag_k,
        reasoning=(True if args.reasoning else None),
    )

    # Resolve ONNX artifact & tokenizer paths (with fallback to models/tiny_llm_quantized.onnx for default 'small')
    onnx_artifact = active_profile.resolve_onnx_artifact()
    if not Path(onnx_artifact).exists():
        if active_profile.name == "small" and Path("models/tiny_llm_quantized.onnx").exists():
            onnx_artifact = "models/tiny_llm_quantized.onnx"
        else:
            print(
                f"❌ Profile '{active_profile.name}' artifact not found at '{onnx_artifact}'.\n"
                f"👉 Train and export it first with:\n"
                f"   python train/all_profiles.py --profiles {active_profile.name}"
            )
            sys.exit(1)

    tok_path = active_profile.resolve_tokenizer_path()

    rag = RAGSystem(
        model_path=onnx_artifact,
        tokenizer_path=tok_path,
        max_seq_len=active_profile.max_seq_len,
        db_uris=args.db,
    )

    if args.prompt:
        res = rag.ask(
            args.prompt,
            use_reasoning=bool(active_profile.runtime.get("reasoning", False)),
            auto_fallback=False,
        )
        print(res.get("answer", ""))
        sys.exit(0)

    cli = ConversationalCLI(rag, profile=active_profile)
    cli.run()


if __name__ == "__main__":
    main()
