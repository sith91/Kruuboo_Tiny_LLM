"""
rStar Self-Play Data Generation Pipeline
=========================================

Runs rStar mutual reasoning self-play over domain knowledge bases, filters
high-confidence verified reasoning trajectories, and exports them into
ChatML format at training_data/rstar_reasoning.json for SFT model training.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag.system import RAGSystem
from src.rstar import RStarEngine


def generate_self_play_dataset(output_path: str = "training_data/rstar_reasoning.json", min_score: float = 0.55):
    print("=" * 60)
    print("🌟 rStar Mutual Reasoning Self-Play Generator")
    print("=" * 60)

    rag = RAGSystem("models/tiny_llm_quantized.onnx", "tokenizer.json", max_seq_len=512)
    rstar = RStarEngine(rag._engine, rag.tokenizer, max_seq_len=512)

    # Seed questions across domains
    seed_questions = [
        "What is the best home remedy for a severe headache?",
        "How should one treat a sudden fever without pills?",
        "What are the immediate steps if lost in a forest with no cell service?",
        "Why is an NDA crucial before sharing trade secrets?",
        "What natural remedies relieve an upset stomach quickly?",
        "How do you signal for help when stranded in the wilderness?",
        "What constitutes a legally binding oral contract?",
        "What is the recommended treatment for a minor kitchen burn?"
    ]

    # Gather additional questions from domain registry
    for domain in rag.get_available_domains():
        for ex_q in domain.get("example_questions", []):
            if ex_q not in seed_questions:
                seed_questions.append(ex_q)

    print(f"📋 Generating self-play reasoning chains for {len(seed_questions)} domain topics...\n")

    verified_samples = []

    for idx, q in enumerate(seed_questions, 1):
        print(f"[{idx}/{len(seed_questions)}] Self-Play Reasoning: '{q}'")
        
        # Retrieve grounding context
        retrieved = rag.search(q, k=2)
        context = " ".join([d.text for d, _ in retrieved]) if retrieved else ""

        # Execute mutual reasoning self-play
        traj = rstar.reason_and_verify(
            question=q,
            context=context,
            num_candidates=3,
            on_step=lambda msg: print(f"   {msg}")
        )

        print(f"   Trajectory Score: {traj.score:.2f} | Final Answer: {traj.final_answer[:60]}...\n")

        if traj.score >= min_score and traj.final_answer:
            # Build step-by-step reasoning thought trace in ChatML
            deductions = [s.thought for s in traj.steps if s.step_type == 'deduction']
            thought_chain = " ".join(deductions)
            
            chatml_sample = (
                f"<|im_start|>system\nYou are a reasoning assistant. Think step-by-step before answering.<|im_end|>\n"
                f"<|im_start|>user\n{q}<|im_end|>\n"
                f"<|im_start|>assistant\n<thought>\n{thought_chain}\n</thought>\n{traj.final_answer}<|im_end|>\n"
            )
            
            verified_samples.append({
                "question": q,
                "context": context,
                "thought_trace": thought_chain,
                "answer": traj.final_answer,
                "score": traj.score,
                "formatted_chatml": chatml_sample
            })

    # Save to disk
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(verified_samples, f, indent=2)

    print("=" * 60)
    print(f"🎉 Generated {len(verified_samples)} verified rStar reasoning samples saved to {out_file}")
    print("=" * 60)


if __name__ == "__main__":
    generate_self_play_dataset()
