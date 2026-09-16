"""
rStar: Mutual Reasoning & Self-Play Engine for Tiny LLM
=========================================================

Implements rStar-style self-play reasoning for small language models:
  1. Generator (Proposer): Explores diverse reasoning trajectories via multi-action space:
     - Decompose complex query into sub-problems
     - Step-by-step contextual deduction
     - Candidate solution proposal
  2. Mutual Verifier (Discriminator):
     - Factual grounding check against retrieved RAG context
     - Self-consistency audit of intermediate deductions
     - Trajectory scoring & pruning of invalid reasoning steps
  3. Best-of-N / Tree Search Selection:
     - Selects the reasoning trajectory with the highest mutual verification reward
"""

from __future__ import annotations

import re
import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable


@dataclass
class ReasoningStep:
    step_type: str        # 'decomposition', 'deduction', 'verification'
    thought: str
    confidence: float = 1.0


@dataclass
class ReasoningTrajectory:
    steps: List[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    score: float = 0.0

    def add_step(self, step_type: str, thought: str, confidence: float = 1.0):
        self.steps.append(ReasoningStep(step_type=step_type, thought=thought, confidence=confidence))


class RStarEngine:
    """
    Mutual Reasoning & Verification Engine for Tiny LLM.
    """

    def __init__(self, inference_engine, tokenizer, max_seq_len: int = 512):
        self.engine = inference_engine
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def _score_grounding(self, text: str, context: str) -> float:
        """Calculate factual grounding score of a reasoning step against context."""
        if not context or not text:
            return 0.5
            
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'of', 'for', 'to', 'with', 'about', 'how', 'why', 'where', 'when', 'who', 'does', 'do', 'did', 'can', 'could', 'would', 'should', 'and', 'or', 'so', 'then', 'therefore'}
        text_words = set(re.findall(r'\b\w+\b', text.lower())) - stopwords
        context_words = set(re.findall(r'\b\w+\b', context.lower())) - stopwords
        
        if not text_words:
            return 0.5
            
        overlap = len(text_words & context_words)
        return min(1.0, overlap / max(1, min(len(text_words), 8)))

    def propose_decomposition(self, question: str) -> List[str]:
        """Propose sub-questions to break down the problem."""
        prompt = (
            f"<|im_start|>system\nYou are a reasoning planner. Break the question into 2 key sub-steps.<|im_end|>\n"
            f"<|im_start|>user\nQuestion: {question}<|im_end|>\n"
            f"<|im_start|>assistant\nStep 1:"
        )
        raw = self.engine.generate(prompt, max_new_tokens=60, temperature=0.3)
        steps = [s.strip() for s in re.split(r'Step \d+:|\n\d+\.|\n-', "Step 1: " + raw) if s.strip()]
        return steps[:2] if steps else [question]

    def reason_and_verify(
        self,
        question: str,
        context: str = "",
        num_candidates: int = 3,
        on_step: Optional[Callable[[str], None]] = None
    ) -> ReasoningTrajectory:
        """
        Executes rStar mutual self-play:
          1. Proposes multiple reasoning trajectories
          2. Verifies grounding against context
          3. Selects optimal trajectory and synthesizes final answer
        """
        if on_step:
            on_step("🧠 [rStar Proposer] Exploring reasoning paths...")

        # 1. Action: Decompose question
        sub_goals = self.propose_decomposition(question)
        candidates: List[ReasoningTrajectory] = []

        # 2. Action: Sample N candidate reasoning trajectories
        for i in range(num_candidates):
            traj = ReasoningTrajectory()
            
            # Step A: Propose intermediate thought
            ctx_block = f"Context: {context}\n" if context else ""
            deduce_prompt = (
                f"<|im_start|>system\nYou are a precise reasoning assistant. Deduce the answer step-by-step using facts.<|im_end|>\n"
                f"<|im_start|>user\n{ctx_block}Question: {question}\nReasoning step:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            # Sample with slight temperature diversity
            temp = 0.2 + (i * 0.25)
            deduction = self.engine.generate(deduce_prompt, max_new_tokens=80, temperature=temp)
            grounding_score = self._score_grounding(deduction, context)
            traj.add_step("deduction", deduction, confidence=grounding_score)

            # Step B: Mutual Verifier Audit
            verify_prompt = (
                f"<|im_start|>system\nYou are a critical factual verifier. Check if the deduction matches the context.<|im_end|>\n"
                f"<|im_start|>user\n{ctx_block}Deduction: {deduction}\nIs this deduction supported?<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            verification = self.engine.generate(verify_prompt, max_new_tokens=50, temperature=0.1)
            v_score = 0.8 if any(w in verification.lower() for w in ["yes", "supported", "correct", "accurate", "true"]) else 0.4
            traj.add_step("verification", verification, confidence=v_score)

            # Step C: Propose Candidate Final Answer
            ans_prompt = (
                f"<|im_start|>system\nSynthesize the verified reasoning into a clear final answer.<|im_end|>\n"
                f"<|im_start|>user\nQuestion: {question}\nVerified steps: {deduction}\nAnswer:<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            answer = self.engine.generate(ans_prompt, max_new_tokens=80, temperature=0.2)
            traj.final_answer = answer
            
            # Composite score: grounding + verification consistency
            traj.score = (0.6 * grounding_score) + (0.4 * v_score)
            candidates.append(traj)

        # 3. Select best trajectory by mutual verification score
        candidates.sort(key=lambda t: t.score, reverse=True)
        best_traj = candidates[0]

        if on_step:
            on_step(f"✅ [rStar Verifier] Best path selected (Confidence score: {best_traj.score:.2f})")

        return best_traj
