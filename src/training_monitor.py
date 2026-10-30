"""
TrainingMonitor — Real-Time Telemetry & On-The-Go Learning Coordinator
=======================================================================

Provides background thread management, step-by-step metric logging, Server-Sent
Events (SSE) metric broadcasting, live probe generation, and state persistence.
"""

import os
import time
import json
import math
import queue
import threading
from pathlib import Path
from typing import List, Dict, Optional, Generator, Callable

import torch
import yaml

from src.model import TinyLLM
from src.tokenizer import TinyTokenizer


WORKSPACE_DIR = Path("workspace")
METRICS_FILE = WORKSPACE_DIR / "training_metrics.json"
WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)


class TrainingMonitor:
    """
    Singleton manager for live model training visualization and interactive learning.
    """

    def __init__(self):
        self.lock = threading.Lock()
        self.status = "idle"  # idle | training | paused | exporting | error
        self.error_message: Optional[str] = None
        self.current_step = 0
        self.total_steps = 0
        self.train_loss = 0.0
        self.val_loss = 0.0
        self.perplexity = 0.0
        self.learning_rate = 0.0
        self.speed_tok_s = 0.0
        self.elapsed_time = 0.0
        self.eta_seconds = 0.0
        self.tokens_processed = 0
        self.best_val_loss = float("inf")
        self.dataset_source = "knowledge_base"
        self.on_training_complete: Optional[Callable] = None

        # History arrays for UI charts
        self.history: List[Dict] = []
        self.probes: List[Dict] = []
        self.logs: List[str] = []

        # In-memory active training model & cached probe references (eliminates 420 MB disk reloads)
        self._active_model: Optional[TinyLLM] = None
        self._active_tokenizer: Optional[TinyTokenizer] = None
        self._active_device: Optional[str] = None
        self._cached_probe_model: Optional[TinyLLM] = None
        self._cached_probe_tokenizer: Optional[TinyTokenizer] = None
        self._cached_probe_device: Optional[str] = None

        # Threading controls
        self._training_thread: Optional[threading.Thread] = None
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._subscribers: List[queue.Queue] = []

        # Load persisted metrics if available
        self._load_state()

    def set_active_model(self, model: TinyLLM, tokenizer: TinyTokenizer, device: Optional[str] = None):
        """Registers the live in-memory training model so probe requests from API or hooks bypass disk I/O."""
        with self.lock:
            self._active_model = model
            self._active_tokenizer = tokenizer
            self._active_device = device

    def clear_active_model(self):
        """Clears the active in-memory training model reference upon training completion."""
        with self.lock:
            self._active_model = None
            self._active_tokenizer = None
            self._active_device = None

    def _load_state(self):
        if METRICS_FILE.exists():
            try:
                with open(METRICS_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.history = data.get("history", [])[-200:]
                    self.probes = data.get("probes", [])[-20:]
                    self.best_val_loss = data.get("best_val_loss", float("inf"))
                    if self.history:
                        latest = self.history[-1]
                        self.current_step = latest.get("step", 0)
                        self.train_loss = latest.get("train_loss", 0.0)
                        self.val_loss = latest.get("val_loss", 0.0)
                        self.perplexity = latest.get("perplexity", 0.0)
            except Exception:
                pass

    def _save_state(self):
        try:
            with open(METRICS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "history": self.history[-200:],
                    "probes": self.probes[-20:],
                    "best_val_loss": self.best_val_loss,
                    "last_updated": time.time(),
                }, f, indent=2)
        except Exception:
            pass

    def subscribe(self) -> queue.Queue:
        q = queue.Queue(maxsize=100)
        with self.lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        with self.lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _broadcast(self, event_data: Dict):
        with self.lock:
            dead_subscribers = []
            for q in self._subscribers:
                try:
                    q.put_nowait(event_data)
                except queue.Full:
                    dead_subscribers.append(q)
            for q in dead_subscribers:
                if q in self._subscribers:
                    self._subscribers.remove(q)

    def log_message(self, msg: str):
        with self.lock:
            timestamp = time.strftime("%H:%M:%S")
            entry = f"[{timestamp}] {msg}"
            self.logs.append(entry)
            if len(self.logs) > 100:
                self.logs.pop(0)
        self._broadcast({"type": "log", "message": entry})

    def record_step(
        self,
        step: int,
        total_steps: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        speed: float,
        tokens: int,
        elapsed: float,
        is_best: bool = False,
    ):
        with self.lock:
            self.current_step = step
            self.total_steps = total_steps
            self.train_loss = round(train_loss, 4)
            self.val_loss = round(val_loss, 4)
            self.perplexity = round(math.exp(min(15.0, train_loss)), 2)
            self.learning_rate = lr
            self.speed_tok_s = round(speed, 1)
            self.tokens_processed = tokens
            self.elapsed_time = round(elapsed, 1)
            
            # Estimate ETA
            if step > 0:
                remaining_steps = max(0, total_steps - step)
                time_per_step = elapsed / step
                self.eta_seconds = round(remaining_steps * time_per_step, 1)
            else:
                self.eta_seconds = 0.0

            if is_best or val_loss < self.best_val_loss:
                self.best_val_loss = round(val_loss, 4)

            entry = {
                "step": step,
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "perplexity": self.perplexity,
                "lr": self.learning_rate,
                "speed": self.speed_tok_s,
                "is_best": is_best,
                "timestamp": time.time(),
            }
            self.history.append(entry)
            if len(self.history) > 300:
                self.history.pop(0)

        self._broadcast({
            "type": "step",
            "data": entry,
            "summary": self.get_status(),
        })

        if step % 20 == 0 or step == total_steps - 1:
            self._save_state()

    def record_probe(self, prompt: str, output: str, step: int):
        with self.lock:
            probe_entry = {
                "step": step,
                "prompt": prompt,
                "output": output,
                "timestamp": time.time(),
            }
            self.probes.append(probe_entry)
            if len(self.probes) > 20:
                self.probes.pop(0)
        self._broadcast({"type": "probe", "data": probe_entry})
        self._save_state()

    def get_status(self) -> Dict:
        with self.lock:
            return {
                "status": self.status,
                "error": self.error_message,
                "current_step": self.current_step,
                "total_steps": self.total_steps,
                "progress_pct": round((self.current_step / max(1, self.total_steps)) * 100, 1),
                "train_loss": self.train_loss,
                "val_loss": self.val_loss,
                "perplexity": self.perplexity,
                "best_val_loss": self.best_val_loss if self.best_val_loss != float("inf") else None,
                "learning_rate": self.learning_rate,
                "speed_tok_s": self.speed_tok_s,
                "elapsed_time": self.elapsed_time,
                "eta_seconds": self.eta_seconds,
                "tokens_processed": self.tokens_processed,
                "dataset_source": self.dataset_source,
                "history": self.history[-100:],
                "probes": self.probes[-10:],
                "logs": self.logs[-20:],
            }

    def start_training(
        self,
        max_iters: int = 100,
        dataset_source: str = "knowledge_base",
        custom_data: Optional[List[str]] = None,
        probe_prompt: str = "How do I treat a mild burn at home?",
    ):
        with self.lock:
            if self.status == "training":
                return False, "Training is already in progress"
            self.status = "training"
            self.error_message = None
            self.current_step = 0
            self.total_steps = max_iters
            self.dataset_source = dataset_source
            self._stop_requested.clear()
            self._pause_requested.clear()

        self.log_message(f"🚀 Starting on-the-go training run: {max_iters} steps from source '{dataset_source}'")

        def _worker():
            try:
                from src.train import run_training_session
                run_training_session(
                    monitor=self,
                    max_iters=max_iters,
                    dataset_source=dataset_source,
                    custom_data=custom_data,
                    probe_prompt=probe_prompt,
                    stop_event=self._stop_requested,
                    pause_event=self._pause_requested,
                )
                with self.lock:
                    self.status = "finished" if not self._stop_requested.is_set() else "idle"
                self.log_message("✅ Training session finished successfully.")

                # Hot-reload updated model weights if hook registered
                if self.on_training_complete and callable(self.on_training_complete):
                    try:
                        self.log_message("🔄 Triggering automatic model hot-reload in runtime...")
                        self.on_training_complete()
                        self.log_message("✅ Runtime session refreshed with newly trained weights.")
                    except Exception as reload_err:
                        self.log_message(f"⚠️ Hot-reload notice: {reload_err}")
            except Exception as e:
                with self.lock:
                    self.status = "error"
                    self.error_message = str(e)
                self.log_message(f"❌ Training error: {e}")
            finally:
                self._save_state()
                self._broadcast({"type": "status", "data": self.get_status()})

        self._training_thread = threading.Thread(target=_worker, daemon=True)
        self._training_thread.start()
        return True, "Training launched successfully"

    def stop_training(self):
        with self.lock:
            if self.status != "training":
                return False, "No active training session"
            self._stop_requested.set()
            self.status = "idle"
        self.log_message("🛑 Training stop requested by user.")
        return True, "Stop signal sent"

    def probe_model(
        self,
        prompt: str,
        max_tokens: int = 60,
        model: Optional[TinyLLM] = None,
        tokenizer: Optional[TinyTokenizer] = None,
        device: Optional[str] = None,
    ) -> str:
        """
        Runs live autoregressive generation on the provided model argument, the active in-memory
        training model, or a cached checkpoint instance, eliminating 420 MB disk reloads and I/O latency.
        """
        try:
            # 1. Resolve model & tokenizer (direct argument > active training model > cached model)
            target_model = model or self._active_model
            target_tokenizer = tokenizer or self._active_tokenizer
            target_device = device or self._active_device

            if target_model is None:
                # Use in-memory cached probe model if already loaded to avoid repetitive 420 MB disk I/O
                if self._cached_probe_model is None:
                    target_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
                    with open("config.yaml", "r") as f:
                        cfg = yaml.safe_load(f)
                    m_cfg = cfg.get("model", {})
                    
                    target_tokenizer = TinyTokenizer.load("tokenizer.json")
                    target_model = TinyLLM(
                        vocab_size=m_cfg.get("vocab_size", 8192),
                        d_model=m_cfg.get("d_model", 768),
                        n_heads=m_cfg.get("n_heads", 12),
                        n_layers=m_cfg.get("n_layers", 14),
                        d_ff=m_cfg.get("d_ff", 3072),
                        max_seq_len=m_cfg.get("max_seq_len", 1024),
                        dropout=0.0
                    )
                    
                    ckpt = Path("models/tiny_llm.pt")
                    if ckpt.exists():
                        target_model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
                    target_model.to(target_device)
                    target_model.eval()

                    self._cached_probe_model = target_model
                    self._cached_probe_tokenizer = target_tokenizer
                    self._cached_probe_device = target_device
                else:
                    target_model = self._cached_probe_model
                    target_tokenizer = self._cached_probe_tokenizer
                    target_device = self._cached_probe_device
            else:
                if target_tokenizer is None:
                    target_tokenizer = self._cached_probe_tokenizer or TinyTokenizer.load("tokenizer.json")
                if target_device is None:
                    try:
                        target_device = next(target_model.parameters()).device
                    except Exception:
                        target_device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

            # 2. Preserve training mode state (probe in eval mode without dropout noise)
            was_training = getattr(target_model, "training", False)
            if was_training:
                target_model.eval()

            # 3. Autoregressive generation
            tokens = target_tokenizer.encode(prompt)
            x = torch.tensor([tokens], dtype=torch.long, device=target_device)
            max_seq_len = getattr(target_model, "max_seq_len", 1024)
            
            with torch.no_grad():
                for _ in range(max_tokens):
                    if x.shape[1] >= max_seq_len:
                        break
                    logits = target_model(x)[:, -1, :] / 0.7
                    probs = torch.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    if next_token.item() in [0, 1]:  # PAD or EOS
                        break
                    x = torch.cat([x, next_token], dim=1)

            # 4. Restore training mode if model was actively training
            if was_training:
                target_model.train()

            generated_ids = x[0, len(tokens):].tolist()
            output = target_tokenizer.decode(generated_ids).strip()
            self.record_probe(prompt, output, self.current_step)
            return output
        except Exception as e:
            return f"[Probe Error: {e}]"


# Global singleton instance
training_monitor = TrainingMonitor()
