# 📡 Kruuboo Tiny LLM (SLM-0.2) — REST API & SSE Telemetry Reference

This document provides a complete specification for all HTTP endpoints, Server-Sent Events (SSE) telemetry streams, request/response schemas, and example invocations for the Kruuboo-SLM-0.2 server (`server.py`).

**Base URL**: `http://localhost:8000`

---

## 📑 Endpoint Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web application (Chat & Live Learning Studio UI) |
| `POST` | `/ask` | Primary conversational chat & grounded RAG query |
| `POST` | `/upload` | Multipart file upload to global semantic knowledge base |
| `GET` | `/api/training/status` | Current training telemetry snapshot |
| `GET` | `/api/training/events` | Real-time Server-Sent Events (SSE) metric stream |
| `POST` | `/api/training/start` | Launch an incremental background fine-tuning run |
| `POST` | `/api/training/stop` | Gracefully terminate an active training run |
| `POST` | `/api/training/probe` | Test model behavior on a prompt without saving |
| `POST` | `/api/model/reload` | Atomically hot-reload updated ONNX weights into RAM |

---

## 1. Conversational Query (`POST /ask`)

Processes a user prompt through the unified conversational pipeline (math resolution, semantic hybrid search, multi-turn dialogue memory, and Min-P neural generation).

### Request Headers
- `Content-Type: application/json`

### Request Body Schema
```json
{
  "question": "What is the recommended treatment for a mild burn?",
  "history": [
    { "role": "user", "content": "Hello!" },
    { "role": "assistant", "content": "Hello! How can I assist you today?" }
  ]
}
```

### Response Schema (`200 OK`)
```json
{
  "answer": "For mild burns, immediately cool the area under running tap water for 10 to 15 minutes. Avoid ice, which can cause tissue damage. Apply aloe vera gel or a mild moisturizer, and protect with a sterile, non-stick dressing.",
  "knowledge_used": 1,
  "retrieved_context": [
    {
      "text": "Home Remedy for Burn: Run cool tap water over the burn for 10-15 minutes...",
      "source": "home_remedies.json:Burn",
      "score": 0.88
    }
  ],
  "validated": true
}
```

### cURL Example
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is 15 * 24 + 18?",
    "history": []
  }'
```

---

## 2. Knowledge File Upload (`POST /upload`)

Ingests a dataset directly into global semantic memory. Automatically parses `.json`, `.csv`, `.txt`, and `.md` files, splits into chunks, and computes native 768d dense SLM embeddings.

### Request Headers
- `Content-Type: multipart/form-data`

### Request Parameters
- `file`: The binary file to upload.

### Response Schema (`200 OK`)
```json
{
  "message": "Successfully uploaded 'survival_manual.txt' to global memory!",
  "chunks_created": 42
}
```

### cURL Example
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@training_data/home_remedies.json"
```

---

## 3. Training Telemetry Snapshot (`GET /api/training/status`)

Returns the current state of the training daemon, step counts, active loss, and throughput.

### Response Schema (`200 OK`)
```json
{
  "is_training": true,
  "current_step": 350,
  "max_iters": 1000,
  "train_loss": 2.418,
  "val_loss": 2.652,
  "perplexity": 11.22,
  "learning_rate": 0.000285,
  "tokens_per_sec": 4210.5,
  "recent_probes": [
    {
      "step": 300,
      "prompt": "How to build a shelter?",
      "output": "A lean-to shelter requires a ridgepole secured between two trees..."
    }
  ]
}
```

### cURL Example
```bash
curl -X GET http://localhost:8000/api/training/status
```

---

## 4. Live Metric SSE Stream (`GET /api/training/events`)

Streams real-time training events using the standard **Server-Sent Events (SSE)** protocol. Used by the Web UI Chart.js charts.

### Stream Event Types
- `step`: Dispatched every training step with loss, learning rate, and throughput.
- `eval`: Dispatched after validation evaluation with validation loss and perplexity.
- `probe`: Dispatched when the background probe prompt is tested.
- `heartbeat`: Dispatched every 500ms when idle to maintain TCP connection.

### Stream Example Payload
```text
data: {"type": "step", "step": 12, "loss": 3.84, "lr": 0.00003, "tokens_per_sec": 4150}

data: {"type": "eval", "step": 50, "val_loss": 3.22, "perplexity": 25.02}

data: {"type": "heartbeat", "time": 1729000000.12}
```

### cURL Example
```bash
curl -N http://localhost:8000/api/training/events
```

---

## 5. Start Background Training (`POST /api/training/start`)

Launches an incremental SFT fine-tuning job in a dedicated background worker thread without blocking the web server.

### Request Body Schema
```json
{
  "max_iters": 500,
  "dataset_source": "all",
  "probe_prompt": "Explain what a non-disclosure agreement is."
}
```

### Response Schema (`200 OK`)
```json
{
  "message": "Incremental training started for 500 iterations."
}
```

### cURL Example
```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -d '{"max_iters": 100, "probe_prompt": "What causes rain?"}'
```

---

## 6. Stop Background Training (`POST /api/training/stop`)

Gracefully requests the active training loop to stop and save the latest checkpoint.

### Response Schema (`200 OK`)
```json
{
  "message": "Training stop signal sent.",
  "success": true
}
```

### cURL Example
```bash
curl -X POST http://localhost:8000/api/training/stop
```

---

## 7. Model Hot-Reloading (`POST /api/model/reload`)

Re-creates the active ONNX Runtime inference session from `models/tiny_llm.onnx` and flushes encoder caches. Enables immediate evaluation of fine-tuned weights without restarting the server process.

### Response Schema (`200 OK`)
```json
{
  "message": "Model successfully hot-reloaded into memory."
}
```

### cURL Example
```bash
curl -X POST http://localhost:8000/api/model/reload
```

---

## 🐍 Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

def ask_assistant(question: str, history=None):
    payload = {
        "question": question,
        "history": history or []
    }
    resp = requests.post(f"{BASE_URL}/ask", json=payload)
    resp.raise_for_status()
    data = resp.json()
    return data["answer"], data.get("retrieved_context", [])

# Example conversational usage
history = []
q1 = "What is the capital of Sri Lanka?"
ans1, _ = ask_assistant(q1, history)
print(f"Assistant: {ans1}")

history.append({"role": "user", "content": q1})
history.append({"role": "assistant", "content": ans1})

q2 = "What is its currency?"
ans2, _ = ask_assistant(q2, history)
print(f"Assistant: {ans2}")
```
