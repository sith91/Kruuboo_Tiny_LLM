import uvicorn
from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
import json
from pathlib import Path
import logging

# Set up logging to avoid cluttering the streaming output if possible
logging.basicConfig(level=logging.ERROR)

# Import our RAG system
from src.rag import RAGSystem

# Load configuration dynamically
import yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Must match config.yaml model.max_seq_len and src/train.py BLOCK_SIZE
MODEL_MAX_SEQ_LEN = config["model"]["max_seq_len"]

app = FastAPI(title="Tiny LLM Mobile")

# Initialize RAG System
model_path = "models/tiny_llm_quantized.onnx"
tokenizer_path = "tokenizer.json"

# In a real scenario, we'd want this to be a singleton or dependency
rag = RAGSystem(model_path, tokenizer_path, max_seq_len=MODEL_MAX_SEQ_LEN)

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>Tiny AI Assistant</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --safe-top: env(safe-area-inset-top);
                --safe-bottom: env(safe-area-inset-bottom);
            }
            body { 
                background-color: #0f172a; 
                color: white; 
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                height: 100vh;
                display: flex;
                flex-direction: column;
            }
            .chat-container { 
                flex: 1;
                overflow-y: auto; 
                scroll-behavior: smooth;
                padding: 1rem;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }
            .message-bubble { 
                max-width: 85%; 
                border-radius: 20px; 
                padding: 0.8rem 1.2rem;
                font-size: 0.95rem;
                line-height: 1.5;
                position: relative;
            }
            .user-msg { 
                background: linear-gradient(135deg, #3b82f6, #2563eb);
                align-self: flex-end; 
                border-bottom-right-radius: 4px;
                color: white;
            }
            .bot-msg { 
                background-color: #1e293b; 
                align-self: flex-start; 
                border-bottom-left-radius: 4px; 
                border: 1px solid #334155;
            }
            .domain-badge {
                font-size: 0.65rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: #94a3b8;
                margin-bottom: 0.25rem;
                display: block;
            }
            #user-input {
                -webkit-appearance: none;
                appearance: none;
            }
            .typing {
                opacity: 0.7;
                font-style: italic;
            }
        </style>
    </head>
    <body>
        <!-- Header -->
        <header class="pt-[var(--safe-top)] bg-slate-900 border-b border-slate-800 shrink-0 z-10">
            <div class="px-4 py-3 flex justify-between items-center">
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-gradient-to-tr from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                        <i class="fas fa-robot text-white text-lg"></i>
                    </div>
                    <div>
                        <h1 class="font-bold text-base leading-tight">Tiny AI Agent</h1>
                        <p class="text-[10px] text-slate-400 uppercase tracking-widest">Active System</p>
                    </div>
                </div>
                <div class="flex items-center gap-2">
                    <!-- Response Mode Selector -->
                    <select id="response-mode" class="bg-slate-800 border border-slate-700 rounded-full px-3 py-1.5 text-[10px] font-medium text-slate-300 focus:outline-none transition-colors cursor-pointer">
                        <option value="rstar">rStar (Self-Play Reasoning)</option>
                        <option value="hybrid">Hybrid (Smart RAG + Tiny LLM)</option>
                        <option value="toy_llm">Tiny LLM (Custom 105M / 100MB)</option>
                        <option value="smart_rag">Extractive RAG</option>
                    </select>

                    <label for="file-upload" class="cursor-pointer bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-full border border-slate-700 text-[10px] font-medium text-slate-300 flex items-center gap-1.5 transition-colors">
                        <i class="fas fa-cloud-upload-alt text-blue-400"></i>
                        <span>UPLOAD DATA</span>
                    </label>
                    <input type="file" id="file-upload" class="hidden" accept=".txt,.json">
                    
                    <div id="connection" class="hidden lg:flex items-center gap-2 bg-slate-800 px-3 py-1.5 rounded-full border border-slate-700">
                        <div class="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse"></div>
                        <span class="text-[10px] font-medium text-slate-300">ONLINE</span>
                    </div>
                </div>
            </div>
        </header>

        <!-- Chat History -->
        <div id="chat-box" class="chat-container">
            <div class="flex flex-col items-start">
                <div class="message-bubble bot-msg shadow-md">
                    <p>Welcome! I'm your Tiny LLM. I can adapt to different domains automatically. Try asking about the alphabet or pricing! ✨</p>
                </div>
            </div>
        </div>

        <!-- Typing Indicator -->
        <div id="typing" class="px-6 py-2 text-[10px] text-slate-500 hidden">
            <i class="fas fa-circle-notch fa-spin mr-1"></i> AI IS SPECIALIZING...
        </div>

        <!-- Footer / Input -->
        <footer class="pb-[var(--safe-bottom)] bg-slate-900 border-t border-slate-800 p-4 shrink-0">
            <form id="chat-form" class="flex gap-2 max-w-2xl mx-auto">
                <input type="text" id="user-input" 
                    placeholder="Message Tiny AI..." 
                    class="flex-grow bg-slate-800 rounded-2xl px-5 py-3.5 text-sm border border-slate-700 text-white placeholder-slate-500 focus:border-blue-500 transition-colors"
                    autocomplete="off">
                <button type="submit" class="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center text-white shadow-lg active:scale-90 transition-transform">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </form>
        </footer>

        <script>
            const form = document.getElementById('chat-form');
            const input = document.getElementById('user-input');
            const chatBox = document.getElementById('chat-box');
            const typing = document.getElementById('typing');

            function addMessage(text, isUser, domainName = null, sources = []) {
                const wrapper = document.createElement('div');
                wrapper.className = `flex flex-col ${isUser ? 'items-end' : 'items-start'} w-full`;
                
                let html = '';
                if (domainName && !isUser) {
                    html += `<span class="domain-badge px-1">${domainName}</span>`;
                }
                
                html += `
                    <div class="message-bubble ${isUser ? 'user-msg' : 'bot-msg'} shadow-xl">
                        <p>${text.replace(/\\n/g, '<br>')}</p>
                    </div>
                `;

                if (sources && sources.length > 0 && !isUser) {
                    html += `
                        <div class="mt-1 text-[10px] text-slate-400 max-w-[85%] px-2">
                            <details class="cursor-pointer">
                                <summary class="font-semibold hover:text-slate-200 transition-colors">📚 View Retrieved Context (${sources.length} sources)</summary>
                                <ul class="list-disc pl-4 mt-1 space-y-1 bg-slate-800/50 p-2 rounded border border-slate-700/50">
                                    ${sources.map(src => `<li>${src}</li>`).join('')}
                                </ul>
                            </details>
                        </div>
                    `;
                }
                
                wrapper.innerHTML = html;
                chatBox.appendChild(wrapper);
                chatBox.scrollTo(0, chatBox.scrollHeight);
            }

            let chatHistory = [];

            form.onsubmit = async (e) => {
                e.preventDefault();
                const question = input.value.trim();
                if (!question) return;

                input.value = '';
                addMessage(question, true);
                
                // Track user query in history
                chatHistory.push({ role: "user", content: question });

                // Show typing
                typing.classList.remove('hidden');
                chatBox.scrollTo(0, chatBox.scrollHeight);

                try {
                    const responseMode = document.getElementById('response-mode').value;
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question, response_mode: responseMode, history: chatHistory })
                    });
                    
                    const data = await res.json();
                    typing.classList.add('hidden');
                    
                    // Track assistant response in history
                    chatHistory.push({ role: "assistant", content: data.answer });
                    
                    addMessage(data.answer, false, "Global RAG", data.retrieved_context || []);
                } catch (err) {
                    typing.classList.add('hidden');
                    addMessage("Communication error. Is the server running?", false);
                }
            };

            // File upload logic
            const fileUpload = document.getElementById('file-upload');

            fileUpload.onchange = async () => {
                const file = fileUpload.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);
                
                addMessage(`Uploading ${file.name} to global knowledge base...`, true);
                
                try {
                    const res = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await res.json();
                    if (res.ok) {
                        addMessage(`🎉 Success: ${data.message} (split into ${data.chunks_created || 1} chunks for RAG)`, false);
                    } else {
                        addMessage(`❌ Upload failed: ${data.error}`, false);
                    }
                } catch (err) {
                    addMessage("❌ Connection error during upload.", false);
                }
                fileUpload.value = '';
            };

            // Prevent zoom on iOS input focus
            document.addEventListener('gesturestart', function (e) {
                e.preventDefault();
            });
        </script>
    </body>
    </html>
    """

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question")
    response_mode = data.get("response_mode", "smart_rag")
    history = data.get("history", [])
    
    # Update response mode in RAG dynamically
    rag.response_mode = response_mode
    rag.use_pretrained = False
    
    use_rag = (response_mode != "toy_llm")
    
    # Use the unified RAG system directly passing the dialogue history
    answer = rag.ask(question, use_rag=use_rag, history=history)
    retrieved = rag.search(question, k=3) if use_rag else []
    
    return JSONResponse(content={
        "answer": answer,
        "knowledge_used": len(retrieved),
        "retrieved_context": [d.text for d, _ in retrieved],
        "validated": True
    })

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...)
):
    try:
        content = await file.read()
        filename = file.filename
        
        if filename.endswith(".json"):
            try:
                data = json.loads(content.decode("utf-8"))
                if isinstance(data, list):
                    added_count = 0
                    for item in data:
                        if isinstance(item, dict):
                            t = item.get("text", item.get("content", str(item)))
                            src = item.get("source", filename)
                        else:
                            t = str(item)
                            src = filename
                        if t.strip():
                            rag.add_text(t, src)
                            added_count += 1
                    return {"message": f"Successfully ingested {added_count} items from JSON dataset.", "chunks_created": added_count}
                elif isinstance(data, dict):
                    for k, v in data.items():
                        rag.add_text(f"{k}: {v}", filename)
                    return {"message": "Successfully ingested JSON dataset.", "chunks_created": len(data)}
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Failed to parse JSON dataset: {e}"})
        
        # Default: treat as text file
        text = content.decode("utf-8", errors="ignore")
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "File is empty"})
            
        # Add text directly to RAG system (it chunks internally using config)
        rag.add_text(text, filename)
        
        return {
            "message": f"Successfully uploaded '{filename}' to the global knowledge base!",
            "chunks_created": len(text) // 500 + 1
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to process file: {str(e)}"})

if __name__ == "__main__":
    # In a local network, host='0.0.0.0' allows other devices (phones) to connect
    uvicorn.run(app, host="0.0.0.0", port=8000)
