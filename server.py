import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

# Import RAG system and Training Monitor
from src.rag import RAGSystem
from src.training_monitor import training_monitor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server")

# Load configuration dynamically
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MODEL_MAX_SEQ_LEN = config["model"]["max_seq_len"]

app = FastAPI(title="Kruuboo Tiny LLM — Conversational AI & Live Learning Studio")

# Initialize Unified Conversational RAG System
model_path = "models/tiny_llm_quantized.onnx"
tokenizer_path = "tokenizer.json"
rag = RAGSystem(model_path, tokenizer_path, max_seq_len=MODEL_MAX_SEQ_LEN)

# Wire live learning model hot-reload: when background training finishes, reload active session
training_monitor.on_training_complete = lambda: rag.reload_model()


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
        <title>Kruuboo Tiny LLM — Studio & Live Learning Monitor</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" rel="stylesheet">
        <style>
            :root {
                --safe-top: env(safe-area-inset-top);
                --safe-bottom: env(safe-area-inset-bottom);
            }
            body { 
                background-color: #0b0f19; 
                color: #f8fafc; 
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
                height: 100vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }
            .mono { font-family: 'JetBrains Mono', monospace; }
            .glass-panel {
                background: rgba(17, 24, 39, 0.75);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }
            .glass-card {
                background: linear-gradient(135deg, rgba(30, 41, 59, 0.6), rgba(15, 23, 42, 0.8));
                border: 1px solid rgba(255, 255, 255, 0.06);
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
                border-radius: 18px; 
                padding: 0.85rem 1.25rem;
                font-size: 0.93rem;
                line-height: 1.55;
                position: relative;
            }
            .user-msg { 
                background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                align-self: flex-end; 
                border-bottom-right-radius: 4px;
                color: white;
                box-shadow: 0 4px 15px rgba(59, 130, 246, 0.25);
            }
            .bot-msg { 
                background-color: #1e293b; 
                align-self: flex-start; 
                border-bottom-left-radius: 4px; 
                border: 1px solid #334155;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
            }
            .pulse-dot {
                box-shadow: 0 0 10px #10b981, 0 0 20px #10b981;
            }
            .pulse-dot-amber {
                box-shadow: 0 0 10px #f59e0b, 0 0 20px #f59e0b;
            }
            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 6px; height: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: #334155; border-radius: 9999px; }
            ::-webkit-scrollbar-thumb:hover { background: #475569; }
        </style>
    </head>
    <body>
        <!-- Top Navbar -->
        <header class="pt-[var(--safe-top)] glass-panel border-b border-slate-800 shrink-0 z-20">
            <div class="max-w-7xl mx-auto px-4 py-3 flex justify-between items-center gap-3">
                <!-- Branding -->
                <div class="flex items-center gap-3">
                    <div class="w-10 h-10 bg-gradient-to-tr from-blue-600 via-indigo-600 to-cyan-400 rounded-xl flex items-center justify-center shadow-lg shadow-blue-500/20">
                        <i class="fas fa-brain text-white text-lg"></i>
                    </div>
                    <div>
                        <div class="flex items-center gap-2">
                            <h1 class="font-bold text-base tracking-tight text-white">Kruuboo SLM-0.2</h1>
                            <span class="text-[10px] bg-blue-500/20 text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded-full font-semibold">105.4M</span>
                        </div>
                        <p class="text-[10px] text-slate-400 font-medium">Conversational AI with Native 768d Semantic RAG</p>
                    </div>
                </div>

                <!-- Main View Tabs -->
                <div class="flex items-center bg-slate-900/80 p-1 rounded-xl border border-slate-800">
                    <button id="tab-btn-chat" onclick="switchTab('chat')" class="px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-blue-600 text-white shadow-md">
                        <i class="fas fa-comments"></i>
                        <span>Conversational Chat</span>
                    </button>
                    <button id="tab-btn-learn" onclick="switchTab('learn')" class="px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all text-slate-400 hover:text-white">
                        <i class="fas fa-chart-line text-cyan-400"></i>
                        <span>Live Learning Monitor</span>
                        <span id="nav-live-badge" class="hidden w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                    </button>
                </div>

                <!-- Right Action Controls -->
                <div class="flex items-center gap-2">
                    <label for="file-upload" class="cursor-pointer bg-slate-800 hover:bg-slate-700 px-3 py-1.5 rounded-xl border border-slate-700 text-xs font-medium text-slate-300 flex items-center gap-1.5 transition-colors">
                        <i class="fas fa-cloud-upload-alt text-blue-400"></i>
                        <span class="hidden sm:inline">Upload Knowledge</span>
                    </label>
                    <input type="file" id="file-upload" class="hidden" accept=".txt,.json">
                    
                    <div id="connection-status" class="flex items-center gap-2 bg-slate-800/80 px-3 py-1.5 rounded-xl border border-slate-700">
                        <div id="status-indicator" class="w-2 h-2 bg-emerald-500 rounded-full pulse-dot"></div>
                        <span id="status-text" class="text-[11px] font-semibold text-slate-300">ONLINE</span>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Workspace Container -->
        <main class="flex-1 flex flex-col overflow-hidden relative max-w-7xl w-full mx-auto">
            
            <!-- ========================================== -->
            <!-- VIEW 1: CONVERSATIONAL CHAT -->
            <!-- ========================================== -->
            <div id="view-chat" class="flex-1 flex flex-col h-full overflow-hidden">
                <!-- Status Header -->
                <div class="px-4 py-2 bg-slate-900/40 border-b border-slate-800/50 flex flex-wrap items-center justify-between text-xs text-slate-400 gap-2">
                    <div class="flex items-center gap-2">
                        <span class="text-slate-300 font-medium">✨ Natural Neural Assistant</span>
                        <span class="text-slate-600">•</span>
                        <span class="text-slate-400">Trained on Unified Knowledge + Multi-Turn Dialogues</span>
                    </div>
                    <div class="text-[11px] text-slate-500 flex items-center gap-2">
                        <i class="fas fa-microchip text-indigo-400"></i>
                        <span>Native 768d SLM Vector Memory</span>
                    </div>
                </div>

                <!-- Chat History Messages -->
                <div id="chat-box" class="chat-container">
                    <div class="flex flex-col items-start w-full">
                        <div class="message-bubble bot-msg shadow-lg">
                            <p class="font-semibold text-blue-400 mb-1">👋 Welcome to Kruuboo Tiny LLM!</p>
                            <p>I am your local conversational AI assistant. You can chat with me naturally about anything—medical home remedies, legal principles, survival guides, mathematics, or casual conversation. I respond using my trained weights and native memory.</p>
                        </div>
                    </div>
                </div>

                <!-- Typing Indicator -->
                <div id="typing" class="px-6 py-2 text-xs text-slate-400 hidden flex items-center gap-2">
                    <i class="fas fa-circle-notch fa-spin text-blue-400"></i>
                    <span>Kruuboo SLM is thinking & generating...</span>
                </div>

                <!-- Footer / Input -->
                <footer class="pb-[var(--safe-bottom)] glass-panel border-t border-slate-800 p-4 shrink-0">
                    <form id="chat-form" class="flex gap-2 max-w-4xl mx-auto">
                        <input type="text" id="user-input" 
                            placeholder="Message Kruuboo SLM (e.g., 'How to treat a burn?', 'What is an NDA?', 'Tell me a story')..." 
                            class="flex-grow bg-slate-800/90 rounded-2xl px-5 py-3.5 text-sm border border-slate-700 text-white placeholder-slate-500 focus:border-blue-500 focus:outline-none transition-all shadow-inner"
                            autocomplete="off">
                        <button type="submit" class="w-12 h-12 bg-gradient-to-tr from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 rounded-2xl flex items-center justify-center text-white shadow-lg shadow-blue-600/30 active:scale-95 transition-all">
                            <i class="fas fa-paper-plane"></i>
                        </button>
                    </form>
                </footer>
            </div>

            <!-- ========================================== -->
            <!-- VIEW 2: LIVE LEARNING MONITOR & DASHBOARD -->
            <!-- ========================================== -->
            <div id="view-learn" class="flex-1 flex flex-col h-full overflow-y-auto p-4 space-y-4 hidden">
                
                <!-- Hero Telemetry Cards -->
                <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
                    <div class="glass-card p-3.5 rounded-2xl">
                        <div class="flex items-center justify-between text-slate-400 text-xs mb-1">
                            <span>Training State</span>
                            <i class="fas fa-heartbeat text-emerald-400"></i>
                        </div>
                        <div class="flex items-center gap-2">
                            <div id="card-status-dot" class="w-2.5 h-2.5 rounded-full bg-emerald-500"></div>
                            <span id="card-status-text" class="text-base font-bold tracking-tight text-white uppercase">IDLE</span>
                        </div>
                        <div id="card-eta-text" class="text-[10px] text-slate-400 mt-1">Ready to train</div>
                    </div>

                    <div class="glass-card p-3.5 rounded-2xl">
                        <div class="flex items-center justify-between text-slate-400 text-xs mb-1">
                            <span>Step Progress</span>
                            <i class="fas fa-forward-step text-blue-400"></i>
                        </div>
                        <div class="text-base font-bold mono text-white">
                            <span id="stat-step">0</span><span class="text-xs text-slate-400 font-normal"> / <span id="stat-total-steps">0</span></span>
                        </div>
                        <div class="w-full bg-slate-800 rounded-full h-1.5 mt-2 overflow-hidden">
                            <div id="stat-progress-bar" class="bg-gradient-to-r from-blue-500 to-indigo-500 h-1.5 rounded-full" style="width: 0%"></div>
                        </div>
                    </div>

                    <div class="glass-card p-3.5 rounded-2xl">
                        <div class="flex items-center justify-between text-slate-400 text-xs mb-1">
                            <span>Training Loss</span>
                            <i class="fas fa-chart-line-down text-rose-400"></i>
                        </div>
                        <div class="text-base font-bold mono text-rose-300" id="stat-train-loss">--</div>
                        <div class="text-[10px] text-slate-400 mt-1">Target Loss Masking Active</div>
                    </div>

                    <div class="glass-card p-3.5 rounded-2xl">
                        <div class="flex items-center justify-between text-slate-400 text-xs mb-1">
                            <span>Validation Loss</span>
                            <i class="fas fa-bullseye text-amber-400"></i>
                        </div>
                        <div class="text-base font-bold mono text-amber-300" id="stat-val-loss">--</div>
                        <div class="text-[10px] text-slate-400 mt-1">Best: <span id="stat-best-loss" class="font-semibold text-emerald-400">--</span></div>
                    </div>

                    <div class="glass-card p-3.5 rounded-2xl">
                        <div class="flex items-center justify-between text-slate-400 text-xs mb-1">
                            <span>Perplexity (PPL)</span>
                            <i class="fas fa-atom text-cyan-400"></i>
                        </div>
                        <div class="text-base font-bold mono text-cyan-300" id="stat-perplexity">--</div>
                        <div class="text-[10px] text-slate-400 mt-1">e^(Loss) measure</div>
                    </div>

                    <div class="glass-card p-3.5 rounded-2xl">
                        <div class="flex items-center justify-between text-slate-400 text-xs mb-1">
                            <span>Throughput</span>
                            <i class="fas fa-bolt text-yellow-400"></i>
                        </div>
                        <div class="text-base font-bold mono text-yellow-300" id="stat-speed">0 <span class="text-xs font-normal text-slate-400">tok/s</span></div>
                        <div class="text-[10px] text-slate-400 mt-1">LR: <span id="stat-lr" class="mono text-slate-300">--</span></div>
                    </div>
                </div>

                <!-- Main Visualizer Grid (Live Chart + Controller) -->
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 flex-1">
                    
                    <!-- Left: Real-Time Dynamic Learning Chart -->
                    <div class="lg:col-span-2 glass-card p-4 rounded-2xl flex flex-col">
                        <div class="flex items-center justify-between mb-3">
                            <div class="flex items-center gap-2">
                                <div class="w-3 h-3 rounded-full bg-cyan-400 animate-pulse"></div>
                                <h2 class="text-sm font-bold tracking-tight text-white">Live Learning Curve & Loss Trajectory</h2>
                            </div>
                            <div class="flex items-center gap-3 text-xs">
                                <span class="flex items-center gap-1.5 text-blue-400 font-medium"><span class="w-2.5 h-2.5 rounded bg-blue-500 inline-block"></span> Train Loss</span>
                                <span class="flex items-center gap-1.5 text-amber-400 font-medium"><span class="w-2.5 h-2.5 rounded bg-amber-500 inline-block"></span> Val Loss</span>
                                <span class="flex items-center gap-1.5 text-cyan-400 font-medium"><span class="w-2.5 h-2.5 rounded bg-cyan-400 inline-block"></span> Perplexity</span>
                            </div>
                        </div>
                        <div class="relative flex-1 min-h-[260px] w-full">
                            <canvas id="liveLossChart"></canvas>
                        </div>
                    </div>

                    <!-- Right: On-The-Go Learning Controller -->
                    <div class="glass-card p-4 rounded-2xl flex flex-col space-y-4">
                        <div class="flex items-center gap-2 pb-2 border-b border-slate-800">
                            <i class="fas fa-sliders-h text-indigo-400"></i>
                            <h3 class="text-sm font-bold text-white">On-The-Go Training Studio</h3>
                        </div>

                        <!-- Training Source -->
                        <div>
                            <label class="block text-xs font-semibold text-slate-300 mb-1">Corpus / Dataset Source</label>
                            <select id="train-source" class="w-full bg-slate-800/90 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-blue-500">
                                <option value="all">Unified Multi-Turn SFT Corpus (Recommended)</option>
                                <option value="conversational">Conversational Dialogues (ChatML)</option>
                                <option value="knowledge_base">Knowledge Base Documents (758 Docs)</option>
                                <option value="survival">US Army Survival Manual</option>
                            </select>
                        </div>

                        <!-- Step Budget -->
                        <div>
                            <div class="flex justify-between items-center text-xs font-semibold text-slate-300 mb-1">
                                <span>Training Steps</span>
                                <span id="steps-val-display" class="mono text-blue-400 font-bold">50 Steps</span>
                            </div>
                            <input type="range" id="train-steps-slider" min="10" max="500" step="10" value="50" class="w-full accent-blue-500 cursor-pointer">
                            <div class="flex justify-between text-[10px] text-slate-500 mt-1">
                                <span>10 (Quick)</span>
                                <span>100</span>
                                <span>500 (Deep)</span>
                            </div>
                        </div>

                        <!-- Action Buttons -->
                        <div class="pt-2 flex flex-col gap-2">
                            <button id="btn-start-train" onclick="startTraining()" class="w-full py-2.5 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-500 hover:to-indigo-500 text-white font-semibold text-xs rounded-xl shadow-lg shadow-blue-500/20 active:scale-98 transition-all flex items-center justify-center gap-2">
                                <i class="fas fa-play"></i>
                                <span>Start On-The-Go Training</span>
                            </button>
                            <button id="btn-stop-train" onclick="stopTraining()" disabled class="w-full py-2 bg-slate-800 hover:bg-rose-900/40 text-slate-400 hover:text-rose-300 border border-slate-700 font-medium text-xs rounded-xl transition-all flex items-center justify-center gap-2 disabled:opacity-40 disabled:pointer-events-none">
                                <i class="fas fa-stop"></i>
                                <span>Stop Training</span>
                            </button>
                        </div>

                        <!-- Info Tip -->
                        <div class="bg-blue-500/10 border border-blue-500/20 p-3 rounded-xl text-[11px] text-blue-300/90 leading-relaxed">
                            <i class="fas fa-info-circle mr-1"></i> As neural weights update on MPS/CPU, loss curves and comparative test probes update live below.
                        </div>
                    </div>
                </div>

                <!-- Bottom Row: Live Generation Probing & Activity Console -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    
                    <!-- Left: Model Probing Lab ("Watch the Model Learn") -->
                    <div class="glass-card p-4 rounded-2xl flex flex-col">
                        <div class="flex items-center justify-between pb-2 mb-3 border-b border-slate-800">
                            <div class="flex items-center gap-2">
                                <i class="fas fa-vial text-cyan-400"></i>
                                <h3 class="text-sm font-bold text-white">Live Generation Probing ("Watch Model Learn")</h3>
                            </div>
                            <span class="text-[10px] text-slate-400">Evaluates in-memory weights</span>
                        </div>

                        <div class="flex gap-2 mb-3">
                            <input type="text" id="probe-input" value="How do I treat a mild burn at home?" class="flex-1 bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-500">
                            <button onclick="runProbe()" class="px-3.5 py-2 bg-cyan-600 hover:bg-cyan-500 text-white rounded-xl text-xs font-semibold shadow-md active:scale-95 transition-all flex items-center gap-1.5">
                                <i class="fas fa-bolt"></i>
                                <span>Probe</span>
                            </button>
                        </div>

                        <!-- Probe Evolution Timeline -->
                        <div id="probe-history-list" class="flex-1 max-h-48 overflow-y-auto space-y-2 pr-1 text-xs">
                            <div class="p-2.5 rounded-xl bg-slate-800/60 border border-slate-700/50">
                                <div class="flex justify-between text-[10px] text-slate-400 mb-1">
                                    <span class="font-bold text-blue-400">Initial Probe Snapshot</span>
                                    <span>Step 0</span>
                                </div>
                                <p class="text-slate-300 italic" id="latest-probe-text">Run training or click 'Probe' to inspect generation evolution.</p>
                            </div>
                        </div>
                    </div>

                    <!-- Right: Live Activity Console Stream -->
                    <div class="glass-card p-4 rounded-2xl flex flex-col">
                        <div class="flex items-center justify-between pb-2 mb-3 border-b border-slate-800">
                            <div class="flex items-center gap-2">
                                <i class="fas fa-terminal text-emerald-400"></i>
                                <h3 class="text-sm font-bold text-white">Live Telemetry & Activity Console</h3>
                            </div>
                            <span class="text-[10px] mono text-emerald-400">SSE STREAM ACTIVE</span>
                        </div>
                        <div id="console-logs" class="flex-1 max-h-48 overflow-y-auto bg-slate-950 p-3 rounded-xl border border-slate-800 mono text-[11px] text-slate-300 space-y-1">
                            <div class="text-slate-500">[System] Visualizer connected to Kruuboo telemetry broadcaster.</div>
                        </div>
                    </div>

                </div>

            </div>

        </main>

        <!-- Application Logic -->
        <script>
            // Tab Switcher
            function switchTab(tab) {
                const viewChat = document.getElementById('view-chat');
                const viewLearn = document.getElementById('view-learn');
                const btnChat = document.getElementById('tab-btn-chat');
                const btnLearn = document.getElementById('tab-btn-learn');

                if (tab === 'chat') {
                    viewChat.classList.remove('hidden');
                    viewLearn.classList.add('hidden');
                    btnChat.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-blue-600 text-white shadow-md';
                    btnLearn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all text-slate-400 hover:text-white';
                } else {
                    viewChat.classList.add('hidden');
                    viewLearn.classList.remove('hidden');
                    btnLearn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all bg-blue-600 text-white shadow-md';
                    btnChat.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-2 transition-all text-slate-400 hover:text-white';
                    if (lossChart) lossChart.resize();
                }
            }

            // Slider update
            const slider = document.getElementById('train-steps-slider');
            const sliderDisplay = document.getElementById('steps-val-display');
            slider.oninput = () => {
                sliderDisplay.textContent = `${slider.value} Steps`;
            };

            // ----------------------------------------------------
            // Chart.js Setup for Live Loss & Perplexity Trajectory
            // ----------------------------------------------------
            let lossChart;
            const ctx = document.getElementById('liveLossChart').getContext('2d');
            
            lossChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [
                        {
                            label: 'Train Loss',
                            data: [],
                            borderColor: '#3b82f6',
                            backgroundColor: 'rgba(59, 130, 246, 0.12)',
                            borderWidth: 2,
                            tension: 0.35,
                            fill: true,
                            pointRadius: 2,
                            pointHoverRadius: 5,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Val Loss',
                            data: [],
                            borderColor: '#f59e0b',
                            backgroundColor: 'transparent',
                            borderWidth: 2,
                            borderDash: [4, 4],
                            tension: 0.35,
                            pointRadius: 3,
                            pointHoverRadius: 6,
                            yAxisID: 'y'
                        },
                        {
                            label: 'Perplexity',
                            data: [],
                            borderColor: '#22d3ee',
                            backgroundColor: 'transparent',
                            borderWidth: 1.5,
                            tension: 0.35,
                            pointRadius: 0,
                            yAxisID: 'y1'
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: { duration: 300 },
                    interaction: { mode: 'index', intersect: false },
                    scales: {
                        x: {
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } }
                        },
                        y: {
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: { display: true, text: 'Cross Entropy Loss', color: '#94a3b8', font: { size: 11 } },
                            grid: { color: 'rgba(255, 255, 255, 0.05)' },
                            ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: { display: true, text: 'Perplexity (PPL)', color: '#22d3ee', font: { size: 11 } },
                            grid: { drawOnChartArea: false },
                            ticks: { color: '#22d3ee', font: { family: 'JetBrains Mono', size: 10 } }
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: 'rgba(15, 23, 42, 0.95)',
                            titleFont: { family: 'JetBrains Mono', size: 11 },
                            bodyFont: { family: 'JetBrains Mono', size: 11 },
                            padding: 10,
                            borderColor: 'rgba(255, 255, 255, 0.1)',
                            borderWidth: 1
                        }
                    }
                }
            });

            // ----------------------------------------------------
            // Server-Sent Events (SSE) Live Stream Connection
            // ----------------------------------------------------
            function setupSSE() {
                const eventSource = new EventSource('/api/training/events');
                
                eventSource.onmessage = (e) => {
                    try {
                        const event = JSON.parse(e.data);
                        handleLiveEvent(event);
                    } catch (err) {}
                };

                eventSource.onerror = () => {
                    setTimeout(setupSSE, 3000);
                };
            }

            function handleLiveEvent(event) {
                if (event.type === 'step') {
                    updateMetrics(event.summary);
                    appendChartPoint(event.data);
                } else if (event.type === 'status') {
                    updateMetrics(event.data);
                } else if (event.type === 'log') {
                    appendConsoleLog(event.message);
                } else if (event.type === 'probe') {
                    appendProbeCard(event.data);
                }
            }

            function appendChartPoint(data) {
                lossChart.data.labels.push(`S${data.step}`);
                lossChart.data.datasets[0].data.push(data.train_loss);
                lossChart.data.datasets[1].data.push(data.val_loss);
                lossChart.data.datasets[2].data.push(data.perplexity);
                
                if (lossChart.data.labels.length > 50) {
                    lossChart.data.labels.shift();
                    lossChart.data.datasets[0].data.shift();
                    lossChart.data.datasets[1].data.shift();
                    lossChart.data.datasets[2].data.shift();
                }
                lossChart.update('none');
            }

            function updateMetrics(status) {
                if (!status) return;

                const isTraining = (status.status === 'training');
                const cardDot = document.getElementById('card-status-dot');
                const cardText = document.getElementById('card-status-text');
                const etaText = document.getElementById('card-eta-text');
                const btnStart = document.getElementById('btn-start-train');
                const btnStop = document.getElementById('btn-stop-train');
                const navBadge = document.getElementById('nav-live-badge');

                cardText.textContent = status.status;
                if (isTraining) {
                    cardDot.className = 'w-2.5 h-2.5 rounded-full bg-amber-400 pulse-dot-amber';
                    etaText.textContent = `ETA: ~${status.eta_seconds}s remaining`;
                    btnStart.disabled = true;
                    btnStart.classList.add('opacity-50');
                    btnStop.disabled = false;
                    navBadge.classList.remove('hidden');
                } else {
                    cardDot.className = 'w-2.5 h-2.5 rounded-full bg-emerald-500';
                    etaText.textContent = status.status === 'finished' ? 'Training complete' : 'Ready to train';
                    btnStart.disabled = false;
                    btnStart.classList.remove('opacity-50');
                    btnStop.disabled = true;
                    navBadge.classList.add('hidden');
                }

                document.getElementById('stat-step').textContent = status.current_step;
                document.getElementById('stat-total-steps').textContent = status.total_steps;
                document.getElementById('stat-progress-bar').style.width = `${status.progress_pct}%`;
                document.getElementById('stat-train-loss').textContent = status.train_loss > 0 ? status.train_loss.toFixed(4) : '--';
                document.getElementById('stat-val-loss').textContent = status.val_loss > 0 ? status.val_loss.toFixed(4) : '--';
                document.getElementById('stat-best-loss').textContent = status.best_val_loss ? status.best_val_loss.toFixed(4) : '--';
                document.getElementById('stat-perplexity').textContent = status.perplexity > 0 ? status.perplexity.toFixed(1) : '--';
                document.getElementById('stat-speed').innerHTML = `${Math.round(status.speed_tok_s)} <span class="text-xs font-normal text-slate-400">tok/s</span>`;
                document.getElementById('stat-lr').textContent = status.learning_rate ? status.learning_rate.toExponential(2) : '--';
            }

            function appendConsoleLog(msg) {
                const logBox = document.getElementById('console-logs');
                const line = document.createElement('div');
                line.textContent = msg;
                logBox.appendChild(line);
                logBox.scrollTop = logBox.scrollHeight;
            }

            function appendProbeCard(probe) {
                const probeList = document.getElementById('probe-history-list');
                const card = document.createElement('div');
                card.className = 'p-2.5 rounded-xl bg-slate-800/80 border border-slate-700/60 shadow-md animate-fade-in';
                card.innerHTML = `
                    <div class="flex justify-between text-[10px] text-slate-400 mb-1">
                        <span class="font-bold text-cyan-400">Probe Step ${probe.step}</span>
                        <span class="mono">${new Date(probe.timestamp * 1000).toLocaleTimeString()}</span>
                    </div>
                    <div class="text-[11px] text-slate-300 font-semibold mb-1">Q: "${probe.prompt}"</div>
                    <div class="text-[11px] text-emerald-300/90 bg-slate-900/80 p-2 rounded-lg mono leading-relaxed">${probe.output || '[Generating...]'}</div>
                `;
                probeList.prepend(card);
            }

            async function startTraining() {
                const source = document.getElementById('train-source').value;
                const steps = parseInt(slider.value, 10);
                const probePrompt = document.getElementById('probe-input').value;

                try {
                    const res = await fetch('/api/training/start', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            max_iters: steps,
                            dataset_source: source,
                            probe_prompt: probePrompt
                        })
                    });
                    const data = await res.json();
                    if (!res.ok) alert(data.detail || 'Training failed to start');
                } catch (err) {
                    alert('Could not start training session.');
                }
            }

            async function stopTraining() {
                try {
                    await fetch('/api/training/stop', { method: 'POST' });
                } catch (err) {}
            }

            async function runProbe() {
                const prompt = document.getElementById('probe-input').value;
                if (!prompt) return;
                try {
                    const res = await fetch('/api/training/probe', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ prompt })
                    });
                    const data = await res.json();
                } catch (err) {}
            }

            // Load initial training status & history
            async function initTrainingStatus() {
                try {
                    const res = await fetch('/api/training/status');
                    const data = await res.json();
                    updateMetrics(data);
                    
                    if (data.history && data.history.length > 0) {
                        data.history.forEach(appendChartPoint);
                    }
                    if (data.probes && data.probes.length > 0) {
                        data.probes.forEach(appendProbeCard);
                    }
                    if (data.logs && data.logs.length > 0) {
                        data.logs.forEach(appendConsoleLog);
                    }
                } catch (err) {}
            }

            // ----------------------------------------------------
            // Conversational Chat System Logic
            // ----------------------------------------------------
            const form = document.getElementById('chat-form');
            const input = document.getElementById('user-input');
            const chatBox = document.getElementById('chat-box');
            const typing = document.getElementById('typing');
            let chatHistory = [];

            function addMessage(text, isUser, sources = []) {
                const wrapper = document.createElement('div');
                wrapper.className = `flex flex-col ${isUser ? 'items-end' : 'items-start'} w-full`;
                
                let html = `
                    <div class="message-bubble ${isUser ? 'user-msg' : 'bot-msg'} shadow-xl">
                        <p>${text.replace(/\\n/g, '<br>')}</p>
                    </div>
                `;

                if (sources && sources.length > 0 && !isUser) {
                    html += `
                        <div class="mt-1 text-[10px] text-slate-400 max-w-[85%] px-2">
                            <details class="cursor-pointer">
                                <summary class="font-semibold hover:text-slate-200 transition-colors">📚 Retrieved Semantic Memory (${sources.length} sources)</summary>
                                <ul class="list-disc pl-4 mt-1 space-y-1 bg-slate-800/60 p-2 rounded-xl border border-slate-700/50">
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

            form.onsubmit = async (e) => {
                e.preventDefault();
                const question = input.value.trim();
                if (!question) return;

                input.value = '';
                addMessage(question, true);
                chatHistory.push({ role: "user", content: question });

                typing.classList.remove('hidden');
                chatBox.scrollTo(0, chatBox.scrollHeight);

                try {
                    const res = await fetch('/ask', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ question, history: chatHistory })
                    });
                    
                    const data = await res.json();
                    typing.classList.add('hidden');
                    
                    chatHistory.push({ role: "assistant", content: data.answer });
                    addMessage(data.answer, false, data.retrieved_context || []);
                } catch (err) {
                    typing.classList.add('hidden');
                    addMessage("Communication error. Is the server running?", false);
                }
            };

            // Knowledge File Upload
            const fileUpload = document.getElementById('file-upload');
            fileUpload.onchange = async () => {
                const file = fileUpload.files[0];
                if (!file) return;

                const formData = new FormData();
                formData.append('file', file);
                
                addMessage(`Uploading ${file.name} to global knowledge memory...`, true);
                
                try {
                    const res = await fetch('/upload', { method: 'POST', body: formData });
                    const data = await res.json();
                    if (res.ok) {
                        addMessage(`🎉 Success: ${data.message} (split into ${data.chunks_created || 1} chunks with native embeddings)`, false);
                    } else {
                        addMessage(`❌ Upload failed: ${data.error}`, false);
                    }
                } catch (err) {
                    addMessage("❌ Connection error during upload.", false);
                }
                fileUpload.value = '';
            };

            // Init on load
            setupSSE();
            initTrainingStatus();
        </script>
    </body>
    </html>
    """


@app.get("/api/training/status")
async def get_training_status():
    """Returns JSON telemetry summary."""
    return JSONResponse(content=training_monitor.get_status())


@app.get("/api/training/events")
async def training_events(request: Request):
    """Server-Sent Events (SSE) streaming live training metrics."""
    q = training_monitor.subscribe()

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = q.get_nowait()
                    yield f"data: {json.dumps(event)}\\n\\n"
                except Exception:
                    await asyncio.sleep(0.5)
                    yield f"data: {json.dumps({'type': 'heartbeat', 'time': time.time()})}\\n\\n"
        finally:
            training_monitor.unsubscribe(q)

    import time
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/training/start")
async def start_training(request: Request):
    """Start an on-the-go incremental training run."""
    body = await request.json()
    max_iters = int(body.get("max_iters", 50))
    dataset_source = body.get("dataset_source", "all")
    probe_prompt = body.get("probe_prompt", "How do I treat a mild burn at home?")

    ok, msg = training_monitor.start_training(
        max_iters=max_iters,
        dataset_source=dataset_source,
        probe_prompt=probe_prompt,
    )
    if not ok:
        return JSONResponse(status_code=400, content={"error": msg})
    return JSONResponse(content={"message": msg})


@app.post("/api/training/stop")
async def stop_training():
    """Stop the active training run."""
    ok, msg = training_monitor.stop_training()
    return JSONResponse(content={"message": msg, "success": ok})


@app.post("/api/training/probe")
async def probe_model(request: Request):
    """Probe the model's current weights with a prompt."""
    body = await request.json()
    prompt = body.get("prompt", "What makes a contract valid?")
    output = training_monitor.probe_model(prompt)
    return JSONResponse(content={"prompt": prompt, "output": output})


@app.post("/api/model/reload")
async def reload_model():
    """Explicitly hot-reload the ONNX model from disk into the active RAG system."""
    success = rag.reload_model()
    if success:
        return JSONResponse(content={"message": "Model successfully hot-reloaded into memory."})
    return JSONResponse(status_code=500, content={"error": "Failed to hot-reload model."})


@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")
    history = data.get("history", []) or data.get("conversation_history", [])

    # Pure conversational pipeline
    res = rag.ask(question, history=history)

    return JSONResponse(content={
        "answer": res.get("answer", ""),
        "knowledge_used": res.get("knowledge_used", 0),
        "retrieved_context": res.get("retrieved_context", []),
        "validated": True,
    })


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
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
                    return {"message": f"Successfully ingested {added_count} items into unified memory.", "chunks_created": added_count}
                elif isinstance(data, dict):
                    for k, v in data.items():
                        rag.add_text(f"{k}: {v}", filename)
                    return {"message": "Successfully ingested JSON dataset.", "chunks_created": len(data)}
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Failed to parse JSON dataset: {e}"})

        text = content.decode("utf-8", errors="ignore")
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "File is empty"})

        rag.add_text(text, filename)
        return {
            "message": f"Successfully uploaded '{filename}' to global memory!",
            "chunks_created": len(text) // 500 + 1,
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to process file: {str(e)}"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
