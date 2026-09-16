# 📊 SignalGraph
*Trace the signal. Understand the decision.*

[![Python Version](https://img.shields.io/badge/Python-3.13+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.6+-FF6F00?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain-ai.github.io/langgraph/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.24+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=ollama&logoColor=white)](https://ollama.ai/)
[![Package Manager](https://img.shields.io/badge/uv-Package_Manager-DE5FE9?style=for-the-badge&logo=astral&logoColor=white)](https://docs.astral.sh/uv/)

A clean, developer-focused financial research dashboard and agentic screening pipeline built with **LangGraph**, **Streamlit**, **Plotly**, and **Yahoo Finance**.

Rather than an opaque AI black-box, **SignalGraph** visualizes every node in the screening lifecycle: from live market ingestion and technical indicators to fundamentals, risk checks, and final evidence-based decisions.

---

## 🏗️ LangGraph Pipeline Architecture

```text
┌─────────────┐     ┌─────────────┐     ┌──────────────┐     ┌────────────┐     ┌───────────────────┐
│ Market Data │ ──> │ Technicals  │ ──> │ Fundamentals │ ──> │ Risk Check │ ──> │ Screener Decision │
│  (yfinance) │     │ (SMA/RSI/   │     │ (Valuation/  │     │   (Beta/   │     │  (PASS / WATCH /  │
│             │     │    MACD)    │     │   Margins)   │     │  Drawdown) │     │       AVOID)      │
└─────────────┘     └─────────────┘     └──────────────┘     └────────────┘     └───────────────────┘
```

---

## ✨ Features

- 🖥️ **Minimalist Financial Dashboard**: Clean, data-first Streamlit UI without AI gimmicks, glowing gradients, or robot illustrations.
- 📈 **Interactive Technical Charts**: Plotly candlestick charts with SMA 20, SMA 50 overlays, volume bars, and 14-period RSI subplots.
- 🔍 **Observable Workflow Execution**: Horizontal node pipeline showing execution status and millisecond latency for each LangGraph stage.
- 📋 **Structured Screening Snapshot**: Summarized technical trend, momentum, valuation, and risk profile in a compact table.
- 🎯 **Evidence-Driven Decision**: Final **PASS / WATCH / AVOID** verdict backed by 2–3 precise analytical bullet reasons.
- 🤖 **Dual Interface**:
  - **Web Dashboard**: `uv run streamlit run app.py`
  - **Conversational CLI Agent**: `uv run flow.py`

---

## 📂 Project Structure

```text
├── app.py                  # Streamlit web dashboard (SignalGraph UI)
├── pipeline.py             # LangGraph 5-node screening workflow & state machine
├── flow.py                 # Conversational CLI agent with checkpointer memory
├── tool.py                 # Financial tools (screener, ticker details, news)
├── pyproject.toml          # Project configuration & dependencies
├── README.md               # Documentation
└── uv.lock                 # Dependency lockfile
```

---

## 🚀 Quick Start

### 1. Install Dependencies

Using [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

### 2. Launch the Streamlit Dashboard

```bash
uv run streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

### 3. (Optional) Run the CLI Agent

To run the conversational ReAct agent in your terminal:
```bash
uv run flow.py
```

---

## ⚙️ Configuration

To customize the local Ollama model for the CLI agent:

```bash
# Windows (PowerShell)
$env:OLLAMA_MODEL = "qwen2.5:7b"
$env:OLLAMA_BASE_URL = "http://localhost:11434"
uv run flow.py
```

---

## 📄 License

Open-source under MIT / Apache-2.0. For educational and research purposes only. Not individualized financial advice.
