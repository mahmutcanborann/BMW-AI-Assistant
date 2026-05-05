# 🚗 BMW AI Assistant — Advanced RAG System

<p align="center">
  <img src="assets/banner.png" width="85%" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/LLM-Ollama-blue" />
  <img src="https://img.shields.io/badge/RAG-Hybrid-green" />
  <img src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b" />
  <img src="https://img.shields.io/badge/Backend-FastAPI-009688" />
</p>

## 🧠 Overview

BMW AI Assistant is a privacy-first, fully local RAG system designed for BMW vehicles.

It combines:

- Owner manuals (PDF)
- OBD-II error codes (CSV)
- Natural language queries

into a single intelligent assistant that prioritizes accuracy, safety, and reliability over guesswork.

The system uses metadata-aware retrieval and controlled generation to avoid speculative or hallucinated responses.

> ⚠️ The system avoids hallucinations and returns safe fallback responses when information is uncertain.

---

## 🖥️ UI Preview

### 🏠 Home Screen

<p align="center">
  <img src="assets/ui-main.png" width="85%" />
</p>

### 🤖 Response View

<p align="center">
  <img src="assets/ui-answer.png" width="85%" />
</p>

---

## ⚡ What Makes This Different

This is not a basic RAG demo.
It is a multi-stage retrieval system with safety-aware design.

### 🔍 Retrieval Layer

- Hybrid retrieval (BM25 + Vector Search)
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking

### 🧩 Query Intelligence

- Query normalization
- LLM-based query rewriting
- Feature-aware query expansion

### 🚗 BMW-Aware Logic

- Model detection
- Year extraction
- Generation mapping
- Error code recognition

### 🛡 Safety Layer

- Guardrails to reduce hallucinations
- Context-grounded answers only
- Controlled fallback responses

---

## 🏗 System Architecture

<p align="center">
  <img src="assets/architecture.png" width="80%" />
</p>

---

## 🛠 Tech Stack

- FastAPI
- Streamlit
- LangChain
- Chroma
- Ollama

### Models

- Embedding: `all-MiniLM-L6-v2`
- Reranker: `ms-marco-MiniLM-L-6-v2`
- LLM: `llama3.2:3b`

---

## 📁 Project Structure

```bash
project/
│
├── main.py
├── app.py
├── data/
├── chroma_db/
├── splits_cache.pkl
├── assets/
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/mahmutcanborann/BMW-AI-Assistant.git
cd BMW-AI-Assistant

python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Mac/Linux

pip install -r requirements.txt
```

---

## 🤖 Setup Local LLM

```bash
ollama pull llama3.2:3b
```

---

## 📦 Run Backend

```bash
uvicorn main:app --reload
```

---

## 🚀 Run UI

```bash
streamlit run app.py
```

---

## 📂 Add Your Own Manuals

Place your PDF manuals inside the `data/` folder:

```bash
data/
├── BMW_3_series_2020.pdf
├── BMW_X5_2019.pdf
```

### Naming format

```bash
BMW_<MODEL>_<YEAR>.pdf
```

---

## 💬 Example Queries

- BMW 320i 2020 how to turn on headlights
- What does error code P0456 mean?
- Apple CarPlay setup BMW G20
- How to reset oil service light

---

## ⚠️ Limitations

- No VIN-level precision
- Depends on available manuals
- Not intended for mechanical diagnosis

---

## 🧠 Key Insight

> A reliable RAG system is not about retrieving more data —
> it is about knowing when not to answer.

---

## 👨‍💻 Author

Mahmut Can Boran
AI Engineer (RAG Systems & Automotive AI)

🔗 https://www.linkedin.com/in/mahmut-can-boran/
