<p align="center"> <img src="assets/banner.png" width="80%" /> </p> <p align="center"> <img src="https://img.shields.io/github/v/release/mahmutcanborann/BMW-AI-Assistant?color=blue" /> <img src="https://img.shields.io/github/stars/mahmutcanborann/BMW-AI-Assistant?style=social" /> <img src="https://img.shields.io/badge/LLM-Ollama-blue" /> <img src="https://img.shields.io/badge/RAG-ChromaDB-green" /> <img src="https://img.shields.io/badge/Frontend-Streamlit-ff4b4b" /> </p>
# 🚗 BMW AI Assistant (Universal RAG Prototype)

BMW AI Assistant is a local, privacy-preserving RAG-based AI assistant prototype designed for BMW vehicles.

It demonstrates how Retrieval-Augmented Generation (RAG) can be applied to automotive documentation and diagnostics using fully offline, on-device AI, without relying on cloud APIs or external services.

The system intentionally prioritizes safety over completeness.
When reliable, vehicle-specific information is unavailable, the assistant avoids speculation and returns a safe fallback response.

🎯 Project Goals

Explore real-world challenges of building automotive RAG systems

Combine BMW owner manuals and OBD-II fault codes into a single assistant

Handle model-year ambiguity and legacy documentation conflicts

Demonstrate guardrails and conservative LLM behavior in a safety-relevant domain

This project is learning-focused and exploratory, not a production-ready system.

🔍 What It Can Do

Retrieve OBD-II fault code explanations (CSV-based)

Read and extract information from BMW owner manuals (PDF)

Smart query routing

Error code only → CSV lookup

Model / feature queries → Manuals RAG

Mixed input → Combines both sources

Return safe fallback responses when confident answers are unavailable

Run entirely offline (no API keys, no internet required)

Provide a modern Streamlit UI for fast Q&A

⚠️ Known Challenge: Legacy Documentation Leakage

Due to limited public access to model- and VIN-specific BMW documentation,
the system may occasionally retrieve outdated or mismatched manuals.

This phenomenon—often referred to as legacy documentation leakage—is a known and realistic challenge in automotive RAG systems.

In this prototype:

The issue is intentionally surfaced, not hidden

Guardrails are used to reduce unsafe answers

The assistant prefers deferring over guessing

🛠 Technical Overview

Built with:

LangChain – orchestration & prompt chaining

ChromaDB – local vector database

Ollama – fully local LLM runtime

Streamlit – interactive UI

The assistant processes:

BMW owner manuals (PDF)

OBD-II fault codes (CSV)

Natural language user queries

to deliver source-grounded diagnostic explanations and practical troubleshooting guidance.

⚙️ Installation

Install dependencies:

pip install -r requirements.txt

Install the LLM model via Ollama:

ollama pull llama3.2:3b

📦 Preparing the Database

Place BMW owner manuals inside:

/data/manuals/

Generate embeddings and build the vector database:

python main.py

This will create:

/chroma_db/

Note:
The Chroma database is not included in the repository and is rebuilt locally for performance and flexibility.

▶ Run the Application

Start the Streamlit app:

streamlit run app.py

💬 Example Queries

What does P0300 mean?

How to change driving modes in a BMW?

How to pair an iPhone in BMW 3 Series?

⚠️ Known Limitations

Page-level citations are not guaranteed and may occasionally appear due to generative model behavior

Feature availability varies by model year and equipment

Newer BMW manuals often require VIN-based access and may not be fully covered

When reliable vehicle-specific data is unavailable, the assistant intentionally returns a safe fallback response

🤝 Contributing

Contributions, feedback, and suggestions are welcome.

If you find this project useful:

⭐ Star the repository

🛠 Open an issue or pull request

💬 Share feedback on RAG behavior and edge cases