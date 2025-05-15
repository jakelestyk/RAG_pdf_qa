Retrieval‑Augmented Generation (RAG) for PDF Question Answering
This repository contains a Streamlit‑based RAG system that
ingests a research‑article PDF, builds both BM25 and vector indexes,
and answers natural‑language questions by retrieving relevant sentences
and generating answers with a sequence‑to‑sequence model.

📁 Repository Structure
kotlin
Copy
Edit
RAG-BSAN6200-Final-Project/
├── data/
│   └── Credit_Card_Fraud_Detection_Using_State‑of‑the‑Art_Machine_Learning_and_Deep_Learning_Algorithms.pdf
├── app.py
└── requirements.txt
data/ — holds the target PDF.

app.py — the Streamlit application.

requirements.txt — pinned dependencies.

🎯 Project Goal
Build a modular RAG pipeline that:

Extracts raw text from a research PDF.

Chunks into sentence‑level passages.

Indexes via BM25 (lexical) and FAISS (semantic).

Retrieves precise fallbacks for common fact queries.

Re‑ranks with sentence‑embeddings.

Generates final answers with Flan‑T5, constrained to retrieved context.

🚀 Quick Start
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/<your‑username>/RAG-BSAN6200-Final-Project.git
cd RAG-BSAN6200-Final-Project
2. Install dependencies
bash
Copy
Edit
python3 -m venv .venv
source .venv/bin/activate        # Linux / macOS
.\.venv\Scripts\activate         # Windows PowerShell

pip install -r requirements.txt
3. Run locally
bash
Copy
Edit
streamlit run app.py
Your browser will open at http://localhost:8501.

☁️ Deploy to Streamlit Cloud
Push this repo (with app.py, requirements.txt, and the data/ folder) to GitHub.

Log in to streamlit.io/cloud and click “New app.”

Connect your GitHub repo, and set:

Main file path: app.py

Branch: main (or your chosen branch)

Click Deploy.

Streamlit will automatically install from requirements.txt and load the PDF from data/.

🛠️ How It Works
Text Extraction

Skips figure/table captions.

Cleans line‑break hyphens & normalizes whitespace.

Sentence Splitting

Regex split on . !? boundaries.

Filters out very short fragments.

Indexing

BM25 for keyword queries.

FAISS with MiniLM embeddings for semantic search.

Fallback Rules

dataset → “benchmark dataset” sentence.

accuracy → first % match.

baseline → methods after “such as …”.

false negative → first sentence mentioning it.

Answer Generation

Flan‑T5 reads only the retrieved context.

Falls back to “Information not found in the document.” if missing.

🔧 Dependencies
See requirements.txt for exact versions:

streamlit

pdfplumber

numpy

faiss‑cpu

rank‑bm25

sentence‑transformers

transformers

torch

📄 License
This project is released under the MIT License.
Feel free to adapt for your own RAG pipelines!
