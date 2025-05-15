#!/usr/bin/env python3
import streamlit as st
import sys, re
import pdfplumber
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ──────────────────── Config & Models ────────────────────
st.set_page_config(page_title="PDF RAG QA", layout="wide")
PDF_PATH = "data/Credit_Card_Fraud_Detection_Using_State-of-the-Art_Machine_Learning_and_Deep_Learning_Algorithms.pdf"

@st.cache_data(show_spinner=False)
def extract_raw_text(path_or_bytes):
    pages = []
    # path_or_bytes can be a file‐path or uploaded BytesIO
    with pdfplumber.open(path_or_bytes) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            lines = [ln for ln in txt.split("\n")
                     if not ln.strip().startswith(("Figure","Table"))]
            pages.append(" ".join(lines))
    full = " ".join(pages)
    full = re.sub(r'(?<=\w)-\s*(?=\w)', '', full)  # fix hyphens
    full = re.sub(r'\s+', ' ', full)               # normalize spaces
    return full

@st.cache_data(show_spinner=False)
def split_sentences(full_text):
    sents = re.split(r'(?<=[\.!?])\s+', full_text)
    return [s.strip() for s in sents if len(s.strip()) >= 50]

@st.cache_data(show_spinner=False)
def build_bm25(sentences):
    tokenized = [re.findall(r"\w+", s.lower()) for s in sentences]
    return BM25Okapi(tokenized), tokenized

# load embedding model once
@st.cache_data(show_spinner=False)
def load_embed_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embed_model = load_embed_model()

@st.cache_data(show_spinner=False)
def embed_sentences(sentences):
    embs = embed_model.encode(sentences, convert_to_numpy=True)
    faiss.normalize_L2(embs)
    return embs

@st.cache_data(show_spinner=False)
def build_faiss_index(embs):
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return idx

# load Flan-T5 once
@st.cache_data(show_spinner=False)
def load_t5():
    tok = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    return tok, model

tokenizer, gen_model = load_t5()

# ──────────────────── Retrieval & Generation ────────────────────
def retrieve(query, sentences, bm25, tokenized, embs, faiss_idx, full_text, top_n=20, k=3):
    ql = query.lower()
    # 1) dataset fallback
    if "dataset" in ql:
        m = re.search(r'([^.]*benchmark dataset[^.]*)\.', full_text, re.IGNORECASE)
        if m: return [m.group(1).strip() + "."]
    # 2) accuracy fallback
    if "accuracy" in ql:
        m = re.search(r'(\d+(?:\.\d+)?%)', full_text)
        if m: return [m.group(1)]
    # 3) baseline methods fallback
    if "baseline" in ql or ("machine learning" in ql and "methods" in ql):
        m = re.search(r'such as ([^.]+)\.', full_text, re.IGNORECASE)
        if m: return [m.group(1).strip() + "."]
    # 4) false negative fallback
    if "false negative" in ql:
        for s in sentences:
            if "false negative" in s.lower(): return [s]
    # 5) BM25 lexical
    tokens = re.findall(r"\w+", ql)
    scores = bm25.get_scores(tokens)
    idxs = np.argsort(scores)[-top_n:][::-1]
    # 6) embedding re-rank
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    sims = [(float(np.dot(embs[i], q_emb[0])), i) for i in idxs]
    sims.sort(reverse=True)
    chosen = [i for _, i in sims[:k]]
    return [sentences[i] for i in chosen]

def generate_answer(question, contexts):
    ctx = " ".join(contexts)
    prompt = (
        "You are a helpful assistant. Answer only from the context below.\n"
        "If not found, say 'Information not found in the document.'\n\n"
        f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:"
    )
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = gen_model.generate(**enc, max_new_tokens=75, no_repeat_ngram_size=2)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ──────────────────── Streamlit UI ────────────────────
st.title("📄 PDF RAG Question–Answering")
st.write("This app answers questions from a pre-loaded research PDF.")

# load & process PDF once
with st.spinner("Loading and indexing PDF…"):
    full_text = extract_raw_text(PDF_PATH)
    sentences = split_sentences(full_text)
    bm25, tokenized = build_bm25(sentences)
    embs = embed_sentences(sentences)
    faiss_idx = build_faiss_index(embs)

# question box
q = st.text_input("Enter your question:")
if q:
    with st.spinner("Retrieving context…"):
        ctxs = retrieve(q, sentences, bm25, tokenized, embs, faiss_idx, full_text)
    st.subheader("🔍 Retrieved Context")
    for i,c in enumerate(ctxs, 1):
        st.write(f"**[{i}]** {c}")
    with st.spinner("Generating answer…"):
        ans = generate_answer(q, ctxs)
    st.subheader("💡 Answer")
    st.write(ans)
