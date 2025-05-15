#!/usr/bin/env python3
import os
import re
import streamlit as st
import pdfplumber
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ─── Streamlit config ─────────────────────────────────────────────────────────────
st.set_page_config(page_title="PDF RAG QA", layout="wide")

# ─── 1. BUNDLED PDF ────────────────────────────────────────────────────────────────
DEFAULT_PDF = "article.pdf"   # put your PDF at repo root

# ─── 2. TEXT EXTRACTION ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_text(src):
    if isinstance(src, str) and not os.path.exists(src):
        st.error(f"PDF not found at `{src}`")
        return ""
    texts = []
    with pdfplumber.open(src) as pdf:
        for pg in pdf.pages:
            t = pg.extract_text() or ""
            lines = []
            for ln in t.split("\n"):
                # drop page numbers, headers/footers, figure/table captions
                if re.fullmatch(r"\s*\d+\s*", ln): continue
                if ln.strip().startswith(("Figure","Table")): continue
                lines.append(ln)
            texts.append(" ".join(lines))
    full = " ".join(texts)
    full = re.sub(r'(?<=\w)-\s*(?=\w)', "", full)
    full = re.sub(r"\s+", " ", full)
    return full

# ─── 3. CHUNKING ───────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def chunk_text(txt, size=800, overlap=100):
    out, i = [], 0
    while i < len(txt):
        out.append(txt[i:i+size])
        i += size - overlap
    return out

# ─── 4. BM25 INDEX ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_bm25(chunks):
    toks = [re.findall(r"\w+", c.lower()) for c in chunks]
    return BM25Okapi(toks), toks

# ─── 5. LOCAL EMBEDDINGS + FAISS ──────────────────────────────────────────────────
st.spinner("Loading local embedder…")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def embed_chunks(chunks):
    embs = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    faiss.normalize_L2(embs)
    return embs

@st.cache_data(show_spinner=False)
def build_faiss(embs):
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return idx

# ─── 6. FLAN‑T5 SETUP ──────────────────────────────────────────────────────────────
st.spinner("Loading Flan‑T5…")
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_answer(question, context):
    prompt = (
        "You are a helpful assistant. Answer only from the context below.\n"
        "If not found, say 'Information not found in the document.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    out = gen_model.generate(**inputs, max_new_tokens=100, no_repeat_ngram_size=2)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ─── 7. HYBRID RETRIEVAL ────────────────────────────────────────────────────────────
def hybrid_retrieve(query, chunks, bm25, toks, embs, faiss_idx, full_text, top_n=20, k=5):
    ql = query.lower()

    # simple fallback for well‑known patterns
    if "dataset" in ql:
        m = re.search(r"([^.]*benchmark dataset[^.]*)\.", full_text, re.IGNORECASE)
        if m: return [m.group(1)+"."]

    # BM25
    q_tokens = re.findall(r"\w+", ql)
    bm25_scores = bm25.get_scores(q_tokens)
    top_idxs = np.argsort(bm25_scores)[-top_n:][::-1]

    # semantic re‑rank
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    sims = [(float(np.dot(embs[i], q_emb[0])), i) for i in top_idxs]
    sims.sort(reverse=True)

    chosen = [i for _,i in sims[:k]]
    return [chunks[i] for i in chosen]

# ─── 8. STREAMLIT UI ──────────────────────────────────────────────────────────────
st.title("📄 PDF RAG QA Pipeline")
st.write("Upload a PDF or leave blank to use the bundled research article.")

uploaded = st.file_uploader("Choose PDF", type="pdf")
source = uploaded if uploaded else DEFAULT_PDF

with st.spinner("Indexing…"):
    text = extract_text(source)
    if not text:
        st.stop()
    chunks = chunk_text(text)
    bm25, toks = build_bm25(chunks)
    embs = embed_chunks(chunks)
    idx = build_faiss(embs)

q = st.text_input("Enter your question:")
if q:
    with st.spinner("Retrieving…"):
        ctxs = hybrid_retrieve(q, chunks, bm25, toks, embs, idx, text)
    st.subheader("🔍 Retrieved Context")
    for i,c in enumerate(ctxs,1):
        st.write(f"**[{i}]** {c}")
    with st.spinner("Answering…"):
        answer = generate_answer(q, "\n".join(ctxs))
    st.subheader("💡 Answer")
    st.write(answer)
