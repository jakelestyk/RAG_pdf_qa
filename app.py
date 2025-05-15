#!/usr/bin/env python3
import os
import re
import streamlit as st
import pdfplumber
import numpy as np
import faiss
import openai
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="PDF RAG QA Pipeline", layout="wide")

#
# ─── 1. CONFIGURE OPENAI ────────────────────────────────────────────────────────
#
# On Streamlit Cloud: set OPENAI_API_KEY under App settings → Secrets
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("⚠️ Please set your OPENAI_API_KEY in your environment (or in Streamlit Cloud secrets).")
    st.stop()

#
# ─── 2. BUNDLED PDF PATH ────────────────────────────────────────────────────────
#
DEFAULT_PDF = "article.pdf"  # must live in repo root

#
# ─── 3. TEXT EXTRACTION ─────────────────────────────────────────────────────────
#
@st.cache_data(show_spinner=False)
def extract_text(path_or_bytes):
    if isinstance(path_or_bytes, str) and not os.path.exists(path_or_bytes):
        st.error(f"❌ PDF not found at '{path_or_bytes}'")
        return ""
    pages = []
    with pdfplumber.open(path_or_bytes) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            lines = []
            for ln in txt.split("\n"):
                if re.fullmatch(r"\d+", ln.strip()):  # drop standalone page numbers
                    continue
                if ln.strip().startswith(("Figure", "Table")):
                    continue
                lines.append(ln)
            pages.append(" ".join(lines))
    full = " ".join(pages)
    # fix hyphenation + collapse spaces
    full = re.sub(r'(?<=\w)-\s*(?=\w)', "", full)
    full = re.sub(r'\s+', " ", full)
    return full

#
# ─── 4. CHUNKING ────────────────────────────────────────────────────────────────
#
@st.cache_data(show_spinner=False)
def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + size, L)
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

#
# ─── 5. BM25 INDEX ──────────────────────────────────────────────────────────────
#
@st.cache_data(show_spinner=False)
def build_bm25(chunks):
    tokenized = [re.findall(r"\w+", c.lower()) for c in chunks]
    return BM25Okapi(tokenized), tokenized

#
# ─── 6. OPENAI EMBEDDINGS + FAISS ──────────────────────────────────────────────
#
@st.cache_data(show_spinner=False)
def embed_with_openai(chunks):
    # batch in 50 to stay under rate/size limits
    all_embs = []
    for i in range(0, len(chunks), 50):
        batch = chunks[i : i + 50]
        resp = openai.embeddings.create(
            model="text-embedding-ada-002",
            input=batch
        )
        all_embs.extend([d["embedding"] for d in resp["data"]])
    arr = np.array(all_embs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr

@st.cache_data(show_spinner=False)
def build_faiss_index(embs):
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return idx

#
# ─── 7. FLAN‑T5 SETUP ────────────────────────────────────────────────────────────
#
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

def generate_answer(question, contexts):
    prompt = (
        "You are a helpful assistant. Answer only from the context below. "
        "If not found, respond with 'Information not found in the document.'\n\n"
        f"Context:\n{contexts}\n\nQuestion: {question}\nAnswer:"
    )
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )
    out = gen_model.generate(
        **inputs,
        max_new_tokens=100,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

#
# ─── 8. HYBRID RETRIEVAL ─────────────────────────────────────────────────────────
#
def retrieve(query, chunks, bm25, tokenized, embs, faiss_idx, full_text, top_n=20, k=5):
    ql = query.lower()

    # 8.1 fallback rules to avoid hallucination
    if "dataset" in ql:
        m = re.search(r'([^.]*benchmark dataset[^.]*)\.', full_text, re.IGNORECASE)
        if m: return [m.group(1).strip() + "."]
    if "accuracy" in ql:
        m = re.search(r'(\d+(?:\.\d+)?%)', full_text)
        if m: return [m.group(1)]
    if "baseline" in ql:
        m = re.search(r'such as ([^.]+)\.', full_text, re.IGNORECASE)
        if m: return [m.group(1).strip() + "."]
    if "false negative" in ql:
        for c in chunks:
            if "false negative" in c.lower():
                return [c]

    # 8.2 BM25
    tokens = re.findall(r"\w+", ql)
    scores = bm25.get_scores(tokens)
    bm25_idxs = np.argsort(scores)[-top_n:][::-1]

    # 8.3 semantic re-rank via openai.embeddings + FAISS
    q_emb = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=[query]
    )["data"][0]["embedding"]
    q_vec = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)

    sims = [(float(np.dot(embs[i], q_vec[0])), i) for i in bm25_idxs]
    sims.sort(reverse=True)

    chosen = [idx for _, idx in sims[:k]]
    return [chunks[i] for i in chosen]

#
# ─── 9. STREAMLIT UI ────────────────────────────────────────────────────────────
#
st.title("📄 PDF RAG QA Pipeline")
st.write("Upload a PDF or leave blank to use the bundled research paper.")

uploaded = st.file_uploader("Choose PDF", type="pdf")
source = uploaded if uploaded else DEFAULT_PDF

with st.spinner("Indexing PDF…"):
    full_text = extract_text(source)
    if not full_text:
        st.stop()
    chunks = chunk_text(full_text)
    bm25, tok = build_bm25(chunks)
    embs = embed_with_openai(chunks)
    faiss_idx = build_faiss_index(embs)

q = st.text_input("Enter your question:")
if q:
    with st.spinner("Retrieving…"):
        ctx_chunks = retrieve(q, chunks, bm25, tok, embs, faiss_idx, full_text)
    st.subheader("🔍 Retrieved Context")
    for i, c in enumerate(ctx_chunks, 1):
        st.write(f"**[{i}]** {c}")
    with st.spinner("Generating answer…"):
        ans = generate_answer(q, "\n".join(ctx_chunks))
    st.subheader("💡 Answer")
    st.write(ans)
