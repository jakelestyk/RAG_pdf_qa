import os, re
import streamlit as st
import pdfplumber
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI

# ─── 0. CONFIG / SECRETS ───────────────────────────────────────────────────────
st.set_page_config(page_title="PDF RAG QA", layout="wide")
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

PDF_PATH = "article.pdf"  # must live alongside this app.py in your repo

# ─── 1. EXTRACT + CLEAN PDF TEXT ───────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def extract_text(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            lines = []
            for ln in txt.split("\n"):
                # drop page numbers, headers/footers, captions
                if re.fullmatch(r"\s*\d+\s*", ln): 
                    continue
                if ln.strip().startswith(("Figure", "Table")):
                    continue
                lines.append(ln)
            pages.append(" ".join(lines))
    full = " ".join(pages)
    full = re.sub(r'(?<=\w)-\s*(?=\w)', '', full)  # fix hyphens
    full = re.sub(r'\s+', ' ', full)
    return full

# ─── 2. CHUNKING ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def chunk_text(text, size=800, overlap=100):
    chunks, i = [], 0
    L = len(text)
    while i < L:
        chunks.append(text[i : i + size])
        i += size - overlap
    return chunks

# ─── 3. BM25 INDEX ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_bm25(chunks):
    tokenized = [re.findall(r"\w+", c.lower()) for c in chunks]
    return BM25Okapi(tokenized), tokenized

# ─── 4. OPENAI EMBEDDINGS + FAISS ─────────────────────────────────────────────
@st.cache_data(show_spinner="Embedding with OpenAI…")
def embed_with_openai(chunks, batch=50):
    all_embs = []
    for i in range(0, len(chunks), batch):
        resp = client.embeddings.create(
            model="text-embedding-ada-002",
            input=chunks[i : i + batch]
        )
        all_embs += [d.embedding for d in resp.data]
    embs = np.array(all_embs, dtype="float32")
    faiss.normalize_L2(embs)
    return embs

@st.cache_data(show_spinner=False)
def build_faiss_index(embs):
    dim = embs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(embs)
    return idx

# ─── 5. HYBRID RETRIEVAL ────────────────────────────────────────────────────────
def retrieve(query, chunks, bm25, tokenized, embs, fidx, full, top_n=20, k=5):
    ql = query.lower()
    # simple regex fallbacks
    if "dataset" in ql:
        m = re.search(r'([^.]*benchmark dataset[^.]*)\.', full, re.IGNORECASE)
        if m: return [m.group(1) + "."]
    if "accuracy" in ql:
        m = re.search(r'(\d+(?:\.\d+)?%)', full)
        if m: return [m.group(1)]
    # BM25 stage
    tokens = re.findall(r"\w+", ql)
    scores = bm25.get_scores(tokens)
    bm25_ids = np.argsort(scores)[-top_n:][::-1]
    # semantic re‑rank
    q_emb = client.embeddings.create(
        model="text-embedding-ada-002", input=[query]
    ).data[0].embedding
    q_vec = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_vec)
    sims = [(float(np.dot(embs[i], q_vec[0])), i) for i in bm25_ids]
    sims.sort(reverse=True)
    chosen = [idx for _, idx in sims[:k]]
    return [chunks[i] for i in chosen]

# ─── 6. LLM ANSWERING ───────────────────────────────────────────────────────────
def generate_answer(question, context):
    prompt = (
        "You are a helpful assistant. Answer only from the context below.\n"
        "If not found, reply exactly: \"Information not found in the document.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content": prompt}],
        temperature=0.0,
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()

# ─── 7. STREAMLIT UI ────────────────────────────────────────────────────────────
st.title("📄 PDF RAG Question‑Answering")
st.write("Ask anything about the bundled research paper (article.pdf).")

# load & index once
with st.spinner("Indexing PDF…"):
    full_text = extract_text(PDF_PATH)
    if not full_text:
        st.error(f"Couldn’t load {PDF_PATH}")
        st.stop()
    chunks = chunk_text(full_text)
    bm25, tok = build_bm25(chunks)
    embs = embed_with_openai(chunks)
    fidx = build_faiss_index(embs)

# user input
q = st.text_input("Enter your question:")
if q:
    with st.spinner("Retrieving context…"):
        ctxs = retrieve(q, chunks, bm25, tok, embs, fidx, full_text)
    st.subheader("🔍 Retrieved Context")
    for i, c in enumerate(ctxs, 1):
        snippet = c.strip()
        if len(snippet) > 200:
            snippet = snippet[:197].rstrip() + "…"
        st.markdown(f"**{i}.** {snippet}")
    with st.spinner("Generating answer…"):
        ans = generate_answer(q, "\n".join(ctxs))
    st.subheader("💡 Answer")
    st.write(ans)
