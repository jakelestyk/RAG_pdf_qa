#!/usr/bin/env python3
import os
import re
import streamlit as st
import pdfplumber
import numpy as np
import faiss
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAIError
from rank_bm25 import BM25Okapi
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

st.set_page_config(page_title="PDF RAG QA Pipeline", layout="wide")


#
# ─── 1. CONFIGURE OPENAI ─────────────────────────────────────────────────────────
#
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("⚠️ Please set your OPENAI_API_KEY in your environment (or in Streamlit Cloud secrets).")
    st.stop()


#
# ─── 2. LOCATE BUNDLED PDF ───────────────────────────────────────────────────────
#
DEFAULT_PDF = "article.pdf"  # your research PDF at repo root


#
# ─── 3. TEXT EXTRACTION ───────────────────────────────────────────────────────────
#
@st.cache_data(show_spinner=False)
def extract_text(path_or_bytes):
    if isinstance(path_or_bytes, str) and not os.path.exists(path_or_bytes):
        st.error(f"Bundled PDF not found at {path_or_bytes}")
        return ""
    pages = []
    with pdfplumber.open(path_or_bytes) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            lines = []
            for ln in txt.split("\n"):
                # drop page numbers, headers/footers & figure/table captions
                if re.match(r"^\s*\d+\s*$", ln):
                    continue
                if ln.strip().startswith(("Figure", "Table")):
                    continue
                lines.append(ln)
            pages.append(" ".join(lines))
    full = " ".join(pages)
    full = re.sub(r'(?<=\w)-\s*(?=\w)', "", full)
    full = re.sub(r"\s+", " ", full)
    return full


#
# ─── 4. CHUNKING ─────────────────────────────────────────────────────────────────
#
@st.cache_data(show_spinner=False)
def chunk_text(text, size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


#
# ─── 5. BM25 INDEX ───────────────────────────────────────────────────────────────
#
@st.cache_data(show_spinner=False)
def build_bm25(chunks):
    tokenized = [re.findall(r"\w+", c.lower()) for c in chunks]
    return BM25Okapi(tokenized), tokenized


#
# ─── 6. OPENAI EMBEDDINGS + FAISS ─────────────────────────────────────────────────
#
@retry(
    retry=retry_if_exception_type(OpenAIError),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(6),
)
def _call_openai_embeddings(batch: list[str]) -> list[list[float]]:
    resp = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=batch,
    )
    return [d["embedding"] for d in resp["data"]]


@st.cache_data(show_spinner=False)
def embed_with_openai(chunks, batch_size=50):
    all_embs = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        embs = _call_openai_embeddings(batch)
        all_embs.extend(embs)
    arr = np.array(all_embs, dtype="float32")
    faiss.normalize_L2(arr)
    return arr


@st.cache_data(show_spinner=False)
def build_faiss_index(embs):
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embs)
    return index


#
# ─── 7. FLAN‑T5 SETUP ─────────────────────────────────────────────────────────────
#
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")


def generate_answer(question: str, contexts: str) -> str:
    prompt = (
        "You are a helpful assistant. Answer only from the context below. "
        "If the answer is not there, respond with 'Information not found in the document.'\n\n"
        f"Context:\n{contexts}\n\nQuestion: {question}\nAnswer:"
    )
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        out = gen_model.generate(**inputs, max_new_tokens=100, no_repeat_ngram_size=2)
        return tokenizer.decode(out[0], skip_special_tokens=True)
    except Exception:
        return "Information not found in the document."


#
# ─── 8. HYBRID RETRIEVAL ───────────────────────────────────────────────────────────
#
def retrieve(
    query: str,
    chunks: list[str],
    bm25: BM25Okapi,
    tokenized: list[list[str]],
    embs: np.ndarray,
    faiss_idx,
    full_text: str,
    top_n=20,
    k=5,
):
    ql = query.lower()
    # rule‑based fallbacks
    if "dataset" in ql:
        m = re.search(r"([^.]*benchmark dataset[^.]*)\.", full_text, flags=re.IGNORECASE)
        if m:
            return [m.group(1) + "."]
    if "accuracy" in ql:
        m = re.search(r"(\d+(?:\.\d+)?%)", full_text)
        if m:
            return [m.group(1)]
    # BM25 pass
    tokens = re.findall(r"\w+", ql)
    scores = bm25.get_scores(tokens)
    bm25_idxs = np.argsort(scores)[-top_n:][::-1]
    # embed query + rerank
    q_emb = openai.Embedding.create(model="text-embedding-ada-002", input=[query])["data"][0]["embedding"]
    q_arr = np.array(q_emb, dtype="float32").reshape(1, -1)
    faiss.normalize_L2(q_arr)
    sims = [(float(np.dot(embs[i], q_arr[0])), i) for i in bm25_idxs]
    sims.sort(reverse=True)
    chosen = [i for _, i in sims[:k]]
    return [chunks[i] for i in chosen]


#
# ─── 9. STREAMLIT UI ───────────────────────────────────────────────────────────────
#
st.title("📄 PDF RAG QA Pipeline")
st.write("Upload a PDF or leave blank to use the bundled research article.")

uploaded = st.file_uploader("Choose PDF", type="pdf")
source = uploaded if uploaded else DEFAULT_PDF

with st.spinner("Indexing PDF..."):
    full_text = extract_text(source)
    if not full_text:
        st.stop()
    chunks = chunk_text(full_text)
    bm25, tok = build_bm25(chunks)
    embs = embed_with_openai(chunks)
    faiss_idx = build_faiss_index(embs)

q = st.text_input("Enter your question:")
if q:
    with st.spinner("Retrieving..."):
        ctx = retrieve(q, chunks, bm25, tok, embs, faiss_idx, full_text)
    st.subheader("🔍 Retrieved Context")
    for i, c in enumerate(ctx, 1):
        st.write(f"**[{i}]** {c}")
    with st.spinner("Generating answer..."):
        ans = generate_answer(q, "\n".join(ctx))
    st.subheader("💡 Answer")
    st.write(ans)
