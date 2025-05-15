@@ -1,6 +1,6 @@
#!/usr/bin/env python3
import streamlit as st
import os, sys, re
import os, re
import pdfplumber
import numpy as np
import faiss
@@ -10,44 +10,43 @@

st.set_page_config(page_title="PDF RAG QA", layout="wide")

# 1. Point at the on‑disk PDF in data/
# ─────────── Configuration ───────────
# Point at the bundled PDF in the repo root
DEFAULT_PDF_PATH = "article.pdf"
# 2. Text extraction

# ─────────── Text Extraction ───────────
@st.cache_data(show_spinner=False)
def extract_raw_text(path_or_bytes):
    # if we got a string path, verify it exists
    # Accept either uploaded BytesIO or on‑disk path
    if isinstance(path_or_bytes, str) and not os.path.exists(path_or_bytes):
        st.error(f"Bundled PDF not found at {path_or_bytes}")
        st.error(f"Bundled PDF not found at '{path_or_bytes}'")
        return ""
    pages = []
    with pdfplumber.open(path_or_bytes) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            lines = [
                ln for ln in txt.split("\n")
                if not ln.strip().startswith(("Figure","Table"))
            ]
            # Skip figure/table captions
            lines = [ln for ln in txt.split("\n") if not ln.strip().startswith(("Figure","Table"))]
            pages.append(" ".join(lines))
    full = " ".join(pages)
    # fix hyphens and normalize whitespace
    # Repair hyphenated line‑breaks and collapse whitespace
    full = re.sub(r'(?<=\w)-\s*(?=\w)', '', full)
    full = re.sub(r'\s+', ' ', full)
    return full

# 3. Sentence splitting
# ─────────── Sentence Splitting ───────────
@st.cache_data(show_spinner=False)
def split_sentences(full_text):
    sents = re.split(r'(?<=[\.!?])\s+', full_text)
    return [s.strip() for s in sents if len(s.strip()) >= 50]

# 4. BM25 index
# ─────────── BM25 Indexing ───────────
@st.cache_data(show_spinner=False)
def build_bm25(sentences):
    tokenized = [re.findall(r"\w+", s.lower()) for s in sentences]
    return BM25Okapi(tokenized), tokenized

# 5. Embeddings & FAISS
st.spinner("Loading embedding model…")
# ─────────── Embeddings & FAISS ───────────
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
@@ -63,15 +62,13 @@ def build_faiss_index(embs):
    idx.add(embs)
    return idx

# 6. Load Flan‑T5 for generation
st.spinner("Loading Flan‑T5 model…")
# ─────────── Answer Generation ───────────
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
gen_model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

# 7. Retrieval + fallbacks
def retrieve(query, sents, bm25, tokenized, embs, faiss_idx, full_text, top_n=20, k=3):
    ql = query.lower()
    # FALLBACKS:
    # Fallbacks:
    if "dataset" in ql:
        m = re.search(r'([^.]*benchmark dataset[^.]*)\.', full_text, re.IGNORECASE)
        if m: return [m.group(1).strip() + "."]
@@ -83,12 +80,13 @@ def retrieve(query, sents, bm25, tokenized, embs, faiss_idx, full_text, top_n=20
        if m: return [m.group(1).strip() + "."]
    if "false negative" in ql:
        for s in sents:
            if "false negative" in s.lower(): return [s]
    # BM25 stage
            if "false negative" in s.lower():
                return [s]
    # BM25 lexical retrieval
    tokens = re.findall(r"\w+", ql)
    scores = bm25.get_scores(tokens)
    idxs = np.argsort(scores)[-top_n:][::-1]
    # embedding re-rank
    # Embedding re-rank
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    sims = [(float(np.dot(embs[i], q_emb[0])), i) for i in idxs]
@@ -107,17 +105,14 @@ def generate_answer(question, contexts):
    out = gen_model.generate(**enc, max_new_tokens=75, no_repeat_ngram_size=2)
    return tokenizer.decode(out[0], skip_special_tokens=True)

# ──────────── Streamlit UI ────────────
# ─────────── Streamlit UI ───────────
st.title("📄 PDF RAG Question–Answering")
st.write("Upload your own PDF or leave blank to use the bundled research paper.")
st.write("Upload your own PDF, or leave blank to use the bundled research paper.")

# File uploader (optional)
# Optional upload
uploaded = st.file_uploader("Choose a PDF file", type="pdf")

# decide source: uploaded or default on‑disk
source = uploaded if uploaded else DEFAULT_PDF_PATH

# Load & index
with st.spinner("Loading and indexing PDF…"):
    full_text = extract_raw_text(source)
    if not full_text:
@@ -127,14 +122,13 @@ def generate_answer(question, contexts):
    embs = embed_sentences(sentences)
    faiss_idx = build_faiss_index(embs)

# Ask & answer
q = st.text_input("Enter your question:")
if q:
    with st.spinner("Retrieving context…"):
        ctxs = retrieve(q, sentences, bm25, tokenized, embs, faiss_idx, full_text)
    st.subheader("🔍 Retrieved Context")
    for i, c in enumerate(ctxs, 1):
        st.write(f"**[{i}]** {c}")
    for idx, c in enumerate(ctxs, 1):
        st.write(f"**[{idx}]** {c}")
    with st.spinner("Generating answer…"):
        ans = generate_answer(q, ctxs)
    st.subheader("💡 Answer")
