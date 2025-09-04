# =============================================
# AI-Powered Resume Screening Tool (Streamlit)
# =============================================
#
# Quick start (in terminal):
#   pip install streamlit pandas scikit-learn PyPDF2 python-docx openpyxl
#   streamlit run app.py
#
# Features:
#   • Upload multiple resumes (PDF/DOCX/TXT)
#   • Paste job description (JD)
#   • Rank resumes using TF‑IDF + cosine similarity
#   • Optional weighting using matched JD keywords
#   • Extract quick facts (degrees, years of experience)
#   • Preview resume text and matched keywords
#   • Export ranked results to CSV/Excel
#
# Note:
#   This app avoids heavy NLP downloads. If you later add spaCy models,
#   you can improve entity extraction, but this works out-of-the-box.

import re
from io import BytesIO
from typing import List, Tuple

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Optional readers
import PyPDF2
try:
    from docx import Document  # python-docx
    HAVE_DOCX = True
except Exception:
    HAVE_DOCX = False

# ----------------------------- UI CONFIG -----------------------------
st.set_page_config(
    page_title="AI Resume Screening Tool",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI-Powered Resume Screening Tool")
st.caption("Upload resumes, paste a JD, and get an instant ranked shortlist.")

# --------------------------- TEXT UTILITIES --------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalize whitespace, strip control chars
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# --------------------------- FILE EXTRACTORS -------------------------

def extract_text_from_pdf(uploaded_file) -> str:
    try:
        uploaded_file.seek(0)
        reader = PyPDF2.PdfReader(uploaded_file)
        pages_text = []
        for p in reader.pages:
            pages_text.append(p.extract_text() or "")
        return clean_text("\n".join(pages_text))
    except Exception:
        return ""

def extract_text_from_docx(uploaded_file) -> str:
    if not HAVE_DOCX:
        return ""
    try:
        uploaded_file.seek(0)
        doc = Document(uploaded_file)
        paras = [p.text for p in doc.paragraphs]
        return clean_text("\n".join(paras))
    except Exception:
        return ""

def extract_text_generic(uploaded_file) -> str:
    """Dispatch by file extension/MIME. Fallback to utf-8 text."""
    name = (uploaded_file.name or "").lower()
    if name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    try:
        uploaded_file.seek(0)
        content = uploaded_file.read()
        # Heuristic: attempt utf‑8 decode; ignore errors
        return clean_text(content.decode("utf-8", errors="ignore"))
    except Exception:
        return ""

# ------------------------------ JD KEYS ------------------------------

def jd_top_terms(job_desc: str, max_terms: int = 30) -> List[Tuple[str, float]]:
    """Get top weighted terms from the JD using TF‑IDF on a single doc.
    We sort by TF (since IDF=1 on single doc); TfidfVectorizer still gives weights.
    """
    vect = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vect.fit_transform([job_desc])
    terms = vect.get_feature_names_out()
    weights = X.toarray()[0]
    pairs = list(zip(terms, weights))
    pairs.sort(key=lambda t: t[1], reverse=True)
    # filter very short tokens
    pairs = [(t, w) for t, w in pairs if len(t) > 2][:max_terms]
    return pairs

# ------------------------ FACTS/REGEX EXTRACTORS ---------------------

DEGREE_PATTERNS = [
    r"ph\.?d\.?", r"doctorate", r"msc", r"m\.sc", r"bsc", r"b\.sc",
    r"mtech", r"m\.tech", r"btech", r"b\.tech", r"be", r"b\.e\.", r"me", r"m\.e\.",
    r"mba", r"pgdm", r"mca", r"m\.c\.a\.", r"bca", r"b\.c\.a\.",
    r"ba", r"b\.a\.", r"ma", r"m\.a\.", r"bcom", r"b\.com", r"mcom", r"m\.com",
    r"llb", r"ll\.b\.", r"llm", r"ll\.m\."
]
DEGREE_REGEX = re.compile(r"(?:" + "|".join(DEGREE_PATTERNS) + r")", re.IGNORECASE)
EXPERIENCE_REGEX = re.compile(r"(\d{1,2})\s*\+?\s*(?:years?|yrs)\b", re.IGNORECASE)


def extract_degrees(text: str) -> List[str]:
    found = DEGREE_REGEX.findall(text or "")
    # Normalize/pretty
    norm = set()
    for f in found:
        t = f.upper().replace(".", "").strip()
        mapping = {
            "PHD": "PhD",
            "DOCTORATE": "Doctorate",
            "MSC": "MSc", "MSC ": "MSc",
            "BSC": "BSc",
            "MTECH": "M.Tech", "BTECH": "B.Tech",
            "BE": "B.E.", "ME": "M.E.",
            "MBA": "MBA", "PGDM": "PGDM",
            "MCA": "MCA", "BCA": "BCA",
            "BA": "B.A.", "MA": "M.A.",
            "BCOM": "B.Com", "MCOM": "M.Com",
            "LLB": "LL.B.", "LLM": "LL.M."
        }
        norm.add(mapping.get(t, t))
    return sorted(norm)


def extract_years_experience(text: str) -> int:
    years = 0
    for m in EXPERIENCE_REGEX.finditer(text or ""):
        try:
            years = max(years, int(m.group(1)))
        except Exception:
            pass
    return years

# ------------------------------ SCORING ------------------------------

def score_resumes(job_desc: str, resume_texts: List[str], ngram_low=1, ngram_high=2) -> Tuple[List[float], TfidfVectorizer]:
    corpus = [job_desc] + resume_texts
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(ngram_low, ngram_high))
    tfidf = vectorizer.fit_transform(corpus)
    jd_vec = tfidf[0:1]
    res_vecs = tfidf[1:]
    sims = cosine_similarity(jd_vec, res_vecs).flatten().tolist()
    return sims, vectorizer

# ------------------------------- SIDEBAR -----------------------------
with st.sidebar:
    st.header("Settings")
    ngram = st.select_slider("N‑gram range", options=[1, 2], value=2, help="Include uni+bi-grams when 2 is selected.")
    top_terms = st.slider("Top JD keywords to match", 10, 60, 30, help="Used for keyword coverage metric")
    w_tfidf = st.slider("Weight: TF‑IDF similarity", 0.0, 1.0, 0.85)
    w_kw = 1.0 - w_tfidf
    st.caption(f"Keyword coverage weight auto-set to {w_kw:.2f}")

# ------------------------------- MAIN UI ----------------------------
colA, colB = st.columns([2, 1])
with colA:
    jd_text = st.text_area("📄 Paste Job Description (JD)", height=220, placeholder="Role overview, required skills, years of experience, tech stack, etc.")
with colB:
    st.markdown("**Upload Resumes** (.pdf / .docx / .txt)")
    files = st.file_uploader("Choose one or more files", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    run = st.button("🚀 Rank Candidates", use_container_width=True)

# ------------------------------ PROCESS -----------------------------

if run:
    if not jd_text.strip():
        st.error("Please paste a Job Description.")
    elif not files:
        st.error("Please upload at least one resume file.")
    else:
        # Extract JD terms for coverage
        jd_terms = jd_top_terms(jd_text, max_terms=top_terms)
        jd_vocab = [t for t, _ in jd_terms]

        # Read resumes
        names: List[str] = []
        texts: List[str] = []
        degrees_list: List[str] = []
        exp_years: List[int] = []
        matched_kw_counts: List[int] = []
        matched_kw_lists: List[List[str]] = []

        for f in files:
            name = f.name
            text = extract_text_generic(f)
            names.append(name)
            texts.append(text)

            degs = extract_degrees(text)
            degrees_list.append(", ".join(degs) if degs else "—")

            yrs = extract_years_experience(text)
            exp_years.append(yrs)

            # Keyword coverage
            text_lc = text.lower()
            matched = []
            for kw in jd_vocab:
                # exact substring match; for n-grams this is OK
                if kw.lower() in text_lc:
                    matched.append(kw)
            matched_kw_lists.append(matched)
            matched_kw_counts.append(len(matched))

        # TF‑IDF + cosine similarity
        sims, vect = score_resumes(jd_text, texts, ngram_low=1, ngram_high=ngram)

        # Keyword coverage score (fraction of matched JD terms)
        coverage_scores = [ (c / max(1, len(jd_vocab))) for c in matched_kw_counts ]

        # Final weighted score
        final_scores = [ w_tfidf * s + w_kw * cov for s, cov in zip(sims, coverage_scores) ]

        # Build dataframe
        rows = []
        for i, name in enumerate(names):
            rows.append({
                "Filename": name,
                "TF‑IDF Similarity": round(sims[i], 4),
                "Keyword Coverage": round(coverage_scores[i], 4),
                "Final Score": round(final_scores[i], 4),
                "Matched JD Keywords": ", ".join(matched_kw_lists[i]) if matched_kw_lists[i] else "—",
                "Highest Degrees": degrees_list[i],
                "Max Years Mentioned": exp_years[i],
            })
        df = pd.DataFrame(rows).sort_values(by="Final Score", ascending=False).reset_index(drop=True)

        st.success("✅ Ranking complete!")
        st.dataframe(df, use_container_width=True)

        # ---------- Downloads ----------
        csv_buf = BytesIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            label="⬇️ Download Ranked Results (CSV)",
            data=csv_buf.getvalue(),
            file_name="ranked_resumes.csv",
            mime="text/csv",
            use_container_width=True,
        )

        xlsx_buf = BytesIO()
        with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")
        st.download_button(
            label="⬇️ Download Ranked Results (Excel)",
            data=xlsx_buf.getvalue(),
            file_name="ranked_resumes.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # ---------- Details / Preview ----------
        st.markdown("---")
        st.subheader("🔎 Candidate Preview & Keyword Matches")
        sel = st.selectbox("Select a resume to preview", options=["(choose)"] + names)
        if sel != "(choose)":
            idx = names.index(sel)
            st.markdown(f"**File:** {sel}")
            st.markdown(f"**Final Score:** {final_scores[idx]:.4f} | **TF‑IDF:** {sims[idx]:.4f} | **Coverage:** {coverage_scores[idx]:.4f}")
            st.markdown(f"**Degrees:** {degrees_list[idx]} | **Max years mentioned:** {exp_years[idx]}")

            # Show matched keywords list
            mk = matched_kw_lists[idx]
            if mk:
                st.markdown("**Matched JD Keywords:** ")
                st.write(", ".join(mk))
            else:
                st.markdown("*No explicit JD keywords matched.*")

            # Simple highlighted preview (first 1500 chars)
            preview = texts[idx][:2000]
            # Wrap matches in **bold** (lightweight highlighting)
            for kw in sorted(mk, key=len, reverse=True)[:30]:
                try:
                    pattern = re.compile(re.escape(kw), re.IGNORECASE)
                    preview = pattern.sub(lambda m: f"**{m.group(0)}**", preview)
                except Exception:
                    pass
            st.markdown("**Resume Preview:**")
            st.write(preview if preview else "(No extractable text — possibly a scanned PDF)")

else:
    with st.expander("Need help?"):
        st.write(
            "Paste a JD, upload PDFs/DOCX/TXT resumes, and click 'Rank Candidates'. "
            "Adjust weights in the sidebar. Export results as CSV/Excel."
        )
