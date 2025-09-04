# 🧠 AI-Powered Resume Screening Tool

Automated resume parsing and ranking against a Job Description using **TF–IDF + cosine similarity** with an intuitive **Streamlit** UI.

## ✨ Features
- Upload multiple **PDF** resumes
- Paste **Job Description** text
- Extract **emails, phone**, heuristic **name**
- **Skills matching** via custom vocabulary (+ optional spaCy PhraseMatcher)
- **Rank candidates** by similarity score
- **Export** ranked results to **CSV/Excel**

## 📂 Project Structure
```
.
├── app.py
├── requirements.txt
├── sample_skills.txt
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Quickstart (Local)
```bash
# 1) (Optional) Create a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) (Optional) Download spaCy model
python -m spacy download en_core_web_sm

# 4) Run
streamlit run app.py
```

## 🛠 Usage
1. Paste the **Job Description** in the text area.
2. Upload one or more **PDF resumes**.
3. (Optional) Upload a custom **skills list** (`.txt`) separated by commas/newlines/semicolons.
4. Click **Process & Rank Candidates**.
5. Download **CSV/Excel** results.

## 🌐 Deploy to Streamlit Community Cloud (Free)
1. Push this project to a **public GitHub repo**.
2. Go to **streamlit.io** → **Sign in** → **New app** → Connect GitHub.
3. Select your repo and branch; **Main file path:** `app.py`.
4. Click **Deploy** → copy the app URL and add it to this README.

## 🧪 Tech Stack
- Python, Streamlit, pandas, scikit-learn
- TF–IDF vectorizer + cosine similarity
- PDF parsing: PyPDF2 (+ pdfminer fallback)
- Optional NLP: spaCy PhraseMatcher for skills

## 📜 License
This project is licensed under the **MIT License** (see `LICENSE`).

## 🙏 Acknowledgments
Built for quick HR screening workflows; extend with embeddings, required-skill weighting, or richer parsing as needed.
