# 🤖 AI-Powered Resume Screening Tool

This project is a **Streamlit app** that automatically parses and ranks resumes against a given **Job Description (JD)** using **TF-IDF + cosine similarity**.  

It helps recruiters save time by quickly shortlisting the most relevant candidates.

---

## 🚀 Features
- Upload multiple resumes (**PDF**, **DOCX**, **TXT**)
- Paste a Job Description (JD) into the app
- Rank resumes using **TF-IDF + cosine similarity**
- Optional keyword coverage weighting
- Extract quick facts:
  - Degrees (B.Tech, MBA, LL.B., etc.)
  - Years of experience
- Preview resume text and matched JD keywords
- Export results to **CSV** and **Excel**

---

## 📂 Project Structure
.
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore


---

## 📦 Tech Stack
- Python  
- Streamlit  
- scikit-learn (TF-IDF + cosine similarity)  
- PyPDF2, python-docx (resume parsing)  
- pandas, openpyxl (data & export)  
- pdfminer.six (optional PDF parsing)

---

## ⚙️ Installation & Running Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ai-resume-screener.git
   cd ai-resume-screener
   
2. Install dependencies:
pip install -r requirements.txt

3. Run the app:
   streamlit run app.py
4. Open your browser at:
   http://localhost:8501

🛠 Usage
1. Paste the Job Description into the text area.
2. Upload one or more resumes (PDF, DOCX, or TXT).
3. Adjust weights in the sidebar (TF-IDF vs Keyword coverage).
4. Click 🚀 Rank Candidates.
5. View the ranked list with scores, degrees, and years of experience.
6. Download results as CSV or Excel.
7. Preview individual resumes with highlighted keyword matches.

   📜 License

This project is licensed under the MIT License.

🙏 Acknowledgments

Built for quick HR screening workflows — extendable with embeddings, required-skill weighting, or richer parsing (OCR for scanned PDFs).
