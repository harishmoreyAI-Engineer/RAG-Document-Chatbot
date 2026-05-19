# 📄 DocBot — RAG-Powered PDF Chatbot

Ask any question about any PDF document. The AI answers strictly from the document content — no hallucination.

**Built with:** Python · LangChain · ChromaDB · OpenAI · Streamlit

---

## 🖥️ Live Demo
> Add your Streamlit Cloud URL here after deployment

---

## 🚀 Step-by-Step Setup Guide

### STEP 1 — Create a GitHub Account (if you don't have one)
1. Go to [github.com](https://github.com)
2. Click **Sign Up** → use your email → create a free account
3. Verify your email

---

### STEP 2 — Create a New GitHub Repository
1. After login, click the **+** icon (top right) → **New repository**
2. Repository name: `rag-doc-chatbot`
3. Set to **Public**
4. Check ✅ **Add a README file**
5. Click **Create repository**

---

### STEP 3 — Upload These Project Files to GitHub
You need to upload 2 files: `app.py` and `requirements.txt`

1. In your new repo, click **Add file** → **Upload files**
2. Drag and drop both `app.py` and `requirements.txt`
3. Scroll down → click **Commit changes**

Your repo should now have:
```
rag-doc-chatbot/
├── app.py
├── requirements.txt
└── README.md
```

---

### STEP 4 — Get Your Free OpenAI API Key
1. Go to [platform.openai.com](https://platform.openai.com)
2. Click **Sign Up** (free) → verify email
3. Go to **API Keys** section (left menu)
4. Click **Create new secret key** → copy and save it somewhere safe
5. New accounts get **$5 free credits** — enough for hundreds of questions

> ⚠️ Never share your API key publicly or upload it to GitHub

---

### STEP 5 — Deploy on Streamlit Cloud (Free Hosting)
1. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
2. Click **Sign up** → use your GitHub account to sign in
3. Click **New app**
4. Fill in the form:
   - **Repository:** `your-github-username/rag-doc-chatbot`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **Deploy!**
6. Wait 2–3 minutes for it to build
7. You'll get a free permanent URL like:
   `https://your-username-rag-doc-chatbot-app-xxxxx.streamlit.app`

---

### STEP 6 — Test Your App
1. Open your Streamlit URL in the browser
2. In the **sidebar**, paste your OpenAI API key
3. Upload any PDF (try a company brochure, resume, or contract)
4. Wait ~15 seconds for indexing
5. Ask questions like:
   - *"Summarize this document"*
   - *"What services are offered?"*
   - *"What is the pricing?"*
   - *"Who should I contact?"*

---

## 🎯 Demo Ideas (for Resume / Portfolio)

Try these PDFs to showcase different use cases:

| PDF Type | Sample Question to Ask |
|----------|----------------------|
| Your own resume | "What technologies does this person know?" |
| Company brochure | "What are the main services?" |
| Legal contract | "What are the termination clauses?" |
| Product manual | "How do I reset the device?" |
| Government policy | "What are the eligibility criteria?" |

---

## 📹 How to Record a Demo Video (for Resume)
1. Download **Loom** free from [loom.com](https://loom.com)
2. Screen record yourself:
   - Opening the app
   - Uploading a PDF
   - Asking 3 smart questions
   - Showing accurate answers with source pages
3. Upload to Loom → copy the link
4. Add it to your resume under this project

---

## 🛠️ How It Works (Technical — Good for Interviews)

```
PDF Upload
    ↓
Split into chunks (600 chars each, 80 char overlap)
    ↓
Convert each chunk to a vector (OpenAI embeddings)
    ↓
Store all vectors in ChromaDB (local vector database)
    ↓
User asks a question
    ↓
Question is also converted to a vector
    ↓
Find 4 most similar chunks (semantic search)
    ↓
Send: question + 4 chunks → GPT-4o-mini
    ↓
GPT answers strictly from those chunks
    ↓
Display answer + source page numbers
```

This is called **Retrieval-Augmented Generation (RAG)** — the same technique used by ChatPDF, Notion AI, and Adobe AI.

---

## 💡 Resume Bullet Point (copy this)

> *Developed and deployed a RAG-powered document chatbot using LangChain, ChromaDB, and OpenAI GPT-4o — processes any PDF and answers natural language queries with source-grounded responses. Live at [your-url]*

---

## 🔧 Troubleshooting

| Problem | Solution |
|---------|----------|
| "Invalid API key" | Double-check you copied the full key from OpenAI |
| App not loading | Wait 3–5 mins after deploy; Streamlit cold starts |
| "No credits" error | Add a payment method on OpenAI (pay-as-you-go, very cheap) |
| PDF not reading | Make sure PDF has actual text (not a scanned image) |
| Slow indexing | Normal for large PDFs (50+ pages). Wait ~30 seconds |

---

## 📞 Contact
**Harish Morey** — AI Engineer & Automation Specialist
harishmorey37@gmail.com
