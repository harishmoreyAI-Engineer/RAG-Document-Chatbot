import streamlit as st
import tempfile
import os
import requests
import numpy as np
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocBot — Ask Your PDF", page_icon="📄", layout="centered")

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .answer-box {
        background: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 16px 20px;
        border-radius: 8px;
        margin-top: 10px;
        font-size: 15px;
        color: #1e3a5f;
    }
    .source-box {
        background: #f1f5f9;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 6px;
        font-size: 13px;
        color: #475569;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/PDF_file_icon.svg/267px-PDF_file_icon.svg.png", width=60)
    st.title("DocBot")
    st.markdown("---")
    openrouter_key = st.secrets.get("OPENROUTER_API_KEY", "")
    if not openrouter_key:
        st.error("⚠️ OpenRouter API key not configured.")
    else:
        st.success("✅ API Key loaded")
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload any PDF\n2. AI reads & indexes it\n3. Ask anything in plain English\n4. Get accurate answers from the doc")
    st.markdown("---")
    st.caption("Built by Harish Morey | AI Engineer")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("📄 DocBot — Ask Your PDF")
st.markdown("Upload any PDF and ask questions. The AI answers **only from your document**.")

uploaded_file = st.file_uploader("📂 Upload a PDF file", type=["pdf"])

# ── Session State ─────────────────────────────────────────────────────────────
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embeddings_matrix" not in st.session_state:
    st.session_state.embeddings_matrix = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Helper: Simple TF-IDF style keyword search (no API needed for embeddings) ─
def simple_search(query, chunks, top_k=4):
    """
    Simple keyword overlap search — works without any embedding API.
    Scores each chunk by how many query words appear in it.
    """
    query_words = set(query.lower().split())
    scores = []
    for chunk in chunks:
        text = chunk.page_content.lower()
        score = sum(1 for word in query_words if word in text)
        scores.append(score)
    top_indices = np.argsort(scores)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

# ── Helper: Ask OpenRouter ────────────────────────────────────────────────────
def ask_openrouter(context, question, api_key):
    """
    Uses openrouter/free — automatically picks the best available free model.
    Falls back to deepseek and llama if needed.
    """
    models_to_try = [
        "openrouter/free",
        "deepseek/deepseek-r1:free",
        "meta-llama/llama-3.3-70b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
    ]

    last_error = None
    for model in models_to_try:
        try:
            response = requests.post(
                url="https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://docbot.streamlit.app",
                    "X-Title": "DocBot"
                },
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant that answers questions strictly based on the provided document context. If the answer is not found in the context, say: 'I couldn't find this information in the uploaded document.' Always be concise and clear."
                        },
                        {
                            "role": "user",
                            "content": f"Context from document:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                        }
                    ]
                },
                timeout=60
            )
            result = response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"], model
            last_error = result
        except Exception as e:
            last_error = str(e)
            continue

    raise Exception(f"All models failed. Last error: {last_error}")

# ── Process PDF ───────────────────────────────────────────────────────────────
if uploaded_file and openrouter_key:
    if st.session_state.doc_name != uploaded_file.name:
        with st.spinner("📖 Reading and indexing your document..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name

                pages = PyPDFLoader(tmp_path).load()
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=80
                )
                chunks = splitter.split_documents(pages)

                st.session_state.chunks = chunks
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []

                st.success(f"✅ **{uploaded_file.name}** ready! {len(pages)} pages · {len(chunks)} chunks indexed.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

elif uploaded_file and not openrouter_key:
    st.warning("⚠️ Add OPENROUTER_API_KEY in Streamlit Cloud → Settings → Secrets.")
else:
    st.info("👆 Upload a PDF above to get started.")

# ── Chat Interface ────────────────────────────────────────────────────────────
if st.session_state.chunks:
    st.markdown("---")
    st.markdown(f"### 💬 Ask about: `{st.session_state.doc_name}`")

    for item in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {item['question']}")
        model_used = item.get('model_used', '')
        st.markdown(f'<div class="answer-box">🤖 <b>DocBot:</b> {item["answer"]}<br><small style="color:#94a3b8">Model: {model_used}</small></div>', unsafe_allow_html=True)
        if item.get("sources"):
            with st.expander("📎 Source pages used"):
                for src in item["sources"]:
                    page_num = src.metadata.get("page", "?")
                    if isinstance(page_num, int):
                        page_num += 1
                    snippet = src.page_content[:250].replace("\n", " ")
                    st.markdown(f'<div class="source-box">📄 Page {page_num}: "{snippet}..."</div>', unsafe_allow_html=True)
        st.markdown("---")

    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g. Summarize this document / What are the main points?"
        )
        col1, col2 = st.columns([1, 4])
        with col1:
            submit = st.form_submit_button("Ask →", use_container_width=True)
        with col2:
            clear = st.form_submit_button("🗑️ Clear Chat")

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if submit and question:
        with st.spinner("🤔 Thinking..."):
            try:
                # Find relevant chunks using keyword search (no API cost)
                relevant = simple_search(question, st.session_state.chunks)
                context = "\n\n".join([c.page_content for c in relevant])

                # Get AI answer with auto-fallback
                answer, model_used = ask_openrouter(context, question, openrouter_key)

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": relevant,
                    "model_used": model_used
                })
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
