import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocBot — Ask Your PDF",
    page_icon="📄",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stTextInput > div > div > input { border-radius: 10px; }
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
    st.title("DocBot Settings")
    st.markdown("---")
    openai_key = st.text_input(
        "🔑 OpenAI API Key",
        type="password",
        placeholder="sk-...",
        help="Get your free key at platform.openai.com"
    )
    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("1. Upload any PDF\n2. AI reads & indexes it\n3. Ask anything in plain English\n4. Get accurate answers from the doc")
    st.markdown("---")
    st.caption("Built by Harish Morey | AI Engineer")

# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("📄 DocBot — Ask Your PDF")
st.markdown("Upload any PDF document and ask questions in plain English. The AI answers **only from your document** — no hallucination.")

uploaded_file = st.file_uploader(
    "📂 Upload a PDF file",
    type=["pdf"],
    help="Supports any PDF: contracts, manuals, reports, brochures, resumes"
)

# ── Session State ─────────────────────────────────────────────────────────────
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

# ── Process PDF ───────────────────────────────────────────────────────────────
if uploaded_file and openai_key:
    if st.session_state.doc_name != uploaded_file.name:
        with st.spinner("📖 Reading and indexing your document... (this takes ~15 seconds)"):
            try:
                os.environ["OPENAI_API_KEY"] = openai_key

                # Save to temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(uploaded_file.read())
                    tmp_path = f.name

                # Load PDF
                loader = PyPDFLoader(tmp_path)
                pages = loader.load()

                # Split into chunks
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,
                    chunk_overlap=80,
                    separators=["\n\n", "\n", ".", " "]
                )
                chunks = splitter.split_documents(pages)

                # Embed and store in ChromaDB
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vectorstore = Chroma.from_documents(chunks, embeddings)
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

                # Custom prompt — answers only from document
                prompt_template = """You are a helpful assistant that answers questions strictly based on the provided document context.
If the answer is not in the document, say: "I couldn't find this information in the uploaded document."
Always be concise and clear.

Context from document:
{context}

Question: {question}

Answer:"""

                PROMPT = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )

                # Build QA chain
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                st.session_state.qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": PROMPT}
                )

                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []

                st.success(f"✅ **{uploaded_file.name}** indexed successfully! {len(pages)} pages · {len(chunks)} chunks ready.")

            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.info("Make sure your OpenAI API key is correct and has available credits.")

elif uploaded_file and not openai_key:
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to continue.")

elif not uploaded_file:
    st.info("👆 Upload a PDF above to get started.")

# ── Chat Interface ─────────────────────────────────────────────────────────────
if st.session_state.qa_chain:
    st.markdown("---")
    st.markdown(f"### 💬 Ask about: `{st.session_state.doc_name}`")

    # Display chat history
    for item in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {item['question']}")
        st.markdown(f'<div class="answer-box">🤖 <b>DocBot:</b> {item["answer"]}</div>', unsafe_allow_html=True)
        if item.get("sources"):
            with st.expander("📎 Source pages used"):
                for src in item["sources"]:
                    page_num = src.metadata.get("page", "?") + 1
                    snippet = src.page_content[:200].replace("\n", " ")
                    st.markdown(f'<div class="source-box">📄 Page {page_num}: "{snippet}..."</div>', unsafe_allow_html=True)
        st.markdown("---")

    # Question input
    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g. What are the main services offered? / Summarize this document / What is the pricing?",
        )
        col1, col2 = st.columns([1, 5])
        with col1:
            submit = st.form_submit_button("Ask →", use_container_width=True)
        with col2:
            clear = st.form_submit_button("🗑️ Clear Chat", use_container_width=False)

    if clear:
        st.session_state.chat_history = []
        st.rerun()

    if submit and question:
        with st.spinner("🤔 Thinking..."):
            try:
                result = st.session_state.qa_chain({"query": question})
                answer = result["result"]
                sources = result.get("source_documents", [])

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": sources
                })
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
