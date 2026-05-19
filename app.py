import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocBot — Ask Your PDF",
    page_icon="📄",
    layout="centered"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
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
    openai_key = st.secrets.get("OPENAI_API_KEY", "")
    if not openai_key:
        st.error("⚠️ API key not configured.")
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

uploaded_file = st.file_uploader(
    "📂 Upload a PDF file",
    type=["pdf"],
    help="Supports any PDF: contracts, manuals, reports, brochures, resumes"
)

# ── Session State ─────────────────────────────────────────────────────────────
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_name" not in st.session_state:
    st.session_state.doc_name = None
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

# ── Process PDF ───────────────────────────────────────────────────────────────
if uploaded_file and openai_key:
    if st.session_state.doc_name != uploaded_file.name:
        with st.spinner("📖 Reading and indexing your document... (~15 seconds)"):
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

                # Embed into ChromaDB
                embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
                vectorstore = Chroma.from_documents(chunks, embeddings)

                st.session_state.retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                st.session_state.doc_name = uploaded_file.name
                st.session_state.chat_history = []
                st.session_state.page_count = len(pages)
                st.session_state.chunk_count = len(chunks)

                st.success(f"✅ **{uploaded_file.name}** ready! {len(pages)} pages · {len(chunks)} chunks indexed.")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Make sure your OpenAI API key is correct and has available credits.")

elif uploaded_file and not openai_key:
    st.warning("⚠️ API key not found. Add it in Streamlit Cloud → Settings → Secrets.")

elif not uploaded_file:
    st.info("👆 Upload a PDF above to get started.")

# ── Chat Interface ────────────────────────────────────────────────────────────
if st.session_state.retriever:
    st.markdown("---")
    st.markdown(f"### 💬 Ask about: `{st.session_state.doc_name}`")

    # Display chat history
    for item in st.session_state.chat_history:
        st.markdown(f"**🧑 You:** {item['question']}")
        st.markdown(f'<div class="answer-box">🤖 <b>DocBot:</b> {item["answer"]}</div>', unsafe_allow_html=True)
        if item.get("sources"):
            with st.expander("📎 Source pages used"):
                for src in item["sources"]:
                    page_num = src.metadata.get("page", "?")
                    if isinstance(page_num, int):
                        page_num += 1
                    snippet = src.page_content[:250].replace("\n", " ")
                    st.markdown(f'<div class="source-box">📄 Page {page_num}: "{snippet}..."</div>', unsafe_allow_html=True)
        st.markdown("---")

    # Question input
    with st.form(key="question_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g. Summarize this document / What are the main services? / What is the pricing?"
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
                os.environ["OPENAI_API_KEY"] = openai_key

                # Retrieve relevant chunks
                docs = st.session_state.retriever.invoke(question)
                context = "\n\n".join([doc.page_content for doc in docs])

                # Build prompt manually
                system_prompt = """You are a helpful assistant that answers questions strictly based on the provided document context.
If the answer is not found in the context, say: "I couldn't find this information in the uploaded document."
Always be concise and clear."""

                user_message = f"""Context from document:
{context}

Question: {question}

Answer:"""

                # Call OpenAI directly
                llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
                from langchain_core.messages import SystemMessage, HumanMessage
                response = llm.invoke([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=user_message)
                ])
                answer = response.content

                st.session_state.chat_history.append({
                    "question": question,
                    "answer": answer,
                    "sources": docs
                })
                st.rerun()

            except Exception as e:
                st.error(f"Error: {str(e)}")
