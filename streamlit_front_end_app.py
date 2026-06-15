# app.py
import os
import streamlit as st
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# load_dotenv()
# ════════════════════════════════════════════════════════════════
# PAGE CONFIG — must be first Streamlit command
# ════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title = "Multi-Query PDF Chat",
    page_icon  = "📄",
    layout     = "wide",
)


# ════════════════════════════════════════════════════════════════
# API KEY — read from Streamlit secrets (not .env on cloud)
# ════════════════════════════════════════════════════════════════

# Streamlit Cloud reads from .streamlit/secrets.toml
# Locally, also works with .env via os.getenv as fallback
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Add it to .streamlit/secrets.toml")
    st.stop()

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ════════════════════════════════════════════════════════════════
# CACHED RESOURCES — models loaded once, reused across reruns
# ════════════════════════════════════════════════════════════════

@st.cache_resource
def get_llm():
    """LLM instance — cached so it's created only once."""
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)


@st.cache_resource
def get_embeddings():
    """Embeddings instance — cached so it's created only once."""
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


# ════════════════════════════════════════════════════════════════
# PDF PROCESSING — cached per uploaded file
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def process_pdf(file_bytes: bytes, file_name: str):
    """
    Saves uploaded PDF temporarily, loads, splits, embeds.
    Cached by file content — same file won't reprocess.
    """
    # save uploaded file to temp location — PyPDFLoader needs a path
    # Specify your desired folder path
    temp_path = "/tmp/"
    folder_path = Path(temp_path)

    # Create the folder safely
    folder_path.mkdir(parents=True, exist_ok=True)
    temp_path = f"/tmp/{file_name}"

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    # load and split
    loader    = PyPDFLoader(temp_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = 1000,
        chunk_overlap = 150,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # embed and store
    embeddings  = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # cleanup temp file
    os.remove(temp_path)

    return vectorstore, len(documents), len(chunks)


# ════════════════════════════════════════════════════════════════
# RAG CHAIN
# ════════════════════════════════════════════════════════════════

def format_docs(docs) -> str:
    """Formats retrieved chunks with page numbers."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "unknown")
        formatted.append(f"[Chunk {i} | Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)


def get_rag_chain():
    """Builds the RAG prompt + chain."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a helpful assistant answering questions about "
            "an uploaded document. Use the provided context to answer. "
            "If the answer isn't in the context, say so clearly. "
            "Consider the conversation history for follow-up questions "
            "that reference 'it', 'that', 'the previous answer', etc."
        )),
        ("placeholder", "{history}"),
        ("human", "Context from document:\n{context}\n\nQuestion: {question}"),
    ])

    return prompt | llm | StrOutputParser()


def ask_question(vectorstore, question: str, history: list[BaseMessage], k: int = 4) -> dict:
    """
    Retrieves chunks, generates answer with history.
    Returns dict with answer, sources, pages.
    """
    retriever      = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)
    context        = format_docs(retrieved_docs)

    chain  = get_rag_chain()
    answer = chain.invoke({
        "history":  history,
        "context":  context,
        "question": question,
    })

    pages = sorted(set(doc.metadata.get("page", "?") for doc in retrieved_docs))

    return {
        "answer": answer,
        "pages":  pages,
        "chunks": retrieved_docs,
    }


# ════════════════════════════════════════════════════════════════
# SESSION STATE — Streamlit's way of persisting data across reruns
# ════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []     # for chat display

if "history" not in st.session_state:
    st.session_state.history = []      # LangChain message history for prompt

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ════════════════════════════════════════════════════════════════
# SIDEBAR — file upload and info
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.title("📄 Multi-Query PDF Chat")
    st.caption("Powered by Google Gemini")

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type = ["pdf"],
        help = "Upload a PDF to start asking questions",
    )

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()

        with st.spinner("Processing PDF..."):
            vectorstore, n_pages, n_chunks = process_pdf(file_bytes, uploaded_file.name)
            st.session_state.vectorstore   = vectorstore

        st.success(f"✓ Processed: {n_pages} pages, {n_chunks} chunks")

    st.divider()

    # ── controls ────────────────────────────────────────────────
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

    st.divider()
    st.caption(
        "Tip: Ask follow-up questions like 'elaborate on that' or "
        "'what about the first point?' — the app remembers context."
    )


# ════════════════════════════════════════════════════════════════
# MAIN CHAT INTERFACE
# ════════════════════════════════════════════════════════════════

st.header("Chat with your PDF")

if st.session_state.vectorstore is None:
    st.info("👈 Upload a PDF from the sidebar to get started")
    st.stop()

# ── display chat history ────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "pages" in msg:
            st.caption(f"📍 Sources: pages {msg['pages']}")


# ── chat input ──────────────────────────────────────────────────
if question := st.chat_input("Ask a question about the PDF..."):

    # show user message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # generate and show answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask_question(
                vectorstore = st.session_state.vectorstore,
                question    = question,
                history     = st.session_state.history,
                k           = 4,
            )

        st.markdown(result["answer"])
        st.caption(f"📍 Sources: pages {result['pages']}")

        # show retrieved chunks in expander
        with st.expander("View source chunks"):
            for i, chunk in enumerate(result["chunks"], 1):
                page = chunk.metadata.get("page", "?")
                st.markdown(f"**Chunk {i} (Page {page}):**")
                st.text(chunk.page_content[:300] + "...")

    # save to session state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "pages":   result["pages"],
    })

    # update LangChain history for follow-up context
    st.session_state.history.append(HumanMessage(content=question))
    st.session_state.history.append(AIMessage(content=result["answer"]))