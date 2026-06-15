#pip install google-generativeai langchain langchain-google-genai langchain-community faiss-cpu pypdf python-dotenv


import os
import operator
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

# ── Gemini models ──────────────────────────────────────────────

llm        = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# ════════════════════════════════════════════════════════════════
# STEP 1 — LOAD AND SPLIT PDF
# ════════════════════════════════════════════════════════════════

def load_and_split_pdf(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 150):
    """
    Loads PDF, splits into chunks for embedding.
    Returns list of Document chunks.
    """
    print(f"\n[1] Loading PDF: {pdf_path}")

    loader    = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"    Loaded {len(documents)} pages")

    # ── split into chunks ──────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size    = chunk_size,
        chunk_overlap = chunk_overlap,
        separators    = ["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"    Split into {len(chunks)} chunks")

    return chunks


# ════════════════════════════════════════════════════════════════
# STEP 2 — BUILD VECTOR STORE
# ════════════════════════════════════════════════════════════════

def build_vectorstore(chunks):
    """
    Embeds chunks and stores in FAISS vector database.
    Returns FAISS vectorstore object.
    """
    print(f"\n[2] Embedding {len(chunks)} chunks with Gemini...")

    db = FAISS.from_documents(chunks, embeddings)
    print(f"    Vector store ready")

    return db


# ════════════════════════════════════════════════════════════════
# STEP 3 — MULTI-QUERY CHAIN WITH CONVERSATION HISTORY
# ════════════════════════════════════════════════════════════════

class PDFChatSession:
    """
    Manages a multi-query chat session over a single PDF.
    Maintains conversation history across multiple questions.
    """

    def __init__(self, vectorstore, k: int = 4):
        self.vectorstore = vectorstore
        self.retriever   = vectorstore.as_retriever(search_kwargs={"k": k})
        self.history: list[BaseMessage] = []   # conversation memory

        # ── prompt template ────────────────────────────────────
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a helpful assistant answering questions about "
                "an uploaded document. Use the provided context to answer. "
                "If the answer isn't in the context, say so clearly. "
                "Consider the conversation history for follow-up questions "
                "that reference 'it', 'that', 'the previous answer', etc."
            )),
            ("placeholder", "{history}"),    # ← previous Q&A pairs
            ("human", (
                "Context from document:\n{context}\n\n"
                "Question: {question}"
            )),
        ])

        self.chain = self.prompt | llm | StrOutputParser()

    def _format_docs(self, docs) -> str:
        """Formats retrieved chunks with source page numbers."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "unknown")
            formatted.append(f"[Chunk {i} | Page {page}]\n{doc.page_content}")
        return "\n\n".join(formatted)

    def ask(self, question: str, verbose: bool = True) -> dict:
        """
        Asks a question — retrieves relevant chunks, generates answer,
        updates conversation history.
        Returns dict with answer and source chunks.
        """
        if verbose:
            print(f"\n{'─'*55}")
            print(f"Q: {question}")
            print(f"{'─'*55}")

        # ── retrieve relevant chunks ────────────────────────────
        retrieved_docs = self.retriever.invoke(question)
        context        = self._format_docs(retrieved_docs)

        if verbose:
            pages = sorted(set(
                doc.metadata.get("page", "?") for doc in retrieved_docs
            ))
            print(f"Retrieved {len(retrieved_docs)} chunks from pages: {pages}")

        # ── generate answer using history + context ────────────
        answer = self.chain.invoke({
            "history":  self.history,
            "context":  context,
            "question": question,
        })

        if verbose:
            print(f"\nA: {answer}")

        # ── update conversation history ─────────────────────────
        self.history.append(HumanMessage(content=question))
        self.history.append(AIMessage(content=answer))

        return {
            "question":      question,
            "answer":        answer,
            "source_chunks": retrieved_docs,
            "pages_used":    sorted(set(
                doc.metadata.get("page", "?") for doc in retrieved_docs
            )),
        }

    def get_history_summary(self) -> str:
        """Returns formatted conversation history."""
        lines = []
        for msg in self.history:
            role = "Q" if isinstance(msg, HumanMessage) else "A"
            lines.append(f"{role}: {msg.content[:80]}")
        return "\n".join(lines)

    def reset(self):
        """Clears conversation history — keeps vectorstore."""
        self.history = []
        print("[Session] History cleared")


# ════════════════════════════════════════════════════════════════
# STEP 4 — MULTI-QUERY BATCH PROCESSING
# ask multiple predefined questions at once
# ════════════════════════════════════════════════════════════════

def run_multi_query_batch(session: PDFChatSession, questions: list[str]) -> list[dict]:
    """
    Runs multiple questions through the session sequentially.
    Each question can reference previous answers via history.
    """
    print(f"\n{'='*55}")
    print(f"  Running {len(questions)} queries")
    print(f"{'='*55}")

    results = []
    for i, q in enumerate(questions, 1):
        print(f"\n[{i}/{len(questions)}]")
        result = session.ask(q)
        results.append(result)

    return results


# ════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ════════════════════════════════════════════════════════════════

def run_pdf_qa(pdf_path: str):
    """Full pipeline — load PDF, build vectorstore, start chat session."""

    # ── load and process PDF ────────────────────────────────────
    chunks      = load_and_split_pdf(pdf_path)
    vectorstore = build_vectorstore(chunks)

    # ── create chat session ─────────────────────────────────────
    session = PDFChatSession(vectorstore, k=4)
    print(f"\n[3] Chat session ready — {len(chunks)} chunks indexed")

    return session


# ════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    print("\n" + "="*55)
    print("  Multi-Query PDF Chat — Google Gemini Pro")
    print("="*55)

    pdf_path = input("\nEnter path to PDF file: ").strip()

    if not Path(pdf_path).exists():
        print(f"File not found: {pdf_path}")
        exit()

    # ── build session ────────────────────────────────────────────
    session = run_pdf_qa(pdf_path)

    # ── demo: predefined multi-query sequence ──────────────────
    print("\n" + "="*55)
    print("  Demo: Multi-Query Sequence")
    print("  (questions reference earlier answers)")
    print("="*55)

    demo_questions = [
        "What is this document about? Give a brief summary.",
        "What are the main topics covered?",
        "Can you elaborate on the first topic you mentioned?",   # ← references history
        "Are there any numbers or statistics mentioned?",
    ]

    use_demo = input("\nRun demo questions? (y/n): ").strip().lower()
    if use_demo == "y":
        run_multi_query_batch(session, demo_questions)

        print(f"\n{'='*55}")
        print("CONVERSATION HISTORY")
        print(f"{'='*55}")
        print(session.get_history_summary())

    # ── interactive mode ──────────────────────────────────────────
    print("\n" + "="*55)
    print("  Interactive Mode")
    print("  Commands: 'history', 'reset', 'quit'")
    print("="*55 + "\n")

    while True:
        try:
            question = input("You: ").strip()

            if not question:
                continue

            if question.lower() in ["quit", "exit"]:
                print("Goodbye!")
                break

            elif question.lower() == "history":
                print(f"\n{session.get_history_summary()}\n")
                continue

            elif question.lower() == "reset":
                session.reset()
                continue

            session.ask(question)

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break