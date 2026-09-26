import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv


st.set_page_config(
    page_title="Zyro Dynamics HR Desk",
    page_icon=":briefcase:",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv(Path(__file__).resolve().parent / ".env")

LLM_MODEL = "openai/gpt-oss-20b"
CORPUS_PATH = Path(__file__).parent / "zyro-dynamics-hr-corpus"
DEFAULT_TOP_K = 3
REFUSAL_MESSAGE = (
    "I'm an HR assistant and can only help with questions about company HR "
    "policies (leave, reimbursement, code of conduct, etc.). I don't have "
    "information to answer that question."
)

@st.cache_resource(show_spinner="Loading HR policy documents and building FAISS index...")
def load_rag_components():
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    loader = PyPDFDirectoryLoader(str(CORPUS_PATH))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


@st.cache_resource
def load_llm():
    from langchain_groq import ChatGroq

    try:
        api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
    except FileNotFoundError:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Community Cloud secrets "
            "or your local .env file."
        )
    return ChatGroq(
        model=LLM_MODEL,
        groq_api_key=api_key,
        temperature=0.6,
        max_tokens=500,
    )


def format_docs(documents):
    return "\n\n".join(document.page_content for document in documents)


def ask_bot(question, top_k, vectorstore, llm):
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    guardrail_prompt = ChatPromptTemplate.from_template(
        """You are a scope classifier for an HR assistant. Decide whether the question below is something an HR assistant should answer (company leave policy, reimbursement, code of conduct, or other internal HR topics) or something out of scope (general knowledge, coding help, unrelated small talk, etc.).
Respond with exactly one word: IN_SCOPE or OUT_OF_SCOPE.
Question: {question}"""
    )
    rag_prompt = ChatPromptTemplate.from_template(
        """You are an HR assistant. Your goal is to accurately answer the employee question using ONLY the provided context below. If the context does not contain the answer, say that the policy documents do not provide enough information.

Context:
{context}

Question:
{question}"""
    )
    guardrail_chain = guardrail_prompt | llm | StrOutputParser()
    verdict = guardrail_chain.invoke({"question": question}).strip().upper()
    if verdict != "IN_SCOPE":
        return REFUSAL_MESSAGE, []

    documents = vectorstore.similarity_search(question, k=top_k)
    answer_chain = rag_prompt | llm | StrOutputParser()
    answer = answer_chain.invoke(
        {"context": format_docs(documents), "question": question}
    )
    return answer, documents


def submit_question(question):
    question = question.strip()
    if question:
        st.session_state.pending_question = question


def reset_chat():
    st.session_state.messages = []
    st.session_state.pending_question = ""


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
        :root { --ink: #17231f; --muted: #66756e; --mint: #dcefe4; --coral: #ef765f; --paper: #fbfaf6; --line: #d9e2dc; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); }
        .stApp { background: radial-gradient(circle at 90% 0%, #f6dfbd 0, transparent 22%), linear-gradient(135deg, #fbfaf6 0%, #edf5ef 100%); color: var(--ink); }
        .main .block-container { max-width: 1100px; padding-top: 2.2rem; padding-bottom: 7rem; }
    [data-testid="stSidebar"] { background: #17231f; }
    [data-testid="stSidebar"] * { color: #f7f4ec; }
    [data-testid="stSidebar"] .stButton button { background: #263831; border: 1px solid #3f574d; color: #f7f4ec; text-align: left; }
    [data-testid="stSidebar"] .stButton button:hover { border-color: #ef765f; color: #fff; }
    .eyebrow { color: #ef765f; font-size: .75rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
        .hero { padding: .4rem 0 1.5rem; border-bottom: 1px solid var(--line); margin-bottom: 1.5rem; }
        .hero h1 { font-size: 3.25rem; line-height: 1.05; margin: .4rem 0 .75rem; }
        .hero p { color: var(--muted); max-width: 680px; font-size: 1.05rem; line-height: 1.65; }
    .status { display: flex; align-items: center; gap: .55rem; margin: .65rem 0; font-size: .9rem; }
        .dot { width: .55rem; height: .55rem; border-radius: 50%; background: #ef765f; display: inline-block; flex: 0 0 auto; }
        .dot.is-ready { background: #63d39a; }
    .chat-label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .12em; font-weight: 700; }
        .empty-state { padding: 1.25rem 0 .8rem; }
        .empty-state h2 { font-size: 1.5rem; margin-bottom: .3rem; }
        .empty-state p { color: var(--muted); margin: 0; }
        div[data-testid="stChatMessage"] { border-radius: 8px; border: 1px solid #dce6df; background: rgba(255,255,255,.78); }
        [data-testid="stChatInput"] { border-color: #bdcec3; }
        @media (max-width: 700px) {
            .main .block-container { padding: 1.4rem 1rem 6rem; }
            .hero h1 { font-size: 2.3rem; }
            .hero p { font-size: 1rem; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = ""
with st.sidebar:
    st.markdown("## Zyro Dynamics")
    st.caption("People Operations / Internal Desk")
    st.divider()
    pdf_count = sum(
        1 for path in CORPUS_PATH.glob("*") if path.is_file() and path.suffix.lower() == ".pdf"
    ) if CORPUS_PATH.exists() else 0
    st.markdown("### Setup status")
    corpus_class = "is-ready" if pdf_count else ""
    corpus_label = f"{pdf_count} policy documents found" if pdf_count else "Policy documents not found"
    st.markdown(
        f'<div class="status"><span class="dot {corpus_class}"></span>{corpus_label}</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    st.button("Reset conversation", use_container_width=True, on_click=reset_chat)

st.markdown(
    '<section class="hero"><div class="eyebrow">Zyro Dynamics · People Operations</div><h1>Answers from the handbook.</h1><p>Get clear guidance on leave, benefits, expenses, and workplace policies, grounded in the company handbook.</p></section>',
    unsafe_allow_html=True,
)

if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state"><h2>What can we help you find?</h2><p>Ask about leave, benefits, reimbursement, or workplace policies.</p></div>',
        unsafe_allow_html=True,
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = ""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Checking the policy library..."):
            try:
                llm = load_llm()
                vectorstore = load_rag_components()
                answer, _ = ask_bot(question, DEFAULT_TOP_K, vectorstore, llm)
                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )
            except Exception as error:
                answer = f"I couldn't complete that request: {error}"
                st.error(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown('<div class="chat-label">Ask the policy desk</div>', unsafe_allow_html=True)
prompt = st.chat_input("Ask a question about an HR policy...")
if prompt:
    submit_question(prompt)
    st.rerun()