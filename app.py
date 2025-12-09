import os
import sys
import re

import streamlit as st

sys.path.append(os.getcwd())

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama   # 🔴 BURASI DEĞİŞTİ
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --------------------------
#  Sabitler
# --------------------------

DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"
CODES_SOURCE_NAME = "bmw_codes.csv"
def inject_custom_css():
    st.markdown(
        """
        <style>
        /* --- APP BACKGROUND (full turquoise) --- */
        .stApp {
            background: radial-gradient(circle at top, #00AEEF 0%, #00CFFD 40%, #009ECF 100%);
            color:#ffffff;
            font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont;
        }
        .bmw-card, .bmw-header {
    background: rgba(0,0,0,0.92);
    backdrop-filter: blur(8px) brightness(1.05);
    box-shadow: 0 8px 18px rgba(0,0,0,0.75);
    border: 1px solid rgba(255,255,255,0.10);
}
        .bmw-header {
    background: linear-gradient(180deg, #000 0%, #111 50%, #000 100%);
}

     .bmw-header-title {
    font-weight: 700;
    letter-spacing: 0.14em;
}
     .bmw-label {
    font-weight: 700;
    opacity: 0.9;
}
   /* --- HEADER --- */
        .bmw-header {
            background: linear-gradient(135deg, #006E9F, #003B55);
            border-radius: 18px;
            padding: 18px 24px;
            display: flex;
            align-items: center;
            gap: 20px;
            border: 1px solid rgba(255, 255, 255, 0.12);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55);
        }

        .bmw-header-title {
            font-size: 26px;
            font-weight: 650;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .bmw-header-subtitle {
            font-size: 14px;
            opacity: 0.9;
        }

        /* --- CARD STYLE --- */
        .bmw-card {
    background: rgba(0, 0, 0, 0.92);
    border-radius: 18px;
    padding: 20px 22px;
    border: 1px solid rgba(255, 255, 255, 0.10);
    box-shadow: 0 12px 35px rgba(0, 0, 0, 0.70);
}

        .bmw-label {
            font-size: 14px;
            font-weight: 600;
            letter-spacing: .04em;
            text-transform: uppercase;
            color: #B7F3FF;
        }

    textarea, input[type="text"] {
    background: rgba(255,255,255,0.75) !important;   /* stays light */
    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.55) !important;
    color: #000000 !important;                       /* << now black */
}



        /* --- BUTTON --- */
        .stButton>button {
            background: linear-gradient(135deg, #00DAFF, #006E9F);
            color: white;
            border-radius: 999px;
            border: none;
            padding: 0.45rem 1.6rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            box-shadow: 0 12px 28px rgba(0, 119, 163, 0.65);
        }
        .stButton>button:hover {
            filter: brightness(1.12);
            transform: scale(1.04);
        }

        /* --- ANSWER TEXT --- */
        .bmw-answer {
            font-size: 15px;
            line-height: 1.55;
            color:#E9FCFF;
        }

        /* --- ROUTING PANEL --- */
        .bmw-routing code {
            font-size: 12px !important;
            color:#000;
        }

        /* --- SIDEBAR TURQUOISE GLASS --- */
        section[data-testid="stSidebar"] {
            background: linear-gradient(175deg, #008CB7, #004E63);
            border-right: 1px solid rgba(255,255,255,0.22);
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# --------------------------
#  Yardımcı fonksiyonlar
# --------------------------

def clean_bmw_text(text: str) -> str:
    text = re.sub(r'Online Edition for Part no.*', '', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = text.replace('-\n', '')
    text = re.sub(r'\n+', '\n', text)
    return text


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


ERROR_CODE_PATTERN = re.compile(r"\b[PBUC]?\d{3,4}\b", re.IGNORECASE)

def extract_error_code(query: str):
    m = ERROR_CODE_PATTERN.search(query.upper())
    return m.group(0) if m else None


MODEL_KEYWORDS = [
    "1 series", "2 series", "3 series", "4 series", "5 series", "7 series",
    "x1", "x3", "x4", "x5", "x6", "x7",
    "i3", "i4", "i5", "i7", "ix", "xm",
    "f30", "f10", "g20", "g30", "m3", "m5", "m4", "m2",
    "bmw"
]

def contains_model(query: str) -> bool:
    lower = query.lower()
    return any(k in lower for k in MODEL_KEYWORDS)

# --------------------------
#  RAG BİLEŞENLERİNİ YÜKLE
# --------------------------

@st.cache_resource
def load_rag():
    # 1) Embedding
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # 2) Chroma DB
    if not os.path.exists(CHROMA_PATH):
        raise RuntimeError(
            f"'{CHROMA_PATH}' klasörü bulunamadı. "
            "Önce terminalde 'python main.py' çalıştırıp veritabanını oluşturmalısın."
        )

    vectorstore = Chroma(
        collection_name="bmw_manuals",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    # 3) Retrievers
    codes_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"source": CODES_SOURCE_NAME}
        }
    )

    manuals_retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 6,
            "filter": {"source": {"$ne": CODES_SOURCE_NAME}}
        }
    )

    # 4) LLM (burada senin kullandığın modele göre ayarla)
    llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_gpu=0)


    # Eğer ana kodunda hala "llama3" kullanıyorsan şunu kullan:
    # llm = ChatOllama(model="llama3", temperature=0)

    system_prompt = (
        "You are an expert BMW technical assistant. "
        "You have access to two types of data: "
        "1. **Owner's Manuals** (PDFs) for specific vehicle details. "
        "2. **Error Code Database** (CSV) for technical diagnostics. "
        "\n\n"
        "INSTRUCTIONS:\n"
        "- If the user asks about an **Error Code** (e.g., P0300), prefer the CSV data for the definition.\n"
        "- If the user asks about a **Specific Model** (e.g., '3 Series'), use the relevant manual context.\n"
        "- If the user provides BOTH a Model AND an Error Code, COMBINE the information:\n"
        "  - First explain the error code meaning (from CSV),\n"
        "  - Then explain what to check or how it applies to that specific BMW model (from PDFs).\n"
        "- Always answer in **ENGLISH**.\n"
        "- If the answer is not found in the context, say 'Information not found in my resources'.\n"
        "\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    qa_chain = prompt | llm | StrOutputParser()

    manuals_rag_chain = (
        {
            "input": RunnablePassthrough(),
            "context": manuals_retriever | RunnableLambda(format_docs),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return {
        "codes_retriever": codes_retriever,
        "manuals_retriever": manuals_retriever,
        "qa_chain": qa_chain,
        "manuals_rag_chain": manuals_rag_chain,
    }

def answer_question(query: str, rag_objects):
    codes_retriever = rag_objects["codes_retriever"]
    manuals_retriever = rag_objects["manuals_retriever"]
    qa_chain = rag_objects["qa_chain"]
    manuals_rag_chain = rag_objects["manuals_rag_chain"]

    error_code = extract_error_code(query)
    has_model = contains_model(query)

    # 1) Sadece hata kodu
    if error_code and not has_model:
        routing = f"Detected error code: {error_code} (code-only route → CSV)"
        codes_docs = codes_retriever.invoke(query)
        context = format_docs(codes_docs)

        if not codes_docs:
            routing += " | No code docs found, fallback to manuals RAG."
            answer = manuals_rag_chain.invoke(query)
        else:
            answer = qa_chain.invoke({"input": query, "context": context})

        return answer, routing, codes_docs

    # 2) Model + hata kodu
    if error_code and has_model:
        routing = f"Detected error code: {error_code} with model info (code + manuals fusion)"
        codes_docs = codes_retriever.invoke(query)
        manuals_docs = manuals_retriever.invoke(query)
        merged_docs = codes_docs + manuals_docs
        context = format_docs(merged_docs)

        answer = qa_chain.invoke({"input": query, "context": context})
        return answer, routing, merged_docs

    # 3) Normal soru
    routing = "No explicit error code detected (manuals RAG route)"
    answer = manuals_rag_chain.invoke(query)
    docs = manuals_retriever.invoke(query)
    return answer, routing, docs

# --------------------------
#  STREAMLIT ARAYÜZ
# --------------------------

def main():
    st.set_page_config(
        page_title="BMW Universal Assistant",
        page_icon="🚗",
        layout="wide"
    )

    # Inject custom CSS
    inject_custom_css()

    # Top header section
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        st.image("b.png", width=110)  # BMW logo (b.png)
    with col_title:
        st.markdown(
            """
            <div class="bmw-header">
                <div>
                    <div class="bmw-header-title">BMW Universal Assistant</div>
                    <div class="bmw-header-subtitle">
                        Local RAG • Ollama • Chroma • Technical Diagnostics & Manuals
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("")  # small vertical space

    # Layout: left = question & answer, right = routing/context debug
    left_col, right_col = st.columns([2.1, 1.4])

    # Sidebar (static information)
    with st.sidebar:
        st.header("⚙️ RAG Status")
        st.write(f"📁 Data folder: `{DATA_FOLDER}`")
        st.write(f"💾 Vector DB path: `{CHROMA_PATH}`")

        st.markdown("---")
        st.caption("🧠 LLM: `llama3.2:3b` (Ollama)")
        st.caption("🔎 Embeddings: `all-MiniLM-L6-v2` (HF)")

    # Load RAG components
    try:
        rag_objects = load_rag()
    except Exception as e:
        st.error(f"RAG initialization failed: {e}")
        st.stop()

    # LEFT COLUMN: question + answer
    with left_col:
        st.markdown('<div class="bmw-card">', unsafe_allow_html=True)
        st.markdown('<div class="bmw-label">Ask your question</div>', unsafe_allow_html=True)

        default_q = ""
        query = st.text_input(
            label="",
            value=default_q,
            placeholder='E.g. "In my BMW 3 Series, how to pair my iPhone?" or "What does error code P0300 mean?"',
        )

        ask_clicked = st.button("Ask")

        if (ask_clicked or query) and query.strip():
            with st.spinner("Analyzing your BMW data..."):
                answer, routing, docs = answer_question(query, rag_objects)

            st.markdown("---")
            st.subheader("🤖 Assistant")
            st.markdown(f'<div class="bmw-answer">{answer}</div>', unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)  # close bmw-card
        else:
            st.markdown("</div>", unsafe_allow_html=True)

    # RIGHT COLUMN: routing + retrieved context
    with right_col:
        st.markdown('<div class="bmw-card">', unsafe_allow_html=True)
        st.markdown('<div class="bmw-label">Routing & Context</div>', unsafe_allow_html=True)

        if (ask_clicked or query) and query.strip():
            st.markdown("**🧠 Routing**")
            st.markdown('<div class="bmw-routing">', unsafe_allow_html=True)
            st.code(routing, language="text")
            st.markdown('</div>', unsafe_allow_html=True)

            with st.expander("📄 Retrieved context (debug)", expanded=False):
                for i, d in enumerate(docs, start=1):
                    st.markdown(f"**Chunk {i} – Source: `{d.metadata.get('source', 'unknown')}`**")
                    st.write(d.page_content[:800] + ("..." if len(d.page_content) > 800 else ""))
        else:
            st.write(
                "You have not asked a question yet. "
                "Once you submit a question on the left, RAG routing and the related manual/code context will appear here."
            )

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
