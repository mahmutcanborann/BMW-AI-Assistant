import os
import sys
import re
import shutil  # klasör silmek/yönetmek için

sys.path.append(os.getcwd())

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm   # ilerleme çubuğu

DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"  # Veritabanının kaydedileceği klasör
CODES_SOURCE_NAME = "bmw_codes.csv"  # CSV dosyanın adı (data klasöründeki)

# --------------------------
#  Yardımcı fonksiyonlar
# --------------------------

def clean_bmw_text(text: str) -> str:
    """PDF metnini sadeleştirir."""
    text = re.sub(r'Online Edition for Part no.*', '', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = text.replace('-\n', '')
    text = re.sub(r'\n+', '\n', text)
    return text


def format_docs(docs) -> str:
    """Doküman listesini tek stringe çevirir."""
    return "\n\n".join(doc.page_content for doc in docs)


ERROR_CODE_PATTERN = re.compile(r"\b[PBUC]?\d{3,4}\b", re.IGNORECASE)

def extract_error_code(query: str):
    """Metinden P0300, P0420, 2A87 gibi hata kodu yakalar."""
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
    """Sorguda model / kasa / BMW ile ilgili bir anahtar kelime var mı?"""
    lower = query.lower()
    return any(k in lower for k in MODEL_KEYWORDS)

# --------------------------
#  Başlangıç
# --------------------------

print("🚗 BMW Universal Assistant (Persistent DB + Smart Routing, HF Embeddings) Starting...")

# 🔹 Embedding için HuggingFace sentence-transformer kullanıyoruz
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# --- 1. VERİTABANI KONTROLÜ ---
if os.path.exists(CHROMA_PATH):
    print(f"📂 Found existing database at '{CHROMA_PATH}'. Loading directly...")
    vectorstore = Chroma(
        collection_name="bmw_manuals",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )
else:
    print("✨ No database found. Building from scratch (This happens only once)...")

    if not os.path.exists(DATA_FOLDER):
        print(f"❌ ERROR: '{DATA_FOLDER}' folder not found.")
        sys.exit(1)

    print(f"📂 Scanning '{DATA_FOLDER}' folder...")
    all_docs = []

    for filename in os.listdir(DATA_FOLDER):
        file_path = os.path.join(DATA_FOLDER, filename)

        if filename.endswith(".pdf"):
            print(f"   📄 Processing PDF: {filename}...")
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.page_content = clean_bmw_text(doc.page_content)
                doc.metadata["source"] = filename
            all_docs.extend(raw_docs)

        elif filename.endswith(".csv"):
            print(f"   📊 Processing CSV: {filename}...")
            loader = CSVLoader(file_path)
            raw_docs = loader.load()
            for doc in raw_docs:
                doc.metadata["source"] = filename
            all_docs.extend(raw_docs)

    if not all_docs:
        print("❌ ERROR: No documents found!")
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    print(f"✅ Created {len(splits)} chunks.")

    vectorstore = Chroma(
        collection_name="bmw_manuals",
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    batch_size = 64  # HF lokal olduğu için daha büyük batch kaldırır
    print("💾 Saving database to disk with batched HF embedding...")
    for i in tqdm(range(0, len(splits), batch_size), desc="Embedding batches"):
        batch = splits[i:i + batch_size]
        vectorstore.add_documents(batch)

    print("✅ Database saved!")

# --------------------------
#  2. RETRIEVER & LLM
# --------------------------

# Hata kodu (CSV) için retriever
codes_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 4,
        "filter": {"source": CODES_SOURCE_NAME}
    }
)

# PDF manuel içerik için retriever
manuals_retriever = vectorstore.as_retriever(
    search_kwargs={
        "k": 6,
        "filter": {"source": {"$ne": CODES_SOURCE_NAME}}
    }
)

# Cevap üretimi için güçlü LLM (Ollama llama3)

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0,
    num_gpu=0  # GPU KAPALI -> CPU'da çalış
)

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

rag_chain = (
    {
        "input": RunnablePassthrough(),
        "context": manuals_retriever | RunnableLambda(format_docs),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# --------------------------
#  3. INTERAKTİF DÖNGÜ
# --------------------------

print("\n🎉 System Ready! (type 'q' to exit, type 'update' to rebuild DB)")
while True:
    query = input("\n🧑 Question: ")
    if query.lower() == "q":
        break

    if query.lower() == "update":
        print("♻️ Deleting old database and rebuilding...")
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        print("Please restart the program to rebuild.")
        break

    error_code = extract_error_code(query)
    has_model = contains_model(query)

    # 1) Sadece hata kodu sorusu
        # 1) Sadece hata kodu sorusu
    if error_code and not has_model:
        print(f"🔎 Detected error code: {error_code} (code-only query)")
        codes_docs = codes_retriever.invoke(query)
        context = format_docs(codes_docs)

        if not codes_docs:
            print("⚠️ No code docs found, falling back to manuals RAG...")
            response = rag_chain.invoke(query)
        else:
            response = qa_chain.invoke({"input": query, "context": context})

        print(f"\r🤖 Answer: {response}")
        continue

    # 2) Model + hata kodu birlikte
    if error_code and has_model:
        print(f"🔎 Detected error code: {error_code} with model info in query")
        codes_docs = codes_retriever.invoke(query)
        manuals_docs = manuals_retriever.invoke(query)
        merged_docs = codes_docs + manuals_docs
        context = format_docs(merged_docs)

        response = qa_chain.invoke({"input": query, "context": context})
        print(f"\r🤖 Answer: {response}")
        continue

    # 3) Normal soru: sadece manuel RAG
    print("🔎 No explicit error code detected. Using manuals RAG...")
    response = rag_chain.invoke(query)
    print(f"\r🤖 Answer: {response}")
