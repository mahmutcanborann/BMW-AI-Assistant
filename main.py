import os
import sys
import re
import shutil
import pickle
from typing import List, Optional, Dict, Any, Tuple

sys.path.append(os.getcwd())

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tqdm import tqdm

from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder


# =========================================================
# CONFIG
# =========================================================
DATA_FOLDER = "data"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "bmw_manuals"
SPLITS_CACHE_PATH = "splits_cache.pkl"
CODES_SOURCE_NAME = "bmw_codes.csv"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
OLLAMA_MODEL_NAME = "llama3.2:3b"

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 64

BM25_K = 8
VECTOR_K = 8
VECTOR_FETCH_K = 24
VECTOR_LAMBDA_MULT = 0.5
FINAL_RERANK_TOP_K =10
RRF_K = 60

_RERANKER = None


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        print("[INIT] Loading reranker model...")
        _RERANKER = CrossEncoder(RERANK_MODEL_NAME)
    return _RERANKER


# =========================================================
# SCHEMAS
# =========================================================
class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    query: str
    normalized_query: str
    route: str
    retrieval_strategy: str
    error_code: Optional[str]
    model: Optional[str]
    year: Optional[str]
    generation: Optional[str]
    has_model: bool
    warning_note: Optional[str]
    answer: str
    sources: List[Dict[str, Any]]


class RetrieveDebugResponse(BaseModel):
    query: str
    normalized_query: str
    route: str
    retrieval_strategy: str
    error_code: Optional[str]
    model: Optional[str]
    year: Optional[str]
    generation: Optional[str]
    has_model: bool
    documents: List[Dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    db_exists: bool
    manual_chunks: int
    code_chunks: int
    total_chunks: int
    model_name: str


# =========================================================
# MODEL / GENERATION KNOWLEDGE
# =========================================================
KNOWN_MODELS = [
    "1 series", "2 series", "3 series", "4 series", "5 series", "6 series", "7 series", "8 series",
    "x1", "x2", "x3", "x4", "x5", "x6", "x7",
    "i3", "i4", "i5", "i7", "ix", "xm",
    "m2", "m3", "m4", "m5", "z4",
    "320i", "330i", "340i", "520i", "520d", "530i", "530d",
    "f30", "f10", "g20", "g30", "g05", "g01", "g06", "g07"
]

GENERATION_TO_FAMILY = {
    "g20": "3 series",
    "f30": "3 series",
    "g30": "5 series",
    "f10": "5 series",
    "g05": "x5",
    "g01": "x3",
    "g06": "x6",
    "g07": "x7",
}

FAMILY_TO_GENERATIONS = {
    "3 series": ["f30", "g20"],
    "5 series": ["f10", "g30"],
    "x3": ["g01"],
    "x5": ["g05"],
    "x6": ["g06"],
    "x7": ["g07"],
}

YEAR_PATTERN = re.compile(r"(20\d{2})")
ERROR_CODE_PATTERN = re.compile(r"\b[PBUC]\d{3,4}\b", re.IGNORECASE)


# =========================================================
# QUERY NORMALIZATION / REWRITE
# =========================================================
def normalize_user_query(query: str) -> str:
    q = query.lower().strip()

    replacements = {
        "deries": "series",
        "seris": "series",
        "car play": "apple carplay",
        "carplay": "apple carplay",
        "head lights": "headlights",
        "open the headlights": "turn on the headlights",
        "open headlights": "turn on headlights",
        "open the lights": "turn on the lights",
        "open lights": "turn on lights",
        "open the doors": "unlock and open the doors",
        "open doors": "unlock and open the doors",
        "open the trunk": "open the tailgate trunk",
        "open trunk": "open the tailgate trunk",
        "boot": "trunk tailgate",
    }

    for wrong, right in replacements.items():
        q = q.replace(wrong, right)
    q = re.sub(r"\bderies\b", "series", q)
    q = re.sub(r"\bseris\b", "series", q)
    q = re.sub(r"\bserie\b", "series", q)
    q = re.sub(r"\bbmw\s+x\s*([1-7])\b", r"bmw x\1", q)
    q = re.sub(r"\bbmw\s+i\s*([357])\b", r"bmw i\1", q)
    q = re.sub(r"\bbmw\s+([1-8])\s+series\b", r"bmw \1 series", q)
    q = re.sub(r"\s+", " ", q).strip()

    return q


def expand_query_for_retrieval(query: str) -> str:
    q = query.lower()

    expansions = []

    if "headlight" in q or "light" in q:
        expansions.extend([
            "lighting",
            "light switch",
            "low beams",
            "high beams",
            "parking lights",
            "automatic headlight control",
            "exterior lighting"
        ])

    if "door" in q or "unlock" in q:
        expansions.extend([
            "unlocking",
            "locking",
            "central locking",
            "vehicle key",
            "comfort access",
            "door handle"
        ])

    if "apple carplay" in q:
        expansions.extend([
            "Apple CarPlay",
            "Bluetooth",
            "mobile devices",
            "communication",
            "iDrive",
            "connect device"
        ])

    if "trunk" in q or "tailgate" in q:
        expansions.extend([
            "tailgate",
            "trunk lid",
            "opening and closing",
            "vehicle key",
            "comfort access"
        ])

    if not expansions:
        return query

    return query + " " + " ".join(expansions)


# =========================================================
# HELPERS
# =========================================================
def normalize_model_name(model: Optional[str]) -> Optional[str]:
    if not model:
        return None
    return model.strip().lower()


def resolve_model_family(model: Optional[str]) -> Optional[str]:
    if not model:
        return None

    model = normalize_model_name(model)

    if model in FAMILY_TO_GENERATIONS:
        return model

    if model in GENERATION_TO_FAMILY:
        return GENERATION_TO_FAMILY[model]

    return model


def extract_model_info(query: str) -> Dict[str, Optional[str]]:
    q = query.lower().strip()

    year_match = YEAR_PATTERN.search(q)
    year = year_match.group(1) if year_match else None

    found_model = None
    found_generation = None

    for model in sorted(KNOWN_MODELS, key=len, reverse=True):
        pattern = rf"\b{re.escape(model)}\b"
        if re.search(pattern, q):
            found_model = normalize_model_name(model)
            break

    if found_model in GENERATION_TO_FAMILY:
        found_generation = found_model
        found_model = GENERATION_TO_FAMILY[found_model]
    elif found_model in FAMILY_TO_GENERATIONS:
        gens = FAMILY_TO_GENERATIONS.get(found_model, [])
        found_generation = gens[0] if len(gens) == 1 else None

    return {
        "model": resolve_model_family(found_model),
        "year": year,
        "generation": found_generation
    }


def extract_error_code(query: str) -> Optional[str]:
    m = ERROR_CODE_PATTERN.search(query.upper())
    return m.group(0) if m else None


def infer_metadata_from_filename(filename: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    lower = filename.lower()
    normalized_lower = lower.replace("_", " ").replace("-", " ")

    detected_model = None
    detected_year = None
    detected_generation = None

    for m in sorted(KNOWN_MODELS, key=len, reverse=True):
        pattern = rf"\b{re.escape(m)}\b"
        if re.search(pattern, normalized_lower):
            detected_model = normalize_model_name(m)
            break

    year_match = YEAR_PATTERN.search(lower)
    if year_match:
        detected_year = year_match.group(1)

    for gen in GENERATION_TO_FAMILY.keys():
        if gen in lower:
            detected_generation = gen
            break

    model_family = resolve_model_family(detected_model)

    return model_family, detected_year, detected_generation


def clean_bmw_text(text: str) -> str:
    text = re.sub(r"Online Edition for Part no.*", "", text)
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = text.replace("-\n", "")
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def format_docs_with_source(docs) -> str:
    formatted = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "N/A")
        doc_type = doc.metadata.get("doc_type", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        model = doc.metadata.get("model", "N/A")
        year = doc.metadata.get("year", "N/A")
        generation = doc.metadata.get("generation", "N/A")

        formatted.append(
            f"[DOC {i} | SOURCE: {source} | PAGE: {page} | TYPE: {doc_type} | "
            f"MODEL: {model} | YEAR: {year} | GENERATION: {generation} | CHUNK: {chunk_id}]\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)

def format_docs_for_llm(docs) -> str:
    formatted = []

    for i, doc in enumerate(docs, start=1):
        formatted.append(
            f"[DOC {i}]\n{doc.page_content}"
        )

    return "\n\n".join(formatted)

def extract_sources(docs):
    results = []

    for doc in docs:
        results.append({
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", "-")
        })

    return results
def deduplicate_by_page(docs):
    unique = {}
    for doc in docs:
        key = (
            doc.metadata.get("source"),
            doc.metadata.get("page")
        )
        if key not in unique:
            unique[key] = doc

    return list(unique.values())

def docs_to_debug_json(docs) -> List[Dict[str, Any]]:
    results = []

    for doc in docs:
        results.append({
            "source": doc.metadata.get("source"),
            "doc_type": doc.metadata.get("doc_type"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "model": doc.metadata.get("model"),
            "year": doc.metadata.get("year"),
            "generation": doc.metadata.get("generation"),
            "preview": doc.page_content[:500]
        })

    return results


def guardrail(context: str, answer: str) -> str:
    if not context or len(context.strip()) < 80:
        return "Information not reliable from retrieved documents."

    risky_phrases = [
        "i assume",
        "probably",
        "it may be",
        "likely means"
    ]

    lower_answer = answer.lower()

    if any(p in lower_answer for p in risky_phrases) and "not found" not in lower_answer:
        return (
            "The retrieved documents do not support a reliable answer strongly enough. "
            "Please refine the question or provide the exact BMW model / year / error code."
        )

    return answer


def build_warning_note(
    requested_model: Optional[str],
    requested_year: Optional[str],
    requested_generation: Optional[str],
    docs: List
) -> str:
    if not docs:
        return ""

    matched_models = {
        normalize_model_name(doc.metadata.get("model"))
        for doc in docs
        if doc.metadata.get("model")
    }

    matched_years = {
        str(doc.metadata.get("year"))
        for doc in docs
        if doc.metadata.get("year") not in [None, "N/A"]
    }

    matched_generations = {
        doc.metadata.get("generation")
        for doc in docs
        if doc.metadata.get("generation")
    }

    same_model = requested_model is not None and requested_model in matched_models
    same_year = requested_year is not None and requested_year in matched_years
    same_generation = requested_generation is not None and requested_generation in matched_generations

    if requested_model and requested_year:
        if same_model and same_year:
            return ""

        if same_model and not same_year:
            if requested_generation and matched_generations and not same_generation:
                return (
                    f"I could not find an exact {requested_year} {requested_model} source. "
                    "The retrieved material appears to come from a different generation, "
                    "so the instructions may vary significantly."
                )

            if matched_years:
                try:
                    req_year = int(requested_year)
                    matched_years_int = [int(y) for y in matched_years if y.isdigit()]

                    if matched_years_int:
                        closest_year = min(matched_years_int, key=lambda y: abs(y - req_year))
                        diff = abs(closest_year - req_year)

                        if diff <= 2:
                            return (
                                f"I could not find an exact {requested_year} {requested_model} source. "
                                f"The closest available year is {closest_year}, and the answer is based on that. "
                                "The systems are likely very similar, but minor UI differences may exist."
                            )

                        return (
                            f"I could not find an exact {requested_year} {requested_model} source. "
                            f"The closest available year is {closest_year}, which may belong to a different generation. "
                            "Instructions and interface details may differ significantly."
                        )
                except Exception:
                    pass

            return (
                f"I could not find an exact {requested_year} {requested_model} source. "
                "The answer below is based on the same model from another available year, "
                "so UI or equipment differences may exist."
            )

    return ""


def doc_key(doc) -> Tuple:
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("chunk_id"),
        hash(doc.page_content)
    )


def reciprocal_rank_fusion(result_lists: List[List], k: int = RRF_K) -> List:
    scores = {}
    docs_by_key = {}

    for docs in result_lists:
        for rank, doc in enumerate(docs, start=1):
            key = doc_key(doc)
            docs_by_key[key] = doc
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    return [docs_by_key[key] for key in ranked_keys]


# =========================================================
# DATA LOADING
# =========================================================
def load_source_documents(data_folder: str):
    if not os.path.exists(data_folder):
        raise FileNotFoundError(f"'{data_folder}' folder not found.")

    manual_docs = []
    code_docs = []

    for filename in os.listdir(data_folder):
        file_path = os.path.join(data_folder, filename)

        if filename.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            raw_docs = loader.load()

            model, year, generation = infer_metadata_from_filename(filename)
            print(f"[LOAD] {filename} -> model={model}, year={year}, generation={generation}")

            for doc in raw_docs:
                doc.page_content = clean_bmw_text(doc.page_content)
                doc.metadata["source"] = filename
                doc.metadata["doc_type"] = "manual"
                doc.metadata["model"] = model
                doc.metadata["year"] = year
                doc.metadata["generation"] = generation

            manual_docs.extend(raw_docs)

        elif filename.lower().endswith(".csv"):
            loader = CSVLoader(file_path)
            raw_docs = loader.load()

            for doc in raw_docs:
                doc.metadata["source"] = filename
                doc.metadata["doc_type"] = "code"
                doc.metadata["model"] = None
                doc.metadata["year"] = None
                doc.metadata["generation"] = None

            code_docs.extend(raw_docs)

    return manual_docs, code_docs


def split_and_tag_documents(manual_raw_docs, code_raw_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    manual_splits = text_splitter.split_documents(manual_raw_docs) if manual_raw_docs else []
    code_splits = text_splitter.split_documents(code_raw_docs) if code_raw_docs else []

    for i, doc in enumerate(manual_splits):
        doc.metadata["chunk_id"] = f"manual_{i}"

    for i, doc in enumerate(code_splits):
        doc.metadata["chunk_id"] = f"code_{i}"

    return manual_splits, code_splits


# =========================================================
# VECTOR DB
# =========================================================
def build_or_load_vectorstore(all_splits, embedding_function):
    if os.path.exists(CHROMA_PATH):
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_function
        )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    for i in tqdm(range(0, len(all_splits), EMBED_BATCH_SIZE), desc="Embedding batches"):
        batch = all_splits[i:i + EMBED_BATCH_SIZE]
        vectorstore.add_documents(batch)

    return vectorstore


# =========================================================
# RETRIEVAL ENGINE
# =========================================================
class BMWRetrievalEngine:
    def __init__(self, vectorstore, manual_splits):
        self.vectorstore = vectorstore
        self.manual_splits = manual_splits
        self.reranker = get_reranker()
        self.bm25_cache = {}
        self.build_bm25_indexes()
        self.codes_retriever = self.vectorstore.as_retriever(
            search_kwargs={
                "k": 4,
                "filter": {"source": CODES_SOURCE_NAME}
            }
        )
    def build_bm25_indexes(self):
        grouped = {}

        for doc in self.manual_splits:
            if doc.metadata.get("doc_type") != "manual":
                continue

            model = normalize_model_name(doc.metadata.get("model"))
            year = str(doc.metadata.get("year")) if doc.metadata.get("year") else None

            grouped.setdefault(("all", None), []).append(doc)

            if model:
                grouped.setdefault((model, None), []).append(doc)

            if model and year:
                grouped.setdefault((model, year), []).append(doc)

        for key, docs in grouped.items():
            retriever = BM25Retriever.from_documents(docs)
            retriever.k = BM25_K
            self.bm25_cache[key] = retriever

        print(f"[INIT] BM25 indexes built: {len(self.bm25_cache)}")
    
    def retrieve_code_docs(self, query: str):
        return self.codes_retriever.invoke(query)

    def filter_docs_by_metadata(
        self,
        docs: List,
        model: Optional[str] = None,
        year: Optional[str] = None
    ) -> List:
        filtered = []

        for doc in docs:
            if doc.metadata.get("doc_type") != "manual":
                continue

            doc_model = normalize_model_name(doc.metadata.get("model"))

            if model:
                if not doc_model:
                    continue

                if model not in doc_model and doc_model not in model:
                    continue

            if year and str(doc.metadata.get("year")) != str(year):
                continue

            filtered.append(doc)

        return filtered

    def build_manual_filter(
        self,
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        conditions = [
            {"doc_type": {"$eq": "manual"}}
        ]

        if model:
            conditions.append({"model": {"$eq": normalize_model_name(model)}})

        if len(conditions) == 1:
            return conditions[0]

        return {"$and": conditions}

    def rerank(self, query: str, docs: List, model: Optional[str] = None) -> List:
        if not docs:
            return []

        pairs = [(query, doc.page_content) for doc in docs]
        scores = self.reranker.predict(pairs)

        ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
        top_docs = [doc for doc, _ in ranked[:FINAL_RERANK_TOP_K]]

        if model:
            filtered = []
            for doc in top_docs:
                doc_model = normalize_model_name(doc.metadata.get("model"))
                if doc_model and (model in doc_model or doc_model in model):
                    filtered.append(doc)

            return filtered

        return top_docs

    def build_hybrid_retriever(
        self,
        model: Optional[str] = None,
        year: Optional[str] = None
    ):
        vector_filter = self.build_manual_filter(model=model)


        vector_retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": VECTOR_K,
                "fetch_k": VECTOR_FETCH_K,
                "lambda_mult": VECTOR_LAMBDA_MULT,
                "filter": vector_filter
            }
        )

        bm25_key = (
        normalize_model_name(model) if model else "all",
        str(year) if year else None
        )   

        bm25_retriever = self.bm25_cache.get(bm25_key)

        if bm25_retriever is None:
            return vector_retriever

        class RRFHybridRetriever:
            def __init__(self, bm25_retriever, vector_retriever):
                self.bm25_retriever = bm25_retriever
                self.vector_retriever = vector_retriever

            def invoke(self, query: str):
                bm25_results = self.bm25_retriever.invoke(query)
                vector_results = self.vector_retriever.invoke(query)

                return reciprocal_rank_fusion([bm25_results, vector_results])

        return RRFHybridRetriever(bm25_retriever, vector_retriever)

    def smart_manual_retrieve(self, query: str):
        normalized_query = normalize_user_query(query)
        retrieval_query = expand_query_for_retrieval(normalized_query)

        info = extract_model_info(normalized_query)
        model = info["model"]
        year = info["year"]

        if model and year:
            exact_retriever = self.build_hybrid_retriever(model=model, year=year)
            exact_docs = exact_retriever.invoke(retrieval_query)
            exact_docs = self.rerank(retrieval_query, exact_docs, model)

            exact_year_docs = [
                doc for doc in exact_docs
                if str(doc.metadata.get("year")) == str(year)
            ]

            if exact_year_docs:
                return exact_year_docs, "exact_match", info

        if model:
            model_retriever = self.build_hybrid_retriever(model=model, year=None)
            model_docs = model_retriever.invoke(retrieval_query)
            model_docs = self.rerank(retrieval_query, model_docs, model)

            if model_docs:
                return model_docs, "model_match", info

        general_retriever = self.build_hybrid_retriever(model=None, year=None)
        general_docs = general_retriever.invoke(retrieval_query)
        general_docs = self.rerank(retrieval_query, general_docs)

        return general_docs, "general_bmw", info


# =========================================================
# ASSISTANT SERVICE
# =========================================================
class BMWAssistantService:
    def __init__(self):
        self.embedding_function = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME
        )

        self.llm = ChatOllama(
            model=OLLAMA_MODEL_NAME,
            temperature=0,
            num_gpu=1
        )

        self.prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an expert BMW technical assistant.\n"
     "You have access to two data sources:\n"
     "1. Owner's Manuals (PDFs)\n"
     "2. Error Code Database (CSV)\n\n"

     "INSTRUCTIONS:\n"
     "- If the user asks about an error code, prioritize the CSV definition.\n"
     "- If the user asks about a BMW model, use the manual context.\n"
     "- If both model and error code exist, combine both.\n"

     "- Always answer in ENGLISH.\n"
     "- Use ONLY the provided context.\n"

     "CRITICAL RULES:\n"
     "- Only include information directly relevant to the question\n"
     "- Do NOT copy full paragraphs from the context\n"
     "- Rewrite in your own words\n"
     "- Keep the answer concise (4-6 sentences max)\n"
     "- Ignore unrelated technical text\n"
     "- Structure the answer as a short step-by-step list when applicable.\n"
     "- Use numbered steps for instructions.\n"
     "- Do NOT include source metadata (DOC, SOURCE, PAGE).\n"
     "- Do NOT output a Sources section\n"

     "- Do not add any steps or details not present in the context.\n"
     "- Do not infer additional BMW features beyond the given text.\n"
     "- Prefer the most specific model/year evidence when available.\n"
     "- If the exact wording is not found, infer from related concepts.\n"
     "- Answer only the exact feature asked (e.g. low beams, fog lights).\n"
     "- Ignore related systems like high beam assistant unless explicitly asked.\n"
     "- Only say 'Information not found in my resources' if absolutely no relevant information exists.\n"
     "- Do not start with 'Warning': unless there is a real safety warning.\n"
     "- Do not mention Automatic High Beam Assistant unless the user explicitly asks about high beams or high beam assistant.\n"
     "- If warning_note is provided, incorporate it naturally at the beginning.\n"

     "\nwarning_note:\n{warning_note}\n\n"
     "Context:\n{context}"
    ),
    ("human", "{input}")
])

        self.qa_chain = self.prompt | self.llm | StrOutputParser()
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user question into a better BMW owner manual search query.\n"
     "Keep BMW model, year, feature names, and user intent.\n"
     "Do NOT add a year unless it is explicitly present in the user query.\n"
     "Fix typos and convert vague wording into BMW manual terminology.\n"
     "Examples:\n"
     "- open headlights -> turn on headlights low beams lighting switch exterior lighting\n"
     "- open doors -> unlock doors central locking vehicle key comfort access\n"
     "- connect car play -> connect Apple CarPlay Bluetooth mobile devices iDrive\n"
     "Do not answer the question.\n"
     "Return only the rewritten search query."
    ),
    ("human", "{query}")
])

        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()

        self.manual_splits = []
        self.code_splits = []
        self.vectorstore = None
        self.retrieval_engine = None

        self.initialize()

    def initialize(self):
        if os.path.exists(CHROMA_PATH) and os.path.exists(SPLITS_CACHE_PATH):
            print("[INIT] Loading Chroma DB and cached splits...")

            with open(SPLITS_CACHE_PATH, "rb") as f:
                cache = pickle.load(f)

            cached_embedding_model = cache.get("embedding_model")

            if cached_embedding_model != EMBEDDING_MODEL_NAME:
                print("[INIT] Embedding model changed. Rebuilding index...")

                if os.path.exists(SPLITS_CACHE_PATH):
                    os.remove(SPLITS_CACHE_PATH)

                if os.path.exists(CHROMA_PATH):
                    shutil.rmtree(CHROMA_PATH)

                return self.initialize()

            self.manual_splits = cache["manual_splits"]
            self.code_splits = cache["code_splits"]

            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=CHROMA_PATH,
                embedding_function=self.embedding_function
            )

        else:
            print("[INIT] Building index from source documents...")

            manual_raw_docs, code_raw_docs = load_source_documents(DATA_FOLDER)

            self.manual_splits, self.code_splits = split_and_tag_documents(
                manual_raw_docs,
                code_raw_docs
            )

            all_splits = self.manual_splits + self.code_splits

            self.vectorstore = build_or_load_vectorstore(
                all_splits,
                self.embedding_function
            )

            with open(SPLITS_CACHE_PATH, "wb") as f:
                pickle.dump({
                    "manual_splits": self.manual_splits,
                    "code_splits": self.code_splits,
                    "embedding_model": EMBEDDING_MODEL_NAME
                }, f)

        self.retrieval_engine = BMWRetrievalEngine(
            vectorstore=self.vectorstore,
            manual_splits=self.manual_splits
        )

    def rebuild(self):
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)

        if os.path.exists(SPLITS_CACHE_PATH):
            os.remove(SPLITS_CACHE_PATH)

        self.initialize()

    def rewrite_query_with_llm(self, query: str) -> str:
        try:
            rewritten = self.rewrite_chain.invoke({"query": query}).strip()
            rewritten = rewritten.replace('"', '')

            if not rewritten:
                return query

            original_info = extract_model_info(query)
            rewritten_info = extract_model_info(rewritten)

            if not original_info.get("year") and rewritten_info.get("year"):
                rewritten = re.sub(r"\b20\d{2}\b", "", rewritten)
                rewritten_info = extract_model_info(rewritten)

            if original_info.get("model") and not rewritten_info.get("model"):
                rewritten += f" {original_info['model']}"

            if original_info.get("year") and not rewritten_info.get("year"):
                rewritten += f" {original_info['year']}"

            return normalize_user_query(rewritten)

        except Exception:
            return normalize_user_query(query)

            

    
    def route_query(self, query: str):
        normalized_query = normalize_user_query(query)

        error_code = extract_error_code(normalized_query)
        model_info = extract_model_info(normalized_query)

        model = model_info["model"]
        year = model_info["year"]
        generation = model_info["generation"]
        has_model = model is not None

        if error_code and not has_model:
            return "code_only", error_code, model, year, generation, has_model, normalized_query
        elif error_code and has_model:
            return "model_and_code", error_code, model, year, generation, has_model, normalized_query
        else:
            return "manual_only", error_code, model, year, generation, has_model, normalized_query

    def retrieve_for_query(self, query: str):
        normalized_query = normalize_user_query(query)
        rewritten_query = self.rewrite_query_with_llm(normalized_query)
        

        (
            route,
            error_code,
            model,
            year,
            generation,
            has_model,
            _
        ) = self.route_query(rewritten_query)

        if route == "code_only":
            code_docs = self.retrieval_engine.retrieve_code_docs(rewritten_query)
            if code_docs:
                docs = code_docs
                retrieval_strategy = "code_lookup"
            else:
                docs, retrieval_strategy, _ = self.retrieval_engine.smart_manual_retrieve(rewritten_query)

            return route, retrieval_strategy, error_code, model, year, generation, has_model, normalized_query, docs

        if route == "model_and_code":
            code_docs = self.retrieval_engine.retrieve_code_docs(rewritten_query)
            manual_docs, retrieval_strategy, _ = self.retrieval_engine.smart_manual_retrieve(rewritten_query)
            docs = code_docs + manual_docs

            return route, retrieval_strategy, error_code, model, year, generation, has_model, rewritten_query, docs

        docs, retrieval_strategy, _ = self.retrieval_engine.smart_manual_retrieve(rewritten_query)
        return route, retrieval_strategy, error_code, model, year, generation, has_model, rewritten_query, docs


    def answer_query(self, query: str):
        (
            route,
            retrieval_strategy,
            error_code,
            model,
            year,
            generation,
            has_model,
            normalized_query,
            docs
        ) = self.retrieve_for_query(query)

        docs = deduplicate_by_page(docs)
        context = format_docs_for_llm(docs)

        warning_note = build_warning_note(
            requested_model=model,
            requested_year=year,
            requested_generation=generation,
            docs=docs
        )

        answer = self.qa_chain.invoke({
            "input": query,
            "context": context,
            "warning_note": warning_note
        })

        answer = guardrail(context, answer)
        sources = extract_sources(docs)

        return {
            "route": route,
            "retrieval_strategy": retrieval_strategy,
            "error_code": error_code,
            "model": model,
            "year": year,
            "generation": generation,
            "has_model": has_model,
            "normalized_query": normalized_query,
            "warning_note": warning_note,
            "answer": answer,
            "sources": sources,
            "docs": docs
        }


# =========================================================
# FASTAPI APP
# =========================================================
app = FastAPI(title="BMW AI Assistant API", version="3.0.0")

assistant_service: Optional[BMWAssistantService] = None


@app.on_event("startup")
def startup_event():
    global assistant_service
    try:
        assistant_service = BMWAssistantService()
    except Exception as e:
        raise RuntimeError(f"Startup failed: {e}")


@app.get("/health", response_model=HealthResponse)
def health():
    global assistant_service

    if assistant_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    return HealthResponse(
        status="ok",
        db_exists=os.path.exists(CHROMA_PATH),
        manual_chunks=len(assistant_service.manual_splits),
        code_chunks=len(assistant_service.code_splits),
        total_chunks=len(assistant_service.manual_splits) + len(assistant_service.code_splits),
        model_name=OLLAMA_MODEL_NAME
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    global assistant_service

    if assistant_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        result = assistant_service.answer_query(request.query)

        return AskResponse(
            query=request.query,
            normalized_query=result["normalized_query"],
            route=result["route"],
            retrieval_strategy=result["retrieval_strategy"],
            error_code=result["error_code"],
            model=result["model"],
            year=result["year"],
            generation=result["generation"],
            has_model=result["has_model"],
            warning_note=result["warning_note"],
            answer=result["answer"],
            sources=result["sources"]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/retrieve-debug", response_model=RetrieveDebugResponse)
def retrieve_debug(request: AskRequest):
    global assistant_service

    if assistant_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        (
            route,
            retrieval_strategy,
            error_code,
            model,
            year,
            generation,
            has_model,
            normalized_query,
            docs
        ) = assistant_service.retrieve_for_query(request.query)

        return RetrieveDebugResponse(
            query=request.query,
            normalized_query=normalized_query,
            route=route,
            retrieval_strategy=retrieval_strategy,
            error_code=error_code,
            model=model,
            year=year,
            generation=generation,
            has_model=has_model,
            documents=docs_to_debug_json(docs)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/rebuild-db")
def rebuild_db():
    global assistant_service

    if assistant_service is None:
        raise HTTPException(status_code=500, detail="Service not initialized")

    try:
        assistant_service.rebuild()
        return {"status": "ok", "message": "Database rebuilt successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))