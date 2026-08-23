from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = BASE_DIR / "data" / "novatech_mini_rag_test.pdf"
docs = PyPDFLoader(str(PDF_PATH)).load()

chunks = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150).split_documents(docs)
for d in chunks:
    d.page_content = d.page_content.encode("utf-8", "ignore").decode("utf-8", "ignore")


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = FAISS.from_documents(chunks, embeddings)
vector_retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 4})

bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4

# Hybrid Retrieval
retriever = EnsembleRetriever(retrievers = [bm25_retriever,vector_retriever],weights = [0.2,0.8])