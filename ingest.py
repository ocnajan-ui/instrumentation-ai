import os
import uuid
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions


# ================= LOAD ENV =================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file")


# ================= PATH =================
DATA_PATH = "data"
DB_PATH = "vector_store"


# ================= INIT CHROMA =================
client = chromadb.PersistentClient(path=DB_PATH)

embedding_function = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-small"
)

collection = client.get_or_create_collection(
    name="instrumentation_docs",
    embedding_function=embedding_function
)


# ================= LOAD DOCUMENTS =================
def load_documents():
    documents = []

    print("📂 Reading PDFs...\n")

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            print(f"📄 Loading: {file}")
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            documents.extend(loader.load())

    print(f"\n✅ Total pages loaded: {len(documents)}\n")
    return documents


# ================= SPLIT =================
def split_documents(documents):
    print("✂️ Splitting into chunks...\n")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    print(f"✅ Total chunks: {len(chunks)}\n")
    return chunks


# ================= STORE =================
def store_documents(chunks):
    print("💾 Storing in Chroma DB...\n")

    batch_size = 200

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]

        texts = [doc.page_content for doc in batch]
        metadatas = [doc.metadata for doc in batch]
        ids = [str(uuid.uuid4()) for _ in batch]

        print(f"➡️ Batch {i//batch_size + 1} ({len(batch)} chunks)")

        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )

    print("\n✅ VECTOR DB CREATED SUCCESSFULLY\n")


# ================= MAIN =================
if __name__ == "__main__":
    print("🚀 INGEST STARTED...\n")

    docs = load_documents()

    if not docs:
        print("❌ No PDFs found in data folder")
        exit()

    chunks = split_documents(docs)
    store_documents(chunks)

    print("🔥 DONE! DATA READY")