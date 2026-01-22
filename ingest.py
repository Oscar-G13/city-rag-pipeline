import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Settings
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from pinecone import Pinecone

load_dotenv()

# CRITICAL: Use the high-resolution embedding model (3072 dimensions)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")

# 1. Setup Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("city-data")

# 2. Connect Vector Store
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# 3. Load and Index Documents
print("Reading PDFs from /data folder...")
documents = SimpleDirectoryReader("./data").load_data()

print(f"Uploading {len(documents)} documents to Pinecone (High-Res)...")
index = VectorStoreIndex.from_documents(
    documents, 
    storage_context=storage_context,
    show_progress=True
)

print("Ingestion Complete. Your high-accuracy database is ready.")
