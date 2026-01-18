import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, StorageContext

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize FastAPI
app = FastAPI()

# --- CRITICAL FIX: ENABLE CORS ---
# This tells the server to allow requests from ANY website (including Lovable)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)
# ---------------------------------

# 3. Initialize Vector DB (Pinecone)
# We do this once when the app starts so it's ready for queries
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("city-data")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
query_engine = index.as_query_engine()

# 4. Define the Data Model
# This matches exactly what Lovable sends: { "query": "text" }
class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "O.S.C.A.R. API is Online and Ready."}

@app.post("/query")
def run_query(request: QueryRequest):
    try:
        # Run the search on your documents
        response = query_engine.query(request.query)
        
        # Return the clean text
        return {"response": str(response)}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # This is for local testing only
    uvicorn.run(app, host="0.0.0.0", port=8000)
