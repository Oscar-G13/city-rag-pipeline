import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 1. Setup
load_dotenv()
app = FastAPI()

# 2. CORS Security (Allows your frontend to talk to this backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, change this to your specific domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Data Models (Strict typing for the API)
class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str

# 4. Global Variables to hold our "Brain"
query_engine = None

# 5. Startup Event (Runs once when the server starts)
@app.on_event("startup")
def startup_event():
    global query_engine
    print("Initializing Vector Database Connection...")
    
    # Connect to Pinecone
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    pinecone_index = pc.Index("city-data")
    
    # Connect to LlamaIndex
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Create the engine
    query_engine = index.as_query_engine()
    print("Query Engine Ready!")

# 6. The Endpoint (The API Route)
@app.post("/query", response_model=QueryResponse)
async def query_index(request: QueryRequest):
    if not query_engine:
        raise HTTPException(status_code=503, detail="Search engine not ready")
    
    # Ask the question
    response = query_engine.query(request.question)
    
    return {"answer": str(response)}

@app.get("/")
def read_root():
    return {"status": "City RAG Pipeline is Active"}
