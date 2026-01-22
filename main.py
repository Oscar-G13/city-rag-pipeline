import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

# Enable logging to see errors in Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# THE ACCURACY UPGRADE
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
Settings.llm = OpenAI(model="gpt-4o", temperature=0)

app = FastAPI()

# Permissive CORS for Lovable
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    pinecone_index = pc.Index("city-data")
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Initialize Reranker
    cohere_key = os.environ.get("COHERE_API_KEY")
    reranker = CohereRerank(api_key=cohere_key, top_n=5)
    
    query_engine = index.as_query_engine(
        similarity_top_k=50, 
        node_postprocessors=[reranker]
    )
    logger.info("Engine initialized successfully.")
except Exception as e:
    logger.error(f"Startup Error: {e}")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "O.S.C.A.R. V2 Online (High Accuracy)"}

@app.post("/query")
async def run_query(request: QueryRequest):
    try:
        logger.info(f"Received query: {request.query}")
        response = query_engine.query(request.query)
        return {"response": str(response)}
    except Exception as e:
        logger.error(f"Query Error: {str(e)}")
        # This will send the actual error back to Lovable so we can see it
        raise HTTPException(status_code=500, detail=f"Engine Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
