import os
import logging
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

query_engine = None

def get_engine():
    global query_engine
    if query_engine is None:
        logger.info("Initializing High-Accuracy Brain...")
        from pinecone import Pinecone
        from llama_index.vector_stores.pinecone import PineconeVectorStore
        from llama_index.core import VectorStoreIndex, Settings
        from llama_index.llms.openai import OpenAI
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.postprocessor.cohere_rerank import CohereRerank

        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        Settings.llm = OpenAI(model="gpt-4o", temperature=0)

        pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
        pinecone_index = pc.Index("city-data")
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
        index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
        
        # Reduced top_n to 3 to save memory/speed on free tier
        reranker = CohereRerank(api_key=os.environ.get("COHERE_API_KEY"), top_n=3)
        
        query_engine = index.as_query_engine(
            similarity_top_k=25, 
            node_postprocessors=[reranker]
        )
        logger.info("Brain initialized and ready.")
    return query_engine

class QueryRequest(BaseModel):
    query: str

@app.head("/")
def handle_head():
    return {"status": "online"}

@app.get("/")
def read_root():
    # If you see this message, you know the new code is live
    return {"status": "online", "message": "O.S.C.A.R. V3 Warmup Edition"}

@app.get("/warmup")
def warmup_brain():
    try:
        # Force the engine to load
        get_engine()
        return {"status": "ready", "message": "Brain is loaded. You can now chat."}
    except Exception as e:
        logger.error(f"Warmup Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def run_query(request: QueryRequest):
    try:
        engine = get_engine()
        logger.info(f"Query received: {request.query}")
        response = engine.query(request.query)
        return {"response": str(response)}
    except Exception as e:
        logger.error(f"Engine Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
