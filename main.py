import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.postprocessor.cohere_rerank import CohereRerank

# 1. Load Environment Variables
load_dotenv()

# 2. THE ACCURACY UPGRADE
# Using text-embedding-3-large for 3072-dimension resolution
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
# Using gpt-4o for high-precision reasoning
Settings.llm = OpenAI(model="gpt-4o", temperature=0)

# 3. Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Initialize Vector DB
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
pinecone_index = pc.Index("city-data")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 5. THE QUALITY FILTER (Reranker)
# This uses the Cohere key you just added to Render
reranker = CohereRerank(api_key=os.environ.get("COHERE_API_KEY"), top_n=5)

# We grab 50 results first, then let Cohere pick the best 5
query_engine = index.as_query_engine(
    similarity_top_k=50, 
    node_postprocessors=[reranker]
)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "O.S.C.A.R. V2 Online (High Accuracy)"}

@app.post("/query")
def run_query(request: QueryRequest):
    try:
        response = query_engine.query(request.query)
        return {"response": str(response)}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
