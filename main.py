import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI

# 1. Load Environment Variables
load_dotenv()

# 2. Configure Global Settings (The Brain Upgrade)
# We switch to GPT-4o-mini for better reasoning at low cost
# We set a strict system prompt to prevent hallucinations
Settings.llm = OpenAI(
    model="gpt-4o-mini",
    temperature=0.1, # Keep it factual, no creativity
    system_prompt="You are O.S.C.A.R., an expert on Banning City records. Always distinguish clearly between City Council Members, City Staff, and the Mayor. If you are unsure of a fact, state that you are unsure rather than guessing."
)

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

# 5. Create the Query Engine (The "Reader")
# similarity_top_k=8 means "Read the top 8 most relevant pages before answering"
# This helps catch lists that might be spread out.
query_engine = index.as_query_engine(similarity_top_k=8)

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"message": "O.S.C.A.R. API is Online (GPT-4o-mini)"}

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
