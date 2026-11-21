import asyncio
import logging
import os
import threading

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from security import RequestSizeLimit, require_api_key

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()
app = FastAPI()
app.add_middleware(RequestSizeLimit)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[value.strip() for value in os.environ.get("CITY_RAG_ALLOWED_ORIGINS", "").split(",") if value.strip() and value.strip() != "*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
query_engine = None
_engine_lock = threading.Lock()
_query_slots = asyncio.Semaphore(2)


def get_engine():
    global query_engine
    with _engine_lock:
        if query_engine is None:
            from pinecone import Pinecone
            from llama_index.vector_stores.pinecone import PineconeVectorStore
            from llama_index.core import VectorStoreIndex, Settings
            from llama_index.llms.openai import OpenAI
            from llama_index.embeddings.openai import OpenAIEmbedding
            from llama_index.postprocessor.cohere_rerank import CohereRerank

            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
            Settings.llm = OpenAI(model="gpt-4o", temperature=0, max_tokens=1024, timeout=30)
            pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
            vector_store = PineconeVectorStore(pinecone_index=pc.Index("city-data"))
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
            reranker = CohereRerank(api_key=os.environ.get("COHERE_API_KEY"), top_n=3)
            query_engine = index.as_query_engine(similarity_top_k=25, node_postprocessors=[reranker])
    return query_engine


class QueryRequest(BaseModel):
    query: str | None = Field(default=None, max_length=2000)
    message: str | None = Field(default=None, max_length=2000)
    prompt: str | None = Field(default=None, max_length=2000)
    question: str | None = Field(default=None, max_length=2000)


@app.head("/")
@app.get("/")
def read_root():
    return {"status": "online"}


@app.get("/warmup", dependencies=[Depends(require_api_key)])
def warmup_brain():
    try:
        get_engine()
        return {"status": "ready"}
    except Exception:
        logger.error("Query engine initialization failed")
        raise HTTPException(503, "Query service unavailable") from None


@app.post("/query", dependencies=[Depends(require_api_key)])
async def run_query(request: QueryRequest):
    user_input = request.query or request.message or request.prompt or request.question
    if not user_input or not user_input.strip():
        raise HTTPException(422, "No query found in request body")
    if _query_slots.locked():
        raise HTTPException(429, "Query service is busy", headers={"Retry-After": "10"})
    async with _query_slots:
        try:
            def query():
                return str(get_engine().query(user_input))
            return {"response": await run_in_threadpool(query)}
        except Exception:
            logger.error("Query provider request failed")
            raise HTTPException(503, "Query service unavailable") from None


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("PORT", 10000)))
