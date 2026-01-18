import os
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 1. Load keys
load_dotenv()

# 2. Connect to Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index("city-data")

# 3. Connect to the existing Vector Store
# Note: We use 'from_vector_store' here. We are NOT re-loading the PDF. 
# We are connecting to the math that already exists in the cloud.
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

# 4. Create the Chat Engine
query_engine = index.as_query_engine()

# 5. Run a Test Query
# Change this question to something specific about the PDF you uploaded!
question = "What is the warehouse square feet for each phase?"

print(f"Asking: {question}")
print("Thinking...")

response = query_engine.query(question)
print("------------------------------------------------")
print(f"Answer: {response}")
print("------------------------------------------------")
