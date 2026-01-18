import os
import glob
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext
from llama_index.vector_stores.pinecone import PineconeVectorStore

# 1. Load API keys
load_dotenv()

# 2. Setup Pinecone
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
pinecone_index = pc.Index("city-data")
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def main():
    print("Scanning data folder for PDFs...")
    # Get all PDF files individually
    pdf_files = glob.glob("./data/*.pdf")
    total_files = len(pdf_files)
    print(f"Found {total_files} PDF files.")

    documents_buffer = []
    BATCH_SIZE = 10  # Smaller batch size for safety

    for i, file_path in enumerate(pdf_files):
        try:
            print(f"Reading file {i+1}/{total_files}: {os.path.basename(file_path)}")
            
            # Load ONLY this single file
            # We turn off parallel processing by loading 1 file at a time
            single_doc = SimpleDirectoryReader(input_files=[file_path]).load_data()
            
            # Add to our temporary list
            documents_buffer.extend(single_doc)

            # If we have enough pages, upload them
            if len(documents_buffer) >= BATCH_SIZE:
                print(f"  >> Uploading buffer of {len(documents_buffer)} pages to Pinecone...")
                VectorStoreIndex.from_documents(
                    documents_buffer,
                    storage_context=storage_context
                )
                print("  >> Upload Success!")
                documents_buffer = [] # Clear buffer
                time.sleep(0.5) # Be nice to the API

        except Exception as e:
            print(f"!!!! ERROR reading {os.path.basename(file_path)}: {e}")
            print("!!!! SKIPPING THIS FILE and continuing...")
            continue

    # Process any remaining files in the buffer
    if len(documents_buffer) > 0:
        print(f"  >> Uploading final {len(documents_buffer)} pages...")
        VectorStoreIndex.from_documents(
            documents_buffer,
            storage_context=storage_context
        )
        print("  >> Final Upload Success!")

    print("------------------------------------------------")
    print("ALL FILES PROCESSED")
    print("------------------------------------------------")

if __name__ == "__main__":
    main()
