import os
import chromadb
from chromadb.errors import NotFoundError
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from dotenv import load_dotenv
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any
import shutil
from datetime import datetime

load_dotenv()

# Minimal Document shim (compatible with functions that only need .page_content and .metadata)
@dataclass
class Document:
    page_content: str
    metadata: Dict[str, Any]

# Updated OpenAI Embeddings Wrapper for openai>=1.0.0
class OpenAIEmbeddingsWrapper:
    def __init__(self, model="text-embedding-3-small", api_key=None):
        from openai import OpenAI
        self.model = model
        # Initialize OpenAI client with API key from parameter or environment
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def embed_documents(self, texts):
        """Embed a list of texts"""
        if not texts:
            return []
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text):
        """Embed a single query text"""
        response = self.client.embeddings.create(
            input=[text],
            model=self.model
        )
        return response.data[0].embedding


def load_documents(docs_path="docs"):
    """Load all text files from the docs directory (robust encoding fallbacks)."""
    print(f"Loading documents from {docs_path}...")
    
    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"The directory {docs_path} does not exist. Please create it and add your company files.")
    
    p = Path(docs_path)
    files = sorted(p.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"No .txt files found in {docs_path}. Please add your company documents.")
    
    documents = []
    for f in files:
        text = None
        last_exc = None
        # try encodings in order
        for enc, errs in [("utf-8", "strict"), ("utf-8", "replace"), ("latin-1", "strict")]:
            try:
                with open(f, "r", encoding=enc, errors=errs) as fh:
                    text = fh.read()
                used = f"{enc} (errors={errs})"
                break
            except Exception as e:
                last_exc = e
        if text is None:
            print(f"Error loading file {f}: {last_exc}")
            continue
        # create a LangChain Document
        documents.append(Document(page_content=text, metadata={"source": str(f)}))
        print(f"Loaded {f.name} with encoding {used}, {len(text)} characters")
    
    # show previews
    for i, doc in enumerate(documents[:2]):  # Show first 2 documents
        print(f"\nDocument {i+1}:")
        print(f"  Source: {doc.metadata['source']}")
        print(f"  Content length: {len(doc.page_content)} characters")
        print(f"  Content preview: {doc.page_content[:100]}...")
        print(f"  metadata: {doc.metadata}")
    
    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=0):
    """Split documents into smaller chunks with overlap"""
    print("Splitting documents into chunks...")
    
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    chunks = text_splitter.split_documents(documents)
    
    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print(f"Content:")
            print(chunk.page_content)
            print("-" * 50)
        
        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")
    
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist a chromadb collection using the OpenAIEmbeddingsWrapper"""
    print("Creating embeddings and storing in ChromaDB...")

    embedding_model = OpenAIEmbeddingsWrapper(model="text-embedding-3-small")

    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Create chromadb client with modern API
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Delete existing collection if it exists (for fresh start)
    try:
        client.delete_collection(name="default")
        print("Deleted existing collection")
    except:
        pass
    
    # Create new collection with modern API
    collection = client.create_collection(
        name="default",
        metadata={"hnsw:space": "cosine"}
    )

    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    ids = [f"doc_{i}" for i in range(len(chunks))]

    print("--- Computing embeddings (this may take a moment) ---")
    try:
        embeddings = embedding_model.embed_documents(texts)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to generate embeddings")
        print(f"Details: {e}")
        print("\nPossible reasons:")
        print("1. No OpenAI API key found in .env file")
        print("2. Invalid API key")
        print("3. No payment method added to OpenAI account")
        print("4. Insufficient credits in OpenAI account")
        print("\nPlease check your OpenAI API setup and try again.")
        raise

    print(f"--- Adding {len(texts)} documents to chromadb collection ---")
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    print(f"✅ Vector store created and saved to {persist_directory}")
    return collection


def _get_chroma_client(persist_directory):
    """Get ChromaDB client with modern configuration"""
    # Ensure directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Use modern PersistentClient API
    client = chromadb.PersistentClient(path=persist_directory)
    return client


def main():
    """Main ingestion pipeline"""
    print("=== RAG Document Ingestion Pipeline ===\n")
    
    # Define paths
    docs_path = "docs"
    persistent_directory = "db/chroma_db"
    
    # Check if vector store already exists
    if os.path.exists(persistent_directory):
        print("⚠️  Vector store directory already exists.")
        user_input = input("Do you want to rebuild it? (yes/no): ").strip().lower()
        if user_input not in ['yes', 'y']:
            print("Loading existing vector store...")
            client = _get_chroma_client(persistent_directory)
            try:
                collection = client.get_collection(name="default")
                print(f"✅ Loaded existing vector store with {collection.count()} documents")
                return collection
            except Exception as e:
                print(f"⚠️  Could not load existing collection: {e}")
                print("Proceeding with fresh ingestion...")
    
    print("Starting fresh ingestion...\n")
    
    # Step 1: Load documents
    try:
        documents = load_documents(docs_path)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return None

    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    if not chunks:
        print("\n❌ ERROR: No chunks created. Please check your documents.")
        return None
    
    # Step 3: Create vector store
    try:
        vectorstore = create_vector_store(chunks, persistent_directory)
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        return None
    
    print("\n✅ Ingestion complete! Your documents are now ready for RAG queries.")
    return vectorstore


if __name__ == "__main__":
    main()


## Key Updates in This Version:


