import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

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


def retrieve_documents(query, persist_directory="db/chroma_db", k=5, score_threshold=None):
    """
    Retrieve relevant documents from ChromaDB
    
    Args:
        query: The search query
        persist_directory: Path to the ChromaDB storage
        k: Number of results to return
        score_threshold: Optional minimum similarity score (0-1, where 1 is perfect match)
    
    Returns:
        List of relevant documents with metadata and scores
    """
    # Initialize embedding model
    embedding_model = OpenAIEmbeddingsWrapper(model="text-embedding-3-small")
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path=persist_directory)
    
    # Get the collection
    try:
        collection = client.get_collection(name="default")
    except Exception as e:
        print(f"Error: Could not load collection. Make sure you've run the ingestion pipeline first.")
        print(f"Details: {e}")
        return []
    
    # Generate query embedding
    print(f"Generating embedding for query: '{query}'")
    query_embedding = embedding_model.embed_query(query)
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    
    # Process results
    documents = []
    if results and results['documents'] and results['documents'][0]:
        for i in range(len(results['documents'][0])):
            # ChromaDB returns distance (lower is better), convert to similarity score
            # For cosine distance: similarity = 1 - distance
            distance = results['distances'][0][i]
            similarity_score = 1 - distance
            
            # Apply score threshold if specified
            if score_threshold is not None and similarity_score < score_threshold:
                continue
            
            doc = {
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'similarity_score': similarity_score,
                'distance': distance
            }
            documents.append(doc)
    
    return documents


def main():
    """Main retrieval function with example queries"""
    
    persistent_directory = "db/chroma_db"
    
    # Check if database exists
    if not os.path.exists(persistent_directory):
        print(f"Error: Vector store not found at {persistent_directory}")
        print("Please run the ingestion pipeline first (1_ingestion_pipeline.py)")
        return
    
    # Example query
    query = "In which year did Tesla begin production of Roadster?"
    
    print("="*80)
    print(f"User Query: {query}")
    print("="*80)
    
    # Retrieve documents (basic similarity search)
    relevant_docs = retrieve_documents(
        query=query,
        persist_directory=persistent_directory,
        k=5
    )
    
    # Display results
    if not relevant_docs:
        print("No relevant documents found.")
        return
    
    print(f"\n--- Found {len(relevant_docs)} Relevant Documents ---\n")
    
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:")
        print(f"  Similarity Score: {doc['similarity_score']:.4f}")
        print(f"  Source: {doc['metadata'].get('source', 'Unknown')}")
        print(f"  Content:\n{doc['content']}\n")
        print("-" * 80)
    
    # Example with score threshold
    print("\n" + "="*80)
    print("Example with Score Threshold (≥ 0.3)")
    print("="*80)
    
    filtered_docs = retrieve_documents(
        query=query,
        persist_directory=persistent_directory,
        k=5,
        score_threshold=0.3
    )
    
    print(f"\n--- Found {len(filtered_docs)} Documents with score ≥ 0.3 ---\n")
    
    for i, doc in enumerate(filtered_docs, 1):
        print(f"Document {i}:")
        print(f"  Similarity Score: {doc['similarity_score']:.4f}")
        print(f"  Content Preview: {doc['content'][:200]}...\n")


def test_synthetic_queries():
    """Test with multiple synthetic questions"""
    
    synthetic_questions = [
        "What was NVIDIA's first graphics accelerator called?",
        "Which company did NVIDIA acquire to enter the mobile processor market?",
        "What was Microsoft's first hardware product release?",
        "How much did Microsoft pay to acquire GitHub?",
        "In what year did Tesla begin production of the Roadster?",
        "Who succeeded Ze'ev Drori as CEO in October 2008?",
        "What was the name of the autonomous spaceport drone ship that achieved the first successful sea landing?",
        "What was the original name of Microsoft before it became Microsoft?"
    ]
    
    persistent_directory = "db/chroma_db"
    
    print("\n" + "="*80)
    print("TESTING SYNTHETIC QUERIES")
    print("="*80)
    
    for idx, question in enumerate(synthetic_questions, 1):
        print(f"\n{'='*80}")
        print(f"Question {idx}: {question}")
        print(f"{'='*80}")
        
        docs = retrieve_documents(
            query=question,
            persist_directory=persistent_directory,
            k=3,
            score_threshold=0.2
        )
        
        if docs:
            print(f"\nTop Result (Score: {docs[0]['similarity_score']:.4f}):")
            print(f"{docs[0]['content'][:300]}...\n")
        else:
            print("No relevant documents found.\n")


if __name__ == "__main__":
    # Run main retrieval example
    main()
    
    # Uncomment to test all synthetic queries
    # test_synthetic_queries()