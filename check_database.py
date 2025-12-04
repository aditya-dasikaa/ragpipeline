import chromadb

# Connect to your database
client = chromadb.PersistentClient(path="db/chroma_db")

# Get the collection
try:
    collection = client.get_collection(name="default")
    
    # Get all documents
    results = collection.get()
    
    print(f"Total documents in database: {collection.count()}")
    print("\n" + "="*80)
    
    # Show first 5 documents
    if results['documents']:
        for i, doc in enumerate(results['documents'][:5]):
            print(f"\nDocument {i+1}:")
            print(f"ID: {results['ids'][i]}")
            print(f"Metadata: {results['metadatas'][i]}")
            print(f"Content preview: {doc[:200]}...")
            print("-"*80)
    else:
        print("No documents found in the database!")
        
except Exception as e:
    print(f"Error: {e}")