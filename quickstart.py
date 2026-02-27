"""
Quick Start Example for RAG Pipeline

This script demonstrates basic usage of the modularized RAG pipeline.
"""

from src.pipeline import RAGPipeline
from pathlib import Path


def main():
    """Main example function."""
    
    print("=" * 60)
    print("RAG Pipeline - Quick Start Example")
    print("=" * 60)
    
    # Initialize the pipeline
    print("\n1. Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    print("✓ Pipeline initialized successfully")
    
    # Check if documents directory exists
    docs_path = "./data/documents"
    if not Path(docs_path).exists():
        print(f"\n⚠ Documents directory not found: {docs_path}")
        print("Please add documents to ./data/documents/ before running")
        return
    
    # Check if directory has files
    doc_files = list(Path(docs_path).glob("*"))
    if not doc_files:
        print(f"\n⚠ No documents found in: {docs_path}")
        print("Please add PDF, DOCX, TXT, MD, or HTML files to the documents directory")
        return
    
    # Ingest documents
    print(f"\n2. Ingesting documents from: {docs_path}")
    result = pipeline.ingest_documents(docs_path)
    
    if result["status"] == "success":
        print(f"✓ Successfully processed {result['documents_processed']} document chunks")
    else:
        print(f"✗ Error during ingestion: {result.get('message', 'Unknown error')}")
        return
    
    # Query the system
    print("\n3. Querying the system...")
    question = "What are the main topics covered in the documents?"
    print(f"Question: {question}")
    
    response = pipeline.query(question, return_sources=True)
    
    if response["status"] == "success":
        print(f"\n✓ Answer:\n{response['answer']}")
        
        if response.get("sources"):
            print(f"\n📚 Based on {len(response['sources'])} source documents")
    else:
        print(f"\n✗ Error during query: {response.get('message', 'Unknown error')}")
    
    # Try conversation mode
    print("\n" + "=" * 60)
    print("Testing Conversation Mode")
    print("=" * 60)
    
    session_id = "example_session"
    
    # First question
    question1 = "Tell me about the key concepts"
    print(f"\nQuestion 1: {question1}")
    response1 = pipeline.conversation_query(question1, session_id=session_id)
    
    if response1["status"] == "success":
        print(f"\n✓ Answer:\n{response1['answer']}")
    
    # Follow-up question
    question2 = "Can you give me more details?"
    print(f"\nQuestion 2 (follow-up): {question2}")
    response2 = pipeline.conversation_query(question2, session_id=session_id)
    
    if response2["status"] == "success":
        print(f"\n✓ Answer:\n{response2['answer']}")
        print(f"\n📊 Conversation has {response2.get('message_count', 0)} messages")
    
    print("\n" + "=" * 60)
    print("✓ Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPlease ensure:")
        print("1. You have set up your API keys in .env file")
        print("2. You have documents in ./data/documents/")
        print("3. All dependencies are installed: pip install -r requirements.txt")
