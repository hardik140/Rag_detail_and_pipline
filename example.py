"""
Example usage of RAG Pipeline

This script demonstrates the basic usage of the RAG pipeline.
"""

from pathlib import Path
from rag_pipeline import RAGPipeline
from loguru import logger


def example_basic_usage():
    """Basic usage example."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Create sample document
    sample_doc_dir = Path("./data/documents")
    sample_doc_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = sample_doc_dir / "sample.txt"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""
# Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

## Types of Machine Learning

1. Supervised Learning: Learning from labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through interaction with environment

## Common Algorithms

- Linear Regression
- Decision Trees
- Neural Networks
- Support Vector Machines

Machine learning is widely used in image recognition, natural language processing, and recommendation systems.
            """)
        print(f"✓ Created sample document: {sample_file}")
    
    # Ingest documents
    print("\n📚 Ingesting documents...")
    result = pipeline.ingest_documents(str(sample_doc_dir))
    print(f"✓ Status: {result['status']}")
    print(f"✓ Documents processed: {result['documents_processed']}")
    
    # Query the system
    print("\n🔍 Querying the system...")
    questions = [
        "What is machine learning?",
        "What are the types of machine learning?",
        "Name some common algorithms"
    ]
    
    for question in questions:
        print(f"\n❓ Question: {question}")
        response = pipeline.query(question, return_sources=True)
        print(f"💡 Answer: {response['answer']}")
        if response.get('sources'):
            print(f"📄 Number of sources: {len(response['sources'])}")


def example_similarity_search():
    """Similarity search example."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Similarity Search")
    print("="*60)
    
    pipeline = RAGPipeline()
    
    # Setup query engine (load existing data)
    pipeline.setup_query_engine()
    
    # Search for similar documents
    query = "neural networks"
    print(f"\n🔍 Searching for documents similar to: '{query}'")
    
    docs = pipeline.get_similar_documents(query, k=3)
    
    print(f"\n✓ Found {len(docs)} similar documents:")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Document {i} ---")
        print(f"Content: {doc['content'][:150]}...")
        print(f"Metadata: {doc['metadata']}")


def example_custom_query():
    """Custom query with specific parameters example."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Custom Query")
    print("="*60)
    
    pipeline = RAGPipeline()
    pipeline.setup_query_engine()
    
    # Custom prompt template
    custom_prompt = """Based on the context below, provide a brief answer.

Context: {context}

Question: {question}

Brief Answer:"""
    
    question = "Explain supervised learning in one sentence"
    print(f"\n❓ Question: {question}")
    
    response = pipeline.query_engine.query(
        question,
        return_sources=False,
        custom_prompt=custom_prompt
    )
    
    print(f"💡 Answer: {response['answer']}")


def example_batch_queries():
    """Batch query example."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Batch Queries")
    print("="*60)
    
    pipeline = RAGPipeline()
    pipeline.setup_query_engine()
    
    questions = [
        "What is machine learning?",
        "What is supervised learning?",
        "What is unsupervised learning?"
    ]
    
    print("\n🔍 Processing batch queries...")
    results = pipeline.query_engine.batch_query(questions, return_sources=False)
    
    for i, result in enumerate(results, 1):
        print(f"\n--- Query {i} ---")
        print(f"Q: {result['question']}")
        print(f"A: {result['answer']}")


def example_stats():
    """Get pipeline statistics example."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Pipeline Statistics")
    print("="*60)
    
    pipeline = RAGPipeline()
    
    stats = pipeline.get_stats()
    
    print("\n📊 Pipeline Statistics:")
    print(f"\nConfiguration:")
    print(f"  - LLM Model: {stats['config']['llm_model']}")
    print(f"  - Embeddings Model: {stats['config']['embeddings_model']}")
    print(f"  - Vector Store Type: {stats['config']['vector_store_type']}")
    print(f"  - Chunk Size: {stats['config']['chunk_size']}")
    
    print(f"\nVector Store:")
    for key, value in stats['vector_store'].items():
        print(f"  - {key}: {value}")


def main():
    """Run all examples."""
    try:
        # Run examples
        example_basic_usage()
        example_similarity_search()
        example_custom_query()
        example_batch_queries()
        example_stats()
        
        print("\n" + "="*60)
        print("✅ All examples completed successfully!")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with API keys")
        print("2. Installed all requirements: pip install -r requirements.txt")
        print("3. Run the basic usage example first to ingest documents")


if __name__ == "__main__":
    main()
