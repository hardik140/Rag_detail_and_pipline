"""
Example demonstrating Hybrid Search (Vector + BM25) with automatic fallback.

This example shows how the RAG pipeline automatically falls back to BM25 keyword search
when vector search fails or returns insufficient results.
"""

from pathlib import Path
from rag_pipeline import RAGPipeline
from config import load_config
from loguru import logger


def example_hybrid_search_basic():
    """Demonstrate basic hybrid search with automatic fallback."""
    print("\n" + "="*70)
    print("EXAMPLE 1: Hybrid Search with Automatic BM25 Fallback")
    print("="*70)
    
    # Initialize pipeline (hybrid search is enabled in config.yaml)
    pipeline = RAGPipeline()
    
    # Create sample documents
    sample_dir = Path("./data/documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = sample_dir / "ai_concepts.txt"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""
# Artificial Intelligence Concepts

## Neural Networks
Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) organized in layers. Deep neural networks have multiple hidden layers and are the foundation of deep learning.

## Natural Language Processing
Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret, and manipulate human language. NLP draws from many disciplines including computer science and computational linguistics.

## Computer Vision
Computer vision is a field of AI that trains computers to interpret and understand the visual world. Using digital images from cameras and videos and deep learning models, machines can accurately identify and classify objects.

## Reinforcement Learning
Reinforcement learning is a machine learning training method based on rewarding desired behaviors and punishing undesired ones. An agent learns to achieve a goal in an uncertain, potentially complex environment through trial and error.

## Transfer Learning
Transfer learning is a machine learning method where a model developed for one task is reused as the starting point for a model on a second task. It's popular in deep learning given the vast compute and time resources required to develop neural network models.
            """)
        print(f"✓ Created sample document: {sample_file}")
    
    # Ingest documents
    print("\n📚 Ingesting documents...")
    result = pipeline.ingest_documents(str(sample_dir))
    print(f"✓ Status: {result['status']}")
    print(f"✓ Documents processed: {result['documents_processed']}")
    
    # Show stats
    stats = pipeline.get_stats()
    print(f"\n📊 Pipeline Configuration:")
    print(f"  - Hybrid Search Enabled: {stats['config']['hybrid_search_enabled']}")
    if 'hybrid_search' in stats:
        print(f"  - BM25 Indexed: {stats['hybrid_search']['bm25']['is_indexed']}")
        print(f"  - BM25 Documents: {stats['hybrid_search']['bm25']['document_count']}")
    
    # Test 1: Normal query (vector search should work)
    print("\n" + "-"*70)
    print("TEST 1: Normal Query (Vector Search)")
    print("-"*70)
    question = "What is neural network?"
    print(f"\n❓ Question: {question}")
    response = pipeline.query(question, return_sources=True)
    print(f"💡 Answer: {response['answer']}")
    if response.get('sources'):
        print(f"📄 Sources: {len(response['sources'])} documents")
        for i, source in enumerate(response['sources'][:2], 1):
            search_type = source['metadata'].get('search_type', 'vector')
            print(f"   {i}. Search type: {search_type}")
    
    # Test 2: Query with rare/specific keywords (may trigger BM25 fallback)
    print("\n" + "-"*70)
    print("TEST 2: Keyword-Heavy Query (May Trigger BM25 Fallback)")
    print("-"*70)
    question = "What specific method involves rewarding desired behaviors?"
    print(f"\n❓ Question: {question}")
    response = pipeline.query(question, return_sources=True)
    print(f"💡 Answer: {response['answer']}")
    if response.get('sources'):
        print(f"📄 Sources: {len(response['sources'])} documents")
        for i, source in enumerate(response['sources'][:2], 1):
            search_type = source['metadata'].get('search_type', 'vector')
            bm25_score = source['metadata'].get('bm25_score', 'N/A')
            print(f"   {i}. Search type: {search_type}", end="")
            if search_type == 'bm25':
                print(f" (BM25 score: {bm25_score:.4f})")
            else:
                print()


def example_direct_bm25_search():
    """Demonstrate direct BM25 keyword search."""
    print("\n" + "="*70)
    print("EXAMPLE 2: Direct BM25 Keyword Search")
    print("="*70)
    
    pipeline = RAGPipeline()
    
    # Make sure we have indexed data
    if not pipeline.hybrid_search or not pipeline.hybrid_search.bm25_search.is_indexed:
        print("\n⚠️  No indexed data. Please run example 1 first to ingest documents.")
        return
    
    # Perform direct BM25 search
    query = "deep learning neural network layers"
    print(f"\n🔍 BM25 Keyword Search for: '{query}'")
    
    results = pipeline.hybrid_search.bm25_search.search(query, k=3)
    
    print(f"\n✓ Found {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        score = doc.metadata.get('bm25_score', 0)
        print(f"\n--- Document {i} (BM25 Score: {score:.4f}) ---")
        print(f"Content: {doc.page_content[:200]}...")


def example_fusion_search():
    """Demonstrate fusion search combining vector and BM25."""
    print("\n" + "="*70)
    print("EXAMPLE 3: Fusion Search (Vector + BM25 Combined)")
    print("="*70)
    
    pipeline = RAGPipeline()
    
    if not pipeline.hybrid_search:
        print("\n⚠️  Hybrid search not enabled in configuration.")
        return
    
    if not pipeline.hybrid_search.bm25_search.is_indexed:
        print("\n⚠️  No indexed data. Please run example 1 first.")
        return
    
    # Perform fusion search
    query = "computer vision image recognition"
    print(f"\n🔍 Fusion Search (combining vector + BM25) for: '{query}'")
    
    results = pipeline.hybrid_search.search(query, k=3, search_type="fusion")
    
    print(f"\n✓ Found {len(results)} documents:")
    for i, doc in enumerate(results, 1):
        fusion_score = doc.metadata.get('fusion_score', 0)
        sources = doc.metadata.get('search_sources', [])
        print(f"\n--- Document {i} (Fusion Score: {fusion_score:.4f}) ---")
        print(f"Sources: {', '.join(sources)}")
        print(f"Content: {doc.page_content[:150]}...")


def example_compare_search_methods():
    """Compare different search methods side by side."""
    print("\n" + "="*70)
    print("EXAMPLE 4: Comparing Search Methods")
    print("="*70)
    
    pipeline = RAGPipeline()
    
    if not pipeline.hybrid_search or not pipeline.hybrid_search.bm25_search.is_indexed:
        print("\n⚠️  Please run example 1 first to ingest documents.")
        return
    
    query = "machine learning reused model task"
    print(f"\n🔍 Query: '{query}'")
    
    # Vector search
    print("\n📊 1. Vector Search Results:")
    vector_results = pipeline.hybrid_search.search(query, k=2, search_type="vector")
    for i, doc in enumerate(vector_results, 1):
        print(f"   {i}. {doc.page_content[:100]}...")
    
    # BM25 search
    print("\n📊 2. BM25 Keyword Search Results:")
    bm25_results = pipeline.hybrid_search.search(query, k=2, search_type="bm25")
    for i, doc in enumerate(bm25_results, 1):
        score = doc.metadata.get('bm25_score', 0)
        print(f"   {i}. (Score: {score:.4f}) {doc.page_content[:100]}...")
    
    # Fusion search
    print("\n📊 3. Fusion Search Results:")
    fusion_results = pipeline.hybrid_search.search(query, k=2, search_type="fusion")
    for i, doc in enumerate(fusion_results, 1):
        score = doc.metadata.get('fusion_score', 0)
        sources = doc.metadata.get('search_sources', [])
        print(f"   {i}. (Score: {score:.4f}, Sources: {sources}) {doc.page_content[:100]}...")


def example_fallback_scenario():
    """Demonstrate the automatic fallback scenario."""
    print("\n" + "="*70)
    print("EXAMPLE 5: Demonstrating Automatic Fallback")
    print("="*70)
    
    print("\nThe pipeline automatically falls back to BM25 when:")
    print("  1. Vector search returns no results")
    print("  2. Vector search returns fewer than min_vector_results")
    print("  3. Vector search encounters an error")
    
    pipeline = RAGPipeline()
    
    if not pipeline.hybrid_search:
        print("\n⚠️  Hybrid search not enabled.")
        return
    
    # Show configuration
    if pipeline.config.hybrid_search:
        print(f"\n⚙️  Current Configuration:")
        print(f"  - Fallback Enabled: {pipeline.config.hybrid_search.enable_fallback}")
        print(f"  - Min Vector Results: {pipeline.config.hybrid_search.min_vector_results}")
        print(f"  - Fusion Enabled: {pipeline.config.hybrid_search.enable_fusion}")
    
    # Example query that might trigger fallback
    query = "specific exact terminology phrases keywords"
    print(f"\n🔍 Testing with query: '{query}'")
    print("\nThis query uses specific keywords that may not have strong semantic")
    print("embeddings, potentially triggering the BM25 fallback mechanism.")
    
    results = pipeline.hybrid_search.search(query, k=3, search_type="auto")
    
    if results:
        print(f"\n✓ Retrieved {len(results)} documents")
        for i, doc in enumerate(results, 1):
            search_type = doc.metadata.get('search_type', 'unknown')
            print(f"   {i}. Search method used: {search_type}")
    else:
        print("\n⚠️  No results found")


def main():
    """Run all hybrid search examples."""
    try:
        print("\n" + "="*70)
        print("🔍 HYBRID SEARCH (Vector + BM25) EXAMPLES")
        print("="*70)
        print("\nThese examples demonstrate the automatic fallback from vector search")
        print("to BM25 keyword search when vector search fails or returns few results.")
        
        # Run examples
        example_hybrid_search_basic()
        example_direct_bm25_search()
        example_fusion_search()
        example_compare_search_methods()
        example_fallback_scenario()
        
        print("\n" + "="*70)
        print("✅ All hybrid search examples completed!")
        print("="*70)
        
        print("\n💡 Key Takeaways:")
        print("  1. Hybrid search combines vector (semantic) and BM25 (keyword) search")
        print("  2. BM25 automatically activates as fallback when vector search fails")
        print("  3. Fusion mode combines both methods for comprehensive results")
        print("  4. This makes the system more robust and handles edge cases better")
        print()
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Set up your .env file with API keys")
        print("2. Installed all requirements: pip install -r requirements.txt")
        print("3. Enabled hybrid_search in config.yaml")


if __name__ == "__main__":
    main()
