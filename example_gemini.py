"""
Example demonstrating Gemini LLM with all-MiniLM-L6-v2 embeddings.

This example shows how to use Google's Gemini model for generation
with local HuggingFace embeddings (all-MiniLM-L6-v2) for efficient,
cost-effective semantic search.
"""

from pathlib import Path
from rag_pipeline import RAGPipeline
from config import config
from loguru import logger


def example_gemini_with_local_embeddings():
    """Demonstrate Gemini LLM with local all-MiniLM-L6-v2 embeddings."""
    print("\n" + "="*70)
    print("EXAMPLE: Gemini with all-MiniLM-L6-v2 Local Embeddings")
    print("="*70)
    
    # Show configuration
    print(f"\n⚙️  Configuration:")
    print(f"  - LLM Provider: {config.llm.provider}")
    print(f"  - LLM Model: {config.llm.model}")
    print(f"  - Embeddings Provider: {config.embeddings.provider}")
    print(f"  - Embeddings Model: {config.embeddings.model}")
    print(f"  - Embeddings Dimension: {config.embeddings.dimension}")
    
    if config.embeddings.enable_fallback:
        print(f"  - Fallback Enabled: Yes")
        print(f"  - Fallback Model: {config.embeddings.fallback_model}")
    
    # Initialize pipeline
    print("\n🚀 Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    
    # Create sample documents
    sample_dir = Path("./data/documents")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    sample_file = sample_dir / "gemini_test.txt"
    if not sample_file.exists():
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write("""
# Google Gemini AI Model

Google Gemini is a family of multimodal large language models developed by Google DeepMind. Gemini models are designed to understand and generate text, code, images, audio, and video.

## Key Features

1. **Multimodal Capabilities**: Gemini can process and understand multiple types of input including text, images, code, audio, and video.

2. **Advanced Reasoning**: The model demonstrates strong reasoning abilities across various tasks including mathematics, coding, and complex problem-solving.

3. **Long Context Window**: Gemini Pro supports up to 1 million tokens in context, allowing it to process very long documents.

4. **Code Generation**: Excellent at generating, debugging, and explaining code across multiple programming languages.

5. **Multilingual Support**: Supports over 100 languages for text understanding and generation.

## Model Variants

- **Gemini Ultra**: Most capable model for highly complex tasks
- **Gemini Pro**: Best balance of capability and efficiency  
- **Gemini Nano**: Optimized for on-device tasks

## Use Cases

- Natural language understanding and generation
- Code development and debugging
- Document analysis and summarization
- Multimodal content creation
- Research assistance
- Educational applications

## Integration

Gemini is available through Google AI Studio and Vertex AI, with APIs for various programming languages including Python, JavaScript, and more.
            """)
        print(f"✓ Created sample document: {sample_file}")
    
    # Ingest documents
    print("\n📚 Ingesting documents...")
    print("  - Using all-MiniLM-L6-v2 for creating embeddings")
    print("  - This runs locally (no API calls for embeddings)")
    
    result = pipeline.ingest_documents(str(sample_dir))
    print(f"\n✓ Status: {result['status']}")
    print(f"✓ Documents processed: {result['documents_processed']}")
    
    # Test embeddings
    print("\n🔍 Testing Embeddings:")
    embeddings_manager = pipeline.embeddings_manager
    test_embedding = embeddings_manager.embed_query("test query")
    print(f"  - Embedding dimension: {len(test_embedding)}")
    print(f"  - Model used: {embeddings_manager.model}")
    
    # Show stats
    stats = pipeline.get_stats()
    print(f"\n📊 Pipeline Statistics:")
    print(f"  - LLM Model: {stats['config']['llm_model']}")
    print(f"  - Embeddings Model: {stats['config']['embeddings_model']}")
    print(f"  - Vector Store: {stats['config']['vector_store_type']}")
    print(f"  - Hybrid Search: {'Enabled' if stats['config']['hybrid_search_enabled'] else 'Disabled'}")
    
    # Query using Gemini
    print("\n" + "-"*70)
    print("Testing Gemini LLM with Local Embeddings")
    print("-"*70)
    
    questions = [
        "What is Google Gemini?",
        "What are the key features of Gemini?",
        "Name the different Gemini model variants"
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n❓ Question {i}: {question}")
        
        try:
            response = pipeline.query(question, return_sources=True)
            print(f"💡 Answer: {response['answer']}")
            
            if response.get('sources'):
                print(f"📄 Sources: {len(response['sources'])} documents")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            print("Note: Make sure GOOGLE_API_KEY is set in your .env file")


def example_embeddings_fallback():
    """Demonstrate embeddings fallback behavior."""
    print("\n" + "="*70)
    print("EXAMPLE: Embeddings Fallback Demonstration")
    print("="*70)
    
    print("\nThe system is configured with:")
    print(f"  - Primary: {config.embeddings.provider}/{config.embeddings.model}")
    
    if config.embeddings.enable_fallback:
        print(f"  - Fallback: {config.embeddings.fallback_provider}/{config.embeddings.fallback_model}")
        print("\n✓ If the primary embeddings provider fails (e.g., API error),")
        print("  the system will automatically fall back to the local model.")
    else:
        print("  - Fallback: Not enabled")
    
    print("\n💡 Benefits of Local Embeddings (all-MiniLM-L6-v2):")
    print("  ✓ No API costs for embeddings")
    print("  ✓ Fast processing (runs locally)")
    print("  ✓ Privacy (data doesn't leave your machine)")
    print("  ✓ No rate limits")
    print("  ✓ Works offline")
    print("  ✓ Good quality for general-purpose semantic search")


def example_hybrid_gemini():
    """Demonstrate hybrid search with Gemini."""
    print("\n" + "="*70)
    print("EXAMPLE: Gemini + Hybrid Search (Vector + BM25)")
    print("="*70)
    
    pipeline = RAGPipeline()
    
    if not pipeline.hybrid_search:
        print("\n⚠️  Hybrid search not enabled. Enable it in config.yaml")
        return
    
    print("\n🔍 Testing Hybrid Search with Gemini:")
    print("  - Vector search uses all-MiniLM-L6-v2 embeddings")
    print("  - BM25 provides keyword fallback")
    print("  - Gemini generates the final answer")
    
    # Test query
    query = "What programming languages does Gemini support?"
    print(f"\n❓ Query: {query}")
    
    try:
        response = pipeline.query(query, return_sources=True)
        print(f"\n💡 Answer from Gemini:")
        print(f"{response['answer']}")
        
        if response.get('sources'):
            print(f"\n📄 Retrieved {len(response['sources'])} source documents")
            for i, source in enumerate(response['sources'][:2], 1):
                search_type = source['metadata'].get('search_type', 'vector')
                print(f"  {i}. Search method: {search_type}")
                
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def example_cost_comparison():
    """Show cost comparison between different setups."""
    print("\n" + "="*70)
    print("EXAMPLE: Cost Comparison")
    print("="*70)
    
    print("\n💰 Cost Analysis:")
    print("\n1. OpenAI GPT-4 + OpenAI Embeddings (text-embedding-3-small)")
    print("   - LLM: ~$0.01 per 1K tokens (input) + $0.03 per 1K tokens (output)")
    print("   - Embeddings: ~$0.00002 per 1K tokens")
    print("   - Total: Higher cost, both services need API calls")
    
    print("\n2. Gemini Pro + all-MiniLM-L6-v2 (Current Setup)")
    print("   - LLM: Free tier available, then ~$0.00025 per 1K chars")
    print("   - Embeddings: FREE (runs locally)")
    print("   - Total: Lower cost, only LLM needs API calls")
    
    print("\n✓ Key Advantages of Current Setup:")
    print("  - 100% free embeddings (no API costs)")
    print("  - Gemini offers competitive pricing")
    print("  - Local embeddings are faster (no network latency)")
    print("  - Better privacy (embeddings data stays local)")
    print("  - No rate limits on embeddings")


def main():
    """Run all Gemini + all-MiniLM-L6-v2 examples."""
    try:
        print("\n" + "="*70)
        print("🤖 GEMINI + ALL-MINILM-L6-V2 EXAMPLES")
        print("="*70)
        print("\nThis demonstrates using Google Gemini for generation")
        print("with local all-MiniLM-L6-v2 embeddings for efficient search.")
        
        # Check if API key is available
        from config import get_api_key
        api_key = get_api_key("google")
        
        if not api_key or api_key == "your_google_gemini_api_key_here":
            print("\n⚠️  WARNING:")
            print("GOOGLE_API_KEY not found or not set in .env file")
            print("\nTo use Gemini:")
            print("1. Get API key from: https://makersuite.google.com/app/apikey")
            print("2. Add to .env file: GOOGLE_API_KEY=your_actual_key")
            print("\nRunning embedding examples only...")
            example_embeddings_fallback()
            example_cost_comparison()
        else:
            # Run all examples
            example_gemini_with_local_embeddings()
            example_hybrid_gemini()
            example_embeddings_fallback()
            example_cost_comparison()
        
        print("\n" + "="*70)
        print("✅ Examples completed!")
        print("="*70)
        
        print("\n💡 Key Takeaways:")
        print("  1. Gemini provides powerful LLM capabilities at competitive pricing")
        print("  2. all-MiniLM-L6-v2 offers free, fast, local embeddings")
        print("  3. Hybrid search (vector + BM25) ensures robust retrieval")
        print("  4. Fallback embeddings provide reliability")
        print("  5. This setup balances cost, performance, and privacy")
        print()
        
    except Exception as e:
        logger.error(f"Error running examples: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed all requirements: pip install -r requirements.txt")
        print("2. Set GOOGLE_API_KEY in your .env file")
        print("3. Configured config.yaml for Gemini and all-MiniLM-L6-v2")


if __name__ == "__main__":
    main()
