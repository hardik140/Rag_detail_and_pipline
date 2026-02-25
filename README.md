# RAG Pipeline

A comprehensive Retrieval-Augmented Generation (RAG) pipeline built with LangChain, supporting multiple LLM providers, vector stores, and embedding models.

> **Latest Update (v1.1.0):** Added conversation memory with persistent chat history! Enable multi-turn conversations with context retention and session management. See the [Conversation Memory](#conversation-memory) section below.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [File Descriptions](#file-descriptions)
- [Usage Examples](#usage-examples)
- [API Endpoints](#api-endpoints)
- [Supported Document Formats](#supported-document-formats)
- [Vector Stores](#vector-stores)
- [Hybrid Search](#hybrid-search-vector--bm25)
- [LLM Providers](#llm-providers)
- [Embedding Models](#embedding-models)
- [Conversation Memory](#conversation-memory)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)

## Features

- 🤖 **Multiple LLM Providers**: OpenAI, Anthropic, Cohere, Google Gemini
- 📚 **Vector Store Options**: ChromaDB, FAISS, Pinecone
- 🔍 **Embedding Models**: OpenAI, HuggingFace, Sentence Transformers
- 🔄 **Hybrid Search**: Vector search with BM25 keyword fallback for robust retrieval
- 💾 **Embeddings Fallback**: Automatic fallback to local models if API fails
- 💬 **Conversation Memory**: Persistent chat history with ChromaDB storage
- 📄 **Document Support**: PDF, DOCX, TXT, Markdown, HTML
- ⚙️ **Configurable**: YAML-based configuration
- 🚀 **API Ready**: FastAPI integration for serving
- 📊 **Monitoring**: Built-in logging and statistics

## Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag_pipleine
```

2. **Create virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Setup environment variables**
```bash
# Copy the example env file
copy .env.example .env

# Edit .env and add your API keys
```

## Quick Start

### 1. Basic Usage

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents
result = pipeline.ingest_documents("./data/documents")
print(result)

# Query the system
response = pipeline.query("What is the main topic?")
print(response["answer"])
```

### 2. Using the API

Start the FastAPI server:

```bash
python app.py
```

Access the API at `http://localhost:8000/docs`

### 3. Conversation Mode (NEW!)

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Start a conversation
session_id = pipeline.start_conversation()

# Multi-turn conversation with context
response1 = pipeline.conversation_query("What is RAG?")
response2 = pipeline.conversation_query("How does it work?")  # Uses previous context
response3 = pipeline.conversation_query("What are its benefits?")  # Continues conversation

# Save and end
pipeline.end_conversation(save=True)

# See example_conversation.py for more examples
```

### 4. Configuration

Edit `config.yaml` to customize:

- LLM settings (provider, model, temperature)
- Embeddings configuration
- Vector store settings
- Document processing parameters
- Retrieval settings

## Project Structure

```
rag_pipleine/
├── rag_pipeline.py           # Main pipeline orchestration
├── document_loader.py        # Document loading and processing
├── vector_store.py           # Vector store management
├── embeddings.py             # Embeddings management
├── query_engine.py           # Query processing and generation
├── conversation_memory.py    # Conversation history management
├── hybrid_search.py          # Hybrid vector + BM25 search
├── keyword_search.py         # BM25 keyword search
├── config.py                 # Configuration management
├── config.yaml               # Configuration file
├── app.py                    # FastAPI application
├── example.py                # Basic usage examples
├── example_hybrid_search.py  # Hybrid search examples
├── example_gemini.py         # Gemini LLM examples
├── example_conversation.py   # Conversation feature examples
├── requirements.txt          # Python dependencies
├── .env.example              # Environment variables template
├── README.md                 # This file
└── data/
    ├── documents/            # Place your documents here
    ├── chroma_db/            # Vector store persistence
    ├── conversations/        # Conversation history storage
    ├── processed/            # Processed documents
    ├── cache/                # Cache directory
    └── logs/                 # Application logs
```

## File Descriptions

### Core Modules

| File | Purpose |
|------|---------|
| `rag_pipeline.py` | Main orchestrator that coordinates all components and provides the primary API |
| `document_loader.py` | Handles loading and parsing documents (PDF, DOCX, TXT, MD, HTML) |
| `vector_store.py` | Manages vector databases (ChromaDB, FAISS, Pinecone) for semantic search |
| `embeddings.py` | Generates embeddings using OpenAI, HuggingFace, or Sentence Transformers |
| `query_engine.py` | Processes queries and generates responses using LLM providers |
| `conversation_memory.py` | **NEW** - Manages conversation sessions with persistent history |
| `hybrid_search.py` | Combines vector search with BM25 keyword search for robust retrieval |
| `keyword_search.py` | Implements BM25 algorithm for keyword-based document search |
| `config.py` | Configuration management with Pydantic models and validation |

### Configuration & Deployment

| File | Purpose |
|------|---------|
| `config.yaml` | YAML configuration file for all pipeline settings |
| `.env.example` | Template for environment variables (API keys) |
| `app.py` | FastAPI application with REST endpoints |
| `requirements.txt` | Python package dependencies |

### Examples

| File | Purpose |
|------|---------|
| `example.py` | Basic usage examples for the RAG pipeline |
| `example_hybrid_search.py` | Demonstrates hybrid search capabilities |
| `example_gemini.py` | Shows how to use Google Gemini LLM |
| `example_conversation.py` | **NEW** - Conversation feature examples and API usage |

### Documentation

| File | Purpose |
|------|---------|
| `README.md` | Comprehensive documentation (this file) |
| `LICENSE` | MIT License information |
| `.gitignore` | Git ignore patterns for cache, logs, and sensitive files |

## Usage Examples

### Document Ingestion

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# Ingest from a directory
pipeline.ingest_documents("./data/documents")

# Ingest a single file
pipeline.ingest_documents("./data/documents/example.pdf")
```

### Querying

```python
# Simple query
response = pipeline.query("What is machine learning?")
print(response["answer"])

# Query with sources
response = pipeline.query("Explain neural networks", return_sources=True)
print(response["answer"])
print(f"Sources: {len(response['sources'])}")
for source in response["sources"]:
    print(f"- {source['metadata']}")
```

### Similarity Search

```python
# Find similar documents
docs = pipeline.get_similar_documents("artificial intelligence", k=5)
for doc in docs:
    print(doc["content"][:200])
```

### Custom Configuration

```python
from config import Config, LLMConfig, EmbeddingsConfig
from rag_pipeline import RAGPipeline

# Create custom config
custom_config = Config(
    llm=LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.5
    ),
    embeddings=EmbeddingsConfig(
        provider="openai",
        model="text-embedding-3-large"
    ),
    # ... other settings
)

# Use custom config
pipeline = RAGPipeline(custom_config)
```

## API Endpoints

### POST /ingest
Ingest documents into the system

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"document_path": "./data/documents"}'
```

### POST /query
Query the RAG system

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is AI?", "return_sources": true}'
```

### GET /health
Health check endpoint

```bash
curl "http://localhost:8000/health"
```

### Conversation Endpoints

**POST /conversation/session**  
Create a new conversation session

```bash
curl -X POST "http://localhost:8000/conversation/session" \
  -H "Content-Type: application/json" \
  -d '{"session_id": "my_session"}'
```

**POST /conversation/query**  
Query with conversation context

```bash
curl -X POST "http://localhost:8000/conversation/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is RAG?",
    "session_id": "my_session",
    "return_sources": true
  }'
```

**GET /conversation/history**  
Get conversation history

```bash
curl "http://localhost:8000/conversation/history?format=list"
```

**GET /conversation/sessions**  
List all saved sessions

```bash
curl "http://localhost:8000/conversation/sessions"
```

**DELETE /conversation/session**  
End current session

```bash
curl -X DELETE "http://localhost:8000/conversation/session?save=true"
```

## Supported Document Formats

- **PDF** (.pdf)
- **Word** (.docx)
- **Text** (.txt)
- **Markdown** (.md)
- **HTML** (.html)

## Vector Stores

### ChromaDB (Default)
```yaml
vector_store:
  type: "chroma"
  persist_directory: "./data/chroma_db"
```

### FAISS
```yaml
vector_store:
  type: "faiss"
  persist_directory: "./data/faiss_index"
```

### Pinecone
```yaml
vector_store:
  type: "pinecone"
  pinecone:
    environment: "gcp-starter"
    index_name: "rag-index"
```

## Hybrid Search (Vector + BM25)

The pipeline includes **hybrid search** that combines semantic vector search with BM25 keyword search for robust retrieval.

### How It Works

1. **Primary**: Vector search using embeddings (semantic similarity)
2. **Fallback**: BM25 keyword search when vector search fails or returns insufficient results
3. **Fusion**: Optional mode that combines both methods with weighted scores

### Configuration

```yaml
hybrid_search:
  enabled: true                    # Enable hybrid search
  enable_fallback: true            # Use BM25 when vector search fails
  enable_fusion: false             # Combine vector and BM25 results
  min_vector_results: 2            # Trigger fallback if fewer results
  fusion_weight_vector: 0.7        # Vector search weight (fusion mode)
  fusion_weight_bm25: 0.3         # BM25 search weight (fusion mode)
```

### Usage Examples

**Automatic Fallback (Default)**
```python
# Automatically uses BM25 if vector search fails
response = pipeline.query("your question")
```

**Direct BM25 Search**
```python
# Use BM25 keyword search directly
results = pipeline.hybrid_search.search(
    "specific keywords",
    k=5,
    search_type="bm25"
)
```

**Fusion Search**
```python
# Combine vector and BM25 results
results = pipeline.hybrid_search.search(
    "your query",
    k=5,
    search_type="fusion"
)
```

### Benefits

- ✅ **Robustness**: Falls back to keyword search when semantic search fails
- ✅ **Better Coverage**: Handles both semantic queries and specific keyword matches
- ✅ **Automatic**: No code changes needed, works transparently
- ✅ **Configurable**: Fine-tune fallback behavior and fusion weights

### Example

```python
from rag_pipeline import RAGPipeline

pipeline = RAGPipeline()  # Hybrid search enabled by default

# Ingest documents (indexes both vector and BM25)
pipeline.ingest_documents("./data/documents")

# Query automatically uses hybrid search with fallback
response = pipeline.query("What is machine learning?")

# Check which search method was used
if response.get('sources'):
    for source in response['sources']:
        search_type = source['metadata'].get('search_type')
        print(f"Search method: {search_type}")  # 'vector' or 'bm25'
```

For detailed examples, see [example_hybrid_search.py](example_hybrid_search.py).

## LLM Providers

### OpenAI
```yaml
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"
  temperature: 0.7
```

### Anthropic
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-opus-20240229"
  temperature: 0.7
```

### Cohere
```yaml
llm:
  provider: "cohere"
  model: "command"
  temperature: 0.7
```

### Google Gemini
```yaml
llm:
  provider: "gemini"
  model: "gemini-pro"
  temperature: 0.7
```

**Note**: Get your Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)

## Embedding Models

### OpenAI
```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
```

### HuggingFace
```yaml
embeddings:
  provider: "huggingface"
  model: "all-MiniLM-L6-v2"
```

### Embeddings with Fallback
```yaml
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
  enable_fallback: true
  fallback_provider: "huggingface"
  fallback_model: "all-MiniLM-L6-v2"
```

**Benefits**:
- Automatic fallback if primary embeddings fail
- Use local models (HuggingFace) as backup for reliability
- No additional API costs for fallback embeddings
- Ensures continuous operation even during API outages

### Recommended Setup: Gemini + all-MiniLM-L6-v2

For cost-effective and efficient RAG:
```yaml
llm:
  provider: "gemini"
  model: "gemini-pro"
  
embeddings:
  provider: "huggingface"
  model: "all-MiniLM-L6-v2"
  dimension: 384
```

**Advantages**:
- ✅ Free local embeddings (no API costs)
- ✅ Competitive Gemini pricing
- ✅ Fast local processing for embeddings
- ✅ Better privacy (embeddings stay local)
- ✅ No rate limits on embeddings
- ✅ Works offline for embedding generation

See [example_gemini.py](example_gemini.py) for demo.

## Advanced Features

### Custom Prompt Templates

```python
custom_prompt = """
Given the context below, answer the question.
Be concise and specific.

Context: {context}
Question: {question}
Answer:
"""

response = pipeline.query(
    "What is RAG?",
    custom_prompt=custom_prompt
)
```

### Batch Queries

```python
questions = [
    "What is machine learning?",
    "Explain deep learning",
    "What are neural networks?"
]

results = pipeline.query_engine.batch_query(questions)
for result in results:
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}\n")
```

### Conversation Memory

Enable persistent conversation history with context-aware responses:

```python
# Initialize pipeline with conversation enabled
pipeline = RAGPipeline()

# Start a conversation session
session_id = pipeline.start_conversation()

# Multi-turn conversation
response1 = pipeline.conversation_query("What is RAG?")
response2 = pipeline.conversation_query("How does it work?")  # Uses previous context
response3 = pipeline.conversation_query("What are its advantages?")  # Continues conversation

# Get conversation history
history = pipeline.get_conversation_history(format="list")

# Save and end session
pipeline.end_conversation(save=True)

# Resume a previous conversation
pipeline.conversation_query(
    "Can you summarize our discussion?", 
    session_id=session_id
)
```

**Configuration** (`config.yaml`):

```yaml
conversation:
  enabled: true
  persist_directory: "./data/conversations"
  max_history: 10  # Number of message pairs to keep in context
  collection_name: "conversation_history"
  memory_type: "buffer"  # or "vector" for semantic search
  session_timeout_minutes: 60
  auto_save: true
```

**API Endpoints**:

```bash
# Create session
POST /conversation/session

# Conversation query
POST /conversation/query
{
  "question": "What is RAG?",
  "session_id": "optional_session_id",
  "return_sources": true
}

# Get history
GET /conversation/history?format=list

# List all sessions
GET /conversation/sessions

# End session
DELETE /conversation/session?save=true
```

**See `example_conversation.py` for more examples.**

## Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure your `.env` file contains valid API keys
   - Check that the `.env` file is in the root directory

2. **Vector Store Issues**
   - Delete the vector store directory and re-ingest documents
   - Ensure sufficient disk space for embeddings

3. **Memory Issues**
   - Reduce `chunk_size` in config.yaml
   - Process documents in smaller batches
   - Use FAISS instead of ChromaDB for large datasets

## Performance Tips

1. **Use GPU for embeddings** (if available)
   ```python
   # In embeddings.py, change device to 'cuda'
   model_kwargs={'device': 'cuda'}
   ```

2. **Optimize chunk size**
   - Smaller chunks: Better precision, more storage
   - Larger chunks: Better context, less storage

3. **Adjust retrieval parameters**
   ```yaml
   retrieval:
     k: 4  # Reduce for faster queries
     search_type: "mmr"  # Use MMR for diverse results
   ```

## Changelog

### Version 1.1.0 (February 2026)

**🎉 New Features:**
- ✅ **Conversation Memory**: Added persistent chat history with ChromaDB storage
  - Multi-turn conversations with context retention
  - Session management with custom IDs
  - Auto-save and timeout support
  - Semantic search across conversation history
  - New API endpoints for conversation management
  - See `example_conversation.py` for usage examples

**🔧 Improvements:**
- Enhanced README with conversation feature documentation
- Added conversation configuration section in `config.yaml`
- Updated project structure with `conversation_memory.py` module
- Added comprehensive conversation examples

**🧹 Maintenance:**
- Removed unnecessary files:
  - `test_rag_pipeline.py` (test dependencies not installed)
  - `setup.py` (not needed for direct usage)
  - `__init__.py` (not used in current setup)
- Cleaned up project structure for easier navigation

### Version 1.0.0 (Initial Release)

**Core Features:**
- Multi-provider LLM support (OpenAI, Anthropic, Cohere, Gemini)
- Multiple vector stores (ChromaDB, FAISS, Pinecone)
- Hybrid search with BM25 keyword fallback
- Document processing for PDF, DOCX, TXT, MD, HTML
- FastAPI REST API
- Embeddings fallback to local models
- YAML-based configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - feel free to use this project for your own purposes.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain)
- Uses [ChromaDB](https://www.trychroma.com/) for vector storage
- Powered by various LLM providers

## Support

For issues or questions, please open an issue on GitHub or contact the maintainers.

---

**Happy RAG-ing! 🚀**
