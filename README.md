# RAG Pipeline

A comprehensive, modular Retrieval-Augmented Generation (RAG) pipeline built with LangChain, supporting multiple LLM providers and advanced features.

> **Version 2.0.0** - Complete modular redesign with simplified, production-ready architecture!

## 📖 Quick Summary

### What This Is
A production-ready RAG (Retrieval-Augmented Generation) system that allows you to:
- Upload documents (PDF, DOCX, TXT, Markdown, HTML)
- Query them using natural language
- Get AI-powered answers with source citations
- Maintain conversation context across multiple queries

### What Changed (v1.x → v2.0)
- **🏗️ Modularized** - Reorganized from flat structure to organized `src/` package
- **🧹 Cleaned** - Removed 4 example files, 6 unused dependencies
- **📝 Simplified** - Config reduced from 125 → 60 lines
- **🔧 Improved** - Virtual env, Python 3.12 support, better docs
- **✅ Production Ready** - Clean imports, proper structure, no errors

### What's Inside
```
├── src/               # Modular source code
│   ├── core/         # Configuration
│   ├── loaders/      # Document loading
│   ├── stores/       # Embeddings & vectors
│   ├── search/       # Query engines
│   └── memory/       # Conversations
├── app.py            # FastAPI web server
├── quickstart.py     # Usage example
├── config.yaml       # Settings
└── venv/            # Virtual environment
```

---

## 📋 Table of Contents

- [What's New in v2.0](#whats-new-in-v20)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Changes from v1.x](#changes-from-v1x)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

---

## 🎉 What's New in v2.0

### Major Improvements

#### 1. **Modularized Architecture**
Complete restructure into organized packages:
```
src/
├── core/          - Configuration management
├── loaders/       - Document loading & processing
├── stores/        - Embeddings & vector storage
├── search/        - Query engines & hybrid search
├── memory/        - Conversation management
└── pipeline.py    - Main RAG orchestrator
```

#### 2. **Simplified & Cleaned**
- ✅ Removed 4 example files (consolidated into `quickstart.py`)
- ✅ Reduced config from 125 lines to 60 lines
- ✅ Removed unused dependencies (FAISS, Pinecone, pandas, etc.)
- ✅ Cleaned up complex configuration options
- ✅ Only 2 files in root: `app.py` and `quickstart.py`

#### 3. **Production Ready**
- ✅ Virtual environment setup
- ✅ Python 3.12 compatible
- ✅ Clean imports with no errors
- ✅ Comprehensive documentation
- ✅ Professional code organization

#### 4. **What Was Removed**
- `example.py`, `example_conversation.py`, `example_gemini.py`, `example_hybrid_search.py`
- Old flat-structure module files (moved to `src/`)
- Unused vector stores (FAISS, Pinecone - can be re-enabled if needed)
- Complex fallback configurations
- Deprecated code patterns

---

## Features

- 🤖 **Multiple LLM Providers**: OpenAI, Anthropic, Cohere, Google Gemini
- 📚 **Vector Store**: ChromaDB for efficient document retrieval
- 🔍 **Hybrid Search**: Vector search with BM25 keyword fallback
- 💬 **Conversation Memory**: Persistent chat history with context retention
- 📄 **Document Support**: PDF, DOCX, TXT, Markdown, HTML
- ⚙️ **Configurable**: Simple YAML-based configuration
- 🚀 **API Ready**: FastAPI integration with full REST API
- 🎯 **Modular Design**: Clean, organized codebase structure
- 📊 **Production Ready**: Comprehensive logging and error handling

---

## 📁 Project Structure

```
rag_pipleine/
├── src/                          # Main source code (modular)
│   ├── core/                     # Core utilities
│   │   ├── config.py            # Configuration management
│   │   └── __init__.py
│   ├── loaders/                  # Document loading
│   │   ├── document_loader.py   # Multi-format loader
│   │   └── __init__.py
│   ├── stores/                   # Storage components
│   │   ├── embeddings.py        # Embeddings management
│   │   ├── vector_store.py      # Vector store handling
│   │   └── __init__.py
│   ├── search/                   # Search & query
│   │   ├── query_engine.py      # LLM query engine
│   │   ├── hybrid_search.py     # Hybrid search orchestrator
│   │   ├── keyword_search.py    # BM25 keyword search
│   │   └── __init__.py
│   ├── memory/                   # Conversation
│   │   ├── conversation_memory.py # Chat history
│   │   └── __init__.py
│   ├── pipeline.py              # Main RAG pipeline
│   └── __init__.py
├── venv/                         # Virtual environment
├── data/                         # Data directory
│   ├── documents/               # Input documents
│   ├── chroma_db/              # Vector database
│   ├── conversations/          # Chat history
│   └── processed/              # Processed files
├── logs/                        # Application logs
├── app.py                       # FastAPI web server
├── quickstart.py               # Usage example
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── .env                        # API keys (create this)
├── .gitignore                  # Git ignore rules
├── LICENSE                     # License
├── README.md                   # This file
├── MIGRATION.md               # v1.x → v2.0 guide
├── VENV_GUIDE.md             # Virtual env help
└── SETUP_COMPLETE.md         # Setup checklist
```

### Key Files

- **[src/pipeline.py](src/pipeline.py)** - Main RAG pipeline orchestrator
- **[app.py](app.py)** - FastAPI REST API server
- **[quickstart.py](quickstart.py)** - Example usage script
- **[config.yaml](config.yaml)** - Configuration settings
- **[requirements.txt](requirements.txt)** - Python dependencies

---

## Installation

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Steps

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd rag_pipleine
```

2. **Create and activate virtual environment**
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
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key_here

# Optional: Add other provider keys as needed
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key
GOOGLE_API_KEY=your_google_key
```

## Quick Start

### 1. Basic Usage

```python
from src.pipeline import RAGPipeline

# Initialize pipeline
pipeline = RAGPipeline()

# Ingest documents
result = pipeline.ingest_documents("./data/documents")
print(f"Status: {result['status']}")
print(f"Documents processed: {result['documents_processed']}")

# Query the system
response = pipeline.query("What is the main topic of the documents?")
print(f"Answer: {response['answer']}")
```

### 2. Using the API Server

Start the FastAPI server:

```bash
python app.py
```

The API server will start at `http://localhost:8000`

Access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

**Example API calls:**

```bash
# Ingest documents
curl -X POST "http://localhost:8000/ingest" \
  -H "Content-Type: application/json" \
  -d '{"document_path": "./data/documents"}'

# Query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "return_sources": true}'

# Conversation query
curl -X POST "http://localhost:8000/conversation/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about the documents", "session_id": "user123"}'
```

### 3. Conversation Mode

```python
from src.pipeline import RAGPipeline

pipeline = RAGPipeline()

# Start a conversation
response = pipeline.conversation_query(
    "Tell me about the documents",
    session_id="user123"
)
print(response["answer"])

# Continue the conversation with context
response = pipeline.conversation_query(
    "Can you elaborate on that?",
    session_id="user123"
)
print(response["answer"])
```

## Project Structure

```
rag_pipleine/
├── src/                          # Main source code (modular architecture)
│   ├── core/                     # Core configuration and utilities
│   │   ├── config.py            # Configuration management
│   │   └── __init__.py
│   ├── loaders/                  # Document loading and processing
│   │   ├── document_loader.py   # Multi-format document loader
│   │   └── __init__.py
│   ├── stores/                   # Storage components
│   │   ├── embeddings.py        # Embeddings management
│   │   ├── vector_store.py      # Vector store management
│   │   └── __init__.py
│   ├── search/                   # Search and query engines
│   │   ├── query_engine.py      # Query handling and LLM integration
│   │   ├── hybrid_search.py     # Hybrid search orchestration
│   │   ├── keyword_search.py    # BM25 keyword search
│   │   └── __init__.py
│   ├── memory/                   # Conversation memory
│   │   ├── conversation_memory.py # Session and history management
│   │   └── __init__.py
│   ├── pipeline.py              # Main RAG pipeline orchestrator
│   └── __init__.py
├── data/                         # Data directory (created automatically)
│   ├── documents/               # Place your documents here
│   ├── chroma_db/              # Vector database storage
│   ├── conversations/          # Conversation history
│   └── processed/              # Processed documents
├── logs/                        # Application logs
├── app.py                       # FastAPI application entry point
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── .env                         # Environment variables (create this)
├── LICENSE                      # License file
└── README.md                    # This file
```

## Configuration

Edit `config.yaml` to customize the pipeline:

```yaml
# LLM Settings
llm:
  provider: "openai"             # Options: openai, anthropic, cohere, gemini
  model: "gpt-4-turbo-preview"
  temperature: 0.7
  max_tokens: 2000

# Embeddings Settings
embeddings:
  provider: "openai"
  model: "text-embedding-3-small"
  dimension: 1536

# Vector Store Settings
vector_store:
  type: "chroma"
  collection_name: "rag_documents"
  persist_directory: "./data/chroma_db"

# Document Processing
document_processing:
  chunk_size: 1000
  chunk_overlap: 200

# Retrieval
retrieval:
  search_type: "similarity"
  k: 4                          # Number of documents to retrieve
  score_threshold: 0.7

# Hybrid Search
hybrid_search:
  enabled: true                 # Enable/disable hybrid search
  min_vector_results: 2         # Trigger BM25 fallback threshold

# Conversation Memory
conversation:
  enabled: true                 # Enable/disable conversation history
  max_history: 10               # Maximum conversation turns to keep

# API Settings
api:
  host: "0.0.0.0"
  port: 8000

# Logging
logging:
  level: "INFO"
  format: "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
```

## API Endpoints

### Document Management

#### `POST /ingest`
Ingest documents from a file or directory.

**Request:**
```json
{
  "document_path": "./data/documents"
}
```

**Response:**
```json
{
  "status": "success",
  "documents_processed": 42,
  "message": "Successfully processed 42 document chunks"
}
```

#### `POST /upload`
Upload a document file directly.

**Request:** Multipart form data with file

**Response:**
```json
{
  "status": "success",
  "filename": "document.pdf",
  "message": "Document uploaded and processed successfully"
}
```

### Query Endpoints

#### `POST /query`
Standard query without conversation context.

**Request:**
```json
{
  "question": "What is RAG?",
  "return_sources": true
}
```

**Response:**
```json
{
  "question": "What is RAG?",
  "answer": "RAG stands for Retrieval-Augmented Generation...",
  "status": "success",
  "sources": [
    {"page_content": "...", "metadata": {...}},
    ...
  ],
  "num_sources": 4
}
```

#### `POST /conversation/query`
Query with conversation context and memory.

**Request:**
```json
{
  "question": "Tell me about RAG",
  "session_id": "user123",
  "return_sources": true
}
```

**Response:**
```json
{
  "question": "Tell me about RAG",
  "answer": "RAG is a technique that...",
  "status": "success",
  "session_id": "user123",
  "message_count": 1,
  "sources": [...]
}
```

### Session Management

#### `POST /conversation/session`
Create a new conversation session.

#### `DELETE /conversation/session/{session_id}`
Clear conversation history for a session.

#### `GET /health`
Health check endpoint.

## Supported Document Formats

- **PDF** (`.pdf`)
- **Word Documents** (`.docx`)
- **Text Files** (`.txt`)
- **Markdown** (`.md`)
- **HTML** (`.html`)

## LLM Providers

### OpenAI
```yaml
llm:
  provider: "openai"
  model: "gpt-4-turbo-preview"  # or gpt-3.5-turbo
```

### Anthropic Claude
```yaml
llm:
  provider: "anthropic"
  model: "claude-3-opus-20240229"
```

### Cohere
```yaml
llm:
  provider: "cohere"
  model: "command"
```

### Google Gemini
```yaml
llm:
  provider: "gemini"
  model: "gemini-pro"
```

## Advanced Features

### Hybrid Search
Combines vector similarity search with BM25 keyword search for better retrieval:
- Primary: Vector search using embeddings
- Fallback: BM25 keyword search when vector results are insufficient
- Configurable threshold for fallback triggering

### Conversation Memory
Persistent chat history with:
- Session-based conversation tracking
- Automatic context inclusion in queries
- ChromaDB-backed storage for semantic search
- Configurable history length

### Logging
Comprehensive logging with:
- Automatic log rotation
- Configurable log levels
- File and console output
- Structured log format

## Development

### Running Tests
```bash
# Install dev dependencies
pip install pytest pytest-cov

# Run tests
pytest

# Run with coverage
pytest --cov=src
```

### Code Style
```bash
# Install formatters
pip install black isort

# Format code
black src/
isort src/
```

## Troubleshooting

### Common Issues

**ModuleNotFoundError**
- Make sure you're in the virtual environment
- Run `pip install -r requirements.txt`

**API Key Errors**
- Check your `.env` file exists and has correct keys
- Verify the provider name matches in `config.yaml`

**Document Loading Issues**
- Ensure documents are in supported formats
- Check file permissions and paths
- View logs in `./logs/` for detailed errors

**ChromaDB Issues**
- Delete `./data/chroma_db/` to reset the vector store
- Ensure sufficient disk space

---

## 🔄 Changes from v1.x

### What Changed in v2.0

#### ✅ Code Organization
**Before (v1.x):** All modules in root directory (flat structure)
```
├── config.py
├── rag_pipeline.py
├── document_loader.py
├── embeddings.py
├── vector_store.py
├── query_engine.py
├── hybrid_search.py
├── keyword_search.py
├── conversation_memory.py
├── example.py
├── example_conversation.py
├── example_gemini.py
└── example_hybrid_search.py
```

**After (v2.0):** Organized modular structure
```
├── src/
│   ├── core/          # Config
│   ├── loaders/       # Document loading
│   ├── stores/        # Embeddings & vectors
│   ├── search/        # Query & search
│   └── memory/        # Conversations
├── app.py             # API server
└── quickstart.py      # Single example
```

#### ✅ Import Changes
**Old:**
```python
from rag_pipeline import RAGPipeline
from config import config
```

**New:**
```python
from src.pipeline import RAGPipeline
from src.core.config import config
```

#### ✅ Configuration Simplified
- Reduced from **125+ lines** to **~60 lines**
- Removed complex options: RAG chain settings, MMR configs, Pinecone specifics
- Made optional LLM providers configurable
- Sensible defaults for most settings

#### ✅ Dependencies Cleaned
**Removed:**
- `faiss-cpu` (not used by default)
- `pinecone-client` (optional, can be re-added)
- `pandas` (not needed)
- `markdown` (covered by unstructured)
- `requests` (redundant)
- `tqdm` (not essential)

**Added:**
- `loguru` (better logging)

**Total:** From 20+ to 17 core dependencies

#### ✅ Files Removed
- `example.py` → Consolidated into `quickstart.py`
- `example_conversation.py` → Consolidated
- `example_gemini.py` → Consolidated
- `example_hybrid_search.py` → Consolidated
- `test_setup.py` → Removed after verification
- All old root-level modules → Moved to `src/`

#### ✅ New Features
- Virtual environment setup guide
- Comprehensive README (this file)
- Migration documentation
- Setup completion guide
- Better error messages
- Cleaner code structure

### Compatibility Notes

#### ✓ Fully Compatible (No Migration Needed)
- All your **data** (vector stores, conversations)
- `.env` file and API keys
- Document formats
- API endpoints
- LLM providers

#### ⚠️ Requires Update
- Python import statements (see above)
- Any custom scripts using old imports
- Configuration file (optional - old still works)

### Benefits of v2.0
1. **Easier to Maintain** - Clear module boundaries
2. **Better Scalability** - Add features in isolation
3. **Simpler Setup** - Less configuration overhead
4. **Professional Structure** - Follows Python best practices
5. **Cleaner Codebase** - Remove unused/example code
6. **Better Testing** - Modular design enables unit tests

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Changelog

### Version 2.0.0 (February 27, 2026) - Current
- ✨ **Complete modular redesign**
  - Organized into `src/` package structure
  - Separated concerns: core, loaders, stores, search, memory
- 🧹 **Cleaned codebase**
  - Removed 4 example files (consolidated to quickstart.py)
  - Removed unused dependencies (6 packages)
  - Simplified configuration (125 → 60 lines)
- 🔧 **Improved infrastructure**
  - Virtual environment setup
  - Python 3.12 compatibility
  - Better error handling
  - Comprehensive logging
- 📚 **Enhanced documentation**
  - Complete README with all changes
  - Migration guide (v1.x → v2.0)
  - Virtual environment guide
  - Setup completion checklist
- 🎯 **Production ready**
  - Clean imports with no errors
  - Professional code organization
  - Proper package structure
  - Type hints and docstrings

### Version 1.1.0
- Added conversation memory
- Hybrid search implementation

### Version 1.0.0
- Initial release
- Basic RAG pipeline
- Multiple LLM provider support
