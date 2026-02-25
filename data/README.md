# Data Directory

This directory contains all data-related files for the RAG pipeline.

## Directory Structure

```
data/
├── documents/       # Place your source documents here
├── chroma_db/       # ChromaDB vector store (auto-created)
├── processed/       # Processed documents cache
├── cache/           # General cache directory
└── README.md        # This file
```

## Usage

### documents/
Place your source documents in this folder. Supported formats:
- PDF (.pdf)
- Word Documents (.docx)
- Text files (.txt)
- Markdown (.md)
- HTML (.html)

Example:
```
documents/
├── paper1.pdf
├── notes.txt
├── documentation.md
└── research/
    ├── study1.pdf
    └── study2.pdf
```

### chroma_db/
This directory is automatically created when you use ChromaDB as your vector store. It contains the persistent vector database with all document embeddings.

**Note:** This directory can be large. Add it to `.gitignore` to avoid committing to version control.

### processed/
Stores processed versions of documents and intermediate results. This helps speed up re-processing.

### cache/
General cache directory for temporary files and cached results.

## Getting Started

1. **Add Documents:**
   ```bash
   # Copy your documents to the documents folder
   cp /path/to/your/docs/* data/documents/
   ```

2. **Ingest Documents:**
   ```python
   from rag_pipeline import RAGPipeline
   
   pipeline = RAGPipeline()
   pipeline.ingest_documents("./data/documents")
   ```

3. **Query:**
   ```python
   response = pipeline.query("Your question here")
   print(response["answer"])
   ```

## Maintenance

### Clear Vector Database
To start fresh, delete the vector database:
```bash
# Windows
rmdir /s data\chroma_db

# Linux/macOS
rm -rf data/chroma_db
```

Or use the API:
```python
pipeline.reset_vector_store()
```

### Backup
To backup your vector database:
```bash
# Windows
xcopy data\chroma_db backup\chroma_db /E /I

# Linux/macOS
cp -r data/chroma_db backup/chroma_db
```

## Size Considerations

- Document embeddings can be large (1-2 KB per chunk)
- A 100-page PDF might generate 200-300 chunks
- Plan for ~500 KB per document on average
- Monitor disk space if processing many documents

## Security

- Do not commit sensitive documents to version control
- Add `data/documents/` to `.gitignore` if needed
- Ensure proper access controls on the data directory
