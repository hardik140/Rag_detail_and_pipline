"""Vector store management module."""

from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger

from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma, FAISS
from langchain.vectorstores.base import VectorStore

try:
    from pinecone import Pinecone
    from langchain_community.vectorstores import Pinecone as LangchainPinecone
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False


class VectorStoreManager:
    """
    Manages vector storage for document embeddings.
    
    Supports multiple vector store backends:
    - ChromaDB (default)
    - FAISS
    - Pinecone
    """
    
    def __init__(self, config):
        """
        Initialize vector store manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.vector_store: Optional[VectorStore] = None
        self.store_type = config.vector_store.type.lower()
        
        logger.info(f"Initializing vector store manager with type: {self.store_type}")
    
    def create_vector_store(
        self,
        documents: List[Document],
        embeddings
    ) -> VectorStore:
        """
        Create a new vector store from documents.
        
        Args:
            documents: List of documents to store
            embeddings: Embeddings instance
            
        Returns:
            Vector store instance
        """
        logger.info(f"Creating {self.store_type} vector store with {len(documents)} documents")
        
        if self.store_type == "chroma":
            self.vector_store = self._create_chroma(documents, embeddings)
        elif self.store_type == "faiss":
            self.vector_store = self._create_faiss(documents, embeddings)
        elif self.store_type == "pinecone":
            self.vector_store = self._create_pinecone(documents, embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
        
        logger.info("Vector store created successfully")
        return self.vector_store
    
    def _create_chroma(
        self,
        documents: List[Document],
        embeddings
    ) -> Chroma:
        """Create ChromaDB vector store."""
        persist_dir = self.config.vector_store.persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name=self.config.vector_store.collection_name,
            persist_directory=persist_dir
        )
        
        return vector_store
    
    def _create_faiss(
        self,
        documents: List[Document],
        embeddings
    ) -> FAISS:
        """Create FAISS vector store."""
        vector_store = FAISS.from_documents(
            documents=documents,
            embedding=embeddings
        )
        
        # Save FAISS index
        persist_dir = self.config.vector_store.persist_directory
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        vector_store.save_local(persist_dir)
        
        return vector_store
    
    def _create_pinecone(
        self,
        documents: List[Document],
        embeddings
    ) -> 'LangchainPinecone':
        """Create Pinecone vector store."""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not installed. Install with: pip install pinecone-client")
        
        from config import get_api_key
        
        api_key = get_api_key("pinecone")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        # Initialize Pinecone
        pc = Pinecone(api_key=api_key)
        
        index_name = self.config.vector_store.pinecone.get("index_name", "rag-index")
        
        vector_store = LangchainPinecone.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=index_name
        )
        
        return vector_store
    
    def load_vector_store(self, embeddings) -> VectorStore:
        """
        Load existing vector store.
        
        Args:
            embeddings: Embeddings instance
            
        Returns:
            Vector store instance
        """
        logger.info(f"Loading existing {self.store_type} vector store")
        
        if self.store_type == "chroma":
            self.vector_store = self._load_chroma(embeddings)
        elif self.store_type == "faiss":
            self.vector_store = self._load_faiss(embeddings)
        elif self.store_type == "pinecone":
            self.vector_store = self._load_pinecone(embeddings)
        else:
            raise ValueError(f"Unsupported vector store type: {self.store_type}")
        
        logger.info("Vector store loaded successfully")
        return self.vector_store
    
    def _load_chroma(self, embeddings) -> Chroma:
        """Load existing ChromaDB vector store."""
        persist_dir = self.config.vector_store.persist_directory
        
        if not Path(persist_dir).exists():
            raise FileNotFoundError(f"Chroma database not found at: {persist_dir}")
        
        vector_store = Chroma(
            collection_name=self.config.vector_store.collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )
        
        return vector_store
    
    def _load_faiss(self, embeddings) -> FAISS:
        """Load existing FAISS vector store."""
        persist_dir = self.config.vector_store.persist_directory
        
        if not Path(persist_dir).exists():
            raise FileNotFoundError(f"FAISS index not found at: {persist_dir}")
        
        vector_store = FAISS.load_local(
            persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
        
        return vector_store
    
    def _load_pinecone(self, embeddings) -> 'LangchainPinecone':
        """Load existing Pinecone vector store."""
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone is not installed")
        
        from config import get_api_key
        
        api_key = get_api_key("pinecone")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found")
        
        index_name = self.config.vector_store.pinecone.get("index_name", "rag-index")
        
        vector_store = LangchainPinecone.from_existing_index(
            index_name=index_name,
            embedding=embeddings
        )
        
        return vector_store
    
    def add_documents(
        self,
        documents: List[Document],
        embeddings
    ):
        """
        Add documents to vector store.
        
        Args:
            documents: List of documents to add
            embeddings: Embeddings instance
        """
        if not self.vector_store:
            self.create_vector_store(documents, embeddings)
        else:
            logger.info(f"Adding {len(documents)} documents to existing vector store")
            self.vector_store.add_documents(documents)
            
            # Persist changes for ChromaDB and FAISS
            if self.store_type == "chroma":
                # Chroma auto-persists
                pass
            elif self.store_type == "faiss":
                persist_dir = self.config.vector_store.persist_directory
                self.vector_store.save_local(persist_dir)
    
    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        k = k or self.config.retrieval.k
        
        logger.debug(f"Searching for {k} similar documents")
        results = self.vector_store.similarity_search(query, k=k)
        
        return results
    
    def similarity_search_with_score(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[tuple]:
        """
        Search for similar documents with relevance scores.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        k = k or self.config.retrieval.k
        
        logger.debug(f"Searching for {k} similar documents with scores")
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        return results
    
    def as_retriever(self, **kwargs):
        """
        Get vector store as a retriever.
        
        Args:
            **kwargs: Additional retriever arguments
            
        Returns:
            Retriever instance
        """
        if not self.vector_store:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = {
            "k": self.config.retrieval.k
        }
        search_kwargs.update(kwargs)
        
        return self.vector_store.as_retriever(
            search_type=self.config.retrieval.search_type,
            search_kwargs=search_kwargs
        )
    
    def reset(self):
        """Reset/clear the vector store."""
        if self.store_type == "chroma":
            persist_dir = Path(self.config.vector_store.persist_directory)
            if persist_dir.exists():
                import shutil
                shutil.rmtree(persist_dir)
                logger.info(f"Deleted Chroma database at: {persist_dir}")
        
        elif self.store_type == "faiss":
            persist_dir = Path(self.config.vector_store.persist_directory)
            if persist_dir.exists():
                import shutil
                shutil.rmtree(persist_dir)
                logger.info(f"Deleted FAISS index at: {persist_dir}")
        
        self.vector_store = None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.
        
        Returns:
            Dictionary with statistics
        """
        if not self.vector_store:
            return {"status": "not_initialized"}
        
        stats = {
            "type": self.store_type,
            "status": "initialized"
        }
        
        # Try to get collection count for Chroma
        if self.store_type == "chroma":
            try:
                collection = self.vector_store._collection
                stats["document_count"] = collection.count()
            except:
                stats["document_count"] = "unknown"
        
        return stats
