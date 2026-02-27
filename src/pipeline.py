"""Main RAG Pipeline - Simplified and Modular."""

from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from .core.config import Config, load_config
from .loaders import DocumentLoader
from .stores import EmbeddingsManager, VectorStoreManager
from .search import QueryEngine, HybridSearch
from .memory import ConversationManager


class RAGPipeline:
    """
    Simplified RAG Pipeline that orchestrates all components.
    
    Features:
    - Document loading and processing
    - Vector storage and retrieval
    - Hybrid search (vector + keyword)
    - Conversation memory
    - Multiple LLM providers
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize RAG Pipeline."""
        self.config = load_config(config_path) if config_path else load_config()
        self._setup_logging()
        self._create_directories()
        
        # Initialize core components
        self.document_loader = DocumentLoader(self.config)
        self.embeddings_manager = EmbeddingsManager(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)
        
        # Initialize optional components
        self.hybrid_search = None
        if self.config.hybrid_search.enabled:
            self.hybrid_search = HybridSearch(self.vector_store_manager, self.config)
        
        self.conversation_manager = None
        if self.config.conversation.enabled:
            embeddings = self.embeddings_manager.get_embeddings()
            self.conversation_manager = ConversationManager(self.config, embeddings)
        
        self.query_engine = None
        logger.info("RAG Pipeline initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logger.remove()
        logger.add(
            f"{self.config.paths.logs}/rag_pipeline.log",
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
            format=self.config.logging.format,
            level=self.config.logging.level
        )
        logger.add(
            lambda msg: print(msg, end=""),
            format=self.config.logging.format,
            level=self.config.logging.level
        )
    
    def _create_directories(self):
        """Create necessary directories."""
        dirs = [
            self.config.paths.documents,
            self.config.paths.processed,
            self.config.paths.logs,
            self.config.vector_store.persist_directory
        ]
        if self.config.conversation.enabled:
            dirs.append(self.config.conversation.persist_directory)
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def ingest_documents(self, path: str) -> Dict[str, Any]:
        """
        Ingest documents from a file or directory.
        
        Args:
            path: Path to file or directory
            
        Returns:
            Dictionary with ingestion results
        """
        try:
            logger.info(f"Starting document ingestion from: {path}")
            
            # Load and split documents
            documents = self.document_loader.load_documents(path)
            logger.info(f"Loaded {len(documents)} document chunks")
            
            # Get embeddings
            embeddings = self.embeddings_manager.get_embeddings()
            
            # Create or update vector store
            self.vector_store_manager.create_vector_store(documents, embeddings)
            
            # Index for hybrid search if enabled
            if self.hybrid_search:
                self.hybrid_search.index_documents(documents)
            
            logger.info("Document ingestion completed successfully")
            return {
                "status": "success",
                "documents_processed": len(documents),
                "message": f"Successfully processed {len(documents)} document chunks"
            }
            
        except Exception as e:
            logger.error(f"Error during document ingestion: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional sources
        """
        try:
            # Initialize query engine if needed
            if not self.query_engine:
                embeddings = self.embeddings_manager.get_embeddings()
                vector_store = self.vector_store_manager.load_vector_store(embeddings)
                
                hybrid_retriever = None
                if self.hybrid_search:
                    hybrid_retriever = self.hybrid_search.get_retriever()
                
                self.query_engine = QueryEngine(
                    self.config, 
                    vector_store,
                    hybrid_retriever
                )
            
            # Get answer
            result = self.query_engine.query(question, return_sources=return_sources)
            
            return {
                "status": "success",
                "question": question,
                "answer": result.get("answer", ""),
                "sources": result.get("sources", []) if return_sources else None
            }
            
        except Exception as e:
            logger.error(f"Error during query: {e}")
            return {
                "status": "error",
                "question": question,
                "message": str(e)
            }
    
    def conversation_query(
        self, 
        question: str, 
        session_id: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query with conversation memory.
        
        Args:
            question: User question
            session_id: Optional session ID
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and conversation context
        """
        if not self.conversation_manager:
            return self.query(question, return_sources)
        
        try:
            # Get or create session
            if not session_id:
                session_id = self.conversation_manager.create_session()
            
            # Get answer with conversation context
            result = self.conversation_manager.query_with_memory(
                question,
                session_id,
                self.query_engine or self._init_query_engine()
            )
            
            return {
                "status": "success",
                "question": question,
                "answer": result.get("answer", ""),
                "session_id": session_id,
                "sources": result.get("sources", []) if return_sources else None
            }
            
        except Exception as e:
            logger.error(f"Error during conversation query: {e}")
            return {
                "status": "error",
                "question": question,
                "message": str(e)
            }
    
    def _init_query_engine(self):
        """Initialize query engine if not already initialized."""
        if not self.query_engine:
            embeddings = self.embeddings_manager.get_embeddings()
            vector_store = self.vector_store_manager.load_vector_store(embeddings)
            
            hybrid_retriever = None
            if self.hybrid_search:
                hybrid_retriever = self.hybrid_search.get_retriever()
            
            self.query_engine = QueryEngine(
                self.config,
                vector_store,
                hybrid_retriever
            )
        return self.query_engine
