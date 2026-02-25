"""Main RAG Pipeline Implementation."""

from typing import List, Dict, Any, Optional
from pathlib import Path
from loguru import logger

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import config, get_api_key
from document_loader import DocumentLoader
from vector_store import VectorStoreManager
from embeddings import EmbeddingsManager
from query_engine import QueryEngine
from hybrid_search import HybridSearch, HybridRetriever
from conversation_memory import ConversationManager


class RAGPipeline:
    """
    Main RAG (Retrieval-Augmented Generation) Pipeline.
    
    This class orchestrates the entire RAG workflow including:
    - Document loading and processing
    - Embedding generation
    - Vector storage
    - Query retrieval and generation
    """
    
    def __init__(self, config_obj=None):
        """
        Initialize RAG Pipeline.
        
        Args:
            config_obj: Configuration object (uses default if None)
        """
        self.config = config_obj or config
        self._setup_logging()
        
        # Initialize components
        self.document_loader = DocumentLoader(self.config)
        self.embeddings_manager = EmbeddingsManager(self.config)
        self.vector_store_manager = VectorStoreManager(self.config)
        self.query_engine = None
        
        # Initialize hybrid search if enabled
        self.hybrid_search = None
        if self.config.hybrid_search and self.config.hybrid_search.enabled:
            logger.info("Initializing hybrid search (vector + BM25)...")
            self.hybrid_search = HybridSearch(self.vector_store_manager, self.config)
        
        # Initialize conversation manager if enabled
        self.conversation_manager = None
        if self.config.conversation and self.config.conversation.enabled:
            logger.info("Initializing conversation manager...")
            embeddings = self.embeddings_manager.get_embeddings()
            self.conversation_manager = ConversationManager(self.config, embeddings)
        
        # Create necessary directories
        self._create_directories()
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.logging
        
        # Remove default logger
        logger.remove()
        
        # Add custom logger
        logger.add(
            f"{self.config.paths.logs}/rag_pipeline.log",
            rotation=log_config.rotation,
            retention=log_config.retention,
            format=log_config.format,
            level=log_config.level
        )
        
        # Also log to console
        logger.add(
            lambda msg: print(msg, end=""),
            format=log_config.format,
            level=log_config.level
        )
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        dirs = [
            self.config.paths.documents,
            self.config.paths.processed,
            self.config.paths.cache,
            self.config.paths.logs,
            self.config.vector_store.persist_directory
        ]
        
        # Add conversation directory if enabled
        if self.config.conversation and self.config.conversation.enabled:
            dirs.append(self.config.conversation.persist_directory)
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {dir_path}")
    
    def ingest_documents(self, document_path: str) -> Dict[str, Any]:
        """
        Ingest documents into the RAG pipeline.
        
        Args:
            document_path: Path to document or directory of documents
            
        Returns:
            Dictionary with ingestion results
        """
        logger.info(f"Starting document ingestion from: {document_path}")
        
        try:
            # Load and process documents
            documents = self.document_loader.load_documents(document_path)
            logger.info(f"Loaded {len(documents)} document chunks")
            
            # Create embeddings
            embeddings = self.embeddings_manager.get_embeddings()
            
            # Store in vector database
            self.vector_store_manager.add_documents(documents, embeddings)
            logger.info("Documents successfully stored in vector database")
            
            # Index for hybrid search (BM25) if enabled
            if self.hybrid_search:
                logger.info("Indexing documents for BM25 keyword search...")
                self.hybrid_search.index_documents(documents)
                logger.info("BM25 indexing complete")
            
            return {
                "status": "success",
                "documents_processed": len(documents),
                "message": "Documents ingested successfully"
            }
            
        except Exception as e:
            logger.error(f"Error during document ingestion: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Document ingestion failed"
            }
    
    def setup_query_engine(self):
        """Setup the query engine for question answering."""
        if not self.vector_store_manager.vector_store:
            logger.warning("Vector store not initialized. Loading existing store...")
            embeddings = self.embeddings_manager.get_embeddings()
            self.vector_store_manager.load_vector_store(embeddings)
        
        # Setup hybrid retriever if hybrid search is enabled
        hybrid_retriever = None
        if self.hybrid_search:
            logger.info("Setting up hybrid retriever (vector + BM25 fallback)")
            k = self.config.retrieval.k
            search_type = "auto"  # Use auto mode with fallback
            
            # Check if fusion mode is enabled
            if self.config.hybrid_search.enable_fusion:
                search_type = "fusion"
                logger.info("Using fusion mode (combining vector and BM25 scores)")
            
            hybrid_retriever = HybridRetriever(
                self.hybrid_search,
                search_kwargs={"k": k, "search_type": search_type}
            )
        
        self.query_engine = QueryEngine(
            self.config,
            self.vector_store_manager.vector_store,
            hybrid_retriever=hybrid_retriever
        )
        logger.info("Query engine setup completed")
    
    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            question: Question to ask
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer and optional source documents
        """
        if not self.query_engine:
            self.setup_query_engine()
        
        logger.info(f"Processing query: {question}")
        
        try:
            result = self.query_engine.query(question, return_sources)
            logger.info("Query processed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Query processing failed"
            }
    
    def get_similar_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve similar documents without generation.
        
        Args:
            query: Search query
            k: Number of documents to retrieve (uses config default if None)
            
        Returns:
            List of similar documents with metadata
        """
        if not self.vector_store_manager.vector_store:
            embeddings = self.embeddings_manager.get_embeddings()
            self.vector_store_manager.load_vector_store(embeddings)
        
        k = k or self.config.retrieval.k
        
        logger.info(f"Retrieving {k} similar documents for query: {query}")
        
        try:
            docs = self.vector_store_manager.similarity_search(query, k)
            
            results = []
            for doc in docs:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                })
            
            logger.info(f"Retrieved {len(results)} similar documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def reset_vector_store(self):
        """Reset/clear the vector store and hybrid search."""
        logger.warning("Resetting vector store...")
        self.vector_store_manager.reset()
        
        if self.hybrid_search:
            logger.warning("Resetting hybrid search (BM25)...")
            self.hybrid_search.reset()
        
        logger.info("Vector store reset completed")
    
    def conversation_query(
        self, 
        question: str, 
        session_id: Optional[str] = None,
        return_sources: bool = True
    ) -> Dict[str, Any]:
        """
        Query with conversation context.
        
        Args:
            question: Question to ask
            session_id: Session ID (creates new if not provided)
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary with answer, sources, and session info
        """
        if not self.conversation_manager:
            logger.warning("Conversation manager not enabled, using standard query")
            return self.query(question, return_sources)
        
        # Start or load session
        if session_id:
            self.conversation_manager.load_conversation(session_id)
        elif not self.conversation_manager.current_session_id:
            session_id = self.conversation_manager.start_session()
        else:
            session_id = self.conversation_manager.current_session_id
        
        # Add user message to history
        self.conversation_manager.add_message("user", question)
        
        # Get conversation context
        context_messages = self.conversation_manager.get_context_window()
        
        # Build context-aware query
        if len(context_messages) > 1:  # Has previous context
            context_str = self.conversation_manager.get_conversation_history(format="string")
            enhanced_question = f"Previous conversation:\n{context_str}\n\nCurrent question: {question}"
        else:
            enhanced_question = question
        
        # Get answer
        try:
            result = self.query(enhanced_question, return_sources)
            
            # Add assistant response to history
            if result.get("status") == "success":
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                
                # Store sources in metadata
                source_metadata = {
                    "source_count": len(sources),
                    "source_files": [s.get("source", "unknown") for s in sources[:3]]
                }
                
                self.conversation_manager.add_message("assistant", answer, source_metadata)
                
                # Auto-save if enabled
                if self.conversation_manager.auto_save:
                    self.conversation_manager.save_conversation()
            
            # Add session info to result
            result["session_id"] = session_id
            result["message_count"] = len(self.conversation_manager.current_history)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in conversation query: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "session_id": session_id
            }
    
    def start_conversation(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if not self.conversation_manager:
            raise ValueError("Conversation manager not enabled")
        
        return self.conversation_manager.start_session(session_id)
    
    def end_conversation(self, save: bool = True):
        """
        End current conversation session.
        
        Args:
            save: Whether to save the conversation
        """
        if not self.conversation_manager:
            raise ValueError("Conversation manager not enabled")
        
        self.conversation_manager.end_session(save)
    
    def get_conversation_history(self, format: str = "list") -> Any:
        """
        Get current conversation history.
        
        Args:
            format: Output format ('list', 'string', 'messages')
            
        Returns:
            Conversation history in requested format
        """
        if not self.conversation_manager:
            raise ValueError("Conversation manager not enabled")
        
        return self.conversation_manager.get_conversation_history(format)
    
    def list_conversations(self) -> List[Dict[str, Any]]:
        """
        List all saved conversation sessions.
        
        Returns:
            List of session metadata
        """
        if not self.conversation_manager:
            raise ValueError("Conversation manager not enabled")
        
        return self.conversation_manager.list_sessions()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.
        
        Returns:
            Dictionary with pipeline statistics
        """
        stats = {
            "config": {
                "llm_model": self.config.llm.model,
                "embeddings_model": self.config.embeddings.model,
                "vector_store_type": self.config.vector_store.type,
                "chunk_size": self.config.document_processing.chunk_size,
                "hybrid_search_enabled": bool(self.hybrid_search),
                "conversation_enabled": bool(self.conversation_manager)
            },
            "vector_store": self.vector_store_manager.get_stats()
        }
        
        # Add hybrid search stats if enabled
        if self.hybrid_search:
            stats["hybrid_search"] = self.hybrid_search.get_stats()
        
        # Add conversation stats if enabled
        if self.conversation_manager:
            stats["conversation"] = self.conversation_manager.get_stats()
        
        return stats


def main():
    """Main function for testing the pipeline."""
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Example: Ingest documents
    # result = pipeline.ingest_documents("./data/documents")
    # print(result)
    
    # Example: Query
    # response = pipeline.query("What is the main topic of the documents?")
    # print(response)
    
    logger.info("RAG Pipeline is ready for use")


if __name__ == "__main__":
    main()
