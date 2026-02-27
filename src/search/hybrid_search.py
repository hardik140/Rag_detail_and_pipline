"""Hybrid search combining vector and keyword search."""

from typing import List, Optional, Dict, Any
from loguru import logger

from langchain.docstore.document import Document
from .keyword_search import BM25Search


class HybridSearch:
    """
    Hybrid search combining vector search and BM25 keyword search.
    
    This class provides:
    1. Vector search as primary method
    2. BM25 keyword search as fallback when vector search fails
    3. Combined search with score fusion (optional)
    """
    
    def __init__(self, vector_store_manager, config):
        """
        Initialize hybrid search.
        
        Args:
            vector_store_manager: VectorStoreManager instance
            config: Configuration object
        """
        self.vector_store_manager = vector_store_manager
        self.config = config
        self.bm25_search = BM25Search(config)
        
        # Get hybrid search config
        hybrid_config = getattr(config, 'hybrid_search', None)
        if hybrid_config:
            self.enable_fallback = getattr(hybrid_config, 'enable_fallback', True)
            self.enable_fusion = getattr(hybrid_config, 'enable_fusion', False)
            self.fusion_weight_vector = getattr(hybrid_config, 'fusion_weight_vector', 0.7)
            self.fusion_weight_bm25 = getattr(hybrid_config, 'fusion_weight_bm25', 0.3)
            self.min_vector_results = getattr(hybrid_config, 'min_vector_results', 2)
        else:
            # Defaults
            self.enable_fallback = True
            self.enable_fusion = False
            self.fusion_weight_vector = 0.7
            self.fusion_weight_bm25 = 0.3
            self.min_vector_results = 2
        
        logger.info(f"Hybrid search initialized (fallback: {self.enable_fallback}, fusion: {self.enable_fusion})")
    
    def index_documents(self, documents: List[Document]):
        """
        Index documents for both vector and BM25 search.
        
        Args:
            documents: List of documents to index
        """
        logger.info(f"Indexing {len(documents)} documents for hybrid search...")
        
        # Index for BM25
        self.bm25_search.index_documents(documents)
        
        logger.info("Hybrid search indexing complete")
    
    def search(
        self,
        query: str,
        k: int = 4,
        search_type: str = "auto"
    ) -> List[Document]:
        """
        Search using hybrid approach.
        
        Args:
            query: Search query
            k: Number of documents to return
            search_type: "auto" (hybrid with fallback), "vector", "bm25", or "fusion"
            
        Returns:
            List of relevant documents
        """
        logger.info(f"Hybrid search (type: {search_type}) for query: {query[:50]}...")
        
        if search_type == "bm25":
            return self._bm25_only_search(query, k)
        
        elif search_type == "vector":
            return self._vector_only_search(query, k)
        
        elif search_type == "fusion":
            return self._fusion_search(query, k)
        
        else:  # auto - vector with BM25 fallback
            return self._auto_search_with_fallback(query, k)
    
    def _vector_only_search(self, query: str, k: int) -> List[Document]:
        """Perform vector search only."""
        try:
            if not self.vector_store_manager.vector_store:
                logger.warning("Vector store not initialized")
                return []
            
            results = self.vector_store_manager.similarity_search(query, k)
            logger.info(f"Vector search returned {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"Vector search failed: {str(e)}")
            return []
    
    def _bm25_only_search(self, query: str, k: int) -> List[Document]:
        """Perform BM25 search only."""
        try:
            results = self.bm25_search.search(query, k)
            logger.info(f"BM25 search returned {len(results)} documents")
            return results
        
        except Exception as e:
            logger.error(f"BM25 search failed: {str(e)}")
            return []
    
    def _auto_search_with_fallback(self, query: str, k: int) -> List[Document]:
        """
        Automatic search with fallback logic.
        
        1. Try vector search first
        2. If vector search fails or returns too few results, use BM25
        """
        # Try vector search first
        vector_results = self._vector_only_search(query, k)
        
        # Check if we need to fallback to BM25
        if not self.enable_fallback:
            return vector_results
        
        # Determine if fallback is needed
        fallback_needed = False
        
        if not vector_results:
            logger.warning("Vector search returned no results, falling back to BM25")
            fallback_needed = True
        elif len(vector_results) < self.min_vector_results:
            logger.warning(
                f"Vector search returned only {len(vector_results)} results "
                f"(min: {self.min_vector_results}), falling back to BM25"
            )
            fallback_needed = True
        
        if fallback_needed:
            logger.info("🔄 Activating BM25 keyword search fallback...")
            bm25_results = self._bm25_only_search(query, k)
            
            if bm25_results:
                logger.info(f"✓ BM25 fallback successful: {len(bm25_results)} documents found")
                return bm25_results
            else:
                logger.warning("BM25 fallback also returned no results")
                return vector_results  # Return whatever we got from vector search
        
        return vector_results
    
    def _fusion_search(self, query: str, k: int) -> List[Document]:
        """
        Fusion search combining vector and BM25 with score weighting.
        
        Uses Reciprocal Rank Fusion (RRF) algorithm to combine results.
        """
        logger.info("Performing fusion search (vector + BM25)...")
        
        # Get results from both methods
        vector_results = self._vector_only_search(query, k * 2)  # Get more for fusion
        bm25_results = self._bm25_only_search(query, k * 2)
        
        if not vector_results and not bm25_results:
            logger.warning("Both vector and BM25 searches returned no results")
            return []
        
        # Use Reciprocal Rank Fusion
        doc_scores = {}
        
        # Add vector search scores
        for rank, doc in enumerate(vector_results, 1):
            doc_id = doc.page_content[:100]  # Use content snippet as ID
            score = self.fusion_weight_vector / (rank + 60)  # RRF formula
            doc_scores[doc_id] = {
                "score": score,
                "document": doc,
                "sources": ["vector"]
            }
        
        # Add BM25 scores
        for rank, doc in enumerate(bm25_results, 1):
            doc_id = doc.page_content[:100]
            bm25_score = self.fusion_weight_bm25 / (rank + 60)
            
            if doc_id in doc_scores:
                doc_scores[doc_id]["score"] += bm25_score
                doc_scores[doc_id]["sources"].append("bm25")
            else:
                doc_scores[doc_id] = {
                    "score": bm25_score,
                    "document": doc,
                    "sources": ["bm25"]
                }
        
        # Sort by combined score
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: x[1]["score"],
            reverse=True
        )
        
        # Return top k with metadata
        results = []
        for doc_id, data in sorted_docs[:k]:
            doc = data["document"]
            # Add fusion metadata
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "fusion_score": data["score"],
                    "search_sources": data["sources"],
                    "search_type": "fusion"
                }
            )
            results.append(doc_copy)
        
        logger.info(f"Fusion search returned {len(results)} documents")
        return results
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to both vector and BM25 indexes.
        
        Args:
            documents: List of documents to add
        """
        logger.info(f"Adding {len(documents)} documents to hybrid search...")
        
        # Add to BM25 index
        self.bm25_search.add_documents(documents)
        
        logger.info("Documents added to hybrid search")
    
    def save_bm25_index(self, file_path: str):
        """
        Save BM25 index to disk.
        
        Args:
            file_path: Path to save the index
        """
        self.bm25_search.save_index(file_path)
    
    def load_bm25_index(self, file_path: str):
        """
        Load BM25 index from disk.
        
        Args:
            file_path: Path to load the index from
        """
        self.bm25_search.load_index(file_path)
    
    def reset(self):
        """Reset both vector and BM25 indexes."""
        logger.info("Resetting hybrid search...")
        self.bm25_search.reset()
        logger.info("Hybrid search reset complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get hybrid search statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "type": "hybrid",
            "enable_fallback": self.enable_fallback,
            "enable_fusion": self.enable_fusion,
            "vector_store": self.vector_store_manager.get_stats(),
            "bm25": self.bm25_search.get_stats(),
            "fusion_weights": {
                "vector": self.fusion_weight_vector,
                "bm25": self.fusion_weight_bm25
            }
        }


class HybridRetriever:
    """
    Retriever interface for hybrid search compatible with LangChain.
    """
    
    def __init__(self, hybrid_search: HybridSearch, search_kwargs: dict = None):
        """
        Initialize hybrid retriever.
        
        Args:
            hybrid_search: HybridSearch instance
            search_kwargs: Search parameters (k, search_type)
        """
        self.hybrid_search = hybrid_search
        self.search_kwargs = search_kwargs or {}
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """
        Get relevant documents for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        k = self.search_kwargs.get("k", 4)
        search_type = self.search_kwargs.get("search_type", "auto")
        
        return self.hybrid_search.search(query, k, search_type)
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents."""
        return self.get_relevant_documents(query)
