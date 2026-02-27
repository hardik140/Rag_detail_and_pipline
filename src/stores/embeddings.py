"""Embeddings management module."""

from typing import List, Optional
from loguru import logger

from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer

from ..core.config import get_api_key


class EmbeddingsManager:
    """
    Manages embeddings for RAG pipeline.
    
    Supports multiple embedding providers:
    - OpenAI
    - HuggingFace
    - Sentence Transformers
    """
    
    def __init__(self, config):
        """
        Initialize embeddings manager.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.provider = config.embeddings.provider.lower()
        self.model = config.embeddings.model
        self.embeddings = None
        self.fallback_embeddings = None
        
        # Fallback configuration
        self.enable_fallback = getattr(config.embeddings, 'enable_fallback', False)
        self.fallback_provider = getattr(config.embeddings, 'fallback_provider', None)
        self.fallback_model = getattr(config.embeddings, 'fallback_model', None)
        
        logger.info(f"Embeddings manager initialized with provider: {self.provider}")
        if self.enable_fallback and self.fallback_provider:
            logger.info(f"Fallback enabled: {self.fallback_provider}/{self.fallback_model}")
    
    def get_embeddings(self):
        """
        Get or create embeddings instance with fallback support.
        
        Returns:
            Embeddings instance
        """
        if self.embeddings is None:
            try:
                self.embeddings = self._create_embeddings()
            except Exception as e:
                logger.error(f"Failed to create primary embeddings: {str(e)}")
                
                # Try fallback if enabled
                if self.enable_fallback and self.fallback_provider:
                    logger.warning(f"Attempting fallback to {self.fallback_provider}/{self.fallback_model}...")
                    try:
                        self.embeddings = self._create_fallback_embeddings()
                        logger.info("✓ Fallback embeddings activated successfully")
                    except Exception as fallback_error:
                        logger.error(f"Fallback embeddings also failed: {str(fallback_error)}")
                        raise
                else:
                    raise
        
        return self.embeddings
    
    def _create_embeddings(self):
        """
        Create embeddings based on configuration.
        
        Returns:
            Embeddings instance
        """
        logger.info(f"Creating {self.provider} embeddings with model: {self.model}")
        
        if self.provider == "openai":
            return self._create_openai_embeddings()
        
        elif self.provider in ["huggingface", "sentence-transformers"]:
            return self._create_huggingface_embeddings()
        
        else:
            raise ValueError(f"Unsupported embeddings provider: {self.provider}")
    
    def _create_openai_embeddings(self):
        """
        Create OpenAI embeddings.
        
        Returns:
            OpenAI embeddings instance
        """
        api_key = get_api_key("openai")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        embeddings = OpenAIEmbeddings(
            model=self.model,
            openai_api_key=api_key
        )
        
        logger.info(f"Created OpenAI embeddings: {self.model}")
        return embeddings
    
    def _create_huggingface_embeddings(self):
        """
        Create HuggingFace embeddings.
        
        Returns:
            HuggingFace embeddings instance
        """
        # Map common model names to HuggingFace model IDs
        model_map = {
            "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
            "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
            "multi-qa-MiniLM-L6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        }
        
        model_name = model_map.get(self.model, self.model)
        
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},  # Use 'cuda' if GPU available
            encode_kwargs={'normalize_embeddings': True}
        )
        
        logger.info(f"Created HuggingFace embeddings: {model_name}")
        return embeddings
    
    def _create_fallback_embeddings(self):
        """
        Create fallback embeddings.
        
        Returns:
            Fallback embeddings instance
        """
        if not self.fallback_provider:
            raise ValueError("Fallback provider not configured")
        
        provider = self.fallback_provider.lower()
        model = self.fallback_model or self.model
        
        logger.info(f"Creating fallback {provider} embeddings with model: {model}")
        
        if provider == "openai":
            api_key = get_api_key("openai")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found for fallback")
            
            embeddings = OpenAIEmbeddings(
                model=model,
                openai_api_key=api_key
            )
            logger.info(f"Created fallback OpenAI embeddings: {model}")
            return embeddings
        
        elif provider in ["huggingface", "sentence-transformers"]:
            model_map = {
                "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
                "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
                "multi-qa-MiniLM-L6-cos-v1": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
            }
            
            model_name = model_map.get(model, model)
            
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(f"Created fallback HuggingFace embeddings: {model_name}")
            return embeddings
        
        else:
            raise ValueError(f"Unsupported fallback embeddings provider: {provider}")
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents
            
        Returns:
            List of embedding vectors
        """
        if not self.embeddings:
            self.embeddings = self._create_embeddings()
        
        logger.debug(f"Embedding {len(texts)} documents")
        
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.debug(f"Successfully embedded {len(embeddings)} documents")
            return embeddings
        
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.
        
        Args:
            text: Query text
            
        Returns:
            Embedding vector
        """
        if not self.embeddings:
            self.embeddings = self._create_embeddings()
        
        logger.debug(f"Embedding query: {text[:50]}...")
        
        try:
            embedding = self.embeddings.embed_query(text)
            logger.debug(f"Successfully embedded query (dimension: {len(embedding)})")
            return embedding
        
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embeddings.
        
        Returns:
            Embedding dimension
        """
        # Test with a sample text to get dimension
        sample_embedding = self.embed_query("test")
        return len(sample_embedding)
    
    def calculate_similarity(
        self,
        text1: str,
        text2: str,
        metric: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            metric: Similarity metric (cosine, euclidean, dot_product)
            
        Returns:
            Similarity score
        """
        import numpy as np
        
        # Get embeddings
        emb1 = np.array(self.embed_query(text1))
        emb2 = np.array(self.embed_query(text2))
        
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        elif metric == "euclidean":
            # Euclidean distance (inverted to similarity)
            distance = np.linalg.norm(emb1 - emb2)
            similarity = 1 / (1 + distance)
        
        elif metric == "dot_product":
            # Dot product
            similarity = np.dot(emb1, emb2)
        
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        logger.debug(f"Similarity ({metric}): {similarity}")
        return float(similarity)
    
    def batch_embed_with_cache(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Embed texts in batches for memory efficiency.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size (uses config default if None)
            
        Returns:
            List of embedding vectors
        """
        batch_size = batch_size or self.config.embeddings.batch_size
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.debug(f"Processing batch {i // batch_size + 1}")
            
            batch_embeddings = self.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)
        
        logger.info("Batch embedding completed")
        return all_embeddings
    
    def get_config(self) -> dict:
        """
        Get embeddings configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.config.embeddings.dimension,
            "batch_size": self.config.embeddings.batch_size
        }
    
    def test_embeddings(self) -> bool:
        """
        Test if embeddings are working correctly.
        
        Returns:
            True if test passes, False otherwise
        """
        try:
            logger.info("Testing embeddings...")
            
            # Test embedding a single query
            test_query = "This is a test query"
            query_embedding = self.embed_query(test_query)
            
            if not query_embedding or len(query_embedding) == 0:
                logger.error("Query embedding test failed")
                return False
            
            # Test embedding documents
            test_docs = ["Test document 1", "Test document 2"]
            doc_embeddings = self.embed_documents(test_docs)
            
            if not doc_embeddings or len(doc_embeddings) != len(test_docs):
                logger.error("Document embedding test failed")
                return False
            
            logger.info(f"Embeddings test passed! Dimension: {len(query_embedding)}")
            return True
        
        except Exception as e:
            logger.error(f"Embeddings test failed with error: {str(e)}")
            return False
