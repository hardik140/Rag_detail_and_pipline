"""Keyword search module using BM25 algorithm."""

import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from loguru import logger

import nltk
from rank_bm25 import BM25Okapi
from langchain.docstore.document import Document


# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)


class BM25Search:
    """
    BM25-based keyword search for document retrieval.
    
    BM25 (Best Matching 25) is a ranking function used for information retrieval.
    It's particularly effective for keyword-based search and serves as a fallback
    when vector search fails or produces poor results.
    """
    
    def __init__(self, config=None):
        """
        Initialize BM25 search.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.documents: List[Document] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self.is_indexed = False
        
        logger.info("BM25 search initialized")
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple tokenization - can be enhanced with stemming/lemmatization
        tokens = nltk.word_tokenize(text.lower())
        
        # Filter out punctuation and very short tokens
        tokens = [
            token for token in tokens 
            if token.isalnum() and len(token) > 2
        ]
        
        return tokens
    
    def index_documents(self, documents: List[Document]):
        """
        Index documents for BM25 search.
        
        Args:
            documents: List of documents to index
        """
        logger.info(f"Indexing {len(documents)} documents for BM25 search...")
        
        self.documents = documents
        
        # Tokenize all documents
        self.tokenized_corpus = [
            self._tokenize(doc.page_content) 
            for doc in documents
        ]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.is_indexed = True
        
        logger.info(f"BM25 index created with {len(documents)} documents")
    
    def search(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None
    ) -> List[Document]:
        """
        Search documents using BM25.
        
        Args:
            query: Search query
            k: Number of documents to return
            score_threshold: Minimum score threshold (optional)
            
        Returns:
            List of relevant documents
        """
        if not self.is_indexed:
            logger.warning("BM25 index not created. No documents to search.")
            return []
        
        logger.debug(f"BM25 search for query: {query[:50]}...")
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            logger.warning("Query tokenization resulted in empty tokens")
            return []
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k documents with scores
        top_indices = scores.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            score = scores[idx]
            
            # Apply score threshold if specified
            if score_threshold and score < score_threshold:
                continue
            
            # Create a copy of the document with score in metadata
            doc = self.documents[idx]
            doc_copy = Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    "bm25_score": float(score),
                    "search_type": "bm25"
                }
            )
            results.append(doc_copy)
        
        logger.debug(f"BM25 search returned {len(results)} documents")
        return results
    
    def search_with_scores(
        self,
        query: str,
        k: int = 4
    ) -> List[tuple]:
        """
        Search documents and return with scores.
        
        Args:
            query: Search query
            k: Number of documents to return
            
        Returns:
            List of (document, score) tuples
        """
        if not self.is_indexed:
            logger.warning("BM25 index not created")
            return []
        
        # Tokenize query
        tokenized_query = self._tokenize(query)
        
        if not tokenized_query:
            return []
        
        # Get scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k
        top_indices = scores.argsort()[-k:][::-1]
        
        results = []
        for idx in top_indices:
            doc = self.documents[idx]
            score = scores[idx]
            results.append((doc, float(score)))
        
        return results
    
    def add_documents(self, documents: List[Document]):
        """
        Add new documents to the BM25 index.
        
        Args:
            documents: List of documents to add
        """
        logger.info(f"Adding {len(documents)} documents to BM25 index...")
        
        self.documents.extend(documents)
        
        # Re-tokenize all documents
        new_tokens = [self._tokenize(doc.page_content) for doc in documents]
        self.tokenized_corpus.extend(new_tokens)
        
        # Recreate BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.is_indexed = True
        
        logger.info(f"BM25 index updated with {len(self.documents)} total documents")
    
    def save_index(self, file_path: str):
        """
        Save BM25 index to disk.
        
        Args:
            file_path: Path to save the index
        """
        logger.info(f"Saving BM25 index to {file_path}...")
        
        index_data = {
            "documents": self.documents,
            "tokenized_corpus": self.tokenized_corpus,
            "bm25": self.bm25,
            "is_indexed": self.is_indexed
        }
        
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info("BM25 index saved successfully")
    
    def load_index(self, file_path: str):
        """
        Load BM25 index from disk.
        
        Args:
            file_path: Path to load the index from
        """
        logger.info(f"Loading BM25 index from {file_path}...")
        
        if not Path(file_path).exists():
            raise FileNotFoundError(f"BM25 index file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.documents = index_data["documents"]
        self.tokenized_corpus = index_data["tokenized_corpus"]
        self.bm25 = index_data["bm25"]
        self.is_indexed = index_data["is_indexed"]
        
        logger.info(f"BM25 index loaded with {len(self.documents)} documents")
    
    def reset(self):
        """Reset the BM25 index."""
        logger.info("Resetting BM25 index...")
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self.is_indexed = False
        logger.info("BM25 index reset complete")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get BM25 search statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "type": "bm25",
            "is_indexed": self.is_indexed,
            "document_count": len(self.documents),
            "corpus_size": len(self.tokenized_corpus)
        }
    
    def get_document_count(self) -> int:
        """
        Get the number of indexed documents.
        
        Returns:
            Number of documents
        """
        return len(self.documents)
