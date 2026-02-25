"""Configuration management for RAG Pipeline."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100
    enable_fallback: bool = False
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    type: str = "chroma"
    collection_name: str = "rag_documents"
    persist_directory: str = "./data/chroma_db"
    similarity_metric: str = "cosine"


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = ["\n\n", "\n", " ", ""]
    supported_formats: list = ["pdf", "docx", "txt", "md", "html"]


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    search_type: str = "similarity"
    k: int = 4
    score_threshold: float = 0.7


class HybridSearchConfig(BaseModel):
    """Hybrid search configuration."""
    enabled: bool = True
    enable_fallback: bool = True
    enable_fusion: bool = False
    min_vector_results: int = 2
    fusion_weight_vector: float = 0.7
    fusion_weight_bm25: float = 0.3


class RAGChainConfig(BaseModel):
    """RAG chain configuration."""
    chain_type: str = "stuff"
    return_source_documents: bool = True
    verbose: bool = True


class ConversationConfig(BaseModel):
    """Conversation memory configuration."""
    enabled: bool = True
    persist_directory: str = "./data/conversations"
    max_history: int = 10
    include_in_context: bool = True
    collection_name: str = "conversation_history"
    memory_type: str = "buffer"
    session_timeout_minutes: int = 60
    auto_save: bool = True


class PathsConfig(BaseModel):
    """Paths configuration."""
    documents: str = "./data/documents"
    processed: str = "./data/processed"
    cache: str = "./data/cache"
    logs: str = "./logs"


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    workers: int = 1


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    rotation: str = "500 MB"
    retention: str = "10 days"


class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig
    embeddings: EmbeddingsConfig
    vector_store: VectorStoreConfig
    document_processing: DocumentProcessingConfig
    retrieval: RetrievalConfig
    hybrid_search: Optional[HybridSearchConfig] = None
    rag_chain: RAGChainConfig
    conversation: Optional[ConversationConfig] = None
    paths: PathsConfig
    api: APIConfig
    logging: LoggingConfig


def load_config(config_path: str = "config.yaml") -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Config object
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(**config_dict)


def get_api_key(provider: str) -> Optional[str]:
    """
    Get API key for specified provider from environment variables.
    
    Args:
        provider: Provider name (openai, anthropic, cohere, pinecone, google, gemini)
        
    Returns:
        API key or None
    """
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "cohere": "COHERE_API_KEY",
        "pinecone": "PINECONE_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY"
    }
    
    env_var = key_map.get(provider.lower())
    if env_var:
        return os.getenv(env_var)
    
    return None


# Load default configuration
try:
    config = load_config()
except FileNotFoundError:
    # Use default values if config file doesn't exist
    config = Config(
        llm=LLMConfig(),
        embeddings=EmbeddingsConfig(),
        vector_store=VectorStoreConfig(),
        document_processing=DocumentProcessingConfig(),
        retrieval=RetrievalConfig(),
        hybrid_search=HybridSearchConfig(),
        rag_chain=RAGChainConfig(),
        conversation=ConversationConfig(),
        paths=PathsConfig(),
        api=APIConfig(),
        logging=LoggingConfig()
    )
