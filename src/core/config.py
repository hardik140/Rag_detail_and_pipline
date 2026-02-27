"""Simplified configuration management for RAG Pipeline."""

import os
import yaml
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class LLMConfig(BaseModel):
    """LLM configuration."""
    provider: str = "openai"
    model: str = "gpt-4-turbo-preview"
    temperature: float = 0.7
    max_tokens: int = 2000


class EmbeddingsConfig(BaseModel):
    """Embeddings configuration."""
    provider: str = "openai"
    model: str = "text-embedding-3-small"
    dimension: int = 1536
    batch_size: int = 100


class VectorStoreConfig(BaseModel):
    """Vector store configuration."""
    type: str = "chroma"
    collection_name: str = "rag_documents"
    persist_directory: str = "./data/chroma_db"


class DocumentProcessingConfig(BaseModel):
    """Document processing configuration."""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list = ["\n\n", "\n", " ", ""]


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""
    search_type: str = "similarity"
    k: int = 4
    score_threshold: float = 0.7


class HybridSearchConfig(BaseModel):
    """Hybrid search configuration."""
    enabled: bool = True
    min_vector_results: int = 2


class ConversationConfig(BaseModel):
    """Conversation memory configuration."""
    enabled: bool = True
    persist_directory: str = "./data/conversations"
    max_history: int = 10


class PathsConfig(BaseModel):
    """Paths configuration."""
    documents: str = "./data/documents"
    processed: str = "./data/processed"
    logs: str = "./logs"


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
    rotation: str = "500 MB"
    retention: str = "10 days"


class Config(BaseModel):
    """Main configuration class."""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    document_processing: DocumentProcessingConfig = Field(default_factory=DocumentProcessingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    hybrid_search: HybridSearchConfig = Field(default_factory=HybridSearchConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


def load_config(config_path: str = "config.yaml") -> Config:
    """Load configuration from YAML file or use defaults."""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        return Config(**config_dict)
    
    return Config()


def get_api_key(provider: str) -> Optional[str]:
    """Get API key for specified provider."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "cohere": "COHERE_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "pinecone": "PINECONE_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY"
    }
    
    env_var = key_map.get(provider.lower())
    return os.getenv(env_var) if env_var else None


# Default configuration instance
config = load_config()
