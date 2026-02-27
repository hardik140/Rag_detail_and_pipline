"""Document loading and processing module."""

import os
from pathlib import Path
from typing import List, Optional
from loguru import logger

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    DirectoryLoader
)


class DocumentLoader:
    """
    Document loader for RAG pipeline.
    
    Supports multiple document formats:
    - PDF
    - DOCX
    - TXT
    - Markdown
    - HTML
    """
    
    LOADERS = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".md": UnstructuredMarkdownLoader,
        ".html": UnstructuredHTMLLoader,
    }
    
    def __init__(self, config):
        """
        Initialize document loader.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.text_splitter = self._create_text_splitter()
        
    def _create_text_splitter(self) -> RecursiveCharacterTextSplitter:
        """
        Create text splitter based on configuration.
        
        Returns:
            Text splitter instance
        """
        doc_config = self.config.document_processing
        
        return RecursiveCharacterTextSplitter(
            chunk_size=doc_config.chunk_size,
            chunk_overlap=doc_config.chunk_overlap,
            separators=doc_config.separators,
            length_function=len,
        )
    
    def load_single_document(self, file_path: str) -> List[Document]:
        """
        Load a single document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of document chunks
            
        Raises:
            ValueError: If file format is not supported
        """
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension not in self.LOADERS:
            raise ValueError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {list(self.LOADERS.keys())}"
            )
        
        logger.info(f"Loading document: {file_path}")
        
        try:
            loader_class = self.LOADERS[file_extension]
            loader = loader_class(file_path)
            documents = loader.load()
            
            logger.info(f"Successfully loaded {len(documents)} pages/sections from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of all document chunks
        """
        logger.info(f"Loading documents from directory: {directory_path}")
        
        all_documents = []
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Load each supported file type
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.LOADERS:
                try:
                    documents = self.load_single_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logger.warning(f"Skipping {file_path} due to error: {str(e)}")
                    continue
        
        logger.info(f"Loaded {len(all_documents)} total documents from directory")
        return all_documents
    
    def load_documents(self, path: str) -> List[Document]:
        """
        Load documents from a file or directory.
        
        Args:
            path: Path to file or directory
            
        Returns:
            List of processed document chunks
        """
        path_obj = Path(path)
        
        if not path_obj.exists():
            raise FileNotFoundError(f"Path not found: {path}")
        
        # Load documents
        if path_obj.is_file():
            documents = self.load_single_document(path)
        else:
            documents = self.load_directory(path)
        
        # Split documents into chunks
        logger.info(f"Splitting {len(documents)} documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Created {len(chunks)} chunks from documents")
        
        return chunks
    
    def load_from_text(self, text: str, metadata: Optional[dict] = None) -> List[Document]:
        """
        Load document from raw text.
        
        Args:
            text: Raw text content
            metadata: Optional metadata for the document
            
        Returns:
            List of document chunks
        """
        logger.info("Loading document from raw text")
        
        document = Document(
            page_content=text,
            metadata=metadata or {}
        )
        
        chunks = self.text_splitter.split_documents([document])
        logger.info(f"Created {len(chunks)} chunks from text")
        
        return chunks
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return list(self.LOADERS.keys())
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if a file is supported and readable.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid, False otherwise
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return False
        
        if not path.is_file():
            logger.warning(f"Path is not a file: {file_path}")
            return False
        
        if path.suffix.lower() not in self.LOADERS:
            logger.warning(f"Unsupported file format: {path.suffix}")
            return False
        
        return True
