"""Query engine module for RAG pipeline."""

from typing import Dict, Any, Optional, List
from loguru import logger

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_community.llms import Anthropic, Cohere
from langchain_google_genai import ChatGoogleGenerativeAI

from ..core.config import get_api_key


class QueryEngine:
    """
    Query engine for question answering using RAG.
    
    Supports multiple LLM providers:
    - OpenAI
    - Anthropic
    - Cohere
    """
    
    # Default prompt template
    DEFAULT_PROMPT_TEMPLATE = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:"""
    
    def __init__(self, config, vector_store, hybrid_retriever=None):
        """
        Initialize query engine.
        
        Args:
            config: Configuration object
            vector_store: Vector store instance
            hybrid_retriever: Optional hybrid retriever (for vector + BM25 search)
        """
        self.config = config
        self.vector_store = vector_store
        self.hybrid_retriever = hybrid_retriever
        self.llm = self._initialize_llm()
        self.retriever = self._initialize_retriever()
        self.qa_chain = None
        
        search_mode = "hybrid" if hybrid_retriever else "vector"
        logger.info(f"Query engine initialized with {config.llm.provider} LLM ({search_mode} search)")
    
    def _initialize_llm(self):
        """
        Initialize LLM based on configuration.
        
        Returns:
            LLM instance
        """
        provider = self.config.llm.provider.lower()
        model = self.config.llm.model
        temperature = self.config.llm.temperature
        max_tokens = self.config.llm.max_tokens
        
        logger.info(f"Initializing {provider} LLM: {model}")
        
        if provider == "openai":
            api_key = get_api_key("openai")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment variables")
            
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=api_key
            )
        
        elif provider == "anthropic":
            api_key = get_api_key("anthropic")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
            
            return Anthropic(
                model=model,
                temperature=temperature,
                max_tokens_to_sample=max_tokens,
                anthropic_api_key=api_key
            )
        
        elif provider == "cohere":
            api_key = get_api_key("cohere")
            if not api_key:
                raise ValueError("COHERE_API_KEY not found in environment variables")
            
            return Cohere(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                cohere_api_key=api_key
            )
        
        elif provider == "gemini" or provider == "google":
            api_key = get_api_key("gemini")
            if not api_key:
                raise ValueError("GOOGLE_API_KEY not found in environment variables")
            
            return ChatGoogleGenerativeAI(
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens,
                google_api_key=api_key
            )
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def _initialize_retriever(self):
        """
        Initialize document retriever.
        
        Returns:
            Retriever instance
        """
        # Use hybrid retriever if available
        if self.hybrid_retriever:
            logger.debug("Using hybrid retriever (vector + BM25 with fallback)")
            return self.hybrid_retriever
        
        # Otherwise use standard vector store retriever
        search_type = self.config.retrieval.search_type
        k = self.config.retrieval.k
        
        search_kwargs = {"k": k}
        
        # Add search type specific parameters
        if search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.config.retrieval.score_threshold
        
        elif search_type == "mmr":
            mmr_config = self.config.retrieval.get("mmr", {})
            search_kwargs["fetch_k"] = mmr_config.get("fetch_k", 20)
            search_kwargs["lambda_mult"] = mmr_config.get("lambda_mult", 0.5)
        
        logger.debug(f"Initializing vector retriever with search_type={search_type}, kwargs={search_kwargs}")
        
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs
        )
    
    def _create_qa_chain(
        self,
        prompt_template: Optional[str] = None,
        chain_type: Optional[str] = None
    ) -> RetrievalQA:
        """
        Create QA chain.
        
        Args:
            prompt_template: Custom prompt template
            chain_type: Type of chain to use
            
        Returns:
            RetrievalQA chain
        """
        chain_type = chain_type or self.config.rag_chain.chain_type
        
        if prompt_template:
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            chain_type_kwargs = {"prompt": PROMPT}
        else:
            chain_type_kwargs = {}
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=self.retriever,
            return_source_documents=self.config.rag_chain.return_source_documents,
            chain_type_kwargs=chain_type_kwargs,
            verbose=self.config.rag_chain.verbose
        )
        
        return qa_chain
    
    def query(
        self,
        question: str,
        return_sources: bool = True,
        custom_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: Question to answer
            return_sources: Whether to return source documents
            custom_prompt: Optional custom prompt template
            
        Returns:
            Dictionary with answer and optional source documents
        """
        logger.info(f"Processing query: {question}")
        
        try:
            # Create or recreate chain if custom prompt is provided
            if custom_prompt or not self.qa_chain:
                self.qa_chain = self._create_qa_chain(prompt_template=custom_prompt)
            
            # Execute query
            result = self.qa_chain.invoke({"query": question})
            
            # Format response
            response = {
                "question": question,
                "answer": result.get("result", ""),
                "status": "success"
            }
            
            # Add source documents if requested
            if return_sources and "source_documents" in result:
                sources = []
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                response["sources"] = sources
                response["num_sources"] = len(sources)
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "question": question,
                "answer": "",
                "status": "error",
                "error": str(e)
            }
    
    def batch_query(
        self,
        questions: List[str],
        return_sources: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            questions: List of questions
            return_sources: Whether to return source documents
            
        Returns:
            List of response dictionaries
        """
        logger.info(f"Processing batch of {len(questions)} queries")
        
        results = []
        for question in questions:
            result = self.query(question, return_sources)
            results.append(result)
        
        logger.info("Batch processing completed")
        return results
    
    def retrieve_documents(
        self,
        query: str,
        k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents without generation.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of retrieved documents
        """
        logger.info(f"Retrieving documents for query: {query}")
        
        try:
            if k:
                # Temporarily override k
                old_k = self.retriever.search_kwargs.get("k")
                self.retriever.search_kwargs["k"] = k
            
            documents = self.retriever.get_relevant_documents(query)
            
            if k:
                # Restore original k
                self.retriever.search_kwargs["k"] = old_k
            
            results = []
            for doc in documents:
                results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Retrieved {len(results)} documents")
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    def update_llm_config(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Update LLM configuration dynamically.
        
        Args:
            temperature: New temperature setting
            max_tokens: New max tokens setting
        """
        if temperature is not None:
            self.llm.temperature = temperature
            logger.info(f"Updated LLM temperature to {temperature}")
        
        if max_tokens is not None:
            self.llm.max_tokens = max_tokens
            logger.info(f"Updated LLM max_tokens to {max_tokens}")
        
        # Recreate chain with updated LLM
        self.qa_chain = self._create_qa_chain()
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current query engine configuration.
        
        Returns:
            Configuration dictionary
        """
        return {
            "llm_provider": self.config.llm.provider,
            "llm_model": self.config.llm.model,
            "temperature": self.llm.temperature,
            "max_tokens": getattr(self.llm, 'max_tokens', None),
            "search_type": self.config.retrieval.search_type,
            "retrieval_k": self.config.retrieval.k,
            "chain_type": self.config.rag_chain.chain_type
        }
