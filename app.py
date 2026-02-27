"""FastAPI application for RAG Pipeline."""

from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import shutil
from loguru import logger

from src.pipeline import RAGPipeline
from src.core.config import config


# Request/Response Models
class IngestRequest(BaseModel):
    """Request model for document ingestion."""
    document_path: str


class QueryRequest(BaseModel):
    """Request model for querying."""
    question: str
    return_sources: bool = True


class QueryResponse(BaseModel):
    """Response model for queries."""
    question: str
    answer: str
    status: str
    sources: Optional[list] = None
    num_sources: Optional[int] = None


class IngestResponse(BaseModel):
    """Response model for ingestion."""
    status: str
    documents_processed: Optional[int] = None
    message: str


class ConversationQueryRequest(BaseModel):
    """Request model for conversation queries."""
    question: str
    session_id: Optional[str] = None
    return_sources: bool = True


class ConversationQueryResponse(BaseModel):
    """Response model for conversation queries."""
    question: str
    answer: str
    status: str
    session_id: str
    message_count: int
    sources: Optional[list] = None
    num_sources: Optional[int] = None


class SessionRequest(BaseModel):
    """Request model for creating a session."""
    session_id: Optional[str] = None


class SessionResponse(BaseModel):
    """Response model for session operations."""
    session_id: str
    status: str
    message: str


# Initialize FastAPI app
app = FastAPI(
    title="RAG Pipeline API",
    description="Retrieval-Augmented Generation Pipeline with multiple LLM providers",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG Pipeline
pipeline: Optional[RAGPipeline] = None


@app.on_event("startup")
async def startup_event():
    """Initialize pipeline on startup."""
    global pipeline
    logger.info("Starting RAG Pipeline API...")
    try:
        pipeline = RAGPipeline()
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down RAG Pipeline API...")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "RAG Pipeline API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_initialized": pipeline is not None
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """
    Ingest documents into the RAG pipeline.
    
    Args:
        request: IngestRequest with document_path
        
    Returns:
        IngestResponse with ingestion results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Ingesting documents from: {request.document_path}")
        result = pipeline.ingest_documents(request.document_path)
        return IngestResponse(**result)
    
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        logger.error(f"Ingestion error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest/upload")
async def upload_and_ingest(file: UploadFile = File(...)):
    """
    Upload and ingest a single document.
    
    Args:
        file: Uploaded file
        
    Returns:
        Ingestion results
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Save uploaded file
        upload_dir = Path(config.paths.documents)
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"File uploaded: {file_path}")
        
        # Ingest the file
        result = pipeline.ingest_documents(str(file_path))
        
        return {
            "status": "success",
            "file": file.filename,
            "ingestion_result": result
        }
    
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_pipeline(request: QueryRequest):
    """
    Query the RAG pipeline.
    
    Args:
        request: QueryRequest with question and options
        
    Returns:
        QueryResponse with answer and sources
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Processing query: {request.question}")
        result = pipeline.query(
            request.question,
            return_sources=request.return_sources
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return QueryResponse(**result)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/similar")
async def get_similar_documents(query: str, k: int = 4):
    """
    Get similar documents without generation.
    
    Args:
        query: Search query
        k: Number of documents to retrieve
        
    Returns:
        List of similar documents
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        logger.info(f"Retrieving similar documents for: {query}")
        results = pipeline.get_similar_documents(query, k)
        
        return {
            "query": query,
            "k": k,
            "results": results,
            "num_results": len(results)
        }
    
    except Exception as e:
        logger.error(f"Similarity search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """
    Get pipeline statistics.
    
    Returns:
        Pipeline statistics
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        stats = pipeline.get_stats()
        return stats
    
    except Exception as e:
        logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset")
async def reset_vector_store():
    """
    Reset/clear the vector store.
    
    Returns:
        Reset confirmation
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        logger.warning("Resetting vector store...")
        pipeline.reset_vector_store()
        
        return {
            "status": "success",
            "message": "Vector store reset successfully"
        }
    
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/config")
async def get_config():
    """
    Get current configuration.
    
    Returns:
        Configuration details
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        return pipeline.get_stats()["config"]
    
    except Exception as e:
        logger.error(f"Config error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Conversation Endpoints ====================

@app.post("/conversation/query", response_model=ConversationQueryResponse)
async def conversation_query(request: ConversationQueryRequest):
    """
    Query with conversation context.
    
    Args:
        request: ConversationQueryRequest with question and session_id
        
    Returns:
        ConversationQueryResponse with answer, sources, and session info
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.conversation_manager:
        raise HTTPException(
            status_code=400, 
            detail="Conversation feature is not enabled in configuration"
        )
    
    try:
        logger.info(f"Processing conversation query: {request.question}")
        result = pipeline.conversation_query(
            request.question,
            session_id=request.session_id,
            return_sources=request.return_sources
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("error"))
        
        return ConversationQueryResponse(**result)
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Conversation query error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/conversation/session", response_model=SessionResponse)
async def create_session(request: SessionRequest):
    """
    Create a new conversation session.
    
    Args:
        request: SessionRequest with optional session_id
        
    Returns:
        SessionResponse with session details
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.conversation_manager:
        raise HTTPException(
            status_code=400, 
            detail="Conversation feature is not enabled in configuration"
        )
    
    try:
        session_id = pipeline.start_conversation(request.session_id)
        
        return SessionResponse(
            session_id=session_id,
            status="success",
            message=f"Session {session_id} created successfully"
        )
    
    except Exception as e:
        logger.error(f"Session creation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/history")
async def get_conversation_history(format: str = "list"):
    """
    Get current conversation history.
    
    Args:
        format: Output format ('list', 'string', 'messages')
        
    Returns:
        Conversation history
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.conversation_manager:
        raise HTTPException(
            status_code=400, 
            detail="Conversation feature is not enabled in configuration"
        )
    
    try:
        history = pipeline.get_conversation_history(format)
        
        return {
            "status": "success",
            "session_id": pipeline.conversation_manager.current_session_id,
            "history": history
        }
    
    except Exception as e:
        logger.error(f"History retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/conversation/sessions")
async def list_sessions():
    """
    List all saved conversation sessions.
    
    Returns:
        List of session metadata
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.conversation_manager:
        raise HTTPException(
            status_code=400, 
            detail="Conversation feature is not enabled in configuration"
        )
    
    try:
        sessions = pipeline.list_conversations()
        
        return {
            "status": "success",
            "total_sessions": len(sessions),
            "sessions": sessions
        }
    
    except Exception as e:
        logger.error(f"Session listing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/conversation/session")
async def end_session(save: bool = True):
    """
    End current conversation session.
    
    Args:
        save: Whether to save the conversation
        
    Returns:
        Confirmation message
    """
    if not pipeline:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    if not pipeline.conversation_manager:
        raise HTTPException(
            status_code=400, 
            detail="Conversation feature is not enabled in configuration"
        )
    
    try:
        session_id = pipeline.conversation_manager.current_session_id
        
        if not session_id:
            raise HTTPException(status_code=400, detail="No active session")
        
        pipeline.end_conversation(save)
        
        return {
            "status": "success",
            "message": f"Session {session_id} ended {'and saved' if save else 'without saving'}"
        }
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Session end error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def main():
    """Run the FastAPI application."""
    uvicorn.run(
        "app:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        workers=config.api.workers
    )


if __name__ == "__main__":
    main()
