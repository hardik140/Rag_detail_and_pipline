"""Conversation memory management for RAG pipeline."""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from loguru import logger

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document


class ConversationManager:
    """
    Manages conversation history with ChromaDB persistence.
    
    Features:
    - Store conversations in ChromaDB for semantic search
    - Persist conversations to disk
    - Session management
    - Multiple memory types (buffer, summary, vector)
    - Conversation retrieval for context
    """
    
    def __init__(self, config, embeddings=None):
        """
        Initialize conversation manager.
        
        Args:
            config: Configuration object
            embeddings: Embeddings instance for vector storage
        """
        self.config = config
        self.embeddings = embeddings
        
        # Get conversation config
        conv_config = getattr(config, 'conversation', None)
        if conv_config:
            self.enabled = conv_config.enabled
            self.persist_dir = conv_config.persist_directory
            self.max_history = conv_config.max_history
            self.include_in_context = conv_config.include_in_context
            self.collection_name = conv_config.collection_name
            self.memory_type = conv_config.memory_type
            self.session_timeout = conv_config.session_timeout_minutes
            self.auto_save = conv_config.auto_save
        else:
            # Defaults
            self.enabled = True
            self.persist_dir = "./data/conversations"
            self.max_history = 10
            self.include_in_context = True
            self.collection_name = "conversation_history"
            self.memory_type = "buffer"
            self.session_timeout = 60
            self.auto_save = True
        
        # Create persist directory
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        
        # Session data
        self.current_session_id: Optional[str] = None
        self.current_history: List[Dict[str, str]] = []
        self.session_start_time: Optional[datetime] = None
        self.last_activity_time: Optional[datetime] = None
        
        # Vector store for conversation history
        self.conversation_vector_store: Optional[Chroma] = None
        
        logger.info(f"Conversation manager initialized (enabled: {self.enabled})")
    
    def start_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new conversation session.
        
        Args:
            session_id: Optional custom session ID
            
        Returns:
            Session ID
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.current_session_id = session_id
        self.current_history = []
        self.session_start_time = datetime.now()
        self.last_activity_time = datetime.now()
        
        logger.info(f"Started conversation session: {session_id}")
        return session_id
    
    def end_session(self, save: bool = True):
        """
        End current conversation session.
        
        Args:
            save: Whether to save the conversation to disk
        """
        if not self.current_session_id:
            logger.warning("No active session to end")
            return
        
        if save and self.auto_save and self.current_history:
            self.save_conversation()
        
        logger.info(f"Ended conversation session: {self.current_session_id}")
        
        self.current_session_id = None
        self.current_history = []
        self.session_start_time = None
        self.last_activity_time = None
    
    def _check_session_timeout(self) -> bool:
        """
        Check if current session has timed out.
        
        Returns:
            True if session timed out, False otherwise
        """
        if not self.last_activity_time:
            return False
        
        timeout_delta = timedelta(minutes=self.session_timeout)
        if datetime.now() - self.last_activity_time > timeout_delta:
            logger.warning(f"Session {self.current_session_id} timed out")
            return True
        
        return False
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        """
        Add a message to the current conversation.
        
        Args:
            role: Message role ('user' or 'assistant')
            content: Message content
            metadata: Optional metadata (sources, etc.)
        """
        if not self.current_session_id:
            self.start_session()
        
        # Check timeout
        if self._check_session_timeout():
            self.end_session()
            self.start_session()
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.current_history.append(message)
        self.last_activity_time = datetime.now()
        
        # Trim history if needed
        if len(self.current_history) > self.max_history * 2:  # *2 for user+assistant pairs
            self.current_history = self.current_history[-(self.max_history * 2):]
        
        logger.debug(f"Added {role} message to conversation (session: {self.current_session_id})")
    
    def get_conversation_history(self, format: str = "list") -> Any:
        """
        Get current conversation history.
        
        Args:
            format: Output format ('list', 'string', 'messages')
            
        Returns:
            Conversation history in requested format
        """
        if format == "list":
            return self.current_history
        
        elif format == "string":
            history_str = ""
            for msg in self.current_history:
                role = "Human" if msg["role"] == "user" else "Assistant"
                history_str += f"{role}: {msg['content']}\n\n"
            return history_str.strip()
        
        elif format == "messages":
            messages = []
            for msg in self.current_history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                else:
                    messages.append(AIMessage(content=msg["content"]))
            return messages
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_context_window(self, max_messages: Optional[int] = None) -> List[Dict]:
        """
        Get recent conversation messages for context.
        
        Args:
            max_messages: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        max_msgs = max_messages or self.max_history * 2
        return self.current_history[-max_msgs:]
    
    def save_conversation(self):
        """Save current conversation to disk."""
        if not self.current_session_id or not self.current_history:
            logger.warning("No conversation to save")
            return
        
        # Create session file
        session_file = Path(self.persist_dir) / f"{self.current_session_id}.json"
        
        conversation_data = {
            "session_id": self.current_session_id,
            "start_time": self.session_start_time.isoformat() if self.session_start_time else None,
            "end_time": datetime.now().isoformat(),
            "messages": self.current_history,
            "metadata": {
                "message_count": len(self.current_history),
                "duration_minutes": (datetime.now() - self.session_start_time).total_seconds() / 60 if self.session_start_time else 0
            }
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(conversation_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved conversation to: {session_file}")
        
        # Also store in vector database if enabled
        if self.embeddings and self.memory_type == "vector":
            self._store_in_vector_db()
    
    def _store_in_vector_db(self):
        """Store conversation in ChromaDB for semantic search."""
        if not self.embeddings:
            logger.warning("No embeddings available for vector storage")
            return
        
        try:
            # Initialize vector store if not exists
            if not self.conversation_vector_store:
                chroma_persist_dir = Path(self.persist_dir) / "chroma_conversations"
                chroma_persist_dir.mkdir(parents=True, exist_ok=True)
                
                self.conversation_vector_store = Chroma(
                    collection_name=self.collection_name,
                    embedding_function=self.embeddings,
                    persist_directory=str(chroma_persist_dir)
                )
            
            # Create documents from conversation
            documents = []
            for msg in self.current_history:
                doc = Document(
                    page_content=msg["content"],
                    metadata={
                        "session_id": self.current_session_id,
                        "role": msg["role"],
                        "timestamp": msg["timestamp"],
                        "type": "conversation"
                    }
                )
                documents.append(doc)
            
            # Add to vector store
            self.conversation_vector_store.add_documents(documents)
            logger.info(f"Stored {len(documents)} conversation messages in ChromaDB")
            
        except Exception as e:
            logger.error(f"Error storing conversation in vector DB: {str(e)}")
    
    def load_conversation(self, session_id: str) -> bool:
        """
        Load a previous conversation.
        
        Args:
            session_id: Session ID to load
            
        Returns:
            True if loaded successfully, False otherwise
        """
        session_file = Path(self.persist_dir) / f"{session_id}.json"
        
        if not session_file.exists():
            logger.warning(f"Conversation file not found: {session_file}")
            return False
        
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)
            
            self.current_session_id = conversation_data["session_id"]
            self.current_history = conversation_data["messages"]
            self.session_start_time = datetime.fromisoformat(conversation_data["start_time"]) if conversation_data.get("start_time") else None
            self.last_activity_time = datetime.now()
            
            logger.info(f"Loaded conversation: {session_id} ({len(self.current_history)} messages)")
            return True
            
        except Exception as e:
            logger.error(f"Error loading conversation: {str(e)}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all saved conversation sessions.
        
        Returns:
            List of session metadata
        """
        sessions = []
        persist_path = Path(self.persist_dir)
        
        for session_file in persist_path.glob("session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                sessions.append({
                    "session_id": data["session_id"],
                    "start_time": data.get("start_time"),
                    "end_time": data.get("end_time"),
                    "message_count": len(data.get("messages", [])),
                    "file": str(session_file)
                })
            except Exception as e:
                logger.warning(f"Error reading session file {session_file}: {str(e)}")
                continue
        
        # Sort by start time (newest first)
        sessions.sort(key=lambda x: x.get("start_time", ""), reverse=True)
        
        return sessions
    
    def search_conversations(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search past conversations semantically.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant conversation snippets
        """
        if not self.conversation_vector_store:
            logger.warning("Conversation vector store not initialized")
            return []
        
        try:
            results = self.conversation_vector_store.similarity_search(query, k=k)
            
            formatted_results = []
            for doc in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "session_id": doc.metadata.get("session_id"),
                    "role": doc.metadata.get("role"),
                    "timestamp": doc.metadata.get("timestamp")
                })
            
            logger.info(f"Found {len(formatted_results)} relevant conversation snippets")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching conversations: {str(e)}")
            return []
    
    def clear_history(self):
        """Clear current conversation history."""
        self.current_history = []
        logger.info("Cleared conversation history")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "enabled": self.enabled,
            "current_session": self.current_session_id,
            "messages_in_session": len(self.current_history),
            "max_history": self.max_history,
            "memory_type": self.memory_type,
            "total_saved_sessions": len(list(Path(self.persist_dir).glob("session_*.json"))),
            "persist_directory": self.persist_dir
        }
