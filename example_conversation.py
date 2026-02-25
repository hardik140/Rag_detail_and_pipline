"""
Example usage of conversation features in RAG Pipeline.

This script demonstrates:
1. Starting a conversation session
2. Multi-turn conversations with context
3. Loading previous conversations
4. Session management
5. Conversation history retrieval
"""

from rag_pipeline import RAGPipeline
from config import config
from loguru import logger


def example_1_basic_conversation():
    """Example 1: Basic multi-turn conversation."""
    print("\n" + "="*80)
    print("Example 1: Basic Multi-turn Conversation")
    print("="*80)
    
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Check if conversation is enabled
    if not pipeline.conversation_manager:
        print("⚠️  Conversation feature is not enabled in config.yaml")
        print("Enable it by setting conversation.enabled: true")
        return
    
    # Ingest some documents first (if needed)
    # pipeline.ingest_documents("./data/documents")
    
    # Start a conversation
    session_id = pipeline.start_conversation()
    print(f"\n✓ Started conversation session: {session_id}")
    
    # First question
    print("\n" + "-"*80)
    print("Question 1: What is RAG?")
    result = pipeline.conversation_query(
        "What is RAG?",
        return_sources=True
    )
    print(f"\nAnswer: {result['answer'][:300]}...")
    print(f"Session: {result['session_id']}")
    print(f"Messages in session: {result['message_count']}")
    
    # Follow-up question (uses previous context)
    print("\n" + "-"*80)
    print("Question 2: How does it work? (follow-up)")
    result = pipeline.conversation_query(
        "How does it work?",
        return_sources=True
    )
    print(f"\nAnswer: {result['answer'][:300]}...")
    print(f"Messages in session: {result['message_count']}")
    
    # Another follow-up
    print("\n" + "-"*80)
    print("Question 3: What are its advantages? (follow-up)")
    result = pipeline.conversation_query(
        "What are its advantages?",
        return_sources=True
    )
    print(f"\nAnswer: {result['answer'][:300]}...")
    print(f"Messages in session: {result['message_count']}")
    
    # Get conversation history
    print("\n" + "-"*80)
    print("Conversation History:")
    history = pipeline.get_conversation_history(format="list")
    for i, msg in enumerate(history, 1):
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        print(f"\n{role_icon} {msg['role'].upper()}: {msg['content'][:100]}...")
    
    # End and save the conversation
    pipeline.end_conversation(save=True)
    print(f"\n✓ Conversation saved")


def example_2_session_management():
    """Example 2: Managing multiple sessions."""
    print("\n" + "="*80)
    print("Example 2: Session Management")
    print("="*80)
    
    pipeline = RAGPipeline()
    
    if not pipeline.conversation_manager:
        print("⚠️  Conversation feature is not enabled")
        return
    
    # Create a custom session
    session_id = pipeline.start_conversation("my_custom_session_001")
    print(f"\n✓ Created custom session: {session_id}")
    
    # Ask a few questions
    pipeline.conversation_query("Tell me about vector databases")
    pipeline.conversation_query("Which one is best for production?")
    
    # End the session
    pipeline.end_conversation(save=True)
    print(f"✓ Session ended and saved")
    
    # List all sessions
    print("\n" + "-"*80)
    print("All Saved Sessions:")
    sessions = pipeline.list_conversations()
    for session in sessions:
        print(f"\n  Session: {session['session_id']}")
        print(f"  Messages: {session['message_count']}")
        print(f"  Started: {session.get('start_time', 'Unknown')}")


def example_3_resume_conversation():
    """Example 3: Resume a previous conversation."""
    print("\n" + "="*80)
    print("Example 3: Resume Previous Conversation")
    print("="*80)
    
    pipeline = RAGPipeline()
    
    if not pipeline.conversation_manager:
        print("⚠️  Conversation feature is not enabled")
        return
    
    # List available sessions
    sessions = pipeline.list_conversations()
    
    if not sessions:
        print("⚠️  No saved sessions found. Run example_1 or example_2 first.")
        return
    
    # Load the most recent session
    latest_session = sessions[0]
    session_id = latest_session['session_id']
    
    print(f"\n✓ Resuming session: {session_id}")
    print(f"  Original messages: {latest_session['message_count']}")
    
    # Continue the conversation
    result = pipeline.conversation_query(
        "Can you summarize what we discussed?",
        session_id=session_id,
        return_sources=False
    )
    
    print(f"\nAnswer: {result['answer'][:300]}...")
    print(f"Total messages now: {result['message_count']}")
    
    # Save and end
    pipeline.end_conversation(save=True)
    print(f"\n✓ Updated conversation saved")


def example_4_context_awareness():
    """Example 4: Demonstrate context awareness."""
    print("\n" + "="*80)
    print("Example 4: Context-Aware Conversations")
    print("="*80)
    
    pipeline = RAGPipeline()
    
    if not pipeline.conversation_manager:
        print("⚠️  Conversation feature is not enabled")
        return
    
    # Start new session
    session_id = pipeline.start_conversation()
    print(f"\n✓ Started session: {session_id}")
    
    # Progressive conversation with references
    questions = [
        "What is ChromaDB?",
        "What are its main features?",  # "its" refers to ChromaDB
        "How does it compare to FAISS?",  # "it" refers to ChromaDB
        "Which one should I use for a small project?"  # References both
    ]
    
    for i, question in enumerate(questions, 1):
        print(f"\n{'-'*80}")
        print(f"Q{i}: {question}")
        
        result = pipeline.conversation_query(question, return_sources=False)
        
        print(f"A{i}: {result['answer'][:200]}...")
        print(f"    [Context: {result['message_count']} messages]")
    
    # Show full history
    print(f"\n{'-'*80}")
    print("Full Conversation Context:")
    history_str = pipeline.get_conversation_history(format="string")
    print(history_str[:500] + "...")
    
    pipeline.end_conversation(save=True)


def example_5_api_usage():
    """Example 5: Using conversation via API endpoints."""
    print("\n" + "="*80)
    print("Example 5: API Usage for Conversations")
    print("="*80)
    
    print("""
To use the conversation feature via API:

1. Start a new session:
   POST http://localhost:8000/conversation/session
   Body: {"session_id": "optional_custom_id"}

2. Ask questions with context:
   POST http://localhost:8000/conversation/query
   Body: {
       "question": "What is RAG?",
       "session_id": "session_20240101_120000",  # Optional
       "return_sources": true
   }

3. Get conversation history:
   GET http://localhost:8000/conversation/history?format=list

4. List all sessions:
   GET http://localhost:8000/conversation/sessions

5. End session:
   DELETE http://localhost:8000/conversation/session?save=true

Example using curl:

# Create session
curl -X POST http://localhost:8000/conversation/session \\
  -H "Content-Type: application/json" \\
  -d '{"session_id": "my_session"}'

# Ask question
curl -X POST http://localhost:8000/conversation/query \\
  -H "Content-Type: application/json" \\
  -d '{
    "question": "What is RAG?",
    "session_id": "my_session",
    "return_sources": true
  }'

# Get history
curl http://localhost:8000/conversation/history

# List sessions
curl http://localhost:8000/conversation/sessions

# End session
curl -X DELETE "http://localhost:8000/conversation/session?save=true"

Example using Python requests:

```python
import requests

base_url = "http://localhost:8000"

# Create session
response = requests.post(f"{base_url}/conversation/session")
session_id = response.json()["session_id"]
print(f"Session created: {session_id}")

# Ask questions
questions = [
    "What is RAG?",
    "How does it work?",
    "What are its benefits?"
]

for question in questions:
    response = requests.post(
        f"{base_url}/conversation/query",
        json={
            "question": question,
            "session_id": session_id,
            "return_sources": True
        }
    )
    result = response.json()
    print(f"\\nQ: {question}")
    print(f"A: {result['answer'][:200]}...")

# Get history
response = requests.get(f"{base_url}/conversation/history")
history = response.json()
print(f"\\nTotal messages: {len(history['history'])}")

# List all sessions
response = requests.get(f"{base_url}/conversation/sessions")
sessions = response.json()
print(f"\\nTotal sessions: {sessions['total_sessions']}")

# End session
response = requests.delete(f"{base_url}/conversation/session?save=true")
print(f"\\nSession ended: {response.json()['message']}")
```
    """)


def example_6_troubleshooting():
    """Example 6: Common issues and solutions."""
    print("\n" + "="*80)
    print("Example 6: Troubleshooting & Best Practices")
    print("="*80)
    
    print("""
Common Issues and Solutions:

1. Conversation feature not enabled:
   ❌ Error: "Conversation feature is not enabled"
   ✓ Solution: Set conversation.enabled: true in config.yaml

2. Persist directory not found:
   ❌ Error: "Directory not found"
   ✓ Solution: Set conversation.persist_directory in config.yaml
   ✓ Default: "./data/conversations"

3. Session timeout:
   ❌ Issue: Session automatically resets
   ✓ Solution: Increase conversation.session_timeout_minutes
   ✓ Default: 60 minutes

4. Too much context in queries:
   ❌ Issue: Slow responses or token limit errors
   ✓ Solution: Reduce conversation.max_history
   ✓ Default: 10 message pairs (20 messages total)

5. Conversations not saving:
   ❌ Issue: Lost conversation history
   ✓ Solution: Enable conversation.auto_save: true
   ✓ Or call pipeline.end_conversation(save=True) explicitly

Best Practices:

✓ Start a new session for different topics
✓ Use descriptive session IDs for important conversations
✓ Regularly save long conversations
✓ Clean up old sessions periodically
✓ Use max_history to control context window size
✓ Enable auto_save for production applications
✓ Monitor persist_directory disk usage

Configuration Example (config.yaml):

conversation:
  enabled: true
  persist_directory: "./data/conversations"
  max_history: 10
  include_in_context: true
  collection_name: "conversation_history"
  memory_type: "buffer"
  session_timeout_minutes: 60
  auto_save: true

Memory Types:

- buffer: Keeps all messages in memory (default, simple)
- vector: Stores in ChromaDB for semantic search (advanced)
- summary: Summarizes old messages to save context (future)

Performance Tips:

- Use buffer memory for short conversations (<20 messages)
- Use vector memory for long conversations or history search
- Set max_history based on your LLM's context window
- Enable auto_save to prevent data loss
- Increase session_timeout for long-running sessions
    """)


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("RAG Pipeline - Conversation Feature Examples")
    print("="*80)
    
    examples = [
        ("Basic Conversation", example_1_basic_conversation),
        ("Session Management", example_2_session_management),
        ("Resume Conversation", example_3_resume_conversation),
        ("Context Awareness", example_4_context_awareness),
        ("API Usage", example_5_api_usage),
        ("Troubleshooting", example_6_troubleshooting),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")
    
    print("\nRunning all examples...")
    print("(To run one example, uncomment the specific function call)")
    
    # Run all examples
    for name, func in examples:
        try:
            func()
        except Exception as e:
            logger.error(f"Error in {name}: {str(e)}")
    
    # Or run specific examples:
    # example_1_basic_conversation()
    # example_2_session_management()
    # example_3_resume_conversation()
    # example_4_context_awareness()
    # example_5_api_usage()
    # example_6_troubleshooting()


if __name__ == "__main__":
    main()
