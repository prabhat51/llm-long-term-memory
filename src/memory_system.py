import os
import numpy as np
from typing import List, Dict, Any, Optional

from .storage import MemoryStorage
from .extractor import MemoryExtractor
from .retriever import MemoryRetriever
from .openai_client import OpenAIClient

class MemorySystem:
    """Main class that ties all memory components together."""
    
    def __init__(self, api_key: Optional[str] = None, db_path: str = "memories.db"):
        """
        Initialize the memory system.
        
        Args:
            api_key: OpenAI API key. If not provided, will try to get from environment variable.
            db_path: Path to the SQLite database file.
        """
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set it as a parameter or OPENAI_API_KEY environment variable.")
        
        # Initialize components
        self.storage = MemoryStorage(db_path)
        self.openai_client = OpenAIClient(self.api_key)
        self.extractor = MemoryExtractor(self.openai_client)
        self.retriever = MemoryRetriever(self.openai_client)
    
    def process_conversation(self, conversation: List[Dict[str, str]], 
                           extract_memories: bool = True, 
                           check_for_deletions: bool = True) -> Dict[str, Any]:
        """
        Process a conversation to extract new memories and check for deletions.
        
        Args:
            conversation: A list of message dictionaries with 'role' and 'content' keys.
            extract_memories: Whether to extract new memories from the conversation.
            check_for_deletions: Whether to check for memories to delete.
            
        Returns:
            A dictionary containing the results of the processing.
        """
        result = {
            'new_memories': [],
            'deleted_memories': [],
            'relevant_memories': []
        }
        
        # Extract new memories if requested
        if extract_memories:
            memories = self.extractor.extract_memories(conversation)
            
            for memory in memories:
                # Get embedding for the memory content
                embedding = self.openai_client.get_embedding(memory['content'])
                
                # Store the memory
                memory_id = self.storage.add_memory(
                    content=memory['content'],
                    embedding=embedding,
                    metadata={
                        'importance': memory.get('importance', 5),
                        'category': memory.get('category', 'general'),
                        'entities': memory.get('entities', [])
                    }
                )
                
                result['new_memories'].append({
                    'id': memory_id,
                    'content': memory['content'],
                    'metadata': {
                        'importance': memory.get('importance', 5),
                        'category': memory.get('category', 'general'),
                        'entities': memory.get('entities', [])
                    }
                })
        
        # Check for memories to delete if requested
        if check_for_deletions:
            all_memories = self.storage.get_all_memories()
            memory_ids_to_delete = self.openai_client.identify_memories_to_delete(
                conversation, all_memories
            )
            
            if memory_ids_to_delete:
                for memory_id in memory_ids_to_delete:
                    if self.storage.delete_memory(memory_id):
                        result['deleted_memories'].append(memory_id)
        
        # Get relevant memories for the last user message
        if conversation:
            last_user_message = None
            for message in reversed(conversation):
                if message['role'] == 'user':
                    last_user_message = message['content']
                    break
            
            if last_user_message:
                all_memories = self.storage.get_all_memories()
                relevant_memories = self.retriever.retrieve_relevant_memories(
                    last_user_message, all_memories
                )
                result['relevant_memories'] = relevant_memories
        
        return result
    
    def get_relevant_memories(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get memories relevant to the given query.
        
        Args:
            query: The query to find relevant memories for.
            limit: Maximum number of memories to return.
            
        Returns:
            A list of memory dictionaries, sorted by relevance.
        """
        all_memories = self.storage.get_all_memories()
        return self.retriever.retrieve_relevant_memories(query, all_memories, limit)
    
    def add_memory(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a new memory.
        
        Args:
            content: The content of the memory.
            metadata: Optional metadata for the memory.
            
        Returns:
            The ID of the newly created memory.
        """
        # Get embedding for the memory content
        embedding = self.openai_client.get_embedding(content)
        
        # Store the memory
        return self.storage.add_memory(
            content=content,
            embedding=embedding,
            metadata=metadata
        )
    
    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to delete.
            
        Returns:
            True if the memory was deleted, False otherwise.
        """
        return self.storage.delete_memory(memory_id)
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Get all stored memories.
        
        Returns:
            A list of all memory dictionaries.
        """
        return self.storage.get_all_memories()
    
    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to retrieve.
            
        Returns:
            The memory dictionary, or None if not found.
        """
        return self.storage.get_memory(memory_id)
    
    def chat_with_memory(self, conversation: List[Dict[str, str]], 
                        model: str = "gpt-4", temperature: float = 0.7, 
                        max_tokens: int = 1000) -> Dict[str, Any]:
        """
        Generate a chat response using relevant memories.
        
        Args:
            conversation: A list of message dictionaries with 'role' and 'content' keys.
            model: The OpenAI model to use.
            temperature: Controls randomness. Lower is more deterministic.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            A dictionary containing the response and relevant memories.
        """
        # Process the conversation to extract new memories and check for deletions
        process_result = self.process_conversation(conversation)
        
        # Get relevant memories for the last user message
        relevant_memories = process_result['relevant_memories']
        
        # Format the memories for inclusion in the context
        memories_context = self.retriever.format_memories_for_context(relevant_memories)
        
        # Create a system message that includes the memories
        system_message = {
            "role": "system",
            "content": f"You are a helpful assistant with access to the user's long-term memories. Use the following memories to inform your responses:\n\n{memories_context}"
        }
        
        # Create a new conversation list with the system message first
        conversation_with_memory = [system_message] + conversation
        
        # Generate the response
        response = self.openai_client.chat_completion(
            messages=conversation_with_memory,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return {
            'response': response,
            'relevant_memories': relevant_memories,
            'new_memories': process_result['new_memories'],
            'deleted_memories': process_result['deleted_memories']
        }
