from typing import List, Dict, Any, Optional
from .openai_client import OpenAIClient

class MemoryExtractor:
    """Extracts relevant memories from conversations."""
    
    def __init__(self, openai_client: OpenAIClient):
        """Initialize with an OpenAI client."""
        self.openai_client = openai_client
    
    def extract_memories(self, conversation: List[Dict[str, str]], 
                        importance_threshold: int = 5) -> List[Dict[str, Any]]:
        """
        Extract potential memories from a conversation.
        
        Args:
            conversation: A list of message dictionaries with 'role' and 'content' keys.
            importance_threshold: Minimum importance score for a memory to be stored.
            
        Returns:
            A list of memory dictionaries that meet the importance threshold.
        """
        # Get potential memories from OpenAI
        potential_memories = self.openai_client.extract_memories(conversation)
        
        # Filter memories based on importance threshold
        memories = [
            memory for memory in potential_memories 
            if memory.get('importance', 0) >= importance_threshold
        ]
        
        return memories
    
    def should_store_memory(self, memory: Dict[str, Any], threshold: int = 5) -> bool:
        """
        Determine if a memory should be stored based on its importance.
        
        Args:
            memory: A memory dictionary.
            threshold: The minimum importance score for storing a memory.
            
        Returns:
            True if the memory should be stored, False otherwise.
        """
        return memory.get('importance', 0) >= threshold
