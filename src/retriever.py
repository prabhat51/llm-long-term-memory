import numpy as np
from typing import List, Dict, Any
from .openai_client import OpenAIClient

class MemoryRetriever:
    """Retrieves relevant memories based on context."""
    
    def __init__(self, openai_client: OpenAIClient):
        """Initialize with an OpenAI client."""
        self.openai_client = openai_client
    
    def retrieve_relevant_memories(self, query: str, memories: List[Dict[str, Any]], 
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the given query.
        
        Args:
            query: The query to find relevant memories for.
            memories: A list of memory dictionaries.
            limit: Maximum number of memories to return.
            
        Returns:
            A list of memory dictionaries, sorted by relevance.
        """
        # Get the embedding for the query
        query_embedding = self.openai_client.get_embedding(query)
        
        # Calculate similarity with each memory
        for memory in memories:
            if 'embedding' in memory and memory['embedding'] is not None:
                memory_embedding = memory['embedding']
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, memory_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                )
                memory['similarity'] = similarity
            else:
                memory['similarity'] = 0.0
        
        # Sort by similarity (descending) and return the top 'limit' memories
        memories.sort(key=lambda x: x['similarity'], reverse=True)
        return memories[:limit]
    
    def format_memories_for_context(self, memories: List[Dict[str, Any]]) -> str:
        """
        Format memories for inclusion in a conversation context.
        
        Args:
            memories: A list of memory dictionaries.
            
        Returns:
            A string containing the formatted memories.
        """
        if not memories:
            return "No relevant memories found."
        
        formatted_memories = []
        for memory in memories:
            content = memory.get('content', '')
            formatted_memories.append(f"- {content}")
        
        return "Relevant memories:\n" + "\n".join(formatted_memories)
