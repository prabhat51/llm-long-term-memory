import openai
import numpy as np
from typing import List, Dict, Any, Optional

class OpenAIClient:
    """Client for interacting with OpenAI APIs."""
    
    def __init__(self, api_key: str):
        """Initialize the client with API key."""
        self.api_key = api_key
        openai.api_key = api_key
    
    def get_embedding(self, text: str, model: str = "text-embedding-ada-002") -> np.ndarray:
        """
        Get the embedding for a given text.
        
        Args:
            text: The text to embed.
            model: The OpenAI embedding model to use.
            
        Returns:
            A numpy array representing the embedding.
        """
        response = openai.Embedding.create(
            model=model,
            input=text
        )
        return np.array(response['data'][0]['embedding'])
    
    def chat_completion(self, messages: List[Dict[str, str]], model: str = "gpt-4", 
                       temperature: float = 0.7, max_tokens: int = 1000) -> str:
        """
        Generate a chat completion response.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys.
            model: The OpenAI model to use.
            temperature: Controls randomness. Lower is more deterministic.
            max_tokens: Maximum number of tokens to generate.
            
        Returns:
            The generated response text.
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message['content']
    
    def extract_memories(self, conversation: List[Dict[str, str]], 
                        model: str = "gpt-4") -> List[Dict[str, Any]]:
        """
        Extract potential memories from a conversation.
        
        Args:
            conversation: A list of message dictionaries with 'role' and 'content' keys.
            model: The OpenAI model to use for extraction.
            
        Returns:
            A list of dictionaries, each representing a potential memory.
        """
        system_prompt = """
        You are an AI assistant that extracts important information from conversations to be stored as long-term memories.
        Your task is to identify statements that contain personal preferences, facts, or important information that the user might want to remember in the future.
        
        For each such statement, create a memory object with the following structure:
        {
            "content": "The exact statement or a concise summary of the information",
            "importance": a score from 1 to 10 indicating how important this memory is,
            "category": a string categorizing the memory (e.g., "preference", "fact", "personal_info"),
            "entities": a list of entities mentioned in the memory (e.g., ["Shram", "Magnet"])
        }
        
        Only extract information that seems personally relevant to the user and might be useful in future conversations.
        Ignore general knowledge, transient information, or casual conversation.
        
        Return your response as a JSON array of memory objects.
        """
        
        # Format the conversation for the prompt
        formatted_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        # Create the full prompt
        full_prompt = f"{system_prompt}\n\nConversation:\n{formatted_conversation}"
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": formatted_conversation}
            ],
            temperature=0.3  # Lower temperature for more consistent extraction
        )
        
        # Parse the response
        try:
            import json
            memories = json.loads(response.choices[0].message['content'])
            return memories
        except json.JSONDecodeError:
            # If the response is not valid JSON, try to extract JSON from the text
            content = response.choices[0].message['content']
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                try:
                    memories = json.loads(content[start_idx:end_idx])
                    return memories
                except json.JSONDecodeError:
                    return []
            else:
                return []
    
    def identify_memories_to_delete(self, conversation: List[Dict[str, str]], 
                                  memories: List[Dict[str, Any]], 
                                  model: str = "gpt-4") -> List[int]:
        """
        Identify memories to delete based on the conversation.
        
        Args:
            conversation: A list of message dictionaries with 'role' and 'content' keys.
            memories: A list of memory dictionaries.
            model: The OpenAI model to use for identification.
            
        Returns:
            A list of memory IDs to delete.
        """
        system_prompt = """
        You are an AI assistant that identifies memories to delete based on the user's input.
        Your task is to analyze the conversation and determine which memories should be deleted.
        
        The user might explicitly ask to delete memories (e.g., "I don't use Magnet anymore") or imply that certain information is no longer valid.
        
        For each memory, decide if it should be deleted based on the conversation.
        Return your response as a JSON array of memory IDs that should be deleted.
        If no memories should be deleted, return an empty array.
        """
        
        # Format the conversation for the prompt
        formatted_conversation = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        
        # Format the memories for the prompt
        formatted_memories = []
        for memory in memories:
            formatted_memories.append(f"ID: {memory['id']}, Content: {memory['content']}")
        formatted_memories_str = "\n".join(formatted_memories)
        
        # Create the full prompt
        full_prompt = f"{system_prompt}\n\nConversation:\n{formatted_conversation}\n\nMemories:\n{formatted_memories_str}"
        
        # Call the OpenAI API
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3  # Lower temperature for more consistent identification
        )
        
        # Parse the response
        try:
            import json
            memory_ids_to_delete = json.loads(response.choices[0].message['content'])
            return memory_ids_to_delete
        except json.JSONDecodeError:
            # If the response is not valid JSON, try to extract JSON from the text
            content = response.choices[0].message['content']
            start_idx = content.find('[')
            end_idx = content.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                try:
                    memory_ids_to_delete = json.loads(content[start_idx:end_idx])
                    return memory_ids_to_delete
                except json.JSONDecodeError:
                    return []
            else:
                return []
