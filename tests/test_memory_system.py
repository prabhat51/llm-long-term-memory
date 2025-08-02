import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock
import numpy as np

from src.memory_system import MemorySystem

class TestMemorySystem(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        os.close(self.db_fd)
        
        # Mock the OpenAI API key
        self.api_key = "test_api_key"
        
        # Initialize the memory system with the temporary database
        with patch.dict(os.environ, {"OPENAI_API_KEY": self.api_key}):
            self.memory_system = MemorySystem(api_key=self.api_key, db_path=self.db_path)
    
    def tearDown(self):
        # Remove the temporary database file
        os.unlink(self.db_path)
    
    @patch('src.memory_system.OpenAIClient.get_embedding')
    def test_add_memory(self, mock_get_embedding):
        # Mock the embedding response
        mock_get_embedding.return_value = [0.1] * 1536
        
        # Add a memory
        memory_id = self.memory_system.add_memory(
            content="Test memory",
            metadata={"category": "test", "importance": 5}
        )
        
        # Check that the memory was added
        self.assertIsInstance(memory_id, int)
        self.assertGreater(memory_id, 0)
        
        # Retrieve the memory
        memory = self.memory_system.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory['content'], "Test memory")
        self.assertEqual(memory['metadata']['category'], "test")
        self.assertEqual(memory['metadata']['importance'], 5)
    
    def test_delete_memory(self):
        # Add a memory
        with patch('src.memory_system.OpenAIClient.get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = [0.1] * 1536
            memory_id = self.memory_system.add_memory("Test memory")
        
        # Delete the memory
        result = self.memory_system.delete_memory(memory_id)
        self.assertTrue(result)
        
        # Check that the memory was deleted
        memory = self.memory_system.get_memory(memory_id)
        self.assertIsNone(memory)
    
    def test_get_all_memories(self):
        # Add some memories
        with patch('src.memory_system.OpenAIClient.get_embedding') as mock_get_embedding:
            mock_get_embedding.return_value = [0.1] * 1536
            memory_id1 = self.memory_system.add_memory("Memory 1")
            memory_id2 = self.memory_system.add_memory("Memory 2")
        
        # Get all memories
        memories = self.memory_system.get_all_memories()
        self.assertEqual(len(memories), 2)
        
        # Check that the memories are in descending order by creation time
        self.assertEqual(memories[0]['content'], "Memory 2")
        self.assertEqual(memories[1]['content'], "Memory 1")
    
    @patch('src.memory_system.OpenAIClient.get_embedding')
    def test_get_relevant_memories(self, mock_get_embedding):
        # Mock the embedding response
        mock_get_embedding.return_value = [0.1] * 1536
        
        # Add some memories
        self.memory_system.add_memory("I like apples")
        self.memory_system.add_memory("I like bananas")
        self.memory_system.add_memory("I like oranges")
        
        # Mock the embedding for the query
        mock_get_embedding.return_value = [0.2] * 1536
        
        # Get relevant memories
        relevant_memories = self.memory_system.get_relevant_memories("What fruits do I like?")
        
        # Check that we got some memories
        self.assertGreater(len(relevant_memories), 0)
        
        # Check that the memories have similarity scores
        for memory in relevant_memories:
            self.assertIn('similarity', memory)
    
    @patch('src.memory_system.OpenAIClient.extract_memories')
    @patch('src.memory_system.OpenAIClient.get_embedding')
    def test_process_conversation(self, mock_get_embedding, mock_extract_memories):
        # Mock the embedding response
        mock_get_embedding.return_value = [0.1] * 1536
        
        # Mock the memory extraction
        mock_extract_memories.return_value = [
            {
                "content": "Test memory",
                "importance": 8,
                "category": "test",
                "entities": ["test"]
            }
        ]
        
        # Process a conversation
        conversation = [
            {"role": "user", "content": "I like test things."}
        ]
        result = self.memory_system.process_conversation(conversation)
        
        # Check the result
        self.assertIn('new_memories', result)
        self.assertIn('deleted_memories', result)
        self.assertIn('relevant_memories', result)
        
        # Check that a new memory was added
        self.assertEqual(len(result['new_memories']), 1)
        self.assertEqual(result['new_memories'][0]['content'], "Test memory")
    
    @patch('src.memory_system.OpenAIClient.chat_completion')
    @patch('src.memory_system.OpenAIClient.get_embedding')
    def test_chat_with_memory(self, mock_get_embedding, mock_chat_completion):
        # Mock the embedding response
        mock_get_embedding.return_value = [0.1] * 1536
        
        # Mock the chat completion
        mock_chat_completion.return_value = "You like test things."
        
        # Add a memory
        self.memory_system.add_memory("I like test things")
        
        # Chat with memory
        conversation = [
            {"role": "user", "content": "What do I like?"}
        ]
        result = self.memory_system.chat_with_memory(conversation)
        
        # Check the result
        self.assertIn('response', result)
        self.assertIn('relevant_memories', result)
        self.assertIn('new_memories', result)
        self.assertIn('deleted_memories', result)
        
        # Check the response
        self.assertEqual(result['response'], "You like test things.")
        
        # Check that relevant memories were found
        self.assertGreater(len(result['relevant_memories']), 0)

if __name__ == '__main__':
    unittest.main()
