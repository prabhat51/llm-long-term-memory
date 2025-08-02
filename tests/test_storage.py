import unittest
import os
import tempfile
import numpy as np
from src.storage import MemoryStorage

class TestMemoryStorage(unittest.TestCase):
    def setUp(self):
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        os.close(self.db_fd)
        
        # Initialize the storage with the temporary database
        self.storage = MemoryStorage(self.db_path)
    
    def tearDown(self):
        # Remove the temporary database file
        os.unlink(self.db_path)
    
    def test_add_and_get_memory(self):
        # Add a memory
        memory_id = self.storage.add_memory(
            content="Test memory",
            metadata={"category": "test", "importance": 5}
        )
        
        # Check that the memory was added
        self.assertIsInstance(memory_id, int)
        self.assertGreater(memory_id, 0)
        
        # Retrieve the memory
        memory = self.storage.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory['content'], "Test memory")
        self.assertEqual(memory['metadata']['category'], "test")
        self.assertEqual(memory['metadata']['importance'], 5)
    
    def test_add_memory_with_embedding(self):
        # Create a mock embedding
        embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        
        # Add a memory with embedding
        memory_id = self.storage.add_memory(
            content="Test memory with embedding",
            embedding=embedding
        )
        
        # Retrieve the memory
        memory = self.storage.get_memory(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory['content'], "Test memory with embedding")
        
        # Check that the embedding was stored correctly
        self.assertIsNotNone(memory['embedding'])
        np.testing.assert_array_equal(memory['embedding'], embedding)
    
    def test_delete_memory(self):
        # Add a memory
        memory_id = self.storage.add_memory("Test memory to delete")
        
        # Delete the memory
        result = self.storage.delete_memory(memory_id)
        self.assertTrue(result)
        
        # Check that the memory was deleted
        memory = self.storage.get_memory(memory_id)
        self.assertIsNone(memory)
    
    def test_find_similar_memories(self):
        # Create mock embeddings
        embedding1 = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        embedding2 = np.array([0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
        embedding3 = np.array([0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
        
        # Add memories with embeddings
        self.storage.add_memory("Memory 1", embedding=embedding1)
        self.storage.add_memory("Memory 2", embedding=embedding2)
        self.storage.add_memory("Memory 3", embedding=embedding3)
        
        # Create a query embedding similar to embedding1 and embedding2
        query_embedding = np.array([0.15, 0.25, 0.35, 0.45, 0.55], dtype=np.float32)
        
        # Find similar memories
        similar_memories = self.storage.find_similar_memories(query_embedding, limit=2)
        
        # Check that we got 2 memories
        self.assertEqual(len(similar_memories), 2)
        
        # Check that the memories are sorted by similarity (descending)
        self.assertGreater(similar_memories[0]['similarity'], similar_memories[1]['similarity'])
    
    def test_search_by_content(self):
        # Add some memories
        self.storage.add_memory("I like apples")
        self.storage.add_memory("I like bananas")
        self.storage.add_memory("I like oranges")
        
        # Search for memories containing "like"
        memories = self.storage.search_by_content("like")
        
        # Check that we got all 3 memories
        self.assertEqual(len(memories), 3)
        
        # Search for memories containing "apples"
        memories = self.storage.search_by_content("apples")
        
        # Check that we got 1 memory
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]['content'], "I like apples")
    
    def test_update_memory(self):
        # Add a memory
        memory_id = self.storage.add_memory("Original content")
        
        # Update the memory
        new_embedding = np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32)
        result = self.storage.update_memory(
            memory_id=memory_id,
            content="Updated content",
            embedding=new_embedding,
            metadata={"category": "updated"}
        )
        
        # Check that the update was successful
        self.assertTrue(result)
        
        # Retrieve the updated memory
        memory = self.storage.get_memory(memory_id)
        self.assertEqual(memory['content'], "Updated content")
        np.testing.assert_array_equal(memory['embedding'], new_embedding)
        self.assertEqual(memory['metadata']['category'], "updated")
    
    def test_get_all_memories(self):
        # Add some memories
        self.storage.add_memory("Memory 1")
        self.storage.add_memory("Memory 2")
        self.storage.add_memory("Memory 3")
        
        # Get all memories
        memories = self.storage.get_all_memories()
        
        # Check that we got all 3 memories
        self.assertEqual(len(memories), 3)
        
        # Check that the memories are in descending order by creation time
        self.assertEqual(memories[0]['content'], "Memory 3")
        self.assertEqual(memories[1]['content'], "Memory 2")
        self.assertEqual(memories[2]['content'], "Memory 1")

if __name__ == '__main__':
    unittest.main()
