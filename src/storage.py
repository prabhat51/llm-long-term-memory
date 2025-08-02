import sqlite3
import json
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

class MemoryStorage:
    """Handles efficient storage and retrieval of conversation memories."""
    
    def __init__(self, db_path: str = "memories.db"):
        """Initialize the storage with a SQLite database."""
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self) -> None:
        """Create the necessary tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    embedding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Create an index for faster similarity searches
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_memories_embedding 
                ON memories(embedding)
            ''')
            
            conn.commit()
    
    def add_memory(self, content: str, embedding: Optional[np.ndarray] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> int:
        """
        Add a new memory to the database.
        
        Args:
            content: The text content of the memory.
            embedding: Optional embedding vector for semantic search.
            metadata: Optional metadata dictionary.
            
        Returns:
            The ID of the newly created memory.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Convert embedding to bytes if provided
            embedding_bytes = None
            if embedding is not None:
                embedding_bytes = embedding.tobytes()
            
            # Convert metadata to JSON string if provided
            metadata_json = None
            if metadata is not None:
                metadata_json = json.dumps(metadata)
            
            cursor.execute('''
                INSERT INTO memories (content, embedding, metadata)
                VALUES (?, ?, ?)
            ''', (content, embedding_bytes, metadata_json))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to retrieve.
            
        Returns:
            A dictionary containing the memory data, or None if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, content, embedding, created_at, updated_at, metadata
                FROM memories
                WHERE id = ?
            ''', (memory_id,))
            
            row = cursor.fetchone()
            if row is None:
                return None
            
            # Convert embedding bytes back to numpy array
            embedding = None
            if row[2] is not None:
                embedding = np.frombuffer(row[2], dtype=np.float32)
            
            # Parse metadata JSON
            metadata = None
            if row[5] is not None:
                metadata = json.loads(row[5])
            
            return {
                'id': row[0],
                'content': row[1],
                'embedding': embedding,
                'created_at': row[3],
                'updated_at': row[4],
                'metadata': metadata
            }
    
    def find_similar_memories(self, query_embedding: np.ndarray, 
                             limit: int = 5) -> List[Dict[str, Any]]:
        """
        Find memories similar to the given embedding using cosine similarity.
        
        Args:
            query_embedding: The embedding to compare against.
            limit: Maximum number of memories to return.
            
        Returns:
            A list of memory dictionaries sorted by similarity.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, content, embedding, created_at, updated_at, metadata
                FROM memories
                WHERE embedding IS NOT NULL
            ''')
            
            memories = []
            for row in cursor.fetchall():
                # Convert embedding bytes back to numpy array
                memory_embedding = np.frombuffer(row[2], dtype=np.float32)
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, memory_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
                )
                
                # Parse metadata JSON
                metadata = None
                if row[5] is not None:
                    metadata = json.loads(row[5])
                
                memories.append({
                    'id': row[0],
                    'content': row[1],
                    'embedding': memory_embedding,
                    'similarity': similarity,
                    'created_at': row[3],
                    'updated_at': row[4],
                    'metadata': metadata
                })
            
            # Sort by similarity (descending) and return the top 'limit' memories
            memories.sort(key=lambda x: x['similarity'], reverse=True)
            return memories[:limit]
    
    def search_by_content(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for memories containing the given query string.
        
        Args:
            query: The string to search for in memory content.
            limit: Maximum number of memories to return.
            
        Returns:
            A list of memory dictionaries.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, content, embedding, created_at, updated_at, metadata
                FROM memories
                WHERE content LIKE ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (f'%{query}%', limit))
            
            memories = []
            for row in cursor.fetchall():
                # Convert embedding bytes back to numpy array
                embedding = None
                if row[2] is not None:
                    embedding = np.frombuffer(row[2], dtype=np.float32)
                
                # Parse metadata JSON
                metadata = None
                if row[5] is not None:
                    metadata = json.loads(row[5])
                
                memories.append({
                    'id': row[0],
                    'content': row[1],
                    'embedding': embedding,
                    'created_at': row[3],
                    'updated_at': row[4],
                    'metadata': metadata
                })
            
            return memories
    
    def update_memory(self, memory_id: int, content: Optional[str] = None, 
                     embedding: Optional[np.ndarray] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Update an existing memory.
        
        Args:
            memory_id: The ID of the memory to update.
            content: New content for the memory.
            embedding: New embedding for the memory.
            metadata: New metadata for the memory.
            
        Returns:
            True if the memory was updated, False if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Check if the memory exists
            cursor.execute('SELECT id FROM memories WHERE id = ?', (memory_id,))
            if cursor.fetchone() is None:
                return False
            
            # Build the update query based on provided parameters
            update_fields = []
            params = []
            
            if content is not None:
                update_fields.append("content = ?")
                params.append(content)
            
            if embedding is not None:
                update_fields.append("embedding = ?")
                params.append(embedding.tobytes())
            
            if metadata is not None:
                update_fields.append("metadata = ?")
                params.append(json.dumps(metadata))
            
            if not update_fields:
                return True  # Nothing to update
            
            # Add the updated_at timestamp
            update_fields.append("updated_at = CURRENT_TIMESTAMP")
            
            # Add the memory_id to the params
            params.append(memory_id)
            
            # Execute the update
            cursor.execute(f'''
                UPDATE memories
                SET {', '.join(update_fields)}
                WHERE id = ?
            ''', params)
            
            conn.commit()
            return True
    
    def delete_memory(self, memory_id: int) -> bool:
        """
        Delete a memory by its ID.
        
        Args:
            memory_id: The ID of the memory to delete.
            
        Returns:
            True if the memory was deleted, False if not found.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM memories WHERE id = ?', (memory_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    def get_all_memories(self) -> List[Dict[str, Any]]:
        """
        Retrieve all memories from the database.
        
        Returns:
            A list of all memory dictionaries.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, content, embedding, created_at, updated_at, metadata
                FROM memories
                ORDER BY created_at DESC
            ''')
            
            memories = []
            for row in cursor.fetchall():
                # Convert embedding bytes back to numpy array
                embedding = None
                if row[2] is not None:
                    embedding = np.frombuffer(row[2], dtype=np.float32)
                
                # Parse metadata JSON
                metadata = None
                if row[5] is not None:
                    metadata = json.loads(row[5])
                
                memories.append({
                    'id': row[0],
                    'content': row[1],
                    'embedding': embedding,
                    'created_at': row[3],
                    'updated_at': row[4],
                    'metadata': metadata
                })
            
            return memories
