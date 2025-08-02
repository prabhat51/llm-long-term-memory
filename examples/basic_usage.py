
## Step 4: Example Usage

### examples/basic_usage.py

import os
from dotenv import load_dotenv
from src.memory_system import MemorySystem

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the memory system
    memory_system = MemorySystem()
    
    # Example 1: Add a memory explicitly
    memory_id = memory_system.add_memory(
        content="I use Shram and Magnet as productivity tools",
        metadata={
            "category": "preference",
            "importance": 8,
            "entities": ["Shram", "Magnet"]
        }
    )
    print(f"Added memory with ID: {memory_id}")
    
    # Example 2: Process a conversation to extract memories
    conversation = [
        {"role": "user", "content": "I really enjoy hiking in the mountains on weekends."}
    ]
    
    result = memory_system.process_conversation(conversation)
    print(f"Extracted {len(result['new_memories'])} new memories:")
    for memory in result['new_memories']:
        print(f"- {memory['content']}")
    
    # Example 3: Retrieve relevant memories
    query = "What are my hobbies?"
    relevant_memories = memory_system.get_relevant_memories(query)
    print(f"\nRelevant memories for query '{query}':")
    for memory in relevant_memories:
        print(f"- {memory['content']} (similarity: {memory['similarity']:.2f})")
    
    # Example 4: Chat with memory
    conversation = [
        {"role": "user", "content": "What productivity tools do I use?"}
    ]
    
    chat_result = memory_system.chat_with_memory(conversation)
    print(f"\nResponse: {chat_result['response']}")
    print(f"Used {len(chat_result['relevant_memories'])} relevant memories:")
    for memory in chat_result['relevant_memories']:
        print(f"- {memory['content']}")
    
    # Example 5: Process a conversation that includes a memory deletion
    conversation = [
        {"role": "user", "content": "I don't use Magnet anymore."}
    ]
    
    result = memory_system.process_conversation(conversation)
    print(f"\nDeleted {len(result['deleted_memories'])} memories with IDs: {result['deleted_memories']}")
    
    # Example 6: Chat again to see if the memory was deleted
    conversation = [
        {"role": "user", "content": "What productivity tools do I use?"}
    ]
    
    chat_result = memory_system.chat_with_memory(conversation)
    print(f"\nResponse: {chat_result['response']}")
    print(f"Used {len(chat_result['relevant_memories'])} relevant memories:")
    for memory in chat_result['relevant_memories']:
        print(f"- {memory['content']}")

if __name__ == "__main__":
    main()
