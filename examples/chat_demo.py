import os
from dotenv import load_dotenv
from src.memory_system import MemorySystem

def main():
    # Load environment variables
    load_dotenv()
    
    # Initialize the memory system
    memory_system = MemorySystem()
    
    print("LLM Long-Term Memory System Demo")
    print("Type 'exit' to end the conversation")
    print("-" * 40)
    
    # Initialize conversation history
    conversation = []
    
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Check if the user wants to exit
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Add user message to conversation
        conversation.append({"role": "user", "content": user_input})
        
        # Get response from the memory system
        result = memory_system.chat_with_memory(conversation)
        
        # Print the response
        print(f"Assistant: {result['response']}")
        
        # Add assistant response to conversation
        conversation.append({"role": "assistant", "content": result['response']})
        
        # Print information about memories
        if result['new_memories']:
            print(f"[Added {len(result['new_memories'])} new memory/memories]")
        
        if result['deleted_memories']:
            print(f"[Deleted {len(result['deleted_memories'])} memory/memories]")
        
        if result['relevant_memories']:
            print(f"[Used {len(result['relevant_memories'])} relevant memory/memories]")
        
        print("-" * 40)

if __name__ == "__main__":
    main()
