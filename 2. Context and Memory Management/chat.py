import json
from typing import Dict, List

from openai import OpenAI
import sys

def initialize_client(use_openai:bool=False)-> OpenAI:
    if use_openai:
        return OpenAI(api_key="openapi")
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

def create_initial_messages()-> List[Dict[str, str]]:
    """Create the initial messages for context memory."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
    ]
    
def chat(
    user_input: str,messages: List[Dict[str, str]], client: OpenAI,model_name: str
)-> str:
    """Handle the user input and generate a response."""
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        assistant_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_response})
        return assistant_response
    except Exception as e:
        print(f"Error with API: {str(e)}")
        return "I'm sorry, I couldn't process your request at the moment."
    
def summarize_messages(messages:List[Dict[str,str]])->List[Dict[str,str]]:
    """Summarize older messages to save tokens"""
    summary="Previous conversation summarized: "+" ".join([m["content"][:50]+"..." for m in messages[-5:]])

    return [{"role":"system","content":summary}] + messages[-5:]

def save_conversation(messages: List[Dict[str, str]], filename: str = "conversation.json") -> None:
    """Save the conversation to a file."""
    with open(filename, "w") as f:
        json.dump(messages, f)
        
def load_conversation(filename: str = "conversation.json") -> List[Dict[str, str]]:
    """Load the conversation from a file."""
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No previous conversation found. Starting a new one.")
        return create_initial_messages()
    
def main():
    print("Select the model you want to use:")
    print("1. Ollama (Local API)")
    print("2. GPT-4 (OpenAI API)")
    choice = input("Enter your choice (1 or 2): ")
    use_openai = choice == "2"
    
    model_name = "gpt-4" if use_openai else "llama3.2"
    client = initialize_client(use_openai)
    
    messages = create_initial_messages()
    
    print("Available commands:")
    print("- 'save': Save the current conversation.")
    print("- 'load': Load the previous conversation.")
    print("- 'summarize': Summarize the conversation to save tokens.")
    print("- 'exit' or 'quit': Exit the chat.")
    while True:
        user_input = input("\nYou $: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the chat. Goodbye!")
            break
        elif user_input.lower() == "save":
            save_conversation(messages)
            print("Conversation saved.")
            continue
        elif user_input.lower() == "load":
            messages = load_conversation()
            print("Conversation loaded.")
            continue
        elif user_input.lower() == "summarize":
            messages = summarize_messages(messages)
            print("Conversation summarized.")
            continue
       
        response = chat(user_input, messages, client, model_name)
        print(f"\nAssistant: {response}")
        
        if len(messages) > 10:
            messages = summarize_messages(messages)
            print("\nConversation summarized to save tokens.")

if __name__ == "__main__":
    main()