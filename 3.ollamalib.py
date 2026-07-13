import ollama

res=ollama.chat(
    model="llama3.2", 
    messages=[
        {
            "role": "user", 
            "content": "Hello, why is the ocean blue?"
        },
    ],
    stream=True,
)

for chunk in res:
    print(chunk["message"]["content"], end="", flush=True)