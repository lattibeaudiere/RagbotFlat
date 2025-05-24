import aiohttp
import asyncio
import json
import os
from dotenv import load_dotenv

class ChutesChatbot:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("CHUTES_API_TOKEN")
        if not self.api_token:
            raise ValueError("Please set CHUTES_API_TOKEN in your .env file")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://llm.chutes.ai/v1/chat/completions"
        self.model = "chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.conversation_history = []

    async def get_response(self, user_message):
        self.conversation_history.append({"role": "user", "content": user_message})
        
        body = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": True,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        full_response = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=body
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            if "choices" in chunk and chunk["choices"]:
                                content = chunk["choices"][0].get("delta", {}).get("content", "")
                                if content:
                                    full_response += content
                                    print(content, end="", flush=True)
                        except Exception as e:
                            print(f"Error parsing chunk: {e}")

        self.conversation_history.append({"role": "assistant", "content": full_response})
        return full_response

async def main():
    chatbot = ChutesChatbot()
    print("Welcome to the Chutes.ai Chatbot! Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        print("\nAssistant: ", end="")
        await chatbot.get_response(user_input)

if __name__ == "__main__":
    asyncio.run(main()) 