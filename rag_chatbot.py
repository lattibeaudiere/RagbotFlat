import asyncio
import aiohttp
import json
import os
from dotenv import load_dotenv
from typing import List, Dict
from retriever import DocumentRetriever

class RAGChatbot:
    def __init__(self):
        load_dotenv()
        self.api_token = os.getenv("CHUTES_API_TOKEN")
        self.headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        self.api_url = "https://llm.chutes.ai/v1/chat/completions"
        self.model = "chutesai/Llama-4-Maverick-17B-128E-Instruct-FP8"
        self.conversation_history = []
        self.retriever = DocumentRetriever()

    def _format_context(self, documents: List[Dict]) -> str:
        """Format retrieved documents into context for the LLM"""
        context = "Relevant information from the knowledge base:\n\n"
        for doc in documents:
            context += f"From {doc['file_name']}:\n{doc['content']}\n\n"
        # Log the context for debugging (show only first 500 chars)
        print("[RAGChatbot] Context sent to LLM (first 500 chars):\n", context[:500])
        return context

    async def get_response(self, user_message: str) -> str:
        """Get response from the LLM with RAG context"""
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve_documents(user_message)
        print(f"[RAGChatbot] Retrieved {len(relevant_docs)} docs for query: {user_message}")
        for doc in relevant_docs:
            print(f"[RAGChatbot] Doc: {doc['file_name']} - First 100 chars: {doc['content'][:100]}")
        context = self._format_context(relevant_docs)

        # Add context to the user message
        augmented_message = f"{context}\n\nUser: {user_message}"

        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": augmented_message})

        # Prepare the request body
        request_body = {
            "model": self.model,
            "messages": self.conversation_history,
            "stream": True,
            "max_tokens": 1024,
            "temperature": 0.7
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=self.headers,
                json=request_body
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"API request failed: {error_text}")

                full_response = ""
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            json_data = json.loads(data)
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                delta = json_data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    print(content, end='', flush=True)
                                    full_response += content
                        except Exception as e:
                            print(f"Error parsing chunk: {e}")

                # Add assistant's response to conversation history
                self.conversation_history.append({"role": "assistant", "content": full_response})
                return full_response

async def main():
    chatbot = RAGChatbot()
    print("RAG Chatbot initialized. Type 'quit' to exit.")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
            
        print("\nAssistant: ", end='')
        await chatbot.get_response(user_input)

if __name__ == "__main__":
    asyncio.run(main()) 