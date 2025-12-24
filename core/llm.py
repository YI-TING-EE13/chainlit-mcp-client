"""
LLM Client module.
Provides a wrapper around the AsyncOpenAI client to interact with Ollama or compatible APIs.
"""

from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional
from .config import OLLAMA_BASE_URL, OLLAMA_API_KEY, MODEL_NAME, CONTEXT_WINDOW

class LLMClient:
    """
    A client for interacting with the Large Language Model (LLM).
    """
    def __init__(self):
        """
        Initialize the LLM client with configuration from config.py.
        """
        self.client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        self.model = MODEL_NAME

    async def chat_completion(self, messages: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None, **kwargs) -> Any:
        """
        Send a chat completion request to the LLM.

        Args:
            messages (List[Dict[str, Any]]): The history of messages for the conversation.
            tools (Optional[List[Dict[str, Any]]]): A list of tools available to the LLM (function calling).
            **kwargs: Additional arguments to pass to the OpenAI client (e.g., temperature, max_tokens).

        Returns:
            Any: The response object from the OpenAI client.
        """
        params = {
            "model": self.model,
            "messages": messages,
            # Ollama specific options to control context window
            "extra_body": {
                "options": {
                    "num_ctx": CONTEXT_WINDOW
                }
            }
        }
        
        if tools:
            params["tools"] = tools

        # Merge additional kwargs (e.g. temperature, max_tokens)
        # This allows overriding defaults per call
        params.update(kwargs)

        return await self.client.chat.completions.create(**params)

