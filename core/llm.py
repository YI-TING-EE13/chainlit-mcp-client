"""
LLM Client module.

Provides a thin wrapper around the AsyncOpenAI client to interact with Ollama
or any OpenAI-compatible API. Centralized defaults are applied via settings.
"""

from openai import AsyncOpenAI
from typing import List, Dict, Any, Optional

from .settings import AppSettings, load_settings

class LLMClient:
    """
    A client for interacting with the Large Language Model (LLM).
    """
    def __init__(self, settings: Optional[AppSettings] = None):
        """Initialize the LLM client with centralized settings."""
        self.settings = settings or load_settings()
        self.client = AsyncOpenAI(
            base_url=self.settings.llm.base_url,
            api_key=self.settings.llm.api_key
        )
        self.model = self.settings.llm.model

    def _merge_extra_body(self, base_params: Dict[str, Any], override_extra: Optional[Dict[str, Any]]) -> None:
        """Merge extra_body options without clobbering existing settings."""
        if not override_extra:
            return

        base_extra = base_params.get("extra_body", {})
        base_options = base_extra.get("options", {})
        override_options = override_extra.get("options", {})

        merged_options = {**base_options, **override_options}
        merged_extra = {**base_extra, **override_extra}
        if merged_options:
            merged_extra["options"] = merged_options

        base_params["extra_body"] = merged_extra

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
        }

        # Apply centralized defaults
        params.update(self.settings.generation.to_request_params())
        
        if tools:
            params["tools"] = tools

        # Merge extra_body options carefully if provided per call
        if "extra_body" in kwargs:
            override_extra = kwargs.pop("extra_body")
            self._merge_extra_body(params, override_extra)

        # Merge additional kwargs (e.g. temperature, max_tokens)
        # This allows overriding defaults per call
        params.update(kwargs)

        return await self.client.chat.completions.create(**params)

