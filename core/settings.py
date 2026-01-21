"""
Centralized application settings for the MCP Client.

This module defines a typed configuration model and loads values from environment
variables. It standardizes LLM connectivity, generation defaults, and sampling
defaults used by MCP tool callbacks.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os

from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()


def _get_str(name: str, default: Optional[str] = None) -> Optional[str]:
    """Read a string environment variable with a fallback."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value


def _get_int(name: str, default: Optional[int] = None) -> Optional[int]:
    """Read an integer environment variable with a fallback."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _get_float(name: str, default: Optional[float] = None) -> Optional[float]:
    """Read a float environment variable with a fallback."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _get_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable with a fallback."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_base_url(url: str) -> str:
    """Normalize API base URLs to include a scheme and the /v1 suffix."""
    if not url.startswith(("http://", "https://")):
        url = f"http://{url}"
    if not url.endswith("/v1"):
        url = f"{url.rstrip('/')}/v1"
    return url


@dataclass(frozen=True)
class LLMConnectionSettings:
    """Connection details for the OpenAI-compatible API endpoint."""
    base_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class GenerationSettings:
    """Model generation parameters and Ollama-specific options."""
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    top_k: Optional[int]
    repeat_penalty: Optional[float]
    num_ctx: Optional[int]
    num_predict: Optional[int]

    def to_ollama_options(self) -> Dict[str, Any]:
        """Convert relevant fields to Ollama options for extra_body."""
        options: Dict[str, Any] = {}
        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx
        if self.top_k is not None:
            options["top_k"] = self.top_k
        if self.repeat_penalty is not None:
            options["repeat_penalty"] = self.repeat_penalty
        if self.num_predict is not None:
            options["num_predict"] = self.num_predict
        return options

    def to_request_params(self) -> Dict[str, Any]:
        """Convert fields to standard OpenAI-compatible request parameters."""
        params: Dict[str, Any] = {}
        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p

        options = self.to_ollama_options()
        if options:
            params["extra_body"] = {"options": options}
        return params


@dataclass(frozen=True)
class AppSettings:
    """Top-level application settings container."""
    assistant_name: str
    llm: LLMConnectionSettings
    generation: GenerationSettings
    sampling: GenerationSettings
    token_usage_enabled: bool
    tokenizer_model: str
    memory_enabled: bool
    memory_db_path: str
    memory_default_incognito: bool
    memory_summary_enabled: bool
    memory_summary_max_tokens: int
    memory_summary_scheduler_enabled: bool
    memory_summary_interval_seconds: int


def load_settings() -> AppSettings:
    """Load settings from environment variables with reasonable defaults."""
    base_url = _normalize_base_url(_get_str("OLLAMA_HOST", "http://localhost:11434/v1"))
    api_key = _get_str("OLLAMA_KEY", "ollama")
    model = _get_str("OLLAMA_MODEL", "nemotron-3-nano:latest")
    assistant_name = _get_str("ASSISTANT_NAME", "Nemotron")

    generation = GenerationSettings(
        max_tokens=_get_int("LLM_MAX_TOKENS"),
        temperature=_get_float("LLM_TEMPERATURE", 0.8),
        top_p=_get_float("LLM_TOP_P"),
        top_k=_get_int("LLM_TOP_K"),
        repeat_penalty=_get_float("LLM_REPEAT_PENALTY"),
        num_ctx=_get_int("LLM_NUM_CTX", 1048576),
        num_predict=_get_int("LLM_NUM_PREDICT"),
    )

    sampling = GenerationSettings(
        max_tokens=_get_int("SAMPLING_MAX_TOKENS", 131072),
        temperature=_get_float("SAMPLING_TEMPERATURE", 0.8),
        top_p=_get_float("SAMPLING_TOP_P"),
        top_k=_get_int("SAMPLING_TOP_K"),
        repeat_penalty=_get_float("SAMPLING_REPEAT_PENALTY"),
        num_ctx=_get_int("SAMPLING_NUM_CTX", 1048576),
        num_predict=_get_int("SAMPLING_NUM_PREDICT"),
    )

    token_usage_enabled = _get_bool("TOKEN_USAGE_ENABLED", True)
    tokenizer_model = _get_str("TOKENIZER_MODEL", "cl100k_base")

    memory_enabled = _get_bool("MEMORY_ENABLED", True)
    memory_db_path = _get_str("MEMORY_DB_PATH", "data/memory.db")
    memory_default_incognito = _get_bool("MEMORY_DEFAULT_INCOGNITO", False)
    memory_summary_enabled = _get_bool("MEMORY_SUMMARY_ENABLED", True)
    memory_summary_max_tokens = _get_int("MEMORY_SUMMARY_MAX_TOKENS", 512) or 512
    memory_summary_scheduler_enabled = _get_bool("MEMORY_SUMMARY_SCHEDULER_ENABLED", True)
    memory_summary_interval_seconds = _get_int("MEMORY_SUMMARY_INTERVAL_SECONDS", 600) or 600

    return AppSettings(
        assistant_name=assistant_name,
        llm=LLMConnectionSettings(
            base_url=base_url,
            api_key=api_key,
            model=model,
        ),
        generation=generation,
        sampling=sampling,
        token_usage_enabled=token_usage_enabled,
        tokenizer_model=tokenizer_model,
        memory_enabled=memory_enabled,
        memory_db_path=memory_db_path,
        memory_default_incognito=memory_default_incognito,
        memory_summary_enabled=memory_summary_enabled,
        memory_summary_max_tokens=memory_summary_max_tokens,
        memory_summary_scheduler_enabled=memory_summary_scheduler_enabled,
        memory_summary_interval_seconds=memory_summary_interval_seconds,
    )