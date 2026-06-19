import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import CLIENT_ROOT, MCP_CONFIG
from core.settings import load_settings


LLM_ENV_KEYS = (
    "LLM_BASE_URL",
    "LLM_API_KEY",
    "LLM_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_KEY",
    "OLLAMA_MODEL",
)


def with_env(overrides: dict[str, str | None]):
    """Apply temporary environment overrides while preserving .env-loaded values."""
    original = {key: os.environ.get(key) for key in LLM_ENV_KEYS}
    for key in LLM_ENV_KEYS:
        os.environ.pop(key, None)
    for key, value in overrides.items():
        if value is not None:
            os.environ[key] = value
    try:
        yield
    finally:
        for key in LLM_ENV_KEYS:
            os.environ.pop(key, None)
            if original[key] is not None:
                os.environ[key] = original[key]


def assert_llm_settings() -> None:
    env_context = with_env({})
    next(env_context)
    try:
        settings = load_settings()
        assert settings.llm.base_url == "http://localhost:1234/v1"
        assert settings.llm.api_key == "lm-studio"
        assert settings.llm.model == "google/gemma-4-e4b"
    finally:
        try:
            next(env_context)
        except StopIteration:
            pass

    env_context = with_env({
        "OLLAMA_HOST": "localhost:11434",
        "OLLAMA_KEY": "ollama",
        "OLLAMA_MODEL": "legacy-model",
    })
    next(env_context)
    try:
        settings = load_settings()
        assert settings.llm.base_url == "http://localhost:11434/v1"
        assert settings.llm.api_key == "ollama"
        assert settings.llm.model == "legacy-model"
    finally:
        try:
            next(env_context)
        except StopIteration:
            pass

    env_context = with_env({
        "LLM_BASE_URL": "http://localhost:1234/v1",
        "LLM_API_KEY": "lm-studio",
        "LLM_MODEL": "google/gemma-4-e4b",
        "OLLAMA_HOST": "http://localhost:11434/v1",
        "OLLAMA_KEY": "ollama",
        "OLLAMA_MODEL": "legacy-model",
    })
    next(env_context)
    try:
        settings = load_settings()
        assert settings.llm.base_url == "http://localhost:1234/v1"
        assert settings.llm.api_key == "lm-studio"
        assert settings.llm.model == "google/gemma-4-e4b"
    finally:
        try:
            next(env_context)
        except StopIteration:
            pass


def main() -> None:
    server = MCP_CONFIG["mcpServers"]["arxiv-insight"]

    if server["command"] != "uv":
        raise SystemExit("arxiv-insight command must be uv")
    if server["args"][:3] != ["--directory", "../mcp-server", "run"]:
        raise SystemExit(f"Unexpected arxiv-insight args: {server['args']}")

    configured_server_dir = os.path.abspath(os.path.join(CLIENT_ROOT, server["args"][1]))
    if not os.path.isdir(configured_server_dir):
        raise SystemExit(f"Configured server directory does not exist: {configured_server_dir}")
    if not os.path.exists(os.path.join(configured_server_dir, "main.py")):
        raise SystemExit("Configured server directory is missing main.py")

    assert_llm_settings()

    print("client smoke test passed")


if __name__ == "__main__":
    main()
