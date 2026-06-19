import os
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import CLIENT_ROOT, MCP_CONFIG
from core.settings import load_settings
from core.lmstudio_models import (
    LMStudioModelManager,
    derive_native_api_root,
    parse_native_models,
    parse_openai_models,
)


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


class FakeRequester:
    def __init__(self, routes: dict[tuple[str, str], tuple[int, object]]):
        self.routes = routes
        self.calls: list[tuple[str, str, object | None]] = []

    async def __call__(self, method: str, url: str, payload: object | None, headers: dict[str, str]):
        self.calls.append((method, url, payload))
        return self.routes.get((method, url), (404, {"error": "not found"}))


def assert_model_manager_helpers() -> None:
    base_url = "http://localhost:1234/v1"
    native_root = "http://localhost:1234/api/v1"
    assert derive_native_api_root(base_url) == native_root

    openai_models = parse_openai_models({"data": [{"id": "google/gemma-4-e4b"}, {"id": "other"}]})
    assert [model.id for model in openai_models] == ["google/gemma-4-e4b", "other"]

    native_models = parse_native_models({
        "models": [
            {"key": "google/gemma-4-e4b", "loaded_instances": [{"id": "inst-1"}]},
            {"key": "other", "loaded_instances": []},
        ]
    })
    assert native_models[0].id == "google/gemma-4-e4b"
    assert native_models[0].loaded is True
    assert native_models[0].instance_id == "inst-1"
    assert native_models[1].id == "other"


async def assert_model_switching() -> None:
    base_url = "http://localhost:1234/v1"
    native_root = "http://localhost:1234/api/v1"

    # Native API unavailable: fall back to runtime model change for visible models.
    fallback_requester = FakeRequester({
        ("GET", f"{base_url}/models"): (200, {"data": [{"id": "current"}, {"id": "new"}]}),
        ("GET", f"{native_root}/models"): (404, {"error": "method not found"}),
    })
    fallback_manager = LMStudioModelManager(base_url, "test-key", fallback_requester)
    fallback_result = await fallback_manager.switch_model("current", "new")
    assert fallback_result.success is True
    assert fallback_result.current_model == "new"
    assert fallback_result.native_available is False

    # Same model is a no-op and should not call any endpoint.
    noop_requester = FakeRequester({})
    noop_manager = LMStudioModelManager(base_url, "test-key", noop_requester)
    noop_result = await noop_manager.switch_model("current", "current")
    assert noop_result.success is True
    assert noop_result.current_model == "current"
    assert noop_requester.calls == []

    # Unload failure prevents load and preserves the original model.
    unload_fail_requester = FakeRequester({
        ("GET", f"{base_url}/models"): (200, {"data": [{"id": "current"}, {"id": "new"}]}),
        ("GET", f"{native_root}/models"): (
            200,
            {"data": [{"id": "current", "loaded": True, "instance_id": "inst-1"}, {"id": "new"}]},
        ),
        ("POST", f"{native_root}/models/unload"): (500, {"error": "unload failed"}),
    })
    unload_fail_manager = LMStudioModelManager(base_url, "test-key", unload_fail_requester)
    unload_fail_result = await unload_fail_manager.switch_model("current", "new")
    assert unload_fail_result.success is False
    assert unload_fail_result.current_model == "current"
    assert ("POST", f"{native_root}/models/load", {"model": "new"}) not in unload_fail_requester.calls

    # Load failure preserves the original model after unload succeeds.
    load_fail_requester = FakeRequester({
        ("GET", f"{base_url}/models"): (200, {"data": [{"id": "current"}, {"id": "new"}]}),
        ("GET", f"{native_root}/models"): (
            200,
            {"data": [{"id": "current", "loaded": True, "instance_id": "inst-1"}, {"id": "new"}]},
        ),
        ("POST", f"{native_root}/models/unload"): (200, {}),
        ("POST", f"{native_root}/models/load"): (500, {"error": "load failed"}),
    })
    load_fail_manager = LMStudioModelManager(base_url, "test-key", load_fail_requester)
    load_fail_result = await load_fail_manager.switch_model("current", "new")
    assert load_fail_result.success is False
    assert load_fail_result.current_model == "current"
    assert load_fail_result.load_attempted is True

    # Successful unload/load returns the selected model.
    success_requester = FakeRequester({
        ("GET", f"{base_url}/models"): (200, {"data": [{"id": "current"}, {"id": "new"}]}),
        ("GET", f"{native_root}/models"): (
            200,
            {"data": [{"id": "current", "loaded": True, "instance_id": "inst-1"}, {"id": "new"}]},
        ),
        ("POST", f"{native_root}/models/unload"): (200, {}),
        ("POST", f"{native_root}/models/load"): (200, {"id": "new"}),
    })
    success_manager = LMStudioModelManager(base_url, "test-key", success_requester)
    success_result = await success_manager.switch_model("current", "new")
    assert success_result.success is True
    assert success_result.current_model == "new"
    assert success_result.unload_attempted is True
    assert success_result.load_attempted is True


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
    assert_model_manager_helpers()
    asyncio.run(assert_model_switching())

    print("client smoke test passed")


if __name__ == "__main__":
    main()
