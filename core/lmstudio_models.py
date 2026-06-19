"""
LM Studio model discovery and switching helpers.

The OpenAI-compatible `/v1/models` endpoint is portable across backends. The
native `/api/v1/models/*` management endpoints are LM Studio-specific and must
be treated as optional.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import Request, urlopen


JsonRequester = Callable[
    [str, str, Optional[dict[str, Any]], dict[str, str]],
    Awaitable[tuple[int, Any]],
]


@dataclass(frozen=True)
class ModelInfo:
    """A model visible to the UI."""

    id: str
    loaded: bool = False
    instance_id: Optional[str] = None


@dataclass(frozen=True)
class ModelListResult:
    """Structured model-list response."""

    models: list[ModelInfo]
    native_available: bool
    error: Optional[str] = None


@dataclass(frozen=True)
class ModelSwitchResult:
    """Structured model-switch response."""

    success: bool
    current_model: str
    previous_model: str
    message: str
    native_available: bool
    unload_attempted: bool = False
    load_attempted: bool = False


def derive_native_api_root(base_url: str) -> str:
    """Derive LM Studio's native API root from an OpenAI-compatible base URL."""
    parsed = urlparse(base_url.rstrip("/"))
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[: -len("/v1")]
    native_path = f"{path}/api/v1" if path else "/api/v1"
    return urlunparse(parsed._replace(path=native_path, params="", query="", fragment=""))


def parse_openai_models(payload: Any) -> list[ModelInfo]:
    """Parse OpenAI-compatible `/v1/models` responses."""
    data = payload.get("data", []) if isinstance(payload, dict) else []
    models: list[ModelInfo] = []
    for item in data:
        if isinstance(item, dict):
            model_id = item.get("id") or item.get("model") or item.get("name")
        else:
            model_id = str(item)
        if model_id:
            models.append(ModelInfo(id=str(model_id)))
    return _dedupe_models(models)


def parse_native_models(payload: Any) -> list[ModelInfo]:
    """Parse LM Studio native model-list responses defensively."""
    if isinstance(payload, dict):
        data = payload.get("data") or payload.get("models") or payload.get("loaded") or []
    elif isinstance(payload, list):
        data = payload
    else:
        data = []

    models: list[ModelInfo] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = (
            item.get("id")
            or item.get("key")
            or item.get("model")
            or item.get("model_id")
            or item.get("modelId")
            or item.get("model_key")
            or item.get("modelKey")
            or item.get("name")
        )
        if not model_id:
            continue
        instance_id = (
            item.get("instance_id")
            or item.get("instanceId")
            or item.get("instance")
            or item.get("loaded_model_id")
            or item.get("loadedModelId")
        )
        loaded_instances = item.get("loaded_instances") or item.get("loadedInstances") or []
        if not instance_id and isinstance(loaded_instances, list) and loaded_instances:
            first_instance = loaded_instances[0]
            if isinstance(first_instance, dict):
                instance_id = (
                    first_instance.get("instance_id")
                    or first_instance.get("instanceId")
                    or first_instance.get("id")
                    or first_instance.get("identifier")
                )
        loaded = bool(
            item.get("loaded")
            or item.get("is_loaded")
            or item.get("isLoaded")
            or item.get("state") == "loaded"
            or instance_id
            or loaded_instances
        )
        models.append(ModelInfo(id=str(model_id), loaded=loaded, instance_id=str(instance_id) if instance_id else None))
    return _dedupe_models(models)


def _dedupe_models(models: list[ModelInfo]) -> list[ModelInfo]:
    """Deduplicate by model id while preserving loaded/native details."""
    by_id: dict[str, ModelInfo] = {}
    for model in models:
        existing = by_id.get(model.id)
        if existing is None or (model.loaded and not existing.loaded) or (model.instance_id and not existing.instance_id):
            by_id[model.id] = model
    return list(by_id.values())


async def _default_request_json(
    method: str,
    url: str,
    payload: Optional[dict[str, Any]],
    headers: dict[str, str],
) -> tuple[int, Any]:
    """Make a small JSON request without adding a project dependency."""

    def _request() -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(url, data=body, method=method.upper())
        for key, value in headers.items():
            request.add_header(key, value)
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        try:
            with urlopen(request, timeout=120) as response:
                text = response.read().decode("utf-8")
                return response.status, json.loads(text) if text else {}
        except HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                data = json.loads(text) if text else {}
            except json.JSONDecodeError:
                data = {"error": text}
            return exc.code, data
        except URLError as exc:
            return 0, {"error": str(exc.reason)}
        except (OSError, TimeoutError) as exc:
            return 0, {"error": str(exc)}

    return await asyncio.to_thread(_request)


class LMStudioModelManager:
    """Discover and switch models with LM Studio-specific native fallback."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        requester: Optional[JsonRequester] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.native_api_root = derive_native_api_root(base_url)
        self._requester = requester or _default_request_json
        self._headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
        }

    async def list_models(self) -> ModelListResult:
        """List models from native LM Studio API when available, plus `/v1/models`."""
        openai_models: list[ModelInfo] = []
        native_models: list[ModelInfo] = []
        native_available = False
        errors: list[str] = []

        status, payload = await self._requester("GET", f"{self.base_url}/models", None, self._headers)
        if 200 <= status < 300:
            openai_models = parse_openai_models(payload)
        else:
            errors.append(f"/v1/models unavailable: {_payload_error(payload) or status}")

        native_status, native_payload = await self._requester("GET", f"{self.native_api_root}/models", None, self._headers)
        if 200 <= native_status < 300:
            native_available = True
            native_models = parse_native_models(native_payload)

        merged = _dedupe_models(native_models + openai_models)
        return ModelListResult(
            models=merged,
            native_available=native_available,
            error="; ".join(errors) if errors and not merged else None,
        )

    async def switch_model(self, current_model: str, selected_model: str) -> ModelSwitchResult:
        """Switch the runtime model, using native unload/load when available."""
        if selected_model == current_model:
            return ModelSwitchResult(
                success=True,
                current_model=current_model,
                previous_model=current_model,
                message=f"Already using {current_model}.",
                native_available=False,
            )

        listed = await self.list_models()
        available_ids = {model.id for model in listed.models}
        if available_ids and selected_model not in available_ids:
            return ModelSwitchResult(
                success=False,
                current_model=current_model,
                previous_model=current_model,
                message=f"Model '{selected_model}' is not listed by the configured endpoint.",
                native_available=listed.native_available,
            )

        if not listed.native_available:
            if available_ids and selected_model in available_ids:
                return ModelSwitchResult(
                    success=True,
                    current_model=selected_model,
                    previous_model=current_model,
                    message=(
                        "LM Studio native model-management API is unavailable; "
                        "updated runtime model only. Switch manually in LM Studio if needed."
                    ),
                    native_available=False,
                )
            return ModelSwitchResult(
                success=False,
                current_model=current_model,
                previous_model=current_model,
                message="Could not verify selected model because model listing is unavailable.",
                native_available=False,
            )

        current_info = self._find_loaded_model(listed.models, current_model)
        unload_attempted = False
        if current_info and current_info.instance_id:
            unload_attempted = True
            unload_status, unload_payload = await self._requester(
                "POST",
                f"{self.native_api_root}/models/unload",
                {"instance_id": current_info.instance_id},
                self._headers,
            )
            if not 200 <= unload_status < 300:
                return ModelSwitchResult(
                    success=False,
                    current_model=current_model,
                    previous_model=current_model,
                    message=f"Failed to unload current model: {_payload_error(unload_payload) or unload_status}",
                    native_available=True,
                    unload_attempted=True,
                )
        elif current_info and current_info.loaded:
            return ModelSwitchResult(
                success=False,
                current_model=current_model,
                previous_model=current_model,
                message="Current model appears loaded, but no native instance_id was available for unload.",
                native_available=True,
            )

        load_status, load_payload = await self._requester(
            "POST",
            f"{self.native_api_root}/models/load",
            {"model": selected_model},
            self._headers,
        )
        if not 200 <= load_status < 300:
            return ModelSwitchResult(
                success=False,
                current_model=current_model,
                previous_model=current_model,
                message=f"Failed to load selected model: {_payload_error(load_payload) or load_status}",
                native_available=True,
                unload_attempted=unload_attempted,
                load_attempted=True,
            )

        return ModelSwitchResult(
            success=True,
            current_model=selected_model,
            previous_model=current_model,
            message=f"Switched model to {selected_model}.",
            native_available=True,
            unload_attempted=unload_attempted,
            load_attempted=True,
        )

    @staticmethod
    def _find_loaded_model(models: list[ModelInfo], model_id: str) -> Optional[ModelInfo]:
        for model in models:
            if model.id == model_id and (model.loaded or model.instance_id):
                return model
        return None


def _payload_error(payload: Any) -> Optional[str]:
    if isinstance(payload, dict):
        error = payload.get("error") or payload.get("message") or payload.get("detail")
        if isinstance(error, dict):
            return error.get("message") or str(error)
        if error:
            return str(error)
    return None
