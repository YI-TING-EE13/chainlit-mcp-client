import asyncio
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters, stdio_client


CLIENT_ROOT = Path(__file__).resolve().parents[1]
MCP_CONFIG_PATH = CLIENT_ROOT / "mcp.json"
SERVER_NAME = "arxiv-insight"
EXPECTED_TOOLS = {"health_check", "search_arxiv"}
EXPECTED_RESOURCES = {"papers://recent"}
EXPECTED_PROMPTS = {"review_paper", "compare_papers"}


class IntegrationFailure(RuntimeError):
    """Raised when the MCP subprocess integration check fails."""


def load_server_config() -> dict[str, Any]:
    """Load the arxiv-insight server entry from mcp.json."""
    if not MCP_CONFIG_PATH.exists():
        raise IntegrationFailure(f"Missing MCP config: {MCP_CONFIG_PATH}")

    with MCP_CONFIG_PATH.open("r", encoding="utf-8") as config_file:
        config = json.load(config_file)

    servers = config.get("mcpServers", {})
    server = servers.get(SERVER_NAME)
    if not server:
        raise IntegrationFailure(
            f"Missing '{SERVER_NAME}' in {MCP_CONFIG_PATH}. Available servers: {sorted(servers)}"
        )

    command = server.get("command")
    args = server.get("args", [])
    if not command or not isinstance(args, list):
        raise IntegrationFailure(f"Invalid '{SERVER_NAME}' command/args in {MCP_CONFIG_PATH}")

    return {"command": command, "args": [str(arg) for arg in args]}


def validate_configured_server_path(args: list[str]) -> None:
    """Validate relative server paths before spawning the subprocess."""
    if "--directory" not in args:
        return

    directory_index = args.index("--directory") + 1
    if directory_index >= len(args):
        raise IntegrationFailure("'--directory' is present but no server path follows it")

    configured_dir = Path(args[directory_index])
    server_dir = configured_dir if configured_dir.is_absolute() else (CLIENT_ROOT / configured_dir).resolve()
    if not server_dir.exists():
        raise IntegrationFailure(f"Configured MCP server directory does not exist: {server_dir}")
    if not (server_dir / "main.py").exists():
        raise IntegrationFailure(f"Configured MCP server directory is missing main.py: {server_dir}")


def names_missing(expected: set[str], actual: set[str]) -> set[str]:
    """Return missing names from an expected set."""
    return expected - actual


def require_names(kind: str, expected: set[str], actual: set[str]) -> None:
    """Assert that expected MCP names are present, with useful diagnostics."""
    missing = names_missing(expected, actual)
    if missing:
        raise IntegrationFailure(
            f"Missing {kind}: {sorted(missing)}. Actual {kind}: {sorted(actual)}"
        )


def health_payload(result: Any) -> dict[str, Any]:
    """Extract a structured health payload from a CallToolResult."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        nested_result = structured.get("result")
        if isinstance(nested_result, dict):
            return nested_result
        return structured

    content = getattr(result, "content", None)
    if isinstance(content, list):
        for item in content:
            text = getattr(item, "text", None)
            if not text:
                continue
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    raise IntegrationFailure(f"health_check did not return structured JSON content: {result!r}")


def stderr_text(stderr: Any) -> str:
    """Read captured server stderr from a seekable text file."""
    try:
        stderr.flush()
        stderr.seek(0)
        return stderr.read().strip()
    except Exception:
        return ""


async def run_check() -> None:
    """Run the MCP subprocess integration check."""
    server = load_server_config()
    args = server["args"]
    validate_configured_server_path(args)

    params = StdioServerParameters(
        command=server["command"],
        args=args,
        cwd=CLIENT_ROOT,
        encoding_error_handler="replace",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )

    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stderr:
        try:
            with anyio.fail_after(45):
                async with stdio_client(params, errlog=stderr) as (read_stream, write_stream):
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=timedelta(seconds=20),
                    ) as session:
                        await session.initialize()

                        tools_result = await session.list_tools()
                        tool_names = {tool.name for tool in tools_result.tools}
                        require_names("tools", EXPECTED_TOOLS, tool_names)

                        health_result = await session.call_tool("health_check")
                        payload = health_payload(health_result)
                        if not payload.get("status"):
                            raise IntegrationFailure(
                                f"health_check payload is missing 'status': {payload}"
                            )

                        resources_result = await session.list_resources()
                        resource_uris = {str(resource.uri) for resource in resources_result.resources}
                        require_names("resources", EXPECTED_RESOURCES, resource_uris)

                        prompts_result = await session.list_prompts()
                        prompt_names = {prompt.name for prompt in prompts_result.prompts}
                        require_names("prompts", EXPECTED_PROMPTS, prompt_names)

                        print("MCP subprocess integration test passed")
                        print(f"tools={sorted(tool_names)}")
                        print(f"resources={sorted(resource_uris)}")
                        print(f"prompts={sorted(prompt_names)}")
                        print(f"health_status={payload.get('status')}")
        except TimeoutError as exc:
            details = stderr_text(stderr)
            suffix = f"\nServer stderr:\n{details}" if details else ""
            raise IntegrationFailure(f"MCP subprocess integration test timed out.{suffix}") from exc
        except Exception as exc:
            if isinstance(exc, IntegrationFailure):
                raise
            details = stderr_text(stderr)
            suffix = f"\nServer stderr:\n{details}" if details else ""
            raise IntegrationFailure(f"MCP subprocess integration test failed: {exc}{suffix}") from exc


def main() -> None:
    try:
        asyncio.run(run_check())
    except IntegrationFailure as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
