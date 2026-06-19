# MCP Client

A Chainlit UI client for local MCP demos. It connects to MCP servers over stdio,
passes discovered tools to an OpenAI-compatible chat model, and renders a
step-by-step ReAct workflow in the browser.

This repo targets the sibling `../mcp-server` ArXiv Insight demo server.
`chainlit-mcp-client` is another working copy of the same remote and currently
contains additional local refactor work.

## Requirements

- Python 3.12+
- `uv`
- LM Studio or another OpenAI-compatible API for the full UI demo
- A model whose API endpoint supports OpenAI-style tool calling

Ollama is not required. Legacy `OLLAMA_*` variables are still supported as a
fallback for existing setups.

## Install

```powershell
uv sync --frozen
```

Copy `.env.example` to `.env` and update it for your backend.

## LLM Backend Configuration

New setups should use the generic OpenAI-compatible variables:

```env
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=google/gemma-4-e4b
```

Read priority:

1. `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`
2. `OLLAMA_HOST`, `OLLAMA_KEY`, `OLLAMA_MODEL`
3. LM Studio-style defaults: `http://localhost:1234/v1`, `lm-studio`, `google/gemma-4-e4b`

`google/gemma-4-e4b` is the current model identifier shown by this LM Studio
Local Server. If you load a different model, use the identifier shown by LM
Studio Local Server.

## MCP Server Configuration

The client reads `mcp.json` from the repo root. The default local demo config is:

```json
{
  "mcpServers": {
    "arxiv-insight": {
      "command": "uv",
      "args": ["--directory", "../mcp-server", "run", "main.py"]
    },
    "fetch": {
      "command": "uvx",
      "args": ["mcp-server-fetch"]
    }
  }
}
```

The `fetch` server is optional and may require network access the first time
`uvx` installs it.

## A. Local Verification Without LM Studio

These checks do not require Ollama, LM Studio, or any LLM backend:

```powershell
uv sync --frozen
uv run python scripts\smoke_test.py
```

The smoke test verifies:

- `mcp.json` points to sibling `../mcp-server`.
- The configured server directory exists.
- `LLM_*` defaults work.
- `OLLAMA_*` fallback still works.
- `LLM_*` wins when both new and legacy variables are set.

Passing these checks does not mean the complete Chainlit + LM Studio + MCP tool
calling demo has succeeded.

## B. Full Demo With LM Studio

This flow requires manual validation with a running LM Studio server:

1. Open LM Studio.
2. Load `google/gemma-4-e4b`, or another available model with stable tool calling.
3. Start LM Studio Local Server / OpenAI-compatible API server.
4. Confirm the endpoint, commonly `http://localhost:1234/v1`.
5. Set `.env`:

   ```env
   LLM_BASE_URL=http://localhost:1234/v1
   LLM_API_KEY=lm-studio
   LLM_MODEL=google/gemma-4-e4b
   ```

6. In `../mcp-server`, run `uv run python scripts\smoke_test.py`.
7. In this repo, run `uv run python scripts\smoke_test.py`.
8. Start Chainlit:

   ```powershell
   uv run main.py
   ```

9. Open http://localhost:8000.
10. In Chainlit, ask:

    ```text
    Run health_check, then search for three recent papers about retrieval augmented generation.
    ```

Expected successful behavior:

- Chainlit starts and connects to configured MCP servers.
- The model actually calls `health_check`.
- The model then calls `search_arxiv`.
- arXiv results come from MCP tool output, not from the model inventing citations.

This full end-to-end demo has not been verified in this repo state because it
requires a running LM Studio server, a loaded model, and manual UI/tool-calling
confirmation.

## Validation Status

1. Local static/structure validation: available.
2. MCP server smoke test: available in `../mcp-server`.
3. MCP client config smoke test: available in this repo.
4. Full Chainlit + LM Studio + MCP tool calling demo: not yet fully verified.

Current precise conclusion:

> The repos pass local smoke tests that do not depend on an LLM backend and are
> ready for demo preparation. The full Chainlit + LM Studio + MCP tool calling
> flow still needs manual validation after LM Studio is started and a model is
> loaded.

## Troubleshooting

### No Ollama Installed

This is not a problem. The recommended setup is LM Studio or another
OpenAI-compatible endpoint. `OLLAMA_*` variables are only legacy fallback
settings.

Use:

```env
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=<LM Studio model identifier>
```

### LM Studio Server Is Not Running

Symptoms:

- `connection refused`
- Chainlit starts but the model does not respond
- The demo prompt never triggers tool calling

Fix:

- Open LM Studio.
- Load a model.
- Start Local Server.
- Confirm the port and URL.
- Make `.env` `LLM_BASE_URL` match the LM Studio server URL.

### Model Name Does Not Match

Symptoms:

- `model not found`
- API says the model is unavailable
- Chainlit cannot get a model response

Fix:

- Check the actual model identifier in LM Studio Local Server.
- Set `.env` `LLM_MODEL` to that exact identifier.
- Do not assume the README example name is the API identifier.

### Model Does Not Reliably Support Tool Calling

Symptoms:

- Chainlit returns ordinary text but does not call MCP tools.
- `health_check` is not triggered.
- The model claims it searched papers without tool output.
- arXiv results look invented.

Fix:

- Run `uv run python scripts\smoke_test.py` in this repo and `../mcp-server`.
- Confirm the client exposes MCP tools as OpenAI-compatible tool definitions.
- Try another LM Studio model with stronger tool calling support.
- Treat tool calling success as dependent on model capability, LM Studio endpoint support, and client implementation.

### `health_check` Works But arXiv Search Fails

This usually means the MCP server is running. The issue is more likely network
connectivity, arXiv API availability, query parameters, or another external
service problem.

Check `health_check` first, then test `search_arxiv` with a small query and
`max_results`.

### Memory Carries Context Between Runs

Set:

```env
MEMORY_ENABLED=false
```

or use incognito settings if memory is enabled.
