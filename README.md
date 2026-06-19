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

## Runtime Model Selection

When Chainlit starts, the UI shows an **LM Studio model** selector in chat
settings. The model list is loaded dynamically from the configured
OpenAI-compatible endpoint:

```text
GET http://localhost:1234/v1/models
```

The startup default remains `LLM_MODEL=google/gemma-4-e4b`. Selecting another
model changes only the current Chainlit session; it does not edit `.env`.

For LM Studio, the client also derives the native model-management API root from
`LLM_BASE_URL`. For example:

```text
http://localhost:1234/v1 -> http://localhost:1234/api/v1
```

When the native API is available, switching models attempts to unload the
currently loaded model before loading the selected model. This is LM
Studio-specific. Other OpenAI-compatible backends may only support `/v1/models`;
in that case the UI keeps working, updates the runtime model when possible, and
asks you to switch models manually in LM Studio if loading fails.

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
- The resource list shows `papers://recent` from `arxiv-insight`.
- The model actually calls `health_check`.
- The model then calls `search_arxiv`.
- arXiv results come from MCP tool output, not from the model inventing citations.

This full end-to-end demo has been verified on Windows with LM Studio Local
Server, Chainlit, MCP tool calling, and the sibling `../mcp-server`.

Verified baseline:

- Endpoint: `http://localhost:1234/v1`
- Model identifier: `google/gemma-4-e4b`
- Resource shown: `papers://recent`
- Successful tool calls: `health_check`, then `search_arxiv`

## Validation Status

1. Local static/structure validation: available.
2. MCP server smoke test: available in `../mcp-server`.
3. MCP client config smoke test: available in this repo.
4. Full Chainlit + LM Studio + MCP tool calling demo: verified on Windows.

Current precise conclusion:

> The repos pass local smoke tests that do not depend on an LLM backend. The
> full Windows + LM Studio + Chainlit + MCP tool calling flow has also been
> manually verified with `google/gemma-4-e4b` at `http://localhost:1234/v1`.

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

### Model List Is Empty

Symptoms:

- The Chainlit model selector only shows the `.env` default.
- `/v1/models` fails or returns no model IDs.

Fix:

- Confirm LM Studio Local Server is running.
- Open `http://localhost:1234/v1/models` and verify it returns model IDs.
- Keep using the default model if the API is temporarily unavailable.

### Model Unload Failed

Symptoms:

- The UI reports that unloading the current model failed.
- The selected model is not loaded.

Fix:

- Check whether LM Studio exposes a loaded model instance ID through its native
  API.
- Unload the model manually in LM Studio, then retry.
- If VRAM is still full, close other loaded models or restart the LM Studio
  Local Server.

### Model Load Failed

Symptoms:

- The UI reports that loading the selected model failed.
- The session keeps using the previous model.

Fix:

- Confirm the selected model is installed and loadable in LM Studio.
- Check available VRAM.
- Try loading the model manually in LM Studio first.

### LM Studio Native API Is Unavailable

Symptoms:

- The model selector works, but the UI says native unload/load is unavailable.

Fix:

- This is expected for non-LM Studio OpenAI-compatible backends.
- For LM Studio, confirm your version exposes `http://localhost:1234/api/v1`.
- If native model management is unavailable, switch models manually in LM Studio
  and then select the matching model in Chainlit.

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

### LM Studio Reports `Model reloaded`

Symptoms:

```text
400 - {'error': 'Model reloaded.'}
```

This can happen briefly while LM Studio reloads the selected model. Wait for the
model to finish loading, then retry the same prompt.

### Fetch Server Resource Warning

Symptoms:

```text
Error listing resources from fetch: Method not found
```

This is a non-blocking warning. The optional `fetch` server does not expose MCP
resources, so resource listing can fail for that server while the `arxiv-insight`
server still works.

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

### Demo Failure Log Checklist

If the full demo fails, collect:

- Chainlit terminal output or `.demo-logs` output.
- LM Studio Local Server log.
- Non-sensitive `.env` values: `LLM_BASE_URL`, `LLM_MODEL`, and whether
  `LLM_API_KEY` is set.
- `mcp.json`.
- Whether Chainlit showed `papers://recent`.
- Whether `health_check` and `search_arxiv` appeared as actual tool calls.
- The exact error message.

Do not paste real API keys, tokens, or private credentials.
