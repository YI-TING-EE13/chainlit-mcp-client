# 🤖 MCP Client (Chainlit + MCP)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Chainlit](https://img.shields.io/badge/UI-Chainlit-orange)](https://github.com/Chainlit/chainlit)
[![Protocol](https://img.shields.io/badge/Protocol-MCP-green)](https://modelcontextprotocol.io/)

A professional **Model Context Protocol (MCP)** client with a modern **Chainlit** UI and
seamless **Ollama** integration (or any OpenAI-compatible API). Built for agentic research
workflows, tool orchestration, and structured reasoning across connected MCP servers.

---

## ✨ Key Features

- **MCP Compatibility**: Connects to any MCP-compliant server.
- **Interactive UI**: Powered by Chainlit for a clean, chat-based experience.
- **ReAct Workflow**: Tool usage, analysis, and response synthesis in one loop.
- **Ollama Integration**: Optimized for local models and OpenAI-compatible APIs.
- **Centralized Configuration**: Unified settings for LLM defaults and sampling.
- **uv-First**: Reproducible dependency management and execution.

---

## 🛠️ Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+**
- **uv** (recommended): https://github.com/astral-sh/uv
- **Ollama** (or any OpenAI-compatible API): https://ollama.com/

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/YI-TING-EE13/chainlit-mcp-client.git
   cd chainlit-mcp-client
   ```

2. **Install dependencies**
  Using `uv` ensures a reproducible environment:
   ```bash
   uv sync
   ```

3. **Configure MCP servers**
  The client reads server configurations from mcp.json in the root directory. Ensure this file exists and points to your MCP servers.

    Example mcp.json:
   ```json
   {
     "mcpServers": {
       "arxiv": {
         "command": "uv",
         "args": ["run", "arxiv_insight.py"],
         "env": {
           "PYTHONPATH": "../ArXiv-Insight-MCP-Server"
         }
       }
     }
   }
   ```

4. **Install SQLite (for long-term memory)**
  SQLite is required if you enable long-term memory. On Ubuntu/Debian:
  ```bash
  sudo apt-get update
  sudo apt-get install -y sqlite3
  ```

---

## ⚙️ Configuration

Configuration is centralized in core/settings.py and controlled via environment variables.
Create a .env file in the chainlit-mcp-client directory to configure the LLM connection
and generation hyperparameters. You can copy .env.example as a starting point.

```bash
# .env

# URL for your Ollama instance (default: http://localhost:11434/v1)
OLLAMA_HOST=http://localhost:11434/v1

# API Key (optional for Ollama, required for OpenAI)
OLLAMA_KEY=ollama

# Model to use (ensure you have pulled this model in Ollama)
OLLAMA_MODEL=nemotron-3-nano:latest

# UI display name
ASSISTANT_NAME=Nemotron

# Default chat generation settings
LLM_NUM_CTX=1048576
LLM_MAX_TOKENS=
LLM_TEMPERATURE=0.8
LLM_TOP_P=
LLM_TOP_K=
LLM_REPEAT_PENALTY=
LLM_NUM_PREDICT=

# MCP sampling defaults (used for tool-driven summarization)
SAMPLING_NUM_CTX=1048576
SAMPLING_MAX_TOKENS=4096
SAMPLING_TEMPERATURE=0.8
SAMPLING_TOP_P=
SAMPLING_TOP_K=
SAMPLING_REPEAT_PENALTY=
SAMPLING_NUM_PREDICT=

# Local token usage reporting
TOKEN_USAGE_ENABLED=true
TOKENIZER_MODEL=cl100k_base

# Long-term memory
MEMORY_ENABLED=true
MEMORY_DB_PATH=data/memory.db
MEMORY_DEFAULT_INCOGNITO=false
MEMORY_SUMMARY_ENABLED=true
MEMORY_SUMMARY_MAX_TOKENS=512
MEMORY_SUMMARY_SCHEDULER_ENABLED=true
MEMORY_SUMMARY_INTERVAL_SECONDS=600
```

To pull the default model in Ollama:
```bash
ollama pull nemotron-3-nano
```

---

## 🚀 Usage

### Run the UI
To start the Chainlit chat interface:

```bash
uv run main.py
```
Or explicitly:
```bash
uv run main.py ui
```

The UI will be available at http://localhost:8000.

### Long-term memory
When MEMORY_DEFAULT_INCOGNITO=false, the app stores conversation history and
injects the latest summary into the system context on startup. When set to true,
only the default system prompt is used and no history is written.

### Token usage
Token usage is calculated locally with a tokenizer when TOKEN_USAGE_ENABLED=true.

### Agent Mode (Headless)
Coming soon: a headless mode for automated tasks.
```bash
uv run main.py agent
```

---

## 🏗️ Architecture

The project follows a modular architecture:

```
mcp-client/
├── core/
│   ├── config.py       # System prompts + MCP configuration
│   ├── settings.py     # Centralized LLM + hyperparameter settings
│   ├── engine.py       # Chat Engine (ReAct Loop)
│   ├── llm.py          # LLM Client Wrapper (OpenAI/Ollama)
│   └── mcp_client.py   # MCP Connection & Tool Management
├── interfaces/
│   └── ui.py           # Chainlit UI Event Handlers
├── main.py             # Entry Point
├── mcp.json            # MCP Server Registry
└── pyproject.toml      # Project Metadata & Dependencies
```

### Key Components

- **ChatEngine (core/engine.py)**: Manages conversation history and the ReAct loop.
- **MCPClientWrapper (core/mcp_client.py)**: Connects to MCP servers, routes tool calls, and handles sampling.
- **LLMClient (core/llm.py)**: Thin wrapper over AsyncOpenAI with centralized defaults.
- **MemoryStore (core/memory_store.py)**: SQLite-backed long-term storage for conversations and summaries.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License. See LICENSE for more information.
