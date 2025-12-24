# MCP Client

A professional, robust **Model Context Protocol (MCP)** client implementation featuring a modern **Chainlit** user interface and seamless integration with **Ollama** (or compatible OpenAI APIs).

This client is designed to connect with various MCP Servers (like the included ArXiv Insight Server) to provide an agentic AI experience capable of tool usage, research, and complex reasoning.

## 🚀 Features

- **Model Context Protocol (MCP) Support**: Fully compliant with the MCP standard to connect with any MCP-compatible server.
- **Interactive UI**: Built with [Chainlit](https://github.com/Chainlit/chainlit) for a beautiful, chat-based user experience.
- **Agentic Workflow**: Implements a ReAct (Reasoning + Acting) loop allowing the AI to autonomously use tools, analyze results, and refine its search.
- **Ollama Integration**: Optimized for local LLMs (e.g., `nemotron-3-nano`, `llama3`) via Ollama, but compatible with any OpenAI-API-compliant provider.
- **Robust Error Handling**: Includes secure input parsing, connection management, and graceful failure recovery.
- **Easy Deployment**: Streamlined dependency management and execution using `uv`.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)**: An extremely fast Python package installer and resolver.
- **[Ollama](https://ollama.com/)**: For running the local LLM (or access to an OpenAI-compatible API).

## 🛠️ Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/YI-TING-EE13/chainlit-mcp-client.git
   cd chainlit-mcp-client
   ```

2. **Install Dependencies**
   Using `uv` ensures a reproducible environment:
   ```bash
   uv sync
   ```

3. **Configure MCP Servers**
   The client reads server configurations from `mcp.json` in the root directory. Ensure this file exists and points to your MCP servers.

   *Example `mcp.json`:*
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

## ⚙️ Configuration

Create a `.env` file in the `mcp-client` directory to configure the LLM connection.

```bash
# .env

# URL for your Ollama instance (default: http://localhost:11434/v1)
OLLAMA_HOST=http://localhost:11434/v1

# API Key (optional for Ollama, required for OpenAI)
OLLAMA_KEY=ollama

# Model to use (ensure you have pulled this model in Ollama)
OLLAMA_MODEL=nemotron-3-nano:latest
```

To pull the default model in Ollama:
```bash
ollama pull nemotron-3-nano
```

## 🚀 Usage

### Running the UI
To start the Chainlit chat interface:

```bash
uv run main.py
```
Or explicitly:
```bash
uv run main.py ui
```

The UI will be available at `http://localhost:8000`.

### Agent Mode (Headless)
*Coming Soon: A headless mode for automated tasks.*
```bash
uv run main.py agent
```

## 🏗️ Architecture

The project follows a modular architecture:

```
mcp-client/
├── core/
│   ├── config.py       # Configuration & System Prompts
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

- **ChatEngine (`core/engine.py`)**: The brain of the application. It manages the conversation history and executes the loop: `LLM Think` -> `Tool Call` -> `MCP Execution` -> `LLM Response`.
- **MCPClientWrapper (`core/mcp_client.py`)**: Handles the complexity of connecting to multiple MCP servers via stdio, managing sessions, and routing tool calls.
- **LLMClient (`core/llm.py`)**: A clean abstraction over the `AsyncOpenAI` client, pre-configured for local Ollama instances.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
