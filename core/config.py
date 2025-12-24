"""
Configuration module for the MCP Client.
Handles environment variables, MCP server configuration, and system prompts.
"""

import os
import json
from dotenv import load_dotenv

# Load environment variables from a .env file if present
load_dotenv()

# --- LLM Configuration ---
# Base URL for the Ollama API (default: localhost)
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")

# Ensure the protocol (http/https) is present in the URL
if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
    OLLAMA_BASE_URL = f"http://{OLLAMA_BASE_URL}"

# Ensure the URL ends with the /v1 API version
if not OLLAMA_BASE_URL.endswith("/v1"):
    OLLAMA_BASE_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/v1"

# API Key for Ollama (usually not required, but good practice for compatibility)
OLLAMA_API_KEY = os.getenv("OLLAMA_KEY", "ollama")

# Model name to use (default: nemotron-3-nano:latest)
# Can be overridden by the OLLAMA_MODEL environment variable
MODEL_NAME = os.getenv("OLLAMA_MODEL", "nemotron-3-nano:latest")

# --- MCP Server Configuration ---
# Determine the root directory of the client to locate mcp.json
CLIENT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
MCP_CONFIG_PATH = os.path.join(CLIENT_ROOT, "mcp.json")

def load_mcp_config() -> dict:
    """
    Load the MCP server configuration from mcp.json.
    
    Returns:
        dict: The configuration dictionary, or an empty dict if loading fails.
    """
    if not os.path.exists(MCP_CONFIG_PATH):
        print(f"Warning: {MCP_CONFIG_PATH} not found.")
        return {}
    try:
        with open(MCP_CONFIG_PATH, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading mcp.json: {e}")
        return {}

# Load the configuration immediately
MCP_CONFIG = load_mcp_config()

# --- System Prompt ---
# Defines the persona and operational guidelines for the AI assistant
SYSTEM_PROMPT = (
    "You are an advanced AI researcher assistant integrated with arXiv search tools. "
    "Your goal is to answer user questions comprehensively, accurately, and professionally using the latest academic papers.\n\n"
    "### TOOL USAGE GUIDELINES\n"
    "- **Search Strategy**: Use specific keywords and boolean operators (AND, OR, NOT) for effective searching. "
    "Avoid long natural language sentences in the search query. "
    "Example: Use `(\"gesture recognition\" OR \"hand pose\") AND \"open set\"` instead of `how to do open set gesture recognition`.\n"
    "- **Iterative Refinement**: If a search returns no results, significantly broaden your query, remove restrictive keywords, or try synonyms. "
    "Never give up after one failed search unless the topic is clearly out of scope.\n"
    "- **Verification**: Do NOT hallucinate. Only discuss papers that are explicitly returned by the search tool. "
    "Verify the paper Title and ID before citing.\n\n"
    "### RESPONSE GUIDELINES\n"
    "- **Synthesis**: Synthesize information from multiple papers to provide a direct, structured answer. Do not just list abstracts.\n"
    "- **Clarity**: Use clear Markdown formatting (headers, bullet points, tables) to organize your response.\n"
    "- **Completeness**: Ensure you address all parts of the user's request (e.g., methods, comparisons, edge deployment).\n"
    "- **User Experience**: When you finish using tools, provide a complete final response. Do not leave the user waiting."
)

# --- Context Window Settings ---
# Maximum context window size (in tokens)
CONTEXT_WINDOW = 1048576  # 1M tokens (large context)
SAMPLING_CONTEXT_WINDOW = 1048576

