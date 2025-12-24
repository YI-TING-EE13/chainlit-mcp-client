"""
MCP Client Wrapper module.
Manages connections to Model Context Protocol (MCP) servers, tool execution, and resource retrieval.
"""

import asyncio
import os
from contextlib import AsyncExitStack
from typing import Any, Optional, Dict, List

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as types

from .config import MCP_CONFIG, CLIENT_ROOT
from .llm import LLMClient

class MCPClientWrapper:
    """
    Wraps the MCP client functionality to manage multiple server connections.
    """
    def __init__(self, llm_client: LLMClient):
        """
        Initialize the MCP Client Wrapper.

        Args:
            llm_client (LLMClient): The LLM client instance to use for sampling requests.
        """
        self.llm_client = llm_client
        self.exit_stack = AsyncExitStack()
        self.sessions: Dict[str, ClientSession] = {}
        self.tool_to_server: Dict[str, str] = {} # Map tool name to server name

    async def connect(self):
        """
        Connect to all MCP servers defined in the configuration (mcp.json).
        Establishes stdio connections and initializes sessions.
        """
        servers = MCP_CONFIG.get("mcpServers", {})
        if not servers:
            print("No servers found in mcp.json")
            return

        for server_name, config in servers.items():
            print(f"Connecting to MCP Server: {server_name}...")
            
            command = config.get("command")
            args = config.get("args", [])
            env = config.get("env")
            
            # Resolve relative paths in args if necessary (simple heuristic)
            # If arg looks like a path and starts with ./ or ../, resolve it relative to CLIENT_ROOT
            resolved_args = []
            for arg in args:
                if arg.startswith(("./", "../")):
                    resolved_args.append(os.path.abspath(os.path.join(CLIENT_ROOT, arg)))
                else:
                    resolved_args.append(arg)

            server_params = StdioServerParameters(
                command=command,
                args=resolved_args,
                env=env
            )

            try:
                # Establish the stdio transport
                stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
                read, write = stdio_transport
                
                # Create and initialize the client session
                session = await self.exit_stack.enter_async_context(
                    ClientSession(
                        read, 
                        write,
                        sampling_callback=self.handle_sampling
                    )
                )
                
                await session.initialize()
                self.sessions[server_name] = session
                print(f"Connected to {server_name}!")
            except Exception as e:
                print(f"Failed to connect to {server_name}: {e}")

    async def cleanup(self):
        """
        Clean up resources and close all server connections.
        """
        try:
            await self.exit_stack.aclose()
        except RuntimeError as e:
            # Ignore anyio task scope error during cleanup in Chainlit
            # This is a known issue with Chainlit's async lifecycle
            if "Attempted to exit cancel scope" in str(e):
                pass
            else:
                raise
        except Exception as e:
            print(f"Error during cleanup: {e}")

    async def list_tools(self) -> types.ListToolsResult:
        """
        List available tools from all connected servers.
        
        Returns:
            types.ListToolsResult: A result object containing the list of tools.
        """
        all_tools = []
        self.tool_to_server.clear()

        for server_name, session in self.sessions.items():
            try:
                result = await session.list_tools()
                for tool in result.tools:
                    # We might want to namespace tools if there are collisions, 
                    # but for now assume unique names or last-wins.
                    all_tools.append(tool)
                    self.tool_to_server[tool.name] = server_name
            except Exception as e:
                print(f"Error listing tools from {server_name}: {e}")
        
        return types.ListToolsResult(tools=all_tools)

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """
        Call a specific tool on the appropriate server.

        Args:
            name (str): The name of the tool to call.
            arguments (dict): The arguments to pass to the tool.

        Returns:
            Any: The result of the tool execution.
        """
        server_name = self.tool_to_server.get(name)
        if not server_name:
            raise ValueError(f"Tool {name} not found in any connected server.")
        
        session = self.sessions.get(server_name)
        if not session:
             raise RuntimeError(f"Session for {server_name} is not active.")

        return await session.call_tool(name, arguments=arguments)

    async def list_resources(self) -> Dict[str, List[types.Resource]]:
        """
        List resources from all connected servers.

        Returns:
            Dict[str, List[types.Resource]]: A dictionary mapping server names to their resources.
        """
        all_resources = {}
        for server_name, session in self.sessions.items():
            try:
                result = await session.list_resources()
                all_resources[server_name] = result.resources
            except Exception as e:
                print(f"Error listing resources from {server_name}: {e}")
        return all_resources

    async def read_resource(self, uri: str) -> Any:
        """
        Read a specific resource by URI.
        Iterates through all servers to find one that can read the resource.

        Args:
            uri (str): The URI of the resource to read.

        Returns:
            Any: The content of the resource.
        """
        # This is tricky because we don't have a map of URI -> Server.
        # We have to try all servers or rely on list_resources having been called.
        # A simple approach: Try all servers, return first success.
        
        for server_name, session in self.sessions.items():
            try:
                return await session.read_resource(uri)
            except Exception:
                continue
        
        raise ValueError(f"Resource {uri} not found or readable on any server.")

    async def handle_sampling(self, context: Any, params: types.CreateMessageRequestParams) -> types.CreateMessageResult:
        """
        Handle sampling requests from the server (e.g., for summarization or agentic behavior).
        This allows the MCP server to ask the client's LLM to generate text.

        Args:
            context (Any): The context of the request.
            params (types.CreateMessageRequestParams): The parameters for the message generation.

        Returns:
            types.CreateMessageResult: The generated message result.
        """
        print(f"Sampling requested by server. Max tokens: {params.maxTokens}")
        
        # Convert MCP messages to OpenAI messages format
        openai_messages = []
        
        # Add system prompt if present
        if params.systemPrompt:
            openai_messages.append({"role": "system", "content": params.systemPrompt})
            
        for msg in params.messages:
            if msg.content.type == "text":
                openai_messages.append({"role": msg.role, "content": msg.content.text})
            # Note: Image content handling omitted for simplicity
            
        try:
            # Call LLM via our wrapper
            completion = await self.llm_client.chat_completion(
                messages=openai_messages,
                max_tokens=params.maxTokens or 4096,
                temperature=params.temperature or 0.8,
            )
            
            text = completion.choices[0].message.content
            
            return types.CreateMessageResult(
                role="assistant",
                content=types.TextContent(type="text", text=text),
                model=self.llm_client.model,
                stopReason="end_turn"
            )
        except Exception as e:
            print(f"Sampling failed: {e}")
            return types.CreateMessageResult(
                role="assistant",
                content=types.TextContent(type="text", text=f"Error during sampling: {str(e)}"),
                model=self.llm_client.model,
                stopReason="error"
            )

