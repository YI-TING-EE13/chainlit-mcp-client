import asyncio
import os
import json
from contextlib import AsyncExitStack
from typing import Any

import chainlit as cl
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import mcp.types as types
from openai import AsyncOpenAI

# Load environment variables
load_dotenv()

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434/v1")

# Ensure protocol is present
if not OLLAMA_BASE_URL.startswith(("http://", "https://")):
    OLLAMA_BASE_URL = f"http://{OLLAMA_BASE_URL}"

if not OLLAMA_BASE_URL.endswith("/v1"):
    OLLAMA_BASE_URL = f"{OLLAMA_BASE_URL.rstrip('/')}/v1"

OLLAMA_API_KEY = os.getenv("OLLAMA_KEY", "ollama")
MODEL_NAME = "nemotron-3-nano"

SYSTEM_PROMPT = (
    "You are a helpful research assistant. You have access to tools to search for arXiv papers and read them. "
    "When asked to find papers, use the 'search_arxiv' tool. "
    "When asked to read a paper, use the 'get_paper_fulltext' tool. "
    "IMPORTANT: When you receive tool results, you MUST answer the user's specific questions based on those results. "
    "Do not just list the papers or topics. Synthesize the information to provide a direct answer. "
    "If the user asks for methods, list the methods found in the summaries. "
    "If the user asks for metrics, list the metrics found. "
    "If the user asks for feasibility, look for keywords like 'lightweight', 'mobile', 'edge', 'real-time' in the summaries."
)

# Server Configuration
SERVER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../ArXiv-Insight-MCP-Server"))
SERVER_SCRIPT = "arxiv_insight.py"

server_params = StdioServerParameters(
    command="uv",
    args=["run", "--directory", SERVER_DIR, SERVER_SCRIPT],
    env=None
)

class NemotronMCPClient:
    def __init__(self):
        self.openai = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY)
        self.exit_stack = AsyncExitStack()
        self.session = None

    async def handle_sampling(self, context: Any, params: types.CreateMessageRequestParams) -> types.CreateMessageResult:
        """
        Handle sampling requests from the server (e.g. for summarization).
        """
        print(f"Sampling requested by server. Max tokens: {params.maxTokens}")
        
        # Convert MCP messages to OpenAI messages
        openai_messages = []
        
        # Add system prompt if present
        if params.systemPrompt:
            openai_messages.append({"role": "system", "content": params.systemPrompt})
            
        for msg in params.messages:
            if msg.content.type == "text":
                openai_messages.append({"role": msg.role, "content": msg.content.text})
            # Note: Image content handling omitted for simplicity
            
        try:
            # Call LLM
            completion = await self.openai.chat.completions.create(
                model=MODEL_NAME,
                messages=openai_messages,
                max_tokens=params.maxTokens or 1048576,
                temperature=params.temperature or 0.8,
                extra_body={
                    "options": {
                        "num_ctx": 1048576 # Ensure we have enough context for the summary
                    }
                }
            )
            
            text = completion.choices[0].message.content
            
            return types.CreateMessageResult(
                role="assistant",
                content=types.TextContent(type="text", text=text),
                model=MODEL_NAME,
                stopReason="end_turn"
            )
        except Exception as e:
            print(f"Sampling failed: {e}")
            return types.CreateMessageResult(
                role="assistant",
                content=types.TextContent(type="text", text=f"Error during sampling: {str(e)}"),
                model=MODEL_NAME,
                stopReason="error"
            )

    async def connect(self):
        print(f"Connecting to MCP Server at {SERVER_DIR}...")
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        
        # Initialize session with sampling callback
        self.session = await self.exit_stack.enter_async_context(
            ClientSession(
                read, 
                write,
                sampling_callback=self.handle_sampling
            )
        )
        
        await self.session.initialize()
        print("Connected to arXiv-Insight Server!")

    async def cleanup(self):
        await self.exit_stack.aclose()

@cl.on_chat_start
async def start():
    client = NemotronMCPClient()
    try:
        await client.connect()
        cl.user_session.set("client", client)
        cl.user_session.set("messages", [{"role": "system", "content": SYSTEM_PROMPT}])
        await cl.Message(content="Connected to ArXiv Insight MCP Server! Ready to help you research.").send()
    except Exception as e:
        await cl.Message(content=f"Failed to connect to MCP Server: {e}").send()

@cl.on_chat_end
async def end():
    client = cl.user_session.get("client")
    if client:
        await client.cleanup()

@cl.on_message
async def main(message: cl.Message):
    client = cl.user_session.get("client")
    if not client:
        await cl.Message(content="Error: Client not initialized.").send()
        return

    user_input = message.content
    
    messages = cl.user_session.get("messages")
    messages.append({"role": "user", "content": user_input})

    # Get tools
    response = await client.session.list_tools()
    tools = [{
        "type": "function",
        "function": {
            "name": t.name,
            "description": t.description,
            "parameters": t.inputSchema
        }
    } for t in response.tools]

    # Loop for handling tool calls (max 10 turns)
    for i in range(10):
        # Create a step to show "Thinking..."
        async with cl.Step(name="Nemotron", type="llm") as step:
            step.input = messages
            
            completion = await client.openai.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=tools,
                extra_body={
                    "options": {
                        "num_ctx": 1048576
                    }
                }
            )
            
            msg = completion.choices[0].message
            step.output = msg.content or "Tool Call Requested"

        # If no tool calls, send the response and break
        if not msg.tool_calls:
            await cl.Message(content=msg.content).send()
            break
        
        # If there are tool calls, process them
        messages.append(msg)
        
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            func_args = tool_call.function.arguments
            
            # Create a step for the tool execution
            async with cl.Step(name=func_name, type="tool") as tool_step:
                tool_step.input = func_args
                
                try:
                    args = json.loads(func_args)
                except json.JSONDecodeError:
                    try:
                        args = eval(func_args)
                    except:
                        args = {}

                # Execute tool
                try:
                    # Check if tool exists
                    available_tool_names = [t["function"]["name"] for t in tools]
                    if func_name not in available_tool_names:
                        # Handle hallucinated tool names
                        if func_name == "search":
                            func_name = "search_arxiv"
                            tool_step.name = "search_arxiv (corrected)"
                        else:
                            raise ValueError(f"Tool '{func_name}' not found. Available tools: {available_tool_names}")

                    result = await client.session.call_tool(func_name, arguments=args)
                    
                    # Extract text content cleanly
                    content_str = ""
                    if isinstance(result.content, list):
                        for item in result.content:
                            if hasattr(item, 'text'):
                                content_str += item.text
                            else:
                                content_str += str(item)
                    else:
                        content_str = str(result.content)

                    # Optimization for search results
                    if func_name == "search_arxiv":
                        try:
                            data = json.loads(content_str)
                            if isinstance(data, list):
                                simplified_data = []
                                for paper in data:
                                    simplified_data.append({
                                        "id": paper.get("id"),
                                        "title": paper.get("title"),
                                        "published": paper.get("published"),
                                        "summary": paper.get("summary", "")[:500]
                                    })
                                content_str = json.dumps(simplified_data, indent=2)
                                tool_step.output = "Search results retrieved (summarized)."
                            else:
                                tool_step.output = content_str[:500] + "..."
                        except:
                            tool_step.output = content_str[:500] + "..."
                    else:
                        tool_step.output = content_str[:500] + "..." # Show preview in UI

                    # Add result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": content_str
                    })
                    
                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })
                    tool_step.output = error_msg

    # If loop finishes and we still have tool results pending, force a final response
    if messages and messages[-1].get("role") == "tool":
        async with cl.Step(name="Nemotron (Final)", type="llm") as step:
            step.input = messages
            completion = await client.openai.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                extra_body={
                    "options": {
                        "num_ctx": 1048576
                    }
                }
            )
            msg = completion.choices[0].message
            step.output = msg.content
            await cl.Message(content=msg.content).send()
