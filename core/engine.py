"""
Chat Engine module.
Orchestrates the interaction between the User, the LLM, and the MCP Tools.
Manages the conversation loop (ReAct pattern).
"""

import json
import asyncio
import ast
from typing import List, Dict, Any, AsyncGenerator

from .llm import LLMClient
from .mcp_client import MCPClientWrapper
from .config import SYSTEM_PROMPT

class ChatEngine:
    """
    The core engine that drives the chat application.
    """
    def __init__(self):
        self.llm = LLMClient()
        self.mcp = MCPClientWrapper(self.llm)
        self.messages: List[Dict[str, Any]] = []
        self.reset_context()

    async def initialize(self):
        """Initialize connections to MCP servers."""
        await self.mcp.connect()

    async def cleanup(self):
        """Cleanup resources and close connections."""
        await self.mcp.cleanup()

    def reset_context(self):
        """Reset conversation history to the initial system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def add_user_message(self, content: str):
        """Add a user message to the history."""
        self.messages.append({"role": "user", "content": content})

    async def get_resources(self) -> Dict[str, Any]:
        """Get available resources from all connected servers."""
        return await self.mcp.list_resources()

    async def process_turn(self, max_turns: int = 10) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a conversation turn using the ReAct pattern (Reasoning + Acting).
        
        Args:
            max_turns (int): Maximum number of tool execution loops to prevent infinite loops.
            
        Yields:
            Dict[str, Any]: Events describing the progress (step_start, step_output, message, error).
        """
        
        # 1. Retrieve available tools from MCP servers
        try:
            response = await self.mcp.list_tools()
            tools = [{
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            } for t in response.tools]
        except Exception as e:
            yield {"type": "error", "content": f"Failed to list tools: {e}"}
            return

        # 2. Main Execution Loop
        for i in range(max_turns):
            # Notify UI that LLM is thinking
            yield {"type": "step_start", "name": "Nemotron", "step_type": "llm", "input": self.messages}
            
            try:
                # Call LLM with current history and tools
                completion = await self.llm.chat_completion(
                    messages=self.messages,
                    tools=tools
                )
                
                # Report token usage if available
                if hasattr(completion, 'usage') and completion.usage:
                    yield {
                        "type": "usage",
                        "input": completion.usage.prompt_tokens,
                        "output": completion.usage.completion_tokens,
                        "total": completion.usage.total_tokens
                    }

                msg = completion.choices[0].message
                yield {"type": "step_output", "output": msg.content or "Tool Call Requested"}
            except Exception as e:
                yield {"type": "error", "content": f"LLM Error: {e}"}
                return

            # 3. Check if LLM wants to call tools
            if not msg.tool_calls:
                # No tool calls -> Final response
                self.messages.append(msg.model_dump()) # Add assistant response to history
                
                # Update the step to show completion
                yield {"type": "step_output", "output": "Response generated."}
                
                # Yield the actual message content to the UI
                yield {"type": "message", "content": msg.content}
                
                # Also print to CLI for debugging/logging
                print(f"\n[Assistant Response]\n{msg.content}\n")
                return

            # 4. Process Tool Calls
            # IMPORTANT: Convert ChatCompletionMessage to dict before appending to messages
            self.messages.append(msg.model_dump()) 
            
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments
                
                yield {"type": "step_start", "name": func_name, "step_type": "tool", "input": func_args}
                
                # Parse arguments safely
                try:
                    args = json.loads(func_args)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for malformed JSON (e.g., single quotes)
                    # This is safer than eval() but still allows Python literals
                    try:
                        args = ast.literal_eval(func_args)
                        if not isinstance(args, dict):
                            raise ValueError("Arguments must be a dictionary")
                    except Exception:
                        args = {} # Fail gracefully or handle error

                # Execute Tool
                try:
                    # Validation: Check if tool exists
                    available_tool_names = [t["function"]["name"] for t in tools]
                    if func_name not in available_tool_names:
                        # Heuristic fix for common hallucination
                        if func_name == "search":
                            func_name = "search_arxiv"
                            yield {"type": "step_update_name", "name": "search_arxiv (corrected)"}
                        else:
                            raise ValueError(f"Tool '{func_name}' not found. Available tools: {available_tool_names}")

                    # Call the tool via MCP
                    result = await self.mcp.call_tool(func_name, arguments=args)
                    
                    # Extract text content from result
                    content_str = ""
                    if isinstance(result.content, list):
                        for item in result.content:
                            if hasattr(item, 'text'):
                                content_str += item.text
                            else:
                                content_str += str(item)
                    else:
                        content_str = str(result.content)

                    # Prepare display output (truncated/summarized for UI)
                    display_output = content_str
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
                                        "summary": paper.get("summary", "")[:200] + "..."
                                    })
                                # Show summarized JSON in UI step
                                display_output = json.dumps(simplified_data, indent=2)
                            else:
                                display_output = content_str[:500] + "..."
                        except:
                            display_output = content_str[:500] + "..."
                    else:
                        display_output = content_str[:500] + "..."

                    yield {"type": "step_output", "output": display_output}

                    # Add result to conversation history
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": content_str
                    })
                    
                except Exception as e:
                    error_msg = f"Error executing tool: {str(e)}"
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": error_msg
                    })
                    yield {"type": "step_output", "output": error_msg}

        # 5. Final Fallback
        # If loop finishes and we still have tool results pending (last message was tool result)
        if self.messages and self.messages[-1].get("role") == "tool":
            yield {"type": "step_start", "name": "Nemotron (Final)", "step_type": "llm", "input": self.messages}
            try:
                completion = await self.llm.chat_completion(messages=self.messages)
                msg = completion.choices[0].message
                yield {"type": "step_output", "output": msg.content}
                self.messages.append(msg.model_dump())
                yield {"type": "message", "content": msg.content}
            except Exception as e:
                yield {"type": "error", "content": f"Final LLM Error: {e}"}

