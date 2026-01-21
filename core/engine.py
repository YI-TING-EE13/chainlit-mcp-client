"""
Chat Engine module.

Orchestrates the interaction between the user, the LLM, and MCP tools.
Implements a ReAct-style loop with tool execution and response synthesis.
"""

import json
import asyncio
import ast
from typing import List, Dict, Any, AsyncGenerator, Optional

from .llm import LLMClient
from .mcp_client import MCPClientWrapper
from .config import SYSTEM_PROMPT
from .settings import AppSettings, load_settings
from .tokenizer import TokenCounter
from .memory_store import MemoryStore

class ChatEngine:
    """
    The core engine that drives the chat application.
    """
    def __init__(self, settings: Optional[AppSettings] = None, memory_store: Optional[MemoryStore] = None):
        self.settings = settings or load_settings()
        self.llm = LLMClient(self.settings)
        self.mcp = MCPClientWrapper(self.llm)
        self.assistant_name = self.settings.assistant_name
        self.messages: List[Dict[str, Any]] = []
        self._token_counter = TokenCounter(self.settings.tokenizer_model)
        self.memory_store = memory_store
        self.conversation_id: Optional[str] = None
        self.persistent_enabled: bool = False
        self._summary_task: Optional[asyncio.Task] = None
        self._memory_dirty: bool = False
        self.reset_context()

    def _build_completion_text(self, msg: Any) -> str:
        """Best-effort string for counting completion tokens."""
        if getattr(msg, "content", None):
            return msg.content or ""

        tool_calls = getattr(msg, "tool_calls", None)
        if not tool_calls:
            return ""

        safe_calls = []
        for call in tool_calls:
            try:
                safe_calls.append({
                    "name": call.function.name,
                    "arguments": call.function.arguments
                })
            except Exception:
                continue

        if not safe_calls:
            return ""

        return json.dumps(safe_calls, ensure_ascii=False)

    async def initialize(self) -> None:
        """Initialize MCP server connections."""
        await self.mcp.connect()

    async def cleanup(self) -> None:
        """Close MCP connections and release resources."""
        await self.mcp.cleanup()

    def reset_context(self) -> None:
        """Reset conversation history to the system prompt."""
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _apply_summary(self, summary: Optional[str]) -> None:
        """Inject a stored summary into the system context when available."""
        if summary:
            self.messages.append({
                "role": "system",
                "content": (
                    "Conversation summary for continuity. "
                    "Use this to maintain context across sessions.\n\n"
                    f"{summary}"
                )
            })

    def start_conversation(self, persistent: bool = True) -> None:
        """Start a new conversation, optionally persisting to storage."""
        self.reset_context()
        self.conversation_id = None
        self.persistent_enabled = False

        if self.memory_store and persistent:
            self.conversation_id = self.memory_store.create_conversation(is_persistent=True)
            self.persistent_enabled = True

    def load_conversation(self, conversation_id: str, persistent: bool = True) -> None:
        """Load an existing conversation, injecting summary and messages as needed."""
        self.reset_context()
        self.conversation_id = conversation_id
        self.persistent_enabled = bool(self.memory_store and persistent)

        summary = self.memory_store.get_summary(conversation_id) if self.memory_store else None
        if self.persistent_enabled:
            self._apply_summary(summary)

        if self.memory_store and self.persistent_enabled:
            stored_messages = self.memory_store.get_messages(conversation_id)
            for msg in stored_messages:
                self.messages.append({"role": msg["role"], "content": msg["content"]})

    def _store_message(self, role: str, content: str) -> None:
        """Persist a message to storage when memory is enabled."""
        if not self.persistent_enabled or not self.memory_store or not self.conversation_id:
            return
        if not content:
            return
        self.memory_store.add_message(self.conversation_id, role, content)
        self._memory_dirty = True

    def _ensure_title(self, user_content: str) -> None:
        """Set a default title from the first user message if missing."""
        if not self.persistent_enabled or not self.memory_store or not self.conversation_id:
            return
        current_title = self.memory_store.get_title(self.conversation_id)
        if current_title:
            return

        title = user_content.strip().split("\n")[0][:60]
        if title:
            self.memory_store.update_title(self.conversation_id, title)

    def add_user_message(self, content: str) -> None:
        """Append a user message to the conversation history."""
        self.messages.append({"role": "user", "content": content})
        self._store_message("user", content)
        self._ensure_title(content)

    async def get_resources(self) -> Dict[str, Any]:
        """Return available resources from all connected servers."""
        return await self.mcp.list_resources()

    async def process_turn(self, max_turns: int = 10) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Process a conversation turn using the ReAct pattern (Reasoning + Acting).
        
        Args:
            max_turns (int): Maximum number of tool execution loops to prevent infinite loops.
            
        Yields:
            Dict[str, Any]: Events describing the progress (step_start, step_output, message, error).
        """
        
        # 1) Retrieve available tools from MCP servers.
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

        # 2) Main execution loop.
        for i in range(max_turns):
            # Notify the UI that the LLM is thinking.
            yield {"type": "step_start", "name": self.assistant_name, "step_type": "llm", "input": self.messages}
            
            try:
                # Call the LLM with current history and tools.
                completion = await self.llm.chat_completion(
                    messages=self.messages,
                    tools=tools
                )
                
                msg = completion.choices[0].message

                # Report token usage (local tokenizer when enabled).
                if self.settings.token_usage_enabled:
                    completion_text = self._build_completion_text(msg)
                    prompt_tokens = self._token_counter.count_messages(self.messages)
                    completion_tokens = self._token_counter.count_text(completion_text)
                    yield {
                        "type": "usage",
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                        "source": "llm",
                        "method": "local"
                    }
                elif hasattr(completion, "usage") and completion.usage:
                    yield {
                        "type": "usage",
                        "input": completion.usage.prompt_tokens,
                        "output": completion.usage.completion_tokens,
                        "total": completion.usage.total_tokens,
                        "source": "llm",
                        "method": "server"
                    }
                yield {"type": "step_output", "output": msg.content or "Tool Call Requested"}
            except Exception as e:
                yield {"type": "error", "content": f"LLM Error: {e}"}
                return

            # 3) Check whether the LLM is requesting tools.
            if not msg.tool_calls:
                # No tool calls: finalize the response.
                self.messages.append(msg.model_dump()) # Add assistant response to history
                self._store_message("assistant", msg.content or "")
                
                # Update the step to show completion.
                yield {"type": "step_output", "output": "Response generated."}
                
                # Yield the actual message content to the UI.
                yield {"type": "message", "content": msg.content}
                
                # Also print to CLI for debugging/logging.
                print(f"\n[Assistant Response]\n{msg.content}\n")
                return

            # 4) Process tool calls.
            # IMPORTANT: Convert ChatCompletionMessage to dict before appending.
            self.messages.append(msg.model_dump()) 
            
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                func_args = tool_call.function.arguments
                
                yield {"type": "step_start", "name": func_name, "step_type": "tool", "input": func_args}
                
                # Parse arguments safely.
                try:
                    args = json.loads(func_args)
                except json.JSONDecodeError:
                    # Fallback to ast.literal_eval for malformed JSON (e.g., single quotes).
                    # This is safer than eval() but still allows Python literals.
                    try:
                        args = ast.literal_eval(func_args)
                        if not isinstance(args, dict):
                            raise ValueError("Arguments must be a dictionary")
                    except Exception:
                        args = {} # Fail gracefully or handle error

                # Execute the tool.
                try:
                    # Validation: ensure the tool exists.
                    available_tool_names = [t["function"]["name"] for t in tools]
                    if func_name not in available_tool_names:
                        # Heuristic fix for common hallucination.
                        if func_name == "search":
                            func_name = "search_arxiv"
                            yield {"type": "step_update_name", "name": "search_arxiv (corrected)"}
                        else:
                            raise ValueError(f"Tool '{func_name}' not found. Available tools: {available_tool_names}")

                    # Call the tool via MCP.
                    result = await self.mcp.call_tool(func_name, arguments=args)
                    
                    # Extract text content from result.
                    content_str = ""
                    if isinstance(result.content, list):
                        for item in result.content:
                            if hasattr(item, 'text'):
                                content_str += item.text
                            else:
                                content_str += str(item)
                    else:
                        content_str = str(result.content)

                    # Prepare display output (truncated/summarized for UI).
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
                                # Show summarized JSON in the UI step.
                                display_output = json.dumps(simplified_data, indent=2)
                            else:
                                display_output = content_str[:500] + "..."
                        except:
                            display_output = content_str[:500] + "..."
                    else:
                        display_output = content_str[:500] + "..."

                    yield {"type": "step_output", "output": display_output}

                    # Emit any buffered sampling token usage events.
                    for usage_event in self.mcp.pop_sampling_usage_events():
                        yield usage_event

                    # Add tool result to conversation history.
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

        # 5) Final fallback.
        # If the loop finishes with a tool result, force a final LLM response.
        if self.messages and self.messages[-1].get("role") == "tool":
            yield {"type": "step_start", "name": f"{self.assistant_name} (Final)", "step_type": "llm", "input": self.messages}
            try:
                completion = await self.llm.chat_completion(messages=self.messages)
                msg = completion.choices[0].message
                if self.settings.token_usage_enabled:
                    completion_text = self._build_completion_text(msg)
                    prompt_tokens = self._token_counter.count_messages(self.messages)
                    completion_tokens = self._token_counter.count_text(completion_text)
                    yield {
                        "type": "usage",
                        "input": prompt_tokens,
                        "output": completion_tokens,
                        "total": prompt_tokens + completion_tokens,
                        "source": "llm",
                        "method": "local"
                    }
                elif hasattr(completion, "usage") and completion.usage:
                    yield {
                        "type": "usage",
                        "input": completion.usage.prompt_tokens,
                        "output": completion.usage.completion_tokens,
                        "total": completion.usage.total_tokens,
                        "source": "llm",
                        "method": "server"
                    }
                yield {"type": "step_output", "output": msg.content}
                self.messages.append(msg.model_dump())
                yield {"type": "message", "content": msg.content}
                self._store_message("assistant", msg.content or "")
            except Exception as e:
                yield {"type": "error", "content": f"Final LLM Error: {e}"}

    async def persist_summary(self) -> None:
        """Summarize the current conversation and store it for future continuity."""
        if not (self.memory_store and self.persistent_enabled and self.conversation_id):
            return
        if not self.settings.memory_summary_enabled:
            return

        messages = self.memory_store.get_messages(self.conversation_id)
        if not messages:
            return

        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        summary_prompt = (
            "Summarize the following conversation for future continuation. "
            "Focus on user goals, decisions, constraints, and open tasks. "
            "Keep it concise.\n\n"
            f"{transcript}"
        )

        completion = await self.llm.chat_completion(
            messages=[
                {"role": "system", "content": "You are a concise conversation summarizer."},
                {"role": "user", "content": summary_prompt}
            ],
            max_tokens=self.settings.memory_summary_max_tokens,
            temperature=0.2
        )

        summary_text = completion.choices[0].message.content or ""
        if summary_text.strip():
            self.memory_store.save_summary(self.conversation_id, summary_text.strip())
            self._memory_dirty = False

    async def start_summary_scheduler(self) -> None:
        """Start the background summary scheduler if enabled."""
        if not self.settings.memory_summary_scheduler_enabled:
            return
        if self._summary_task is not None:
            return

        async def _loop() -> None:
            while True:
                await asyncio.sleep(self.settings.memory_summary_interval_seconds)
                if not self.persistent_enabled or not self._memory_dirty:
                    continue
                await self.persist_summary()

        self._summary_task = asyncio.create_task(_loop())

    async def stop_summary_scheduler(self) -> None:
        """Stop the background summary scheduler."""
        if self._summary_task is None:
            return
        self._summary_task.cancel()
        self._summary_task = None

