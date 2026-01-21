"""
Chainlit UI module.

Defines the user interface for the MCP Client using Chainlit and delegates
core logic to the ChatEngine. The UI renders step events and the final response.
"""

import chainlit as cl
import sys
import os

# Ensure core is importable when Chainlit is executed from a different CWD.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import ChatEngine
from core.settings import load_settings
from core.memory_store import MemoryStore


@cl.on_chat_start
async def start() -> None:
    """Initialize ChatEngine and establish MCP server connections."""
    settings = load_settings()
    memory_store = MemoryStore(settings.memory_db_path) if settings.memory_enabled else None
    engine = ChatEngine(settings=settings, memory_store=memory_store)
    try:
        await engine.initialize()
        cl.user_session.set("engine", engine)
        cl.user_session.set("memory_store", memory_store)

        incognito = settings.memory_default_incognito
        cl.user_session.set("incognito", incognito)

        if settings.memory_enabled and not incognito:
            engine.start_conversation(persistent=True)
        else:
            engine.start_conversation(persistent=False)

        await engine.start_summary_scheduler()

        # Resource Viewer: list available server resources on startup.
        resources = await engine.get_resources()
        resource_msg = "### Available Resources\n"
        if resources:
            for server, res_list in resources.items():
                resource_msg += f"**Server: {server}**\n"
                for res in res_list:
                    resource_msg += f"- `{res.uri}`: {res.name}\n"
        else:
            resource_msg += "No resources found."

        await cl.Message(content=f"Connected to MCP Servers!\n\n{resource_msg}").send()
    except Exception as e:
        await cl.Message(content=f"Failed to connect to MCP Server: {e}").send()


@cl.on_chat_end
async def end() -> None:
    """Tear down resources when the chat session ends."""
    engine = cl.user_session.get("engine")
    memory_store = cl.user_session.get("memory_store")
    if engine:
        await engine.stop_summary_scheduler()
        await engine.persist_summary()
        await engine.cleanup()
    if memory_store:
        memory_store.close()


@cl.on_message
async def main(message: cl.Message) -> None:
    """Handle user input and stream step events from the ChatEngine."""
    engine = cl.user_session.get("engine")
    if not engine:
        await cl.Message(content="Error: Engine not initialized.").send()
        return

    engine.add_user_message(message.content)

    current_step = None

    # Process the conversation turn and render step-by-step updates.
    async for event in engine.process_turn():
        if event["type"] == "step_start":
            # Start a new step in the UI.
            current_step = cl.Step(name=event["name"], type=event["step_type"])
            current_step.input = event.get("input")
            await current_step.send()

        elif event["type"] == "step_update_name":
            # Update the name of the current step (e.g., correction).
            if current_step:
                current_step.name = event["name"]
                await current_step.update()

        elif event["type"] == "step_output":
            # Update the output of the current step.
            if current_step:
                current_step.output = event["output"]
                await current_step.update()

        elif event["type"] == "message":
            # Send the final response message.
            await cl.Message(content=event["content"]).send()

        elif event["type"] == "usage":
            # Display token usage information.
            source = event.get("source", "llm")
            method = event.get("method", "server")
            usage_info = (
                f"📊 **Token Usage** ({source}, {method}): "
                f"Input: {event['input']} | Output: {event['output']} | Total: {event['total']}"
            )
            await cl.Message(content=usage_info, author="System").send()

        elif event["type"] == "error":
            # Display error messages.
            await cl.Message(content=f"Error: {event['content']}").send()
