"""
Chainlit UI module.
Defines the user interface for the MCP Client using Chainlit.
Handles chat events, message rendering, and step visualization.
"""

import chainlit as cl
import sys
import os

# Ensure we can import from core
# When running via `chainlit run interfaces/ui.py`, the CWD is usually the project root if run from there.
# But to be safe, let's add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.engine import ChatEngine

@cl.on_chat_start
async def start():
    """
    Called when a new chat session starts.
    Initializes the ChatEngine and connects to MCP servers.
    """
    engine = ChatEngine()
    try:
        await engine.initialize()
        cl.user_session.set("engine", engine)
        
        # Resource Viewer: List available resources at startup
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
async def end():
    """
    Called when the chat session ends.
    Cleans up resources.
    """
    engine = cl.user_session.get("engine")
    if engine:
        await engine.cleanup()

@cl.on_message
async def main(message: cl.Message):
    """
    Called when a user sends a message.
    Passes the message to the ChatEngine and renders the response.
    """
    engine = cl.user_session.get("engine")
    if not engine:
        await cl.Message(content="Error: Engine not initialized.").send()
        return

    engine.add_user_message(message.content)

    current_step = None

    # Process the conversation turn
    async for event in engine.process_turn():
        if event["type"] == "step_start":
            # Start a new step in the UI
            current_step = cl.Step(name=event["name"], type=event["step_type"])
            current_step.input = event.get("input")
            await current_step.send()
            
        elif event["type"] == "step_update_name":
            # Update the name of the current step (e.g., correction)
            if current_step:
                current_step.name = event["name"]
                await current_step.update()

        elif event["type"] == "step_output":
            # Update the output of the current step
            if current_step:
                current_step.output = event["output"]
                await current_step.update()
                # Note: We don't explicitly "close" the step here because Chainlit 
                # handles step lifecycle mostly via context managers. 
                # However, since we are manually sending, it stays in the UI history.

        elif event["type"] == "message":
            # Send the final response message
            await cl.Message(content=event["content"]).send()

        elif event["type"] == "usage":
            # Display token usage information
            usage_info = f"📊 **Token Usage**: Input: {event['input']} | Output: {event['output']} | Total: {event['total']}"
            await cl.Message(content=usage_info, author="System").send()

        elif event["type"] == "error":
            # Display error messages
            await cl.Message(content=f"Error: {event['content']}").send()

