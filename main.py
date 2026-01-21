"""
Application entrypoint for the MCP Client.

This module provides a small CLI wrapper to launch the Chainlit UI (default)
or a future headless agent mode. It is intentionally lightweight and delegates
all application logic to the core and interfaces packages.
"""

import argparse
import subprocess
import sys
import os

def run_ui() -> None:
    """Launch the Chainlit UI via uv-managed execution."""
    print("Starting UI mode...")
    # Resolve the UI module path explicitly to avoid CWD ambiguity.
    ui_path = os.path.join(os.path.dirname(__file__), "interfaces", "ui.py")
    
    # Use uv to ensure execution within the managed environment.
    cmd = ["uv", "run", "chainlit", "run", ui_path, "--port", "8000"]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nUI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running UI: {e}")

def run_agent() -> None:
    """Placeholder for a future headless agent mode."""
    print("Agent mode is not yet implemented.")
    print("This mode will allow autonomous execution of tasks without a GUI.")

def main() -> None:
    """Parse CLI arguments and execute the requested mode."""
    parser = argparse.ArgumentParser(description="MCP Client Application")
    parser.add_argument("mode", choices=["ui", "agent"], nargs="?", default="ui", help="Mode to run the application in (default: ui)")
    
    args = parser.parse_args()
    
    if args.mode == "ui":
        run_ui()
    elif args.mode == "agent":
        run_agent()

if __name__ == "__main__":
    main()

