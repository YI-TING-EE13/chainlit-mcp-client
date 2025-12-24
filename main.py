"""
Main entry point for the MCP Client Application.
Supports running in UI mode (Chainlit) or Agent mode (Headless - Future Work).
"""

import argparse
import subprocess
import sys
import os

def run_ui():
    """
    Run the Chainlit UI.
    Launches the Chainlit server using the `uv` package manager.
    """
    print("Starting UI mode...")
    # Path to the UI file
    ui_path = os.path.join(os.path.dirname(__file__), "interfaces", "ui.py")
    
    # Use 'uv run' to ensure we are in the correct environment
    # This assumes 'uv' is installed and available in the system PATH
    cmd = ["uv", "run", "chainlit", "run", ui_path, "--port", "8000"]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nUI stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running UI: {e}")

def run_agent():
    """
    Run the Agent mode (Placeholder).
    This mode is intended for autonomous execution without a GUI.
    """
    print("Agent mode is not yet implemented.")
    print("This mode will allow autonomous execution of tasks without a GUI.")

def main():
    """
    Parse command line arguments and execute the requested mode.
    """
    parser = argparse.ArgumentParser(description="MCP Client Application")
    parser.add_argument("mode", choices=["ui", "agent"], nargs="?", default="ui", help="Mode to run the application in (default: ui)")
    
    args = parser.parse_args()
    
    if args.mode == "ui":
        run_ui()
    elif args.mode == "agent":
        run_agent()

if __name__ == "__main__":
    main()

