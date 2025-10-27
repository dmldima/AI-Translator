"""
Main entry point for running as module: python -m src
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli.cli_interface import cli

if __name__ == '__main__':
    cli()
