"""Pytest configuration to add project root to Python path."""
import sys
from pathlib import Path

# Add project root to Python path so 'src' package can be imported
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))