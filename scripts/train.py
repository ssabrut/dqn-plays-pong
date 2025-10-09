import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core import wrappers, agent, config

if __name__ == "__main__":
    print(f"Using device: {config.DEVICE}")