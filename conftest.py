import sys, os

# Get the absolute path to the project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Add it to sys.path if not already added
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
