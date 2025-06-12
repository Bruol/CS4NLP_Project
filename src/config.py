import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the Google API key from the environment
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

DEEP_SEEK_API_KEY = os.getenv("DEEP_SEEK_API_KEY")

if not DEEP_SEEK_API_KEY:
    raise ValueError("DEEP_SEEK_API_KEY not found in environment variables. Please set it in your .env file.")