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

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

OPENAI_AZURE_API_KEY = os.getenv("OPENAI_AZURE_API_KEY")

if not OPENAI_AZURE_API_KEY:
    raise ValueError("OPENAI_AZURE_API_KEY not found in environment variables. Please set it in your .env file.")
