"""
Configuration module for RAG application.
Loads environment variables and defines application constants.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_DIR = DATA_DIR / "pdfs"
CHROMA_DIR = DATA_DIR / "chroma"

# Ensure directories exist
PDF_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# LangSmith tracking (optional)
LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "DOT-Maintenance-RAG")

# Model configurations
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-base-en-v1.5")

# ChromaDB configuration
COLLECTION_NAME = "road_maintenance_manuals"

# Chunking configuration
CHUNK_SIZE = 5000  # characters (approximately 1200 tokens)
CHUNK_OVERLAP = 0.125  # 12.5% overlap
CHUNK_OVERLAP_CHARS = int(CHUNK_SIZE * CHUNK_OVERLAP)

# Time-related keywords for tagging
TIME_KEYWORDS = [
    "night",
    "nighttime",
    "daytime",
    "off-peak",
    "peak",
    "curfew",
    "hours of work",
    "work hours",
    "lane closure",
    "closure window",
]

# State configuration
SUPPORTED_STATES = ["CA", "TX", "WA"]
STATE_NAMES = {
    "CA": "California",
    "TX": "Texas",
    "WA": "Washington"
}

# Document type (for phase 1, only maintenance manuals)
DOC_TYPE = "maintenance_manual"

# RAG configuration
DEFAULT_TOP_K = 10
MAX_TOP_K = 20
MIN_TOP_K = 5

