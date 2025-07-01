# config.py
"""
Centralized configuration for YouTube Transcript Summarizer
"""

# Path to yt-dlp executable
YTDLP_PATH = "yt-dlp.exe"

# Ollama API settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:27b"
OLLAMA_TIMEOUT = 30  # seconds

# Default Whisper model
DEFAULT_WHISPER_MODEL = "turbo"

# Default chunk size for transcript splitting
DEFAULT_CHUNK_SIZE = 80000

# Supported transcript sources
TRANSCRIPT_SOURCES = ["yt-dlp", "whisper", "api"]
