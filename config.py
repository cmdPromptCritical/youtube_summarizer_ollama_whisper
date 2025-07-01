# config.py
"""
Centralized configuration for YouTube Transcript Summarizer
"""

# Path to yt-dlp executable
YTDLP_PATH = "yt-dlp.exe"

# Ollama API settings
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:27b"
OLLAMA_TIMEOUT = 900  # seconds

# Default Whisper model
DEFAULT_WHISPER_MODEL = "turbo"

# Default chunk size for transcript splitting. 80,000 characters is roughly 1h of audio
DEFAULT_CHUNK_SIZE = 80000

TRANSCRIPT_SOURCES = ["yt-dlp", "whisper", "api"]

# Prompt Template to summarize YouTube video transcripts
SUMMARY_INSTRUCTIONS = """Include:
1. Main topics discussed
2. Key points and insights
3. Important conclusions or takeaways"""







# Advanced Prompt templates for summarization. This is called on the first chunk of text
PROMPT_INITIAL = """Please provide a comprehensive summary of the following YouTube video transcript chunk (part {chunk_number} of {total_chunks}).
{instructions}

Transcript chunk:
{chunk}

Summary:"""

# Advanced Prompt templates for summarization. This is called on subsequent chunks of text (usually for videos >1h in duration)
PROMPT_CONTINUATION = """You are continuing to summarize a YouTube video transcript. Below is the summary so far and the next chunk of the transcript.

PREVIOUS SUMMARY:
{running_summary}

CURRENT TRANSCRIPT CHUNK ({chunk_number} of {total_chunks}):
{chunk}

Please update and expand the summary by integrating the new information from this chunk. Maintain the same structure.
{instructions}

UPDATED SUMMARY:"""
