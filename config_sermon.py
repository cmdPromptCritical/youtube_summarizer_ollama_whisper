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
SUMMARY_INSTRUCTIONS = """Please analyze the provided sermon transcript and create a comprehensive summary with the following structure:
1. Main Theme: Identify and clearly state the central message or primary theme of the sermon in 2-3 sentences.
2. Comprehensive Talking Points: Create a detailed list capturing all significant elements, including:
* Key theological concepts and teachings discussed
* Biblical passages referenced (book, chapter, verse)
* Notable quotes, phrases, or memorable statements by the speaker
* Analogies, metaphors, or illustrations used
* Personal stories or anecdotes shared
* Practical applications or life lessons mentioned
* Questions posed to the congregation
* Any historical or cultural context provided
* Supporting arguments or sub-points that reinforce the main theme
3. Closing Remarks: Summarize how the sermon concluded, including:
* Final thoughts or challenges presented
* Calls to action or invitations extended
* Closing prayers or benedictions mentioned
* Any final encouragement or warnings given
Instructions:
* Be thorough and capture even brief mentions or passing references
* Maintain the original meaning and context of all points
* Use bullet points or numbered lists for clarity
* Include approximate timestamps if they would be helpful for reference
* Note any particularly emphasized or repeated concepts
Please ensure nothing significant is overlooked in your analysis.
"""







# Advanced Prompt templates for summarization. This is called on the first chunk of text
PROMPT_INITIAL = """{instructions}

Transcript chunk {chunk_number} of {total_chunks}:
{chunk}

Summary:"""

# Advanced Prompt templates for summarization. This is called on subsequent chunks of text (usually for videos >1h in duration)
PROMPT_CONTINUATION = """You are continuing to summarize a YouTube video transcript. Below is the summary so far and the next chunk of the transcript.

PREVIOUS SUMMARY:
{running_summary}

CURRENT TRANSCRIPT CHUNK ({chunk_number} of {total_chunks}):
{chunk}

Please update and expand the summary by integrating the new information from this chunk.
{instructions}

UPDATED SUMMARY:"""
