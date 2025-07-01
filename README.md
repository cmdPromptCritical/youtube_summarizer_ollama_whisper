# YouTube Transcript Summarizer

Extracts transcripts from YouTube videos using yt-dlp, Whisper, or youtube-transcript-api and summarizes them using Ollama (local LLM).

## Features
- Extracts subtitles/transcripts from YouTube videos
- Supports three transcript sources: yt-dlp (default), Whisper, and youtube-transcript-api
- Summarizes transcripts using Ollama (requires local Ollama server and model)
- Handles long transcripts by chunking
- Optional saving of transcript and audio files

## Requirements
- Python 3.8+
- yt-dlp.exe (in the same directory)
- Ollama running locally (http://localhost:11434) with the `gemma3:27b` model installed
- See `requirements.txt` for Python dependencies

## Installation
1. Download `yt-dlp.exe` from [yt-dlp releases](https://github.com/yt-dlp/yt-dlp/releases) and place it in this directory.
2. Install Python dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Install and start Ollama, and pull the `gemma3:27b` model:
   ```sh
   ollama pull gemma3:27b
   ollama serve
   ```

## Usage Examples

Extract transcript using subtitles (default, fastest):
```
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

Force Whisper transcription (for videos without subtitles):
```
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-source whisper
```

Use a specific Whisper model:
```
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-source whisper --whisper-model large
```

Save transcript and audio files:
```
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-source whisper --save-transcript --save-audio
```

Use the YouTube Transcript API (may work when yt-dlp fails):
```
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-source api
```

Adjust chunk size for large transcripts (default: 80000):
```
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --chunk-size 50000
```

## Notes
- The `--use-whisper` flag is deprecated. Use `--transcript-source whisper` instead.
- Ollama must be running and the required model installed before summarization will work.

## License
MIT License
