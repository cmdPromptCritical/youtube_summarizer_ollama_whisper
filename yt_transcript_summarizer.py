#!/usr/bin/env python3
"""
YouTube Transcript Summarizer
Extracts transcript from YouTube video using yt-dlp, Whisper, or youtube-transcript-api and summarizes it using Ollama

USAGE EXAMPLES:

# 1. Extract transcript using subtitles (default, fastest)
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID"

# 2. Force Whisper transcription (for videos without subtitles)
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-source whisper

# 3. Use a specific Whisper model for better accuracy or speed
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-source whisper --whisper-model large

# 4. Save transcript and audio files
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-source whisper --save-transcript --save-audio

# 5. Use the YouTube Transcript API (may work when yt-dlp fails)
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript-source api

# 6. Adjust chunk size for large transcripts (default: 80000)
python yt_transcript_summarizer.py "https://www.youtube.com/watch?v=VIDEO_ID" --chunk-size 50000

# Note: --use-whisper is deprecated, use --transcript-source whisper instead.

"""

import subprocess
import json
import requests
import sys
import os
import tempfile
import argparse
from pathlib import Path

def check_dependencies(use_whisper=False, transcript_source="yt-dlp"):
    """Check if required dependencies are available"""
    # Check if yt-dlp.exe exists for relevant sources
    if transcript_source == "yt-dlp" and not os.path.exists("yt-dlp.exe"):
        print("Error: yt-dlp.exe not found in current directory")
        print("Please download yt-dlp.exe from https://github.com/yt-dlp/yt-dlp/releases")
        return False
    
    # Check for youtube-transcript-api if selected
    if transcript_source == "api":
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            print("Error: youtube-transcript-api not installed")
            print("Please install with: pip install youtube-transcript-api")
            return False

    # Check Whisper if needed
    if use_whisper or transcript_source == "whisper":
        try:
            import whisper
            print("OpenAI Whisper found")
        except ImportError:
            print("Error: OpenAI Whisper not installed")
            print("Please install with: pip install openai-whisper")
            return False
    
    # Check if Ollama is running
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            if not any("gemma3:27b" in model.get("name", "") for model in models):
                print("Warning: gemma3:27b model not found. Available models:")
                for model in models:
                    print(f"  - {model.get('name', 'Unknown')}")
                return False
        else:
            print("Error: Could not connect to Ollama API")
            return False
    except requests.exceptions.RequestException:
        print("Error: Ollama is not running on localhost:11434")
        print("Please start Ollama and ensure gemma3:27b model is installed")
        return False
    
    return True

def extract_transcript(youtube_url):
    """Extract transcript from YouTube video using yt-dlp"""
    print(f"Extracting transcript from: {youtube_url}")
    
    # Create temporary directory for output
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "transcript"
        
        try:
            # Run yt-dlp to extract transcript
            cmd = [
                "./yt-dlp.exe",
                "--write-subs",
                "--write-auto-subs",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--skip-download",
                "--output", str(output_path),
                youtube_url
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Transcript extraction completed")
            
            # Find the generated subtitle file
            subtitle_files = list(Path(temp_dir).glob("*.vtt"))
            if not subtitle_files:
                print("No subtitle file found. Trying alternative approach...")
                return extract_transcript_alternative(youtube_url)
            
            # Read the subtitle file
            subtitle_file = subtitle_files[0]
            with open(subtitle_file, 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            
            # Parse VTT content to extract text
            transcript = parse_vtt_content(vtt_content)
            return transcript
            
        except subprocess.CalledProcessError as e:
            print(f"Error running yt-dlp: {e}")
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

def extract_transcript_alternative(youtube_url):
    """Alternative method to extract transcript using yt-dlp's JSON output"""
    try:
        print("Attempting alternative transcript extraction...")
        cmd = [
            "./yt-dlp.exe",
            "--write-info-json",
            "--write-subs",
            "--write-auto-subs",
            "--sub-lang", "en",
            "--skip-download",
            "--output", "temp_video",
            youtube_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Look for subtitle files
        subtitle_files = [f for f in os.listdir('.') if f.startswith('temp_video') and f.endswith('.vtt')]
        
        if subtitle_files:
            with open(subtitle_files[0], 'r', encoding='utf-8') as f:
                vtt_content = f.read()
            
            # Clean up temporary files
            for f in os.listdir('.'):
                if f.startswith('temp_video'):
                    os.remove(f)
            
            return parse_vtt_content(vtt_content)
        else:
            print("No transcript available for this video")
            return None
            
    except Exception as e:
        print(f"Alternative extraction failed: {e}")
        return None

def download_audio(youtube_url, output_dir="."):
    """Download audio from YouTube video using yt-dlp"""
    print(f"Downloading audio from: {youtube_url}")
    
    try:
        # Use yt-dlp to download audio only
        cmd = [
            "./yt-dlp.exe",
            "--extract-audio",
            "--audio-format", "wav",
            "--audio-quality", "0",  # Best quality
            "--output", os.path.join(output_dir, "%(title)s.%(ext)s"),
            youtube_url
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Audio download completed")
        
        # Find the downloaded audio file
        audio_files = [f for f in os.listdir(output_dir) if f.endswith('.wav')]
        if not audio_files:
            print("No audio file found after download")
            return None
        
        # Return the path to the most recently created audio file
        audio_file = max([os.path.join(output_dir, f) for f in audio_files], 
                        key=os.path.getctime)
        print(f"Audio file: {audio_file}")
        return audio_file
        
    except subprocess.CalledProcessError as e:
        print(f"Error downloading audio: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return None
    except Exception as e:
        print(f"Unexpected error downloading audio: {e}")
        return None

def transcribe_with_whisper(audio_file, model_size="base"):
    """Transcribe audio file using OpenAI Whisper"""
    try:
        import whisper
        print(f"Loading Whisper model: {model_size}")
        
        # Load the Whisper model
        model = whisper.load_model(model_size)
        
        print("Transcribing audio... This may take a while depending on the audio length.")
        
        # Transcribe the audio
        result = model.transcribe(audio_file, verbose=False)
        
        transcript = result["text"]
        print(f"Transcription completed ({len(transcript)} characters)")
        
        return transcript
        
    except ImportError:
        print("Error: OpenAI Whisper not installed")
        return None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None

def extract_transcript_with_api(youtube_url):
    """Extract transcript using youtube_transcript_api"""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        video_id = None
        if "v=" in youtube_url:
            video_id = youtube_url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in youtube_url:
            video_id = youtube_url.split("youtu.be/")[1].split("?")[0]

        if not video_id:
            print("Could not extract video ID from URL")
            return None

        print(f"Fetching transcript for video ID: {video_id} using API...")
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        
        # Combine transcript parts into a single string
        transcript = ' '.join([item['text'] for item in transcript_list])
        return transcript
        
    except Exception as e:
        print(f"Error using YouTubeTranscriptApi: {e}")
        return None

def get_transcript(youtube_url, transcript_source="yt-dlp", whisper_model="base", save_audio=False):
    """Get transcript using the specified source"""
    
    if transcript_source == "api":
        print("Using YouTube Transcript API...")
        return extract_transcript_with_api(youtube_url)
        
    elif transcript_source == "whisper":
        # Use Whisper transcription
        print("Using Whisper for audio transcription...")
        
        # Create temporary directory for audio
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download audio
            audio_file = download_audio(youtube_url, temp_dir)
            
            if not audio_file:
                print("Failed to download audio")
                return None
            
            # Optionally save audio file permanently
            if save_audio:
                import shutil
                permanent_audio = f"audio_{os.path.basename(audio_file)}"
                shutil.copy2(audio_file, permanent_audio)
                print(f"Audio saved to: {permanent_audio}")
            
            # Transcribe with Whisper
            transcript = transcribe_with_whisper(audio_file, whisper_model)
            
            return transcript
            
    else: # Default to yt-dlp
        # Try subtitle extraction first
        print("Attempting to extract existing transcript/subtitles with yt-dlp...")
        transcript = extract_transcript(youtube_url)
        
        if transcript:
            return transcript
        else:
            print("No subtitles found with yt-dlp.")
            return None

def parse_vtt_content(vtt_content):
    """Parse VTT subtitle content to extract clean text"""
    lines = vtt_content.split('\n')
    transcript_lines = []
    
    for line in lines:
        line = line.strip()
        # Skip empty lines, WEBVTT header, and timestamp lines
        if (line and 
            not line.startswith('WEBVTT') and 
            not line.startswith('NOTE') and
            '-->' not in line and
            not line.isdigit()):
            
            # Remove HTML tags if present
            import re
            clean_line = re.sub(r'<[^>]+>', '', line)
            if clean_line.strip():
                transcript_lines.append(clean_line.strip())
    
    return ' '.join(transcript_lines)

def chunk_transcript(transcript, chunk_size):
    """Split transcript into chunks of specified size"""
    chunks = []
    words = transcript.split()
    
    current_chunk = []
    current_length = 0
    
    for word in words:
        word_length = len(word) + 1  # +1 for space
        
        if current_length + word_length > chunk_size and current_chunk:
            # Save current chunk and start new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def summarize_chunk_with_ollama(chunk, running_summary="", chunk_number=1, total_chunks=1):
    """Summarize a transcript chunk using Ollama API with running summary context"""
    print(f"Summarizing chunk {chunk_number} of {total_chunks}...")
    
    # Prepare the prompt based on whether this is the first chunk or not
    if running_summary:
        prompt = f"""You are continuing to summarize a YouTube video transcript. Below is the summary so far and the next chunk of the transcript.

PREVIOUS SUMMARY:
{running_summary}

CURRENT TRANSCRIPT CHUNK ({chunk_number} of {total_chunks}):
{chunk}

Please update and expand the summary by integrating the new information from this chunk. Maintain the same structure:
1. Main topics discussed
2. Key points and insights  
3. Important conclusions or takeaways

UPDATED SUMMARY:"""
    else:
        prompt = f"""Please provide a comprehensive summary of the following YouTube video transcript chunk (part {chunk_number} of {total_chunks}). Include:
1. Main topics discussed
2. Key points and insights
3. Important conclusions or takeaways

Transcript chunk:
{chunk}

Summary:"""
    
    # Prepare API request
    api_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "gemma3:27b",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=300)  # 5 minute timeout
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "No summary generated")
        else:
            print(f"Error from Ollama API: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("Request timed out. The chunk might be too long.")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None

def summarize_with_ollama(transcript, chunk_size=80000):
    """Summarize transcript using Ollama API with chunking support"""
    if len(transcript) <= chunk_size:
        # Single chunk processing
        print("Processing transcript as single chunk...")
        return summarize_chunk_with_ollama(transcript, "", 1, 1)
    
    # Multi-chunk processing
    print(f"Transcript is long ({len(transcript)} chars), processing in chunks...")
    chunks = chunk_transcript(transcript, chunk_size)
    print(f"Split into {len(chunks)} chunks")
    
    running_summary = ""
    
    for i, chunk in enumerate(chunks, 1):
        chunk_summary = summarize_chunk_with_ollama(
            chunk, 
            running_summary, 
            chunk_number=i, 
            total_chunks=len(chunks)
        )
        
        if chunk_summary is None:
            print(f"Failed to summarize chunk {i}")
            return None
        
        running_summary = chunk_summary
        print(f"Completed chunk {i}/{len(chunks)}")
    
    return running_summary

def main():
    """Main function"""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='YouTube Transcript Summarizer')
    parser.add_argument('url', nargs='?', help='YouTube URL to process')
    parser.add_argument('--save-transcript', action='store_true', 
                        help='Save the extracted transcript to a text file')
    parser.add_argument('--chunk-size', type=int, default=80000,
                        help='Maximum characters per chunk (default: 80000)')
    parser.add_argument('--transcript-source', type=str, default='yt-dlp',
                        choices=['yt-dlp', 'whisper', 'api'],
                        help='Source for transcript extraction (default: yt-dlp)')
    parser.add_argument('--use-whisper', action='store_true',
                        help='(DEPRECATED) Use --transcript-source=whisper instead.')
    parser.add_argument('--whisper-model', type=str, default='turbo',
                        choices=['tiny', 'base', 'small', 'medium', 'large', 'turbo'],
                        help='Whisper model size (default: base). Larger models are more accurate but slower.')
    parser.add_argument('--save-audio', action='store_true',
                        help='Save the downloaded audio file (only when using --use-whisper)')
    
    args = parser.parse_args()
    
    print("YouTube Transcript Summarizer")
    print("=" * 40)

    # Handle deprecated --use-whisper flag
    if args.use_whisper:
        print("Warning: --use-whisper is deprecated. Use --transcript-source=whisper instead.")
        args.transcript_source = "whisper"
    
    # Check dependencies
    if not check_dependencies(transcript_source=args.transcript_source):
        sys.exit(1)
    
    # Get YouTube URL
    if args.url:
        youtube_url = args.url
    else:
        youtube_url = input("Enter YouTube URL: ").strip()
    
    if not youtube_url:
        print("No URL provided")
        sys.exit(1)
    
    # Validate URL
    if "youtube.com" not in youtube_url and "youtu.be" not in youtube_url:
        print("Please provide a valid YouTube URL")
        sys.exit(1)
    
    # Show processing method
    print(f"Transcript source: {args.transcript_source}")
    if args.transcript_source == 'whisper':
        print(f"Whisper model: {args.whisper_model}")
        if args.save_audio:
            print("Audio file will be saved")

    # Get transcript
    transcript = get_transcript(
        youtube_url,
        transcript_source=args.transcript_source,
        whisper_model=args.whisper_model,
        save_audio=args.save_audio
    )
    
    if not transcript:
        print("Failed to extract/transcribe content")
        if not args.use_whisper:
            print("Try using --use-whisper flag to transcribe audio instead")
        sys.exit(1)
    
    print(f"Transcript obtained successfully ({len(transcript)} characters)")
    
    # Save transcript if requested
    if args.save_transcript:
        transcript_filename = "transcript.txt"
        with open(transcript_filename, "w", encoding="utf-8") as f:
            f.write(f"YouTube URL: {youtube_url}\n")
            f.write(f"Transcript Length: {len(transcript)} characters\n")
            f.write(f"Method: {args.transcript_source}\n")
            if args.transcript_source == 'whisper':
                f.write(f"Whisper Model: {args.whisper_model}\n")
            f.write("=" * 50 + "\n")
            f.write("TRANSCRIPT:\n")
            f.write("=" * 50 + "\n")
            f.write(transcript)
        print(f"Transcript saved to: {transcript_filename}")
    
    # Determine processing approach
    if len(transcript) > args.chunk_size:
        print(f"Transcript exceeds chunk size ({args.chunk_size} chars), will process in chunks")
    
    # Summarize with Ollama
    summary = summarize_with_ollama(transcript, args.chunk_size)
    
    if summary:
        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(summary)
        
        # Save summary to file
        summary_filename = "summary.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(f"YouTube URL: {youtube_url}\n")
            f.write(f"Transcript Length: {len(transcript)} characters\n")
            f.write(f"Chunk Size Used: {args.chunk_size} characters\n")
            f.write(f"Processing Method: {'Chunked' if len(transcript) > args.chunk_size else 'Single Pass'}\n")
            f.write(f"Transcription Method: {args.transcript_source}\n")
            if args.transcript_source == 'whisper':
                f.write(f"Whisper Model: {args.whisper_model}\n")
            f.write("\nSUMMARY:\n")
            f.write("=" * 50 + "\n")
            f.write(summary)
        
        print(f"\nSummary saved to: {summary_filename}")
        
        if args.save_transcript:
            print(f"Transcript saved to: transcript.txt")
    else:
        print("Failed to generate summary")
        sys.exit(1)

if __name__ == "__main__":
    main()
