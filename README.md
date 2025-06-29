# Lyric Ninja

A Python library for automatic lyric-to-audio alignment using machine learning. It can process audio files, separate vocals, and align them with provided lyric text to generate synchronized lyric files (`.lrc`) and embed them directly into the audio's metadata.

This project leverages powerful models for high-quality results:
- **Vocal Separation**: Uses `UVR-MDX-NET-Inst_full_292.onnx` to isolate vocals from the audio track, improving alignment accuracy.
- **Speech Recognition**: Employs `WAV2VEC2_ASR_BASE_960H` for generating character-level timestamps from the vocal track.
- **Apple Silicon Support**: Automatically converts and utilizes a Core ML version of the Wav2Vec2 model for significantly faster performance on MPS devices.

## Installation

### From GitHub (Recommended)

You can install the latest version directly from the GitHub repository using `pip`. This is the easiest way to get started. It is recommended to use Python 3.10.

```sh
pip install git+https://github.com/DeeepNeuralDog/lyric_ninja.git
```

### For Development

If you want to contribute to the project, clone the repository and install it in editable mode.

1.  **Clone the repository:**
    ```sh
    git clone https://github.com/DeeepNeuralDog/lyric_ninja.git
    cd lyric_ninja
    ```

2.  **Set up a virtual environment:**
    ```sh
    python3.10 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install in editable mode:**
    ```sh
    pip install -e .
    ```

## Usage

You can use Lyric Ninja either as a command-line tool or as a Python library.

### Command-Line Interface (CLI)

The CLI is the simplest way to process a single song.

**Example:**
```sh
lyric-ninja "data/songs/The Scientist.mp3" "data/raw_lyrics/Papercut.txt" \
  --create-lrc \
  --embed-lyrics \
  --artist "Linkin Park" \
  --title "Papercut" \
  --album "forgot"
```

### As a Python Library

For more control, you can use Lyric Ninja directly in your Python code.

**Example:**
This script processes a single song and generates a synchronized `.lrc` file.

````python
from lyric_ninja import TorchaudioAligner

# 1. Initialize the aligner
aligner = TorchaudioAligner()

# 2. Define file paths and metadata
audio_file = "data/songs/The Scientist.mp3"
lyric_file = "data/raw_lyrics/The Scientist.txt"
audio_name = "The Scientist"
metadata = {
    "artist": "Linkin Park",
    "title": "Papercut",
    "album": "forgot"
}

# 3. Process the song
print(f"Processing: {audio_name}")

aligner.process_song(
    audio_file=audio_file,
    lyric_file=lyric_file,
    metadata=metadata,
    audio_name=audio_name,
    create_lrc=True,      # Create a .lrc file
    embed_lyrics=True,    # Embed lyrics into the audio file
)

print(f"✅ Successfully processed