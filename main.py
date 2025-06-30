import os
import sys
import traceback
from time import time
from typing import Dict, Any, List

# from lyric_ninja.lyric_aligner.aligner_new_new import LyricAligner
from lyric_ninja.lyric_aligner.aligner import LyricAligner


def main() -> None:
    mdx_params: Dict[str, Any] = {
        "hop_length": 1024,
        "segment_size": 1024,
        "overlap": 0.25,
        "batch_size": 1,
        "enable_denoise": True,
    }
    metadata: Dict[str, str] = {
        "artist": "Various Artists",
        "title": "Unknown Title",
        "album": "Unknown Album",
    }

    data_path: str = "data"
    generation_path: str = "generation"
    audio_path: str = os.path.join(data_path, "songs")
    lyric_path: str = os.path.join(data_path, "raw_lyrics")

    if not os.path.isdir(audio_path):
        print(f"Error: Audio directory not found at '{audio_path}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(lyric_path):
        print(f"Error: Lyric directory not found at '{lyric_path}'", file=sys.stderr)
        sys.exit(1)

    audio_files: List[str] = [
        f for f in os.listdir(audio_path)
        if os.path.isfile(os.path.join(audio_path, f)) and not f.startswith('.')
    ]

    try:
        aligner = LyricAligner(
            sep_model_filename='UVR-MDX-NET-Inst_full_292.onnx',
            # mdx_params=mdx_params,
            generation_path=generation_path,
        )
    except Exception as e:
        print(f"Fatal: Could not initialize the aligner: {e}", file=sys.stderr)
        sys.exit(1)

    for audio_file in audio_files:
        audio_name, _ = os.path.splitext(audio_file)
        current_metadata = metadata.copy()
        current_metadata["title"] = audio_name
        audio_file_path = os.path.join(audio_path, audio_file)
        lyric_file_path = os.path.join(lyric_path, f"{audio_name}.txt")

        if not os.path.exists(lyric_file_path):
            print(f"Info: Lyric file not found for {audio_name}, skipping...")
            continue

        try:
            print("\n" + "="*25 + f" Processing: {audio_name} " + "="*25 + "\n")
            start = time()
            aligner.process_song(
                audio_file=audio_file_path,
                lyric_file=lyric_file_path,
                metadata=current_metadata,
                create_lrc=True,
                embed_lyrics=True,
                audio_name=audio_name,
            )
            end = time()
            print(f"✅ Successfully processed {audio_name} in {end - start:.2f} seconds")
        except Exception as e:
            print(f"❌ Error processing {audio_name}: {e}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    main()