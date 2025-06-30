import argparse
import os
import sys
import traceback
from time import time
from typing import Dict, Any, Optional

from .lyric_aligner.aligner import LyricAligner

def main() -> None:
    parser = argparse.ArgumentParser(description="Align lyrics with audio files using machine learning")
    parser.add_argument("audio_file", type=str, help="Path to the audio file")
    parser.add_argument("lyric_file", type=str, help="Path to the lyrics text file")
    parser.add_argument("--chunk_duration", type=int, default=120, help="chunk duration(seconds) for wav2vec2")
    parser.add_argument("--segment_size", type=int, default=1024, help="Segment size for processing")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory for generated files (default: output)")
    parser.add_argument("--artist", type=str, default="Unknown Artist", help="Artist name for metadata")
    parser.add_argument("--title", type=str, help="Song title for metadata (default: derived from filename)")
    parser.add_argument("--album", type=str, default="Unknown Album", help="Album name for metadata")
    parser.add_argument("--create_lrc", action="store_true", help="Create LRC format lyrics file")
    parser.add_argument("--embed_lyrics", action="store_true", help="Embed lyrics into the audio file metadata")
    
    args = parser.parse_args()
    
    if not os.path.isfile(args.audio_file):
        print(f"Error: Audio file not found: '{args.audio_file}'", file=sys.stderr)
        sys.exit(1)
        
    if not os.path.isfile(args.lyric_file):
        print(f"Error: Lyric file not found: '{args.lyric_file}'", file=sys.stderr)
        sys.exit(1)
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create output directory '{args.output_dir}': {e}", file=sys.stderr)
        sys.exit(1)
    
    title: str = args.title or os.path.splitext(os.path.basename(args.audio_file))[0]
    
    metadata: Dict[str, str] = {
        "artist": args.artist,
        "title": title,
        "album": args.album
    }
    
    mdx_params: Dict[str, Any] = {
        "hop_length": 1024,
        "segment_size": args.segment_size,
        "overlap": 0.25,
        "batch_size": 1,
        "enable_denoise": True
    }
    
    try:
        aligner = LyricAligner(
            sep_model_filename='UVR-MDX-NET-Inst_full_292.onnx',
            mdx_params=mdx_params,
            generation_path=args.output_dir,
            chunk_duration=args.chunk_duration,
        )
    except Exception as e:
        print(f"Error initializing aligner: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    
    try:
        print(f"Processing: {title}")
        start_time = time()
        
        aligner.process_song(
            audio_file=args.audio_file,
            lyric_file=args.lyric_file,
            metadata=metadata,
            create_lrc=args.create_lrc,
            embed_lyrics=args.embed_lyrics,
            audio_name=title,
        )
        
        end_time = time()
        print(f"✅ Successfully processed '{title}' in {end_time - start_time:.2f} seconds")
        print(f"Output saved to: {os.path.abspath(args.output_dir)}")
        
    except Exception as e:
        print(f"❌ Error processing '{title}': {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()