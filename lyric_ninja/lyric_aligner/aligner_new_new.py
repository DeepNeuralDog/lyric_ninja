import torch
import torchaudio
import re
import os
import shutil
import traceback
from dataclasses import dataclass
from mutagen.id3 import ID3, USLT, Encoding
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from rich import print
from audio_separator.separator import Separator
from time import time
import numpy as np
import coremltools as ct
import librosa
from numba import njit
from ..converter.coreml_converter import Wav2Vec2Wrapper, convert_to_coreml
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
from audio_separator.separator.architectures.mdx_separator import MDXSeparator
from audio_separator.separator.separator import Separator as SeparatorWrapper
from tqdm import tqdm

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float

@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float

    @property
    def length(self) -> int:
        return self.end - self.start

@njit
def _get_trellis_np_jit(emission: np.ndarray, tokens: List[int], blank_id: int) -> np.ndarray:
    num_frame = emission.shape[0]
    num_tokens = len(tokens)
    trellis = np.full((num_frame, num_tokens), -np.inf, dtype=np.float32)
    trellis[0, 0] = emission[0, blank_id]
    trellis[0, 1] = emission[0, tokens[1]]
    for t in range(1, num_frame):
        trellis[t, 0] = trellis[t - 1, 0] + emission[t, blank_id]
        for j in range(1, num_tokens):
            stay = trellis[t - 1, j] + emission[t, blank_id]
            change = trellis[t - 1, j - 1] + emission[t, tokens[j]]
            trellis[t, j] = np.logaddexp(stay, change)
    return trellis

def get_trellis_np(emission: np.ndarray, tokens: List[int], blank_id: int = 0) -> np.ndarray:
    return _get_trellis_np_jit(emission.astype(np.float32), tokens, blank_id)

@njit
def _backtrack_np_jit(trellis: np.ndarray, emission: np.ndarray, tokens: List[int], blank_id: int) -> List[Tuple[int, int, float]]:
    t, j = trellis.shape[0] - 1, trellis.shape[1] - 1
    path: List[Tuple[int, int, float]] = []
    path.append((j, t, trellis[t, j]))
    while j > 0 and t > 0:
        p_stay = trellis[t - 1, j] + emission[t, blank_id]
        p_change = trellis[t - 1, j - 1] + emission[t, tokens[j]]
        t -= 1
        if p_change > p_stay:
            j -= 1
        path.append((j, t, float(p_change if p_change > p_stay else p_stay)))
    while t > 0:
        t -= 1
        path.append((0, t, float(trellis[t, 0])))
    return path[::-1]

def backtrack_np(trellis: np.ndarray, emission: np.ndarray, tokens: List[int], blank_id: int = 0) -> List[Point]:
    raw_path = _backtrack_np_jit(trellis.astype(np.float32), emission.astype(np.float32), tokens, blank_id)
    return [Point(token_index, time_index, score) for token_index, time_index, score in raw_path]

def merge_repeats(path: List[Point], transcript: str) -> List[Segment]:
    i1, i2 = 0, 0
    segments: List[Segment] = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(transcript[path[i1].token_index], path[i1].time_index, path[i2 - 1].time_index + 1, score)
        )
        i1 = i2
    return segments

def merge_words(segments: List[Segment], separator: str = "|") -> List[Segment]:
    words: List[Segment] = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(seg.length for seg in segs)
                words.append(Segment(word, segments[i1].start, segments[i2 - 1].end, score))
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words





















class LyricAligner:
    def __init__(
        self,
        sep_model_filename: Optional[str] = None,
        mdx_params: Optional[Dict[str, Any]] = None,
        generation_path: str = "generation",
    ) -> None:
        self.device: str
        self.use_coreml: bool = False
        self.coreml_model_path: str = "models/wav2_vec2.mlpackage"
        self.model: Any

        temp_separator = SeparatorWrapper(log_level=logging.WARNING)
        self.torch_device = temp_separator.torch_device
        self.onnx_execution_provider = temp_separator.onnx_execution_provider

        if torch.backends.mps.is_available():
            self.device = "mps"
            if not os.path.exists(self.coreml_model_path):
                print("CoreML model not found, converting...")
                convert_to_coreml(Wav2Vec2Wrapper(), self.coreml_model_path)
            self.model = ct.models.MLModel(self.coreml_model_path)
            self.use_coreml = True
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
            self.model = bundle.get_model().to(self.device)

        self.bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.labels: List[str] = self.bundle.get_labels()
        self.dictionary: Dict[str, int] = {c: i for i, c in enumerate(self.labels)}
        self.blank_id: int = self.dictionary.get("|", 0)

        self.lrc_path: str = os.path.join(generation_path, "lrc")
        self.synced_path: str = os.path.join(generation_path, "synced")
        self.sep_path: str = os.path.join(generation_path, "sep")
        os.makedirs(self.lrc_path, exist_ok=True)
        os.makedirs(self.synced_path, exist_ok=True)
        os.makedirs(self.sep_path, exist_ok=True)

        self.mdx_model: Optional[MDXSeparator] = None
        try:
            print("Initializing source separation model...")
            _, _, _, model_path, _ = temp_separator.download_model_files(sep_model_filename)
            model_data = temp_separator.load_model_data_using_hash(model_path)
            common_config = {
                "logger": temp_separator.logger,
                "log_level": logging.WARNING,
                "torch_device": self.torch_device,
                "torch_device_cpu": torch.device("cpu"),
                "torch_device_mps": torch.device("mps") if torch.backends.mps.is_available() else None,
                "onnx_execution_provider": self.onnx_execution_provider,
                "model_name": os.path.splitext(sep_model_filename)[0],
                "model_path": model_path,
                "model_data": model_data,
                "output_format": "WAV",
                "output_bitrate": None,
                "output_dir": generation_path,
                "normalization_threshold": 0.9,
                "amplification_threshold": 0.0,
                "output_single_stem": None,
                "invert_using_spec": False,
                "sample_rate": self.bundle.sample_rate,
                "use_soundfile": False,
            }
            arch_config = mdx_params or {
                "hop_length": 1024,
                "segment_size": 1024,
                "overlap": 0.25,
                "batch_size": 1,
                "enable_denoise": True,
            }

            self.mdx_model = MDXSeparator(common_config=common_config, arch_config=arch_config)
            print("Source separation model loaded successfully.")

        except Exception as e:
            print(f"❌ Could not initialize source separation model: {e}")
            traceback.print_exc()




    # def separate_vocals(self, audio_file: str, audio_name: str) -> str:
    #     start = time()
    #     try:
    #         self.separator.separate(audio_file_path=audio_file)
    #         if self.sep_model_filename:
    #             vocals_file = os.path.join(
    #                 self.sep_path, f"{audio_name}_(Vocals)_{self.sep_model_filename.replace('.onnx', '')}.wav"
    #             )
    #             end = time()
    #             print(f"Vocal separation took {end - start:.2f} seconds")
    #             if os.path.exists(vocals_file):
    #                 return vocals_file
    #     except Exception as e:
    #         print(f"Vocal separation failed: {e}")
    #     print("Vocal separation failed or output file not found. Using original audio.")
    #     return audio_file
    
    def pipelined_inference(self, mix: np.ndarray) -> Optional[np.ndarray]:
        if not self.mdx_model:
            print("MDX model not available for pipelined inference.")
            return None
        try:
            # This function now correctly receives a stereo (2, N) array
            chunk_size = self.bundle.sample_rate * 60
            total_samples = mix.shape[1]
            all_emissions = []
            print(f"Total samples in mix: {total_samples}, chunk size: {chunk_size} Total chunks: {total_samples // chunk_size}  ")
            if chunk_size > total_samples:
                print(f"Chunk size {chunk_size} is larger than total samples {total_samples}. Using full mix.")
                chunk_size = total_samples
            print("Starting pipelined inference...")
            for i in range(0, total_samples, chunk_size):
                chunk = mix[:, i:i+chunk_size]
                print(f'MDX chunk shape: {chunk.shape}')
                separated_chunk = self.mdx_model.demix(chunk)
                mono_chunk = librosa.to_mono(separated_chunk)
                waveform_np = np.expand_dims(mono_chunk, axis=0)
                print(f"Wav2Vec2 inference on chunk shape: {waveform_np.shape}")
                emissions_chunk = self.wav2vec2_inference(waveform_np)
                all_emissions.append(emissions_chunk.cpu())

            if not all_emissions:
                return None
            
            full_emissions = torch.cat(all_emissions, dim=1)
            log_probs = torch.nn.functional.log_softmax(full_emissions, dim=-1)
            return log_probs[0].cpu().numpy()

        except Exception as e:
            print(f"❌ Error during pipelined inference: {e}")
            traceback.print_exc()
            return None

        except Exception as e:
            print(f"❌ Error during pipelined inference: {e}")
            traceback.print_exc()
            return None

    def wav2vec2_inference(self, waveform_np: np.ndarray) -> torch.Tensor:
        # This function is now simpler as it doesn't need to chunk anymore
        if self.use_coreml:
            output_key = self.model._spec.description.output[0].name
            input_dict = {"waveform": waveform_np}
            emissions = self.model.predict(input_dict)[output_key]
            return torch.from_numpy(emissions)
        else:
            waveform = torch.from_numpy(waveform_np).to(self.device)
            with torch.no_grad():
                emissions, _ = self.model(waveform)
            return emissions

    def get_timed_words(self, audio_file: str, transcript: str) -> List[Dict[str, Any]]:
        try:
            # 1. Load audio as stereo
            waveform, _ = librosa.load(audio_file, sr=self.bundle.sample_rate, mono=False)
            
            # 2. Ensure waveform is always stereo (2, N) for the separator
            if waveform.ndim == 1:
                # If librosa loaded a mono file, duplicate the channel
                waveform = np.asfortranarray([waveform, waveform])

            start = time()
            # 3. Pass the stereo waveform to the pipeline
            emission_np = self.pipelined_inference(waveform)
            end = time()
            print(f"Pipelined inference took {end - start:.2f} seconds")

            if emission_np is None:
                print("❌ Pipelined inference failed.")
                return []

            tokens = [self.dictionary[c] for c in transcript]
            
            trellis = get_trellis_np(emission_np, tokens, self.blank_id)
            path = backtrack_np(trellis, emission_np, tokens, self.blank_id)
            char_segments = merge_repeats(path, transcript)
            word_segments = merge_words(char_segments)

            # 4. Calculate ratio based on original sample count
            original_samples = waveform.shape[1]
            ratio = original_samples / emission_np.shape[0]
            timed_words = [
                {"text": word.label, "start": (word.start * ratio) / self.bundle.sample_rate, "end": (word.end * ratio) / self.bundle.sample_rate}
                for word in word_segments
            ]
            return timed_words
        except Exception as e:
            print(f"❌ Error during alignment: {e}")
            traceback.print_exc()
            return []

    def _remove_consecutive_duplicate_lyrics(self, lyric_lines: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not lyric_lines:
            return []
        curated_lines = [lyric_lines[0]]
        for i in range(1, len(lyric_lines)):
            if lyric_lines[i]["text"].strip().casefold() != curated_lines[-1]["text"].strip().casefold():
                curated_lines.append(lyric_lines[i])
        return curated_lines

    def create_lrc_file(self, audio_name: str, lyric_lines: List[Dict[str, Any]], metadata: Dict[str, str]) -> str:
        lrc_file = os.path.join(self.lrc_path, f"{audio_name}.lrc")
        lrc_content = self.create_lrc_content(lyric_lines, metadata)
        with open(lrc_file, "w", encoding="utf-8") as f:
            f.write(lrc_content)
        return lrc_file

    def embed_lyrics_to_audio(self, audio_file: str, lyric_lines: List[Dict[str, Any]], metadata: Dict[str, str]) -> Optional[str]:
        audio_name, extension = os.path.splitext(os.path.basename(audio_file))
        output_file = os.path.join(self.synced_path, f"{audio_name}_synced{extension}")
        shutil.copy2(audio_file, output_file)
        file_ext = extension.lower()
        try:
            if file_ext == ".mp3":
                audio = MP3(output_file, ID3=ID3)
                if audio.tags is None:
                    audio.add_tags()
                lrc_content = self.create_lrc_content(lyric_lines, metadata)
                audio.tags.add(USLT(encoding=Encoding.UTF8, lang="eng", desc="", text=lrc_content))
                audio.save()
            elif file_ext in [".m4a", ".mp4", ".aac"]:
                audio = MP4(output_file)
                lrc_content = self.create_lrc_content(lyric_lines, metadata)
                audio["\xa9lyr"] = lrc_content
                audio.save()
            else:
                print(f"Unsupported format for embedding: {file_ext}")
                return None
            return output_file
        except Exception as e:
            print(f"Error embedding lyrics: {e}")
            return None

    def create_lrc_content(self, lyric_lines: List[Dict[str, Any]], metadata: Dict[str, str]) -> str:
        lrc_content = f"[ar:{metadata.get('artist', 'Unknown Artist')}]\n"
        lrc_content += f"[ti:{metadata.get('title', 'Unknown Title')}]\n"
        lrc_content += f"[al:{metadata.get('album', 'Unknown Album')}]\n"
        lrc_content += "[by:LyricNinja]\n[offset:0]\n\n"
        for line in lyric_lines:
            start_time = line["start"]
            minutes, seconds = divmod(start_time, 60)
            lrc_content += f"[{int(minutes):02d}:{seconds:05.2f}]{line['text']}\n"
        return lrc_content

    def process_song(
        self, audio_file: str, lyric_file: str, metadata: Dict[str, str], audio_name: str, create_lrc: bool, embed_lyrics: bool
    ) -> Optional[Dict[str, Any]]:
        with open(lyric_file, "r", encoding="utf-8") as f:
            original_lyrics_text = f.read().strip()
        if not original_lyrics_text:
            print(f"No lyrics found in {lyric_file}, skipping...")
            return None

        original_lines = [line.strip() for line in original_lyrics_text.split('\n') if line.strip()]
        lines_with_words: List[Dict[str, Any]] = []
        all_words_for_transcript: List[str] = []
        for line in original_lines:
            text = re.sub(r'\(.*?\)|\[.*?\]', ' ', line).upper()
            words = [word for word in text.split() if word]
            line_words = ["".join(c for c in word if c in self.dictionary) for word in words]
            line_words = [w for w in line_words if w]
            if line_words:
                lines_with_words.append({'original_line': line, 'words': line_words})
                all_words_for_transcript.extend(line_words)
        
        transcript = "|".join(all_words_for_transcript)
        if not transcript:
            print(f"Could not find any valid characters for alignment in {lyric_file}")
            return None
        transcript = f"|{transcript}|"

        # align_file = self.separate_vocals(audio_file, audio_name=audio_name)
        timed_words = self.get_timed_words(audio_file, transcript)
        if not timed_words:
            print("❌ Alignment failed to produce timed words.")
            return None

        final_lyric_lines: List[Dict[str, Any]] = []
        timed_word_cursor = 0
        for line_info in lines_with_words:
            num_words = len(line_info['words'])
            if timed_word_cursor + num_words > len(timed_words):
                break
            start_time = timed_words[timed_word_cursor]['start']
            end_time = timed_words[timed_word_cursor + num_words - 1]['end']
            final_lyric_lines.append({'start': start_time, 'end': end_time, 'text': line_info['original_line']})
            timed_word_cursor += num_words
        
        final_lyric_lines = self._remove_consecutive_duplicate_lyrics(final_lyric_lines)
        
        lrc_file_path, synced_file_path = None, None
        if create_lrc:
            lrc_file_path = self.create_lrc_file(audio_name=audio_name, lyric_lines=final_lyric_lines, metadata=metadata)
            print(f"\n✓ LRC file: {lrc_file_path}")
        if embed_lyrics:
            synced_file_path = self.embed_lyrics_to_audio(audio_file, final_lyric_lines, metadata=metadata)
            if synced_file_path:
                print(f"✓ Synced audio: {synced_file_path}")

        return {"lyric_lines": final_lyric_lines, "lrc_file": lrc_file_path, "synced_file": synced_file_path}