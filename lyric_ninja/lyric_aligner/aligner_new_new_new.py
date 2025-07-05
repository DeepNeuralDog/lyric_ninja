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
from demucs.api import Separator
from time import time, sleep
import numpy as np
import coremltools as ct
import librosa
from numba import njit
from ..converter.coreml_converter import Wav2Vec2Wrapper, convert_to_coreml
from ..vocal_enhancer.enhancer import enhance
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
from scipy.special import log_softmax
from torchaudio.pipelines._wav2vec2.impl import Wav2Vec2Bundle

def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time()
        result = func(*args, **kwargs)
        end_time = time()
        print(f"Function '{func.__name__}' executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper


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
            trellis[t, j] = np.maximum(stay, change)
    return trellis
@timer
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
@timer
def backtrack_np(trellis: np.ndarray, emission: np.ndarray, tokens: List[int], blank_id: int = 0) -> List[Point]:
    raw_path = _backtrack_np_jit(trellis.astype(np.float32), emission.astype(np.float32), tokens, blank_id)
    return [Point(token_index, time_index, score) for token_index, time_index, score in raw_path]
@timer
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
@timer
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
        sep_model_name: Optional[str] = None,
        generation_path: str = "generation",
        chunk_duration: int = 120
    ) -> None:
        self.device: str
        self.use_coreml: bool = False
        self.coreml_model_path: str = "models/wav2_vec2.mlpackage"
        self.model: Union[ct.models.CompiledMLModel, Wav2Vec2Bundle, ct.models.MLModel]
        self.chunk_duration: int = chunk_duration
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

        self.sep_model_name = sep_model_name
        self.separator = None

    def _initialize_separator(self):
        if self.separator is None:
            print("Initializing vocal separator model...")
            self.separator = Separator(model=self.sep_model_name)
            self.separator.update_parameter(device=self.device)

    @timer
    def separate_vocals(self, audio: Union[torch.Tensor, np.ndarray, str], sr:int) -> np.ndarray:
        self._initialize_separator()
        if audio is None:
            print("No audio provided for separation.")
            return audio
        if audio.shape[0] == 1:
            audio = np.repeat(audio, 2, axis=0) 
        if isinstance(audio, str):
            audio, sr = torchaudio.load(audio)
        if not isinstance(audio, torch.Tensor):
            audio = torch.from_numpy(audio)
        if sr != self.separator.samplerate:
            resampler = torchaudio.transforms.Resample(sr, self.separator.samplerate)
            audio = resampler(audio)   
        try:
            audio = audio.to(self.device)
            _, separated_tensors = self.separator.separate_tensor(audio)
            return separated_tensors["vocals"].cpu().numpy()
        except Exception as e:
            print(f"Vocal separation failed: {e}")
            traceback.print_exc()
        return audio

    @timer
    def wav2vec2_inference(self, waveform: Union[np.ndarray, torch.Tensor], sr:int) -> np.ndarray:
        print(f"Wav shape: {waveform.shape}")
       
        if self.use_coreml:
            if not isinstance(waveform, np.ndarray):
                waveform = waveform.cpu().numpy()
            if waveform.ndim == 1:
                waveform = np.expand_dims(waveform, axis=0)
            if waveform.ndim == 2 and waveform.shape[0] == 2:
                waveform = waveform.mean(axis=0, keepdims=True)
            if sr != self.bundle.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.bundle.sample_rate)
            print(f"Waveform shape after processing: {waveform.shape}")
            output_key = "var_846"
            input_dict = {"waveform": waveform}
            all_emissions = self.model.predict(input_dict)[output_key]
            return all_emissions
        
        else:
            if not isinstance(waveform, torch.Tensor):
                waveform = torch.from_numpy(waveform)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.ndim == 2 and waveform.shape[0] == 2:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != self.bundle.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.bundle.sample_rate)
                waveform = resampler(waveform)
            waveform = waveform.to(self.device)
            with torch.no_grad():
                emissions, _ = self.model(waveform)
            return emissions  
        
    @timer
    def get_timed_words(self, waveform: Union[np.ndarray,torch.Tensor], transcript: str, sr:int) -> List[Dict[str, Any]]:
        try:
            emissions = self.wav2vec2_inference(waveform=waveform, sr=sr)
            log_probs = log_softmax(emissions, axis=-1)
            emission_np = log_probs[0]
            tokens = [self.dictionary[c] for c in transcript]
            
            trellis = get_trellis_np(emission_np, tokens, self.blank_id)
            path = backtrack_np(trellis, emission_np, tokens, self.blank_id)
            char_segments = merge_repeats(path, transcript)
            word_segments = merge_words(char_segments)

            ratio = waveform.shape[1] / emission_np.shape[0]
            timed_words = [
                {"text": word.label, "start": (word.start * ratio) / self.bundle.sample_rate, "end": (word.end * ratio) / self.bundle.sample_rate}
                for word in word_segments
            ]
            return timed_words
        except Exception as e:
            print(f"❌ Error during alignment: {e}")
            traceback.print_exc()
            return []
        
    @timer
    def create_lrc_file(self, audio_name: str, lyric_lines: List[Dict[str, Any]], metadata: Dict[str, str]) -> str:
        lrc_file = os.path.join(self.lrc_path, f"{audio_name}.lrc")
        lrc_content = self.create_lrc_content(lyric_lines, metadata)
        with open(lrc_file, "w", encoding="utf-8") as f:
            f.write(lrc_content)
        return lrc_file

    @timer
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
            traceback.print_exc()
            return None

    @timer
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

    @timer
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

        waveform, sr = torchaudio.load(audio_file)
        print(f"Waveform shape: {waveform.shape}")
        
        # Step 1: Separate vocals
        waveform = self.separate_vocals(audio=waveform, sr=sr)
        
        # Step 2: Release separator model and original waveform from memory
        print("Releasing vocal separator model from memory...")
        del self.separator
        self.separator = None
        if self.device == 'mps':
            torch.mps.empty_cache()
        elif self.device == 'cuda':
            torch.cuda.empty_cache()
        print("Sleeping for 30 seconds to chkup for resources...")
        sleep(30)

        timed_words = self.get_timed_words(waveform, transcript, sr)
        
        print(f"Timed words: {timed_words[:20]}...")
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
            
        lrc_file_path, synced_file_path = None, None
        if create_lrc:
            lrc_file_path = self.create_lrc_file(audio_name=audio_name, lyric_lines=final_lyric_lines, metadata=metadata)
            print(f"\n✓ LRC file: {lrc_file_path}")
        if embed_lyrics:
            synced_file_path = self.embed_lyrics_to_audio(audio_file, final_lyric_lines, metadata=metadata)
            if synced_file_path:
                print(f"✓ Synced audio: {synced_file_path}")

        return {"lyric_lines": final_lyric_lines, "lrc_file": lrc_file_path, "synced_file": synced_file_path}



















