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
from audio_separator2.separator import Separator
from time import time
import numpy as np
import coremltools as ct
import librosa
from numba import njit
from ..converter.coreml_converter import Wav2Vec2Wrapper, convert_to_coreml
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
from audio_separator2.separator.architectures.mdx_separator import MDXSeparator
from audio_separator2.separator.separator import Separator as SeparatorWrapper
from audio_separator2.separator.uvr_lib_v5.stft import STFT
from tqdm import tqdm
import torch.nn.functional as F


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
            temp_separator = SeparatorWrapper(log_level=logging.WARNING)
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
                "overlap": 0.25,
                "segment_size": 256,  # FIX 1: Use a sensible default for segment_size
                "batch_size": 1,
                "enable_denoise": True,
            }

            self.mdx_model = MDXSeparator(common_config=common_config, arch_config=arch_config)
            
            # FIX 1: Explicitly set segment_size to the model's native dim_t if not provided
            if self.mdx_model.segment_size is None:
                self.mdx_model.segment_size = self.mdx_model.dim_t
            
            self.mdx_model.initialize_model_settings()
            print("Source separation model loaded successfully.")

        except Exception as e:
            print(f"❌ Could not initialize source separation model: {e}")
            traceback.print_exc()


    def run_mdx_on_gpu(self, mix_gpu: torch.Tensor) -> torch.Tensor:
        """
        Runs the MDX model on a tensor that is already on the GPU.
        This is a simplified, GPU-only version of the demix logic.
        """
        # The MDX model's internal demixing logic is complex and still uses numpy.
        # For a true GPU pipeline, we must replicate its core logic in PyTorch.
        # This is a simplified version focusing on the model call.
        # NOTE: This assumes the input `mix_gpu` is a single, manageable chunk.
        
        # 1. Convert to Spectrogram (on GPU)
        spek = self.mdx_model.stft(mix_gpu)
        # print(f"Spek shape : ---->{spek.shape}")
        spek = spek.unsqueeze(0)  # Ensure batch dimension is present
        # print(f"Spek shape : ---->{spek.shape}")
        spek[:, :, :3, :] *= 0  # Zero out low-frequency bins

        # 2. Run the model (on GPU)
        if self.mdx_model.enable_denoise:
            spec_pred_neg = self.mdx_model.model_run(-spek)
            spec_pred_pos = self.mdx_model.model_run(spek)
            spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)
        else:
            spec_pred = self.mdx_model.model_run(spek)

        # 3. Inverse STFT to get audio back (on GPU)
        separated_audio_gpu = self.mdx_model.stft.inverse(spec_pred)
        
        return separated_audio_gpu
    def pipelined_inference_on_gpu(self, audio_file: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        Performs a high-performance pipeline optimized for CPU pre-processing and hardware-accelerated model inference.
        1. Loads and resamples audio efficiently on the CPU with Librosa.
        2. Creates overlapping chunks as a NumPy array.
        3. For each batch, performs a minimal CPU->GPU->CPU roundtrip for STFT operations.
        4. Feeds the separated audio into the final model (CoreML or Wav2Vec2).
        """
        if not self.mdx_model:
            print("MDX model not available for pipelined inference.")
            return None
        
        try:
            # 1. Load and resample audio entirely on CPU with Librosa
            waveform_np, sr = librosa.load(audio_file, mono=False, sr=self.bundle.sample_rate)
            
            # Ensure waveform is stereo
            if waveform_np.ndim == 1:
                waveform_np = np.stack([waveform_np, waveform_np])
            
            original_samples = waveform_np.shape[1]

            # 2. Set up parameters and pad on CPU with NumPy
            chunk_size = self.mdx_model.chunk_size
            trim = self.mdx_model.trim
            overlap = self.mdx_model.overlap
            step = int((1 - overlap) * chunk_size)
            
            pad_amount = step - ((original_samples - 1) % step)
            padded_waveform_np = np.pad(waveform_np, ((0, 0), (trim, trim + pad_amount)), 'constant')
            
            # 3. Create batches of overlapping chunks using NumPy
            num_chunks = (padded_waveform_np.shape[1] - chunk_size) // step + 1
            chunks = np.array([
                padded_waveform_np[:, i*step : i*step+chunk_size]
                for i in range(num_chunks)
            ])
            
            # 4. Prepare for overlap-add on CPU
            separated_output_np = np.zeros_like(padded_waveform_np, dtype=np.float32)
            divider_np = np.zeros_like(padded_waveform_np, dtype=np.float32)
            hanning_window_np = np.hanning(chunk_size)

            print("Starting optimized batched inference...")
            mini_batch_size = 4 
            for i in tqdm(range(0, chunks.shape[0], mini_batch_size), desc="Processing Batches"):
                batch_np = chunks[i:i+mini_batch_size]
                
                # --- Minimal CPU -> GPU -> CPU for STFT operations ---
                batch_gpu = torch.from_numpy(batch_np).to(self.torch_device)
                
                # a. STFT (GPU)
                spek = self.mdx_model.stft(batch_gpu)
                
                # b. Model Run (CPU/ANE via NumPy)
                # input_spek_np = spek.cpu().numpy()
                if self.mdx_model.enable_denoise:
                    spec_pred_neg = self.mdx_model.model_run(-spek)
                    spec_pred_pos = self.mdx_model.model_run(spek)
                    spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)
                else:
                    spec_pred = self.mdx_model.model_run(spek)
                
                # c. Inverse STFT (GPU)
                spec_pred_gpu = torch.from_numpy(spec_pred).to(self.torch_device)
                separated_batch = self.mdx_model.stft.inverse(spec_pred_gpu)
                
                # --- Overlap-add back on the CPU ---
                separated_batch_np = separated_batch.cpu().numpy()
                for j, separated_chunk_np in enumerate(separated_batch_np):
                    start_index = (i + j) * step
                    separated_output_np[:, start_index : start_index + chunk_size] += separated_chunk_np * hanning_window_np
                    divider_np[:, start_index : start_index + chunk_size] += hanning_window_np

            # 5. Normalize the result and unpad on CPU
            divider_np[divider_np == 0] = 1.0
            separated_full_np = separated_output_np / divider_np
            separated_mono_np = separated_full_np[0, trim : trim + original_samples] # Use one channel for mono

            # 6. Wav2Vec2 Stage
            # The input is already a NumPy array, perfect for CoreML
            if self.use_coreml:
                emissions, _ = self.model(np.expand_dims(separated_mono_np, axis=0))
            else:
                # For CUDA/CPU, we do one final conversion
                mono_gpu = torch.from_numpy(separated_mono_np).unsqueeze(0).to(self.torch_device)
                emissions, _ = self.model(mono_gpu)
            
            log_probs = torch.nn.functional.log_softmax(emissions, dim=-1)
            return log_probs[0].cpu().numpy(), original_samples

        except Exception as e:
            print(f"❌ Error during pipelined inference: {e}")
            traceback.print_exc()
            return None
            
    # def pipelined_inference_on_gpu(self, audio_file: str) -> Optional[Tuple[np.ndarray, int]]:
    #     if not self.mdx_model:
    #         print("MDX model not available for pipelined inference.")
    #         return None
        
    #     try:
    #         waveform, sr = torchaudio.load(audio_file)
            
    #         # FIX 2: Perform resampling on CPU to avoid MPS error, then move to device
    #         if sr != self.bundle.sample_rate:
    #             resampler = torchaudio.transforms.Resample(sr, self.bundle.sample_rate)
    #             waveform = resampler(waveform.cpu())
            
    #         waveform = waveform.to(self.torch_device)

    #         if waveform.shape[0] == 1:
    #             waveform = waveform.repeat(2, 1)
            
    #         original_samples = waveform.shape[1]

    #         # Replicate the padding and overlap-add logic from demix on the GPU
    #         chunk_size = self.mdx_model.chunk_size
    #         trim = self.mdx_model.trim
    #         gen_size = chunk_size - 2 * trim
    #         step = int((1 - self.mdx_model.overlap) * chunk_size)

    #         pad_amount = gen_size + trim - ((original_samples - 1) % gen_size)
    #         padded_waveform = F.pad(waveform, (trim, pad_amount), 'constant', 0)

    #         total_len = padded_waveform.shape[1]
    #         result = torch.zeros((1, 2, total_len), device=self.torch_device)
    #         divider = torch.zeros((1, 2, total_len), device=self.torch_device)
    #         hanning_window = torch.hann_window(chunk_size, device=self.torch_device)

    #         print("Starting GPU-centric pipelined inference...")
    #         for i in tqdm(range(0, total_len - chunk_size + 1, step), desc="Processing GPU Chunks"):
    #             chunk = padded_waveform[:, i:i+chunk_size].unsqueeze(0) # Add batch dimension
                
    #             spek = self.mdx_model.stft(chunk)
                
    #             # The model_run expects a numpy array if using the fast ONNX path
    #             is_onnx_path = not isinstance(self.mdx_model.model_run, torch.nn.Module)
                
    #             if self.mdx_model.enable_denoise:
    #                 if is_onnx_path:
    #                     spec_pred_neg = torch.from_numpy(self.mdx_model.model_run(-spek)).to(self.torch_device)
    #                     spec_pred_pos = torch.from_numpy(self.mdx_model.model_run(spek)).to(self.torch_device)
    #                 else:
    #                     spec_pred_neg = self.mdx_model.model_run(-spek)
    #                     spec_pred_pos = self.mdx_model.model_run(spek)
    #                 spec_pred = (spec_pred_neg * -0.5) + (spec_pred_pos * 0.5)
    #             else:
    #                 if is_onnx_path:
    #                     spec_pred = torch.from_numpy(self.mdx_model.model_run(spek)).to(self.torch_device)
    #                 else:
    #                     spec_pred = self.mdx_model.model_run(spek)

    #             separated_chunk = self.mdx_model.stft.inverse(spec_pred)

    #             result[:, :, i:i+chunk_size] += separated_chunk * hanning_window
    #             divider[:, :, i:i+chunk_size] += hanning_window

    #         divider[divider == 0] = 1.0 # Avoid division by zero
    #         separated_gpu_full = result / divider
    #         separated_gpu_full = separated_gpu_full[:, :, trim : trim + original_samples]

    #         mono_gpu = torch.mean(separated_gpu_full, dim=1, keepdim=True)
            
    #         if self.use_coreml:
    #             emissions, _ = self.model(mono_gpu.cpu())
    #         else:
    #             emissions, _ = self.model(mono_gpu)
            
    #         log_probs = torch.nn.functional.log_softmax(emissions, dim=-1)
    #         return log_probs[0].cpu().numpy(), original_samples

    #     except Exception as e:
    #         print(f"❌ Error during GPU-centric pipelined inference: {e}")
    #         traceback.print_exc()
    #         return None

    def get_timed_words(self, audio_file: str, transcript: str) -> List[Dict[str, Any]]:
        try:
            start = time()
            
            # Use the new, much faster GPU-centric pipeline
            inference_result = self.pipelined_inference_on_gpu(audio_file)
            
            end = time()
            print(f"Pipelined inference took {end - start:.2f} seconds")

            if inference_result is None:
                print("❌ Pipelined inference failed.")
                return []

            emission_np, original_samples = inference_result

            tokens = [self.dictionary[c] for c in transcript]
            
            trellis = get_trellis_np(emission_np, tokens, self.blank_id)
            path = backtrack_np(trellis, emission_np, tokens, self.blank_id)
            char_segments = merge_repeats(path, transcript)
            word_segments = merge_words(char_segments)

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

    def get_timed_words_past(self, audio_file: str, transcript: str) -> List[Dict[str, Any]]:
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