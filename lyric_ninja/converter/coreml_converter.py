import torch
import torchaudio
import coremltools as ct
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class BaseWrapper(torch.nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[torch.nn.Module] = None
        self.sample_rate: Optional[int] = None
        self._load_model()
        if self.model is None or self.sample_rate is None:
            raise NotImplementedError("Model and sample_rate must be set in _load_model")

    @abstractmethod
    def _load_model(self) -> None:
        pass
    
    @abstractmethod
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        pass

class Wav2Vec2Wrapper(BaseWrapper):
    def _load_model(self) -> None:
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        self.model = bundle.get_model()
        self.model.eval()
        self.sample_rate = bundle.sample_rate
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        emissions, _ = self.model(waveform)
        return emissions

class HuBERTWrapper(BaseWrapper):
    def _load_model(self) -> None:
        bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        self.model = bundle.get_model()
        self.model.eval()
        self.sample_rate = bundle.sample_rate
    
    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        emissions, _ = self.model(waveform)
        return emissions

def convert_to_coreml(wrapped_model: BaseWrapper, output_path: str, max_duration_minutes: int = 15) -> None:
    if wrapped_model.sample_rate is None:
        raise ValueError("Wrapped model must have a sample_rate.")
        
    example_input = torch.randn(1, wrapped_model.sample_rate * 10)
    traced_model = torch.jit.trace(wrapped_model, example_input)
    
    variable_shape = ct.TensorType(
        name="waveform",
        shape=(1, ct.RangeDim(1, wrapped_model.sample_rate * max_duration_minutes * 60))
    )
    
    coreml_model = ct.convert(
        traced_model,
        inputs=[variable_shape],
        compute_units=ct.ComputeUnit.ALL
    )
    
    coreml_model.save(output_path)
    print(f"Model successfully converted and saved as {output_path}")

if __name__ == "__main__":
    import os
    os.makedirs("models", exist_ok=True)
    
    print("Converting Wav2Vec2 model...")
    wav2vec2_wrapper = Wav2Vec2Wrapper()
    convert_to_coreml(wav2vec2_wrapper, "models/wav2_vec2.mlpackage")
    
    print("\nConverting HuBERT model...")
    hubert_wrapper = HuBERTWrapper()
    convert_to_coreml(hubert_wrapper, "models/hubert_model.mlpackage")