"""
Lyric Ninja - A Python library for automatic lyric-to-audio alignment using machine learning.

This library provides tools to automatically align lyrics with audio files using
speech recognition and machine learning models. It supports various audio formats
and can generate synchronized lyrics in LRC format.
"""

__version__ = "0.1.0"
__author__ = "Md. Hasibul Hasan"
__email__ = "md.hasibul.hasan.codel@gmail.com"
__license__ = "MIT"

from .lyric_aligner.aligner import TorchaudioAligner
from .converter.coreml_converter import Wav2Vec2Wrapper, convert_to_coreml, HuBERTWrapper

__all__ = [
    "TorchaudioAligner",
    "Wav2Vec2Wrapper", 
    "convert_to_coreml",
    "HuBERTWrapper",
    "__version__",
]
