import librosa
import soundfile as sf
import numpy as np
from  scipy.signal import butter, sosfilt
import noisereduce as nr

def enhance(y : np.ndarray, sr):
    y= nr.reduce_noise(
            y=y,
            sr=16000,
            stationary=False, 
            prop_decrease=1.0,
            time_constant_s=0.5,
            freq_mask_smooth_hz=300,
            time_mask_smooth_ms=100,
            thresh_n_mult_nonstationary=1.5,
            sigmoid_slope_nonstationary=10, 
            n_fft=1024,  
            win_length=1024,
            hop_length=256, 
            use_tqdm=True
)
    y = y / max(abs(y))
    return y

if __name__ == '__main__':
    input_filename = '/Users/hasibulhasan/github/lyric_ninja/data/songs/separated/htdemucs/Papercut/vocals.wav'
    output_filename = 'enhanced_vocals.wav'

    try:

        waveform, sr = librosa.load(input_filename, sr=None, mono=True)
        final_vocals = enhance(waveform, sr)
        sf.write(output_filename, final_vocals, sr)

    except Exception as e:
        print(f"An error occurred: {e}")