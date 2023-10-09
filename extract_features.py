import torchaudio.transforms as T
import numpy as np
import torch
import soundfile as sf

def extract_audio_features(signal, freq=16000, n_mfcc=13, size=512, step=160):
    """
    Extracts MFCC, MFCC delta, MFCC delta2, and RMSE features from an audio signal.

    Args:
        signal (np.ndarray): Audio signal as a NumPy array.
        freq (int, optional): Sampling rate for MFCC features. Defaults to 16000.
        n_mfcc (int, optional): Number of MFCC coefficients to extract. Defaults to 13.
        size (int, optional): FFT size. Defaults to 512.
        step (int, optional): Hop length for STFT. Defaults to 160.

    Returns:
        mfcc (numpy.ndarray): MFCC features.
        mfcc_delta (numpy.ndarray): MFCC delta features.
        mfcc_delta2 (numpy.ndarray): MFCC delta-delta features.
        rmse (numpy.ndarray): Root Mean Square Energy (RMSE) features.
    """
    # Convert signal to torch tensor
    waveform = torch.tensor(signal, dtype=torch.float32)

    # Create a list of transforms for feature extraction
    transforms = [
        T.Resample(orig_freq=freq, new_freq=freq),  # No resampling needed when using signal directly
        T.MFCC(
            sample_rate=freq,
            n_mfcc=n_mfcc,
            melkwargs={'n_fft': size, 'hop_length': step, 'n_mels': n_mfcc + 2}
        ),
        T.ComputeDeltas(),
        T.ComputeDeltas(),
    ]

    # Apply the transforms to the waveform sequentially
    for transform in transforms:
        waveform = transform(waveform)

    # Calculate RMSE directly from the waveform
    rmse = torch.sqrt(torch.mean(waveform**2, dim=0))

    # Convert the features to NumPy arrays
    mfcc = waveform[0].numpy()
    mfcc_delta = waveform[1].numpy()
    mfcc_delta2 = waveform[2].numpy()
    rmse = rmse.numpy()

    return mfcc, mfcc_delta, mfcc_delta2, rmse



"""
# Example usage:
audio_file_flac = '/home/harry/Downloads/1188-133604-0044.flac'
#audio_file_wav = 'path_to_audio_file.wav'

mfcc_flac, mfcc_delta_flac, mfcc_delta2_flac, rmse_flac = extract_audio_features(audio_file_flac)
print("MFCC shape (FLAC):", mfcc_flac.shape)
print("MFCC Delta shape (FLAC):", mfcc_delta_flac.shape)
print("MFCC Delta2 shape (FLAC):", mfcc_delta2_flac.shape)
print("RMSE shape (FLAC):", rmse_flac.shape)
"""

