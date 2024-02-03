import matplotlib.pyplot as plt
import numpy as np


def plot_audio_channel(samples: np.ndarray, sample_rate: float):
    x = [i / sample_rate for i in range(samples.shape[0])]
    plt.plot(x, samples)
    plt.title("Audio Waveform")
    plt.xlabel("Seconds")


def plot_spectrogram(buckets: np.ndarray):  # sample_rate: float, hop_length: int
    # hop_rate = sample_rate / hop_length
    # x = [i / hop_rate for i in range(buckets.shape[0])]
    plt.pcolormesh(buckets, cmap="inferno")
    plt.title("Spectrogram")
    plt.xlabel("Seconds")
