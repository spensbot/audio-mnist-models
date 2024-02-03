from dataclasses import dataclass
from tools.math import snap_to_power_of_2


@dataclass
class AudioSettings:
    def __init__(
        self,
        sample_rate: float = 8000,  # samples per second
        spectrogram_time_step: float = 0.02,  # time step between stft bins (seconds). Resulting fft width will be snapped to a power of 2
        melspec_bins: int = 80,
    ) -> None:
        self.sample_rate = sample_rate
        self.melspec_bins = melspec_bins
        requested_hop_length = int(self.sample_rate * spectrogram_time_step)
        requested_n_fft = requested_hop_length * 2
        self.n_fft = snap_to_power_of_2(requested_n_fft)
        self.hop_length = self.n_fft // 2

    def __str__(self):
        return f"Sample Rate: {self.sample_rate} | bins: {self.melspec_bins} | n_fft: {self.n_fft} | hop: {self.hop_length} | freq ({self.min_freq()} - {self.max_freq()})hz"

    def min_freq(self):
        stft_seconds = self.n_fft / self.sample_rate
        return 1 / stft_seconds

    def max_freq(self):
        return self.sample_rate / 2
