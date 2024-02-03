import torch
import torchaudio
from tools.AudioSettings import AudioSettings


def set_num_samples(waveform: torch.Tensor, target_num_samples: int):
    num_samples = waveform.shape[-1]
    if num_samples > target_num_samples:
        return waveform[..., :3]
    elif num_samples < target_num_samples:
        samples_to_add = target_num_samples - num_samples
        return torch.nn.functional.pad(waveform, [0, samples_to_add], value=0)
    else:
        return waveform


def standardize(tensor: torch.Tensor):
    centered = tensor - torch.mean(tensor)
    return centered / torch.std(tensor)


# https://pytorch.org/audio/stable/transforms.html
class PrepPipeline(torch.nn.Module):
    def __init__(self, s: AudioSettings, orig_sr: float) -> None:
        super().__init__()
        self.audio_settings = s
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=s.sample_rate,
            n_fft=s.n_fft,
            hop_length=s.hop_length,
            n_mels=s.melspec_bins,
        )
        self.orig_sr = orig_sr

    def forward(self, waveform: torch.Tensor):
        resampled = torchaudio.functional.resample(
            waveform, orig_freq=self.orig_sr, new_freq=self.audio_settings.sample_rate
        )
        spec = self.mel_spec(resampled)
        spec = torchaudio.functional.amplitude_to_DB(
            spec, multiplier=10.0, amin=1e-7, db_multiplier=10
        )
        spec = standardize(spec)
        return spec
