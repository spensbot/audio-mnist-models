from torch.utils.data import Dataset, DataLoader
import torchaudio
from tools.AudioSettings import AudioSettings
from tools.io import get_all_files
from tools.transforms import set_num_samples
import os
import librosa


def get_label(filename: str):
    extension = filename.split(".")[-1]
    if extension != "wav":
        print(f"Not wav: {extension} | {filename}")
        return None
    try:
        return int(filename[0])
    except:
        print("File name error")
        return None


class AudioMnistDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        audio_settings: AudioSettings,
        num_samples: int,
        transform=None,
    ) -> None:
        super().__init__()
        self.audio_settings = audio_settings
        self.data_path = data_path
        self.files = self.__get_files()
        self.transform = transform
        self.num_samples = num_samples
        self.cache = {}

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        if idx not in self.cache:
            path, name, label = self.files[idx]
            (multi_channel_waveform, sample_rate) = torchaudio.load(path)
            multi_channel_waveform = set_num_samples(
                multi_channel_waveform, self.num_samples
            )
            self.cache[idx] = (multi_channel_waveform, sample_rate, label)

        return self.cache[idx]

    def __get_files(self) -> list[(str, str)]:
        all_files = get_all_files(self.data_path)
        print(len(all_files))
        data_files = [
            (path, name, get_label(name))
            for (path, name) in all_files
            if get_label(name) != None
        ]
        print(len(data_files))
        return data_files
