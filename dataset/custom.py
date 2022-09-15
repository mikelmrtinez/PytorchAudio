import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
import pandas as pd


class UrbanSoundDataset(Dataset):
    def __init__(self, annotations_file, audio_dir, transformation, target_sample_rate, num_samples, device):
        self.annotations = pd.read_csv(Path(annotations_file))
        self.audio_dir = Path(audio_dir)
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self.__get_audio_sample_path(index)
        label = self.__get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)

        signal = self.__resample_if_necessary(signal, sr)
        signal = self.__mixdown_if_necessary(signal)
        signal = self.__cut_if_necessary(signal)
        signal = self.__right_pad_if_necessary(signal)
        signal = self.transformation(signal)

        return signal, label

    def __cut_if_necessary(self, signal):
        return signal[:, :self.num_samples] if signal.shape[1] > self.num_samples else signal

    def __right_pad_if_necessary(self, signal):
        num_missing_elements = self.num_samples - signal.shape[1]
        last_dim_pad = (0, num_missing_elements)
        return torch.nn.functional.pad(signal, last_dim_pad)

    def __resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resample = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resample(signal)
        return signal

    def __mixdown_if_necessary(self, signal):

        return torch.mean(signal, dim=0, keepdim=True) if signal.shape[0] > 1 else signal

    def __get_audio_sample_path(self, index):
        fold = f'fold{self.annotations.iloc[index, 5]}'
        path = self.audio_dir / fold / self.annotations.iloc[index, 0]
        return path

    def __get_audio_sample_label(self, index):
        return self.annotations.iloc[index, 6]


if __name__ == '__main__':
    ANNOTATION_FILE = '/home/mikel/Documents/music/datasets/UrbanSound8K/metadata/UrbanSound8K.csv'
    AUDIO_DIR = '/home/mikel/Documents/music/datasets/UrbanSound8K/audio'
    SAMPLE_RATE = 22050
    NUM_SAMPLES = 22050

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512, n_mels=64)
    usd = UrbanSoundDataset(ANNOTATION_FILE, AUDIO_DIR,
                            mel_spectrogram, target_sample_rate=SAMPLE_RATE,
                            num_samples=NUM_SAMPLES, device=device)
    print(len(usd))
    signal, sr = usd[0]
    print(signal.shape)

