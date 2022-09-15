from dataset.custom import UrbanSoundDataset
from models.cnn import CNNNetwork
import torchaudio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

# ANNOTATION_FILE = '/home/mikel/Documents/music/datasets/UrbanSound8K/metadata/UrbanSound8K.csv'
# AUDIO_DIR = '/home/mikel/Documents/music/datasets/UrbanSound8K/audio'
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-4


def create_data_loader(train_data, batch_size):
    return DataLoader(train_data, batch_size=batch_size)


def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    for input, target in tqdm(data_loader):
        input, target = input.to(device), target.to(device)

        prediction = model(input)
        loss = loss_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'loss: {loss.item()}')


def train(model, data_loader, loss_fn, optimizer, device, epochs):
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        train_single_epoch(model, data_loader, loss_fn, optimizer, device)
        print('--------------------------')
    print('Finish training!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN for Audio Classification')
    parser.add_argument('--data_dir',  type=str, default=None,
                        help='an integer for the accumulator')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    print(device)

    ANNOTATION_FILE = Path(args.data_dir) / 'metadata/UrbanSound8K.csv'
    AUDIO_DIR = Path(args.data_dir) / 'audio'

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=512,
                                                           n_mels=64)
    usd = UrbanSoundDataset(ANNOTATION_FILE, AUDIO_DIR,
                            mel_spectrogram, target_sample_rate=SAMPLE_RATE,
                            num_samples=NUM_SAMPLES, device=device)

    train_data = create_data_loader(usd, BATCH_SIZE)

    cnn = CNNNetwork().to(device)
    print(cnn)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(cnn, train_data, loss_fn, optimizer, device, EPOCHS)

    torch.save(cnn.state_dict(), 'cnn.pth')