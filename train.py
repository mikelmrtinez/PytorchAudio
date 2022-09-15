from dataset.custom import UrbanSoundDataset
from models.cnn import CNNNetwork
import torchaudio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import datetime


SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

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
    parser.add_argument('--bs', type=int, default=64,
                        help='an integer for the accumulator')
    parser.add_argument('--epochs', type=int, default=100,
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

    train_data_dataloader = create_data_loader(usd, args.bs)

    train_data = [next(iter(train_data_dataloader)) for _ in range(len(train_data_dataloader))]

    cnn = CNNNetwork().to(device)
    print(cnn)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(cnn, train_data, loss_fn, optimizer, device, args.epochs)

    save_dir = Path().absolute() / 'outputs'
    save_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now()
    save_dir = save_dir / f'{date.year}_{date.month}_{date.day}_{date.hour}_{date.minute}_model.pth'

    torch.save(cnn.state_dict(), save_dir)