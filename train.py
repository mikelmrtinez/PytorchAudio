from dataset.custom import UrbanSoundDataset, SimpleDataset
from models.cnn import CNNNetwork
import torchaudio
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path
import datetime
from sklearn.model_selection import train_test_split
from torchmetrics import Accuracy

SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

LEARNING_RATE = 1e-4


def create_data_loader(train_data, batch_size):
    return DataLoader(train_data, batch_size=batch_size)


def train_single_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    accum_loss, accum_acc = 0., 0.
    accuracy = Accuracy()
    for input, target in tqdm(data_loader):
        input, target = input.to(device), target.to(device)

        prediction = model(input)

        prediction_class = torch.argmax(prediction, dim=1)
        loss = loss_fn(prediction, target)
        accum_loss += loss.item()
        accum_acc += accuracy(prediction_class, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'loss: {accum_loss / len(data_loader)}')
    print(f'acc: {accum_acc / len(data_loader)}')


def evaluate_model(model, data_loader, loss_fn, device):

    model.eval()
    accum_loss, accum_acc = 0., 0.
    accuracy = Accuracy()
    for input, target in tqdm(data_loader):
        input, target = input.to(device), target.to(device)
        with torch.no_grad():
            prediction = model(input)

            prediction_class = torch.argmax(prediction, dim=1)
            loss = loss_fn(prediction, target)
            accum_loss += loss.item()
            accum_acc += accuracy(prediction_class, target)

    print(f'val_loss: {accum_loss / len(data_loader)}')
    print(f'val_acc: {accum_acc / len(data_loader)}')


def train(model, data, loss_fn, optimizer, device, epochs):
    train_data, val_data = data
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}')
        train_single_epoch(model, train_data, loss_fn, optimizer, device)
        evaluate_model(model, val_data, loss_fn, device)
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
    print('Processing data...')
    data_dataloader = create_data_loader(usd, 8732)

    print('Storing data in RAM memory...')
    X, y = next(iter(data_dataloader))

    print('Splitting data...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5)

    train_data = create_data_loader(SimpleDataset(X_train, y_train), args.bs)
    test_data = create_data_loader(SimpleDataset(X_test, y_test), args.bs)
    val_data = create_data_loader(SimpleDataset(X_val, y_val), args.bs)

    cnn = CNNNetwork().to(device)
    print(cnn)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LEARNING_RATE)

    train(cnn, (train_data, val_data), loss_fn, optimizer, device, args.epochs)

    evaluate_model(cnn, test_data, loss_fn, device)

    save_dir = Path().absolute() / 'outputs'
    save_dir.mkdir(parents=True, exist_ok=True)
    date = datetime.datetime.now()
    save_dir = save_dir / f'{date.year}_{date.month}_{date.day}_{date.hour}_{date.minute}_model.pth'

    torch.save(cnn.state_dict(), save_dir)