import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pickle

class SOCDataset(Dataset):
    def __init__(self,data) -> None:
        self.data = pickle.load(open(data, 'rb'))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :returns: x (timesteps, 3), y (timesteps, 1)
        """
        x, y = self.data[idx]
        return [torch.tensor(z, dtype=torch.float32) for z in [x, y]]

class SOCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=3,
            hidden_size=10,
            num_layers=1,
            batch_first=False,
        )
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        x = torch.clamp(x, 0, 1)
        return x

def smooth_predictions(predictions, window_size):
    # Apply a moving average filter to the predictions over a sliding window of size window_size
    window = np.ones(window_size) / window_size
    smoothed_predictions = np.convolve(predictions, window, mode='same')
    return smoothed_predictions


def train():
    model = SOCModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()

    dataset = SOCDataset('data.pkl')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    best_val_loss = float('inf')
    best_model_state_dict = None
    early_stopping_counter = 0

    for epoch in range(100):
        losses = []
        model.train()
        for x, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss = torch.sqrt(loss)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        
        scheduler.step() # reduce learning rate

        avg_train_loss = sum(losses) / len(losses)

        model.eval()
        val_losses = []
        for x, y in val_dataloader:
            y_hat = model(x)
            val_loss = criterion(y_hat, y)
            val_loss = torch.sqrt(val_loss)
            val_losses.append(val_loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f'Epoch {epoch}: Train loss {avg_train_loss}, Validation loss {avg_val_loss}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= 10:
                print('Validation loss did not improve for 10 epochs. Stopping early.')
                break

    x, y = dataset[0]
    model.load_state_dict(best_model_state_dict)
    model.eval()
    y_hat = model(x.unsqueeze(0))

    print(y[:100, 0])
    print(y_hat[0, :100, 0])

    smooth_predictions(y_hat[0, :, 0].detach().numpy(), 10)

    plt.plot(y[:, 0].detach().numpy())
    plt.plot(y_hat[0, :, 0].detach().numpy())
    plt.legend(['y', 'y_hat'])
    plt.show()

train()
