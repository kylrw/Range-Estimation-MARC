import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pickle
from scipy.signal import savgol_filter

# Define a PyTorch Dataset to load the input data
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

# Define a PyTorch model for predicting the output
class SOCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 16
        self.num_layers = 1
        self.lstm = nn.LSTM(3, 16, 1, batch_first=False)
        self.ln = nn.LayerNorm(16)
        self.fc2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )


    def forward(self, x):
        out = x
        # Pass the input through the LSTM
        out, _ = self.lstm(out)
        # Apply a layer normalization to the LSTM output
        out = self.ln(out)
        # Apply a dropout layer to the LSTM output
        out = self.fc2(out)
        # Apply the linear layer to the LSTM output
        #out = self.linear(out)
        # Apply a sigmoid activation function to the output to ensure that it is between 0 and 1
        out = torch.clamp(out, 0, 1)
        return out

# Define a function to smooth the model's predictions using Savitzky-Golay Filter
def smooth_predictions(predictions, window_size):
    np.random.seed(1)
    # Create a Savitzky-Golay filter
    filter = savgol_filter(predictions, window_size, 3)
    return filter

# Define a function to plot the model's predictions
def train():
    model = SOCModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    criterion = nn.MSELoss()

    dataset = SOCDataset('data.pkl')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)])
    # Create DataLoaders to load batches of input-output pairs during training and validation
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)

    best_val_loss = float('inf')
    best_model_state_dict = None
    early_stopping_counter = 0

    for epoch in range(200):
        losses = []
        model.train()
        for x, y in tqdm(train_dataloader):
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss = torch.sqrt(loss)
            # Backpropagate the error and update model weights
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
        
        # Update the learning rate scheduler to reduce learning rate
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

        # If current validation loss is the best seen so far, save model state dict
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state_dict = model.state_dict()
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            # If validation loss did not improve for 10 epochs, stop early
            if early_stopping_counter >= 10:
                print('Validation loss did not improve for 10 epochs. Stopping early.')
                break

    # Use the best model state dict to make predictions on a sample input
    x, y = dataset[0]
    model.load_state_dict(best_model_state_dict)
    model.eval()
    y_hat = model(x.unsqueeze(0))

    print(y[:100, 0])
    print(y_hat[0, :100, 0])

    smooth_predictions(y_hat[0, :, 0].detach().numpy(), 1000)

    plt.plot(y[:, 0].detach().numpy())
    plt.plot(y_hat[0, :, 0].detach().numpy())
    plt.legend(['y', 'y_hat'])
    plt.show()

    return model

model = train()

# Save the model in Models folder
#torch.save(model, 'Models/model.pt')




