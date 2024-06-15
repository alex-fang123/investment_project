import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler

if torch.cuda.is_available():
    device = torch.device('cuda')  # 选择第一个可用的GPU
else:
    device = torch.device('cpu')  # 若GPU不可用，则选择CPU


class StockDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]
        return x, y


class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 500, d_model))  # 500 is a max sequence length
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.embedding(src) * np.sqrt(self.d_model)
        src += self.pos_encoder[:, :src.size(1), :]
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return output


# Hyperparameters
input_seq_size = 30
output_seq_size = 1
input_dim = input_seq_size * 32  # 特征数量
d_model = 1024
nhead = 64
num_encoder_layers = 3
dim_feedforward = 128
dropout = 0.1
seq_length = 30
batch_size = 32
num_epochs = 10
learning_rate = 0.000001



def get_src(data, num_samples, input_seq_size):
    if len(data) - input_seq_size - output_seq_size + 1 <= 0:
        return

    # src_data = torch.tensor(np.array([data[i:i + input_seq_size] for i in
    #                          range(num_samples)])).float().to(device)
    # src_data = src_data.reshape(src_data.shape[0], 960)
    # tgt_data = torch.tensor(np.array([data[i + input_seq_size:i + input_seq_size + output_seq_size] for i in
    #                          range(num_samples)])).float().to(device)  # 0 means daily_stock_return

    src_data = np.array([data[i:i + input_seq_size] for i in range(num_samples)])
    # tgt_data = np.array([data[i + input_seq_size:i + input_seq_size + output_seq_size] for i in range(num_samples)])
    return src_data


def get_tgt(data, num_samples, input_seq_size, output_seq_size):
    if len(data) - input_seq_size - output_seq_size + 1 <= 0:
        return
    tgt_data = np.array([data['monthly_stock_return'][i + input_seq_size:i + input_seq_size + output_seq_size] for i in range(num_samples)])
    return tgt_data


# use my data
data = pd.read_feather('temp/before1.6_monthly_stock_return.feather')

# change data type
data['size_tag'] = data['size_tag'].astype('float64')
data['ROE_tag'] = data['ROE_tag'].astype('float64')
data['bm_tag'] = data['bm_tag'].astype('float64')
data['INV_tag'] = data['INV_tag'].astype('float64')
data['momentum_tag'] = data['momentum_tag'].astype('float64')
data['reversal_tag'] = data['reversal_tag'].astype('float64')

feature_list = ['monthly_stock_return', 'rf', 'excess_return',
                'market_value', 'JAN', 'total_assets', 'total_shareholders_equity',
                'BM ratio', 'ROE', 'assets_increasing_rate', 'momentum', 'reversal',
                'size_tag', 'ROE_tag', 'bm_tag', 'INV_tag', 'momentum_tag',
                'reversal_tag', 'mkt_risk_premium', 'SMB', 'HML', 'RMW', 'CMA',
                't+1_excess_return', 'JAN t+1', 'size*JAN', 'BM*JAN',
                'ROE*JAN', 'INV*JAN', 'MOM*JAN', 'REV*JAN',
                'mkt_risk_premium*JAN']
scaler = StandardScaler()
mean_list = []
std_list = []
data[feature_list] = scaler.fit_transform(data[feature_list])
normalized_data = data
normalized_mean = scaler.mean_
normalized_std = scaler.scale_

# normalized_data = normalized_data[(normalized_data['Stkcd'] == "000001") | (normalized_data['Stkcd'] == "000002")]

train_data = normalized_data[normalized_data['month'] < '2015-01-01']


# num_samples = len(train_data) - input_seq_size - output_seq_size + 1

test_data = normalized_data[normalized_data['month'] >= '2015-01-01']

# a = normalized_data[normalized_data['Stkcd'] == "000001"]

src_data = train_data.groupby('Stkcd', observed=False).apply(
    lambda x: get_src(x[feature_list], len(x) - input_seq_size - output_seq_size + 1, input_seq_size), include_groups=False)
src_data.dropna(inplace=True)
src_data = np.concatenate(src_data)
src_data = src_data.reshape(src_data.shape[0], input_dim)
src_data = torch.tensor(src_data).float().to(device)

tgt_data = train_data.groupby('Stkcd', observed=False).apply(
    lambda x: get_tgt(x[feature_list], len(x) - input_seq_size - output_seq_size + 1, input_seq_size, output_seq_size), include_groups=False)
tgt_data.dropna(inplace=True)
tgt_data = np.concatenate(tgt_data)
tgt_data = tgt_data.reshape(tgt_data.shape[0])
tgt_data = torch.tensor(tgt_data).float().to(device)


# Create DataLoader
dataset = StockDataset(src_data, tgt_data, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Initialize model, loss function, and optimizer
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward, dropout).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    start_time = time.perf_counter()
    for batch in dataloader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.float(), y_batch.float()

        optimizer.zero_grad()
        output = model(x_batch)
        loss = criterion(output.squeeze(), y_batch)

        if torch.isnan(loss):
            print('Loss is NaN. Aborting training')
            break
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        if any(torch.isnan(param.grad).any() for param in model.parameters()):
            print("Gradient is NaN. Aborting training.")
            break

        optimizer.step()

        total_loss += loss.item()
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, time-consumed: {total_time:.4f}s")

# Evaluation
model.eval()
with torch.no_grad():
    total_loss = 0
    for batch in dataloader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.float(), y_batch.float()
        # print(x_batch.shape)
        output = model(x_batch)
        loss = criterion(output.squeeze(), y_batch)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")

# on test set
src_data = test_data.groupby('Stkcd', observed=False).apply(
    lambda x: get_src(x[feature_list], len(x) - input_seq_size - output_seq_size + 1, input_seq_size), include_groups=False)
src_data.dropna(inplace=True)
src_data = np.concatenate(src_data)
src_data = src_data.reshape(src_data.shape[0], input_dim)
src_data = torch.tensor(src_data).float().to(device)

tgt_data = test_data.groupby('Stkcd', observed=False).apply(
    lambda x: get_tgt(x[feature_list], len(x) - input_seq_size - output_seq_size + 1, input_seq_size, output_seq_size), include_groups=False)
tgt_data.dropna(inplace=True)
tgt_data = np.concatenate(tgt_data)
# tgt_data = tgt_data.reshape(tgt_data.shape[0], 32)
tgt_data = torch.tensor(tgt_data).float().to(device)

# Create DataLoader
dataset = StockDataset(src_data, tgt_data, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.eval()
with torch.no_grad():
    total_loss = 0
    for batch in dataloader:
        x_batch, y_batch = batch
        x_batch, y_batch = x_batch.float(), y_batch.float()

        output = model(x_batch)
        loss = criterion(output.squeeze(), y_batch)

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Loss on Test set: {avg_loss:.4f}")

# save the model
torch.save(model.state_dict(), 'models/new_10_epochs.pth')

# load the model
# model = TransformerModel(input_dim=960, d_model=512, nhead=64, num_encoder_layers=3, dim_feedforward=128)
# model.load_state_dict(torch.load('models/10_epochs.pth'))
# model.to(device)
