import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
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
input_dim = 1140  # 特征数量
d_model = 64
nhead = 4
num_encoder_layers = 3
dim_feedforward = 128
dropout = 0.1
seq_length = 30
batch_size = 32
num_epochs = 100
learning_rate = 0.001

# def get_src_and_tgt(data, seq_length=30):
#     train_data = data[data['date'] < '2015-01-01']
#     feature_list = ['daily_stock_return', 'rf', 'excess_return',
#                     'market_value', 'day_kind', 'total_assets', 'total_shareholders_equity',
#                     'BM ratio', 'ROE', 'assets_increasing_rate', 'momentum', 'reversal',
#                     'size_tag', 'ROE_tag', 'bm_tag', 'INV_tag', 'momentum_tag',
#                     'reversal_tag', 'mkt_risk_premium', 'SMB', 'HML', 'RMW', 'CMA',
#                     't+1_excess_return', 'Monday t+1', 'size*Monday', 'BM*Monday',
#                     'ROE*Monday', 'INV*Monday', 'MOM*Monday', 'REV*Monday',
#                     'mkt_risk_premium*Monday', 'new_size_tag', 'new_size_tag t - 1',
#                     'new_bm_tag', 'new_bm_tag t - 1', 'rule', 'rule t - 1']
#     normalized_data = train_data[feature_list]
#     scaler = StandardScaler()
#     normalized_data = scaler.fit_transform(normalized_data)
#     normalized_mean = scaler.mean_
#     normalized_std = scaler.scale_
#
#     src_data = torch.tensor([normalized_data[i:i + seq_length] for i in
#                                 range(len(train_data) - seq_length)]).float().to(device)
#     src_data = src_data.reshape(src_data.shape[0], 1140)
#     tgt_data = torch.tensor([normalized_data[i + seq_length][0] for i in
#                                 range(len(train_data) - seq_length)]).float().to(device)  # 0 means daily_stock_return

# use my data
data = pd.read_feather('temp/before2.6_daily_stock_return.feather')

# change data type
data['day_kind'] = data['day_kind'].apply(lambda
                                              x: 1 if x == 'Monday' else 2 if x == 'Tuesday' else 3 if x == 'Wednesday' else 4 if x == 'Thursday' else 5 if x == 'Friday' else 6 if x == 'Saturday' else 7)
data['size_tag'] = data['size_tag'].astype('float64')
data['ROE_tag'] = data['ROE_tag'].astype('float64')
data['bm_tag'] = data['bm_tag'].astype('float64')
data['INV_tag'] = data['INV_tag'].astype('float64')
data['momentum_tag'] = data['momentum_tag'].astype('float64')
data['reversal_tag'] = data['reversal_tag'].astype('float64')
data['new_size_tag t - 1'] = data['new_size_tag t - 1'].astype('float64')
data['new_bm_tag t - 1'] = data['new_bm_tag t - 1'].astype('float64')
data['rule'] = data['rule'].apply(lambda x: 1 if x == '10.0' else 2 if x == '11.0' else 3 if x == '00.0' else 4)
data['rule t - 1'] = data['rule t - 1'].apply(
    lambda x: 1 if x == '10.0' else 2 if x == '11.0' else 3 if x == '00.0' else 4 if x == '01.0' else 5)

tmp_data = data[data['Stkcd'] == "000001"]

train_data = tmp_data[tmp_data['date'] < '2015-01-01']
# train_data = train_data.iloc[:3102, :]

input_seq_size = 30
output_seq_size = 1
num_samples = len(train_data) - input_seq_size - output_seq_size + 1

test_data = tmp_data[tmp_data['date'] >= '2015-01-01']
# test_data = test_data.iloc[:3073, :]

feature_list = ['daily_stock_return', 'rf', 'excess_return',
                'market_value', 'day_kind', 'total_assets', 'total_shareholders_equity',
                'BM ratio', 'ROE', 'assets_increasing_rate', 'momentum', 'reversal',
                'size_tag', 'ROE_tag', 'bm_tag', 'INV_tag', 'momentum_tag',
                'reversal_tag', 'mkt_risk_premium', 'SMB', 'HML', 'RMW', 'CMA',
                't+1_excess_return', 'Monday t+1', 'size*Monday', 'BM*Monday',
                'ROE*Monday', 'INV*Monday', 'MOM*Monday', 'REV*Monday',
                'mkt_risk_premium*Monday', 'new_size_tag', 'new_size_tag t - 1',
                'new_bm_tag', 'new_bm_tag t - 1', 'rule', 'rule t - 1']

normalized_data = train_data[feature_list].copy()
scaler = StandardScaler()
normalized_data = scaler.fit_transform(normalized_data)
normalized_mean = scaler.mean_
normalized_std = scaler.scale_

src_data = torch.tensor([normalized_data[i:i + input_seq_size] for i in
                         range(num_samples)]).float().to(device)
src_data = src_data.reshape(src_data.shape[0], 1140)
tgt_data = torch.tensor([normalized_data[i + input_seq_size:i + input_seq_size + output_seq_size][0][0] for i in
                         range(num_samples)]).float().to(device)  # 0 means daily_stock_return
tgt_data = tgt_data.reshape(tgt_data.shape[0])

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

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation
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
    print(f"Validation Loss: {avg_loss:.4f}")
