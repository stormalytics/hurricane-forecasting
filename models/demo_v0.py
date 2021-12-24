from typing import AsyncIterable, List
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from torch._C import _tracer_warn_use_python
from torch.nn.modules import loss
from torch.nn.modules.dropout import Dropout
import tqdm

from pathlib import Path
from pprint import pprint as pp

import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam

from tcn import TemporalConvNet
# from dialate_loss.dilate_loss import dilate_loss
from soft_dtw.soft_dtw_cuda import SoftDTW


class HURDAT(Dataset):
    def __init__(self, hurdat_table, input_vars: List[str], target_vars: List[str], grouping_var: str, time_idx: str, past_horizon: int = 1, future_horizon: int = 1):

        self.hurdat_table = None
        if isinstance(hurdat_table, pd.DataFrame):
            self.hurdat_table = hurdat_table
        if isinstance(hurdat_table, str):
            hurdat_table_path = Path(hurdat_table)
            self.hurdat_table = pd.read_csv(hurdat_table_path)

        self.input_vars = input_vars
        self.target_vars = target_vars

        self.past_horizon = past_horizon
        self.future_horizon = future_horizon

        self.grouping_var = grouping_var
        self.time_idx = time_idx

        self.generated_samples = self.generate_all_ts_samples()

    def generate_all_ts_samples(self) -> List[dict]:
        all_samples = []
        for atcf_code, hurricane_df in tqdm.tqdm(self.hurdat_table.groupby(self.grouping_var, sort=False)):
            storm_samples = self.generate_storm_ts_samples(hurricane_df)
            all_samples.extend(storm_samples)
        return all_samples

    def generate_storm_ts_samples(self, storm_df) -> List[dict]:
        data = []
        for w in storm_df.rolling(window=self.past_horizon+self.future_horizon):
            if w.shape[0] != self.past_horizon+self.future_horizon:
                continue
            if w.head(self.past_horizon).iloc[-1]["system_status"] not in ["TS", "HU"]:
                continue

            data_window = {}

            data_window["input"] = torch.tensor(
                w.head(self.past_horizon)[self.input_vars].values, dtype=torch.float)
            data_window["input_time_idx"] = torch.tensor(
                w.head(self.past_horizon)[self.time_idx].values, dtype=torch.float)

            data_window["output"] = torch.tensor(w.tail(self.future_horizon)[
                                                    self.target_vars].values, dtype=torch.float)
            data_window["output_time_idx"] = torch.tensor(
                w.tail(self.future_horizon)[self.time_idx].values, dtype=torch.float)

            data_window["window_time_idx"] = torch.tensor(w[self.time_idx].values, dtype=torch.float)
            data_window["atcf_code"] = w['atcf_code'].iloc[0]

            data.append(data_window)
        return data

    def __len__(self):
        return len(self.generated_samples)

    def __getitem__(self, idx):
        sample = self.generated_samples[idx]
        # input_data = sample["input"]
        # output_data = sample["output"]
        return sample


class Skip(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class MLP(nn.Module):
    def __init__(self, input_vars_len: int, target_vars_len: int, past_horizon: int = 1, future_horizon: int = 1, hidden_size=128):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(past_horizon*input_vars_len, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-4),

            Skip(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                               nn.ReLU(),
                               nn.LayerNorm(hidden_size, eps=1e-4))),

            Skip(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                               nn.ReLU(),
                               nn.LayerNorm(hidden_size, eps=1e-4))),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size, eps=1e-4),

            nn.Linear(hidden_size, target_vars_len*future_horizon),
            nn.Unflatten(1, torch.Size([future_horizon, target_vars_len]))
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class TCN_MLP(nn.Module):
    def __init__(self, input_vars_len: int, target_vars_len: int, past_horizon: int = 1, future_horizon: int = 1, hidden_size=32):
        super(TCN_MLP, self).__init__()
        self.layers = nn.Sequential(
            TemporalConvNet(past_horizon, [input_vars_len, hidden_size//2, hidden_size], kernel_size=6, dropout=0.0),
            nn.Flatten(),

            nn.Linear(hidden_size*input_vars_len, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),

            Skip(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                               nn.ReLU(),
                               nn.LayerNorm(hidden_size)
                               )),

            Skip(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                               nn.ReLU(),
                               nn.LayerNorm(hidden_size)
                               )),

            nn.Linear(hidden_size, target_vars_len*future_horizon),
            nn.Unflatten(1, torch.Size([future_horizon, target_vars_len]))
        )

    def forward(self, x):
        x = self.layers(x)
        return x


def haversine(pred, actual, batch=True):
    R = 6371
    if batch:
        lon_actual, lon_pred = actual[:, :, 0], pred[:, :, 0]
        lat_actual, lat_pred = actual[:, :, 1], pred[:, :, 1]
    else:
        lon_actual, lon_pred = actual[:, 0], pred[:, 0]
        lat_actual, lat_pred = actual[:, 1], pred[:, 1]

    lon_actual, lon_pred = torch.deg2rad(lon_actual), torch.deg2rad(lon_pred)
    lat_actual, lat_pred = torch.deg2rad(lat_actual), torch.deg2rad(lat_pred)

    alpha = torch.sin((lat_pred-lat_actual)/2)**2 + torch.cos(lat_pred) * \
        torch.cos(lat_actual)*torch.sin((lon_pred-lon_actual)/2)**2
    d = 2*R*torch.arcsin(torch.sqrt(alpha))

    d = d*0.539957

    d = d.unsqueeze(-1)
    return d


def equirectangular_approx(pred, actual, batch=True):
    R = 6371
    if batch:
        lon_actual, lon_pred = actual[:, :, 0], pred[:, :, 0]
        lat_actual, lat_pred = actual[:, :, 1], pred[:, :, 1]
    else:
        lon_actual, lon_pred = actual[:, 0], pred[:, 0]
        lat_actual, lat_pred = actual[:, 1], pred[:, 1]

    lon_actual, lon_pred = torch.deg2rad(lon_actual), torch.deg2rad(lon_pred)
    lat_actual, lat_pred = torch.deg2rad(lat_actual), torch.deg2rad(lat_pred)

    x = (lon_pred-lon_actual) * torch.cos((lat_pred+lat_actual)/2)
    y = (lat_pred-lat_pred)
    d = torch.sqrt(x**2 + y**2) * R

    d = d*0.539957

    d = d.unsqueeze(-1)
    return d


# def displacement_seq_to_lat_lon_seq(start_lon_lat, displacement_seq):
#     cumulative_displacement = torch.cumsum(displacement_seq, dim=1)



def path_distance_error_location(pred, actual):
    distances = haversine(pred, actual)
    sum_distance = torch.sum(distances, dim=-1)
    average_error = torch.mean(sum_distance)
    return average_error


class TSExpLoss:
    def __init__(self, alpha, loss_fn) -> None:
        self.alpha = alpha
        self.loss_fn = loss_fn

    def __call__(self, pred, actual):
        loss_result = self.loss_fn(pred, actual)

        exp_weights_t = torch.arange(loss_result.shape[-2], dtype=float, requires_grad=True).to(
            pred.get_device()).unsqueeze(1).repeat(1, loss_result.shape[-1])
        exp_weights = torch.exp(exp_weights_t/self.alpha)/torch.e
        # print(exp_weights)

        loss_result_exp = loss_result * exp_weights
        loss_result_weighted_average = torch.sum(loss_result_exp, dim=[1, 2]) / torch.sum(exp_weights_t)
        batch_loss = torch.mean(loss_result_weighted_average)
        return batch_loss

# class DIALATELoss:
#     def __init__(self, device, alpha=0.5, gamma=0.001) -> None:
#         self.alpha = alpha
#         self.gamma = gamma
#         self.device = device

#     def __call__(self, pred, actual):
#         channels = pred.shape[-1]
#         seq_losses = torch.zeros((channels,))
#         for i in range(channels):
#             loss, loss_shape, loss_temporal = dilate_loss(actual[:, :, i].unsqueeze(-1),
#                                                           pred[:, :, i].unsqueeze(-1),
#                                                           self.alpha, self.gamma,
#                                                           self.device)
#             seq_losses[i] = loss
#         loss = seq_losses.sum()
#         return loss


class SDTWLoss:
    def __init__(self, gamma=0.1, use_cuda=False) -> None:
        self.gamma = gamma
        self.use_cuda = use_cuda
        self.sdtw = SoftDTW(use_cuda=self.use_cuda, gamma=self.gamma)

    def __call__(self, pred, actual):
        loss = self.sdtw(pred, actual)
        return loss.mean()


def smape(pred, actual):
    return torch.mean(2*torch.abs(pred-actual)/(torch.abs(pred)+torch.abs(actual)))

########################
### DEMO STARTS HERE ###
########################

device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
print(device)

data = pd.read_csv('./data/hurdat/hurdat2_processed.csv')


input_vars = ["longitude", "latitude", "min_pressure", "max_sus_wind", "landfall", "hour",
              "jday", "time_idx", "delta_distance", "delta_distance_x", "delta_distance_y", "azimuth",
              "x", "y", "vpre"]
output_vars = ["longitude", "latitude"]


past_horizon = 12
future_horizon = 12
hurdat_dataset = HURDAT(data, input_vars=input_vars, target_vars=output_vars,
                        grouping_var="atcf_code", time_idx="time_idx",
                        past_horizon=past_horizon, future_horizon=future_horizon)

test_percent = 0.3
test_count = int(len(hurdat_dataset) * 0.3)
train_count = len(hurdat_dataset) - test_count

print(f"# of training samples: {train_count}")
print(f"# of testing samples: {test_count}")

train_data, test_data = random_split(
    hurdat_dataset, [train_count, test_count], generator=torch.Generator().manual_seed(42))

batch_size = 256
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
test_dataloader_viewing = DataLoader(test_data, batch_size=1)


# for sample in hurdat_dataset:
#     x, y = sample["input"].to(device), sample["output"].to(device)
#     print(x.shape)
#     y_1 = y[:1]
#     y_2 = y[1:2]
#     print(y_1)
#     print(y_2)
#     print(y_1.shape)
#     print(y_2.shape)
#     d_haversine = haversine(y_1, y_2, batch=False)
#     d_fcc = equirectangular_approx(y_1, y_2, batch=False)
#     print(d_haversine.shape)
#     print(d_fcc.shape)
#     print(d_haversine)
#     print(d_fcc)
#     print(d_fcc-d_haversine)
#     print()


# exit()


# model = MLP(past_horizon=past_horizon, future_horizon=future_horizon,
#             input_vars_len=len(input_vars), target_vars_len=2, hidden_size=128)
model = TCN_MLP(past_horizon=past_horizon, future_horizon=future_horizon,
                input_vars_len=len(input_vars), target_vars_len=2, hidden_size=256)
model = model.to(device)
print(model)

# loss_fn = smape
# loss_fn = nn.L1Loss()
# loss_fn = TSExpLoss(10, loss.L1Loss(reduction='none'))
loss_fn = path_distance_error_location
# loss_fn = TSExpLoss(12, haversine)
# loss_fn = DIALATELoss(device)
# loss_fn = SDTWLoss(use_cuda=True, gamma=0.1)
optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)


def train_loop(dataloader):
    model.train()

    num_batches = len(dataloader)
    train_loss = 0
    train_error_distance = 0

    for batch_idx, data in enumerate(dataloader):
        # Compute prediction and loss
        x, y = data["input"].to(device), data["output"].to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss_fn(pred, y).item()
        train_error_distance += haversine(pred, y, batch=True).mean().detach().item()

    train_loss /= num_batches
    train_error_distance /= num_batches
    print(f"Train Loss: {train_loss:>8f}")
    print(f"Train Error Distance: {train_error_distance:>8f}")
    print()


def test_loop(dataloader):
    model.eval()

    num_batches = len(dataloader)
    test_loss = 0
    test_error_distance = 0

    with torch.no_grad():
        for data in dataloader:
            x, y = data["input"].to(device), data["output"].to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
            test_error_distance += haversine(pred, y, batch=True).mean().detach().item()

    test_loss /= num_batches
    test_error_distance /= num_batches
    print(f"Test Loss: {test_loss:>8f}")
    print(f"Test Error Distance: {test_error_distance:>8f}")
    print()


# with torch.autograd.detect_anomaly():

epochs = 200
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader)
    test_loop(test_dataloader)
print("Done!")


error_distances_list = []
for sample in hurdat_dataset:
    x, y = sample["input"].to(device), sample["output"].to(device)
    pred = model(x.unsqueeze(0))[0]
    actual = y
    error_dist = haversine(pred, actual, batch=False)
    error_distances_list.append(error_dist.cpu().detach())

error_distances = torch.stack(error_distances_list)
error_distances = error_distances[..., 0]

error_distances_df = pd.DataFrame(error_distances.numpy().astype("float"))
columns = [f'future_horizon_{(i+1)*6}' for i in range(future_horizon)]
error_distances_df.columns = columns
print(error_distances_df.describe())
error_distances_df.to_csv("./results/track_error_tcn.csv", index=False)

sns.displot(data=error_distances_df, kind="kde")
plt.xlim(0, 400)
plt.gca().set_xlabel('Forcast Error (n mi)')
plt.show()
