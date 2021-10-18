from typing import AsyncIterable, List
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from pathlib import Path
from pprint import pprint as pp

import torch
from torch import nn
from torch.utils.data import Dataset, random_split
from torch.utils.data.dataloader import DataLoader
from torch.optim import Adam


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

    def generate_all_ts_samples(self) -> list[dict]:
        all_samples = []
        for atcf_code, hurricane_df in tqdm.tqdm(self.hurdat_table.groupby(self.grouping_var, sort=False)):
            storm_samples = self.generate_storm_ts_samples(hurricane_df)
            all_samples.extend(storm_samples)
        return all_samples

    def generate_storm_ts_samples(self, storm_df) -> list[dict]:
        data = []
        for w in storm_df.rolling(window=self.past_horizon+self.future_horizon):
            if w.shape[0] == self.past_horizon+self.future_horizon:
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


class MLP(nn.Module):
    def __init__(self, input_vars_len: int, target_vars_len: int, past_horizon: int = 1, future_horizon: int = 1, hidden_size=256):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(past_horizon*input_vars_len, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
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

    alpha = torch.sin((lat_pred-lat_actual)/2)**2 + torch.cos(lat_pred)*torch.cos(lat_actual)*torch.sin((lon_pred-lon_actual)/2)**2
    d = 2*R*torch.arcsin(torch.sqrt(alpha))

    return d


def path_distance_error(pred, actual):
    distances = haversine(pred, actual)

    sum_distance = torch.sum(distances, dim=-1)
    average_error = torch.mean(sum_distance)

    return average_error


########################
### DEMO STARTS HERE ###
########################

data = pd.read_csv('./data/hurdat/hurdat2_processed.csv')

# data = data[data['year'] == 2015]

input_vars = ["longitude", "latitude", "min_pressure", "max_sus_wind", "landfall", "hour",
              "jday", "time_idx", "delta_distance", "delta_distance_x", "delta_distance_y", "azimuth"]
output_vars = ["longitude", "latitude"]


past_horizon=8
future_horizon=4


hurdat_dataset = HURDAT(data, input_vars=input_vars, target_vars=output_vars,
                        grouping_var="atcf_code", time_idx="time_idx", past_horizon=past_horizon, future_horizon=future_horizon)

test_percent = 0.3
test_count = int(len(hurdat_dataset) * 0.3)
train_count = len(hurdat_dataset) - test_count

print(f"# of training samples: {train_count}")
print(f"# of testing samples: {test_count}")

train_data, test_data = random_split(
    hurdat_dataset, [train_count, test_count], generator=torch.Generator().manual_seed(42))

batch_size = 64
train_dataloader = DataLoader(train_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)
test_dataloader_viewing = DataLoader(test_data, batch_size=1)


model = MLP(past_horizon=past_horizon, future_horizon=future_horizon, input_vars_len=12, target_vars_len=2)
print(model)

loss_fn = nn.L1Loss()
optimizer = Adam(model.parameters(), lr=1e-3)


def train_loop(dataloader):
    model.train()
    size = len(dataloader.dataset)
    for batch_idx, data in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(data["input"])
        loss = loss_fn(pred, data["output"])
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 50 == 0:
            loss, current = loss.item(), batch_idx * len(data["input"])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for data in dataloader:
            pred = model(data["input"])
            test_loss += loss_fn(pred, data["output"]).item()

    test_loss /= num_batches
    print(f"Test Loss: {test_loss:>8f} \n")


epochs = 100
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader)
    test_loop(test_dataloader)
print("Done!")


error_distances_list = []
for sample in hurdat_dataset:
    pred = model(sample['input'].unsqueeze(0))[0]
    actual = sample['output']
    error_dist = haversine(pred, actual, batch=False)
    error_distances_list.append(error_dist)

error_distances = torch.vstack(error_distances_list)

error_distances_df = pd.DataFrame(error_distances.detach().numpy().astype("float"))
columns = [f'future_horizon_{(i+1)*6}' for i in range(future_horizon)]
error_distances_df.columns = columns

sns.boxplot(data=error_distances_df)
plt.show()
