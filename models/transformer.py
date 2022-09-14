from typing import Union
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

from transformers import PerceiverModel, PerceiverConfig
from transformers.models.perceiver.modeling_perceiver import PerceiverBasicDecoder


class HURDAT(Dataset):
    def __init__(self, hurdat_table, input_vars: list[str], target_vars: list[str], grouping_var: str, time_idx: str, past_horizon: int = 1, future_horizon: int = 1):

        self.hurdat_table: pd.DataFrame = None
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

class HurricaneForcastTransformer(nn.Module):
    def __init__(self, input_vars_len: int, target_vars_len: int, past_horizon: int, future_horizon: int, hidden_size=512) -> None:
        super().__init__()

        self.input_vars_len = input_vars_len
        self.target_vars_len = target_vars_len
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon

        # build and encoder / decoder model
        # input encoder seq length = past_horizon
        # output decoder seq length = future_horizon
        # input encoder features = input_vars_len
        # output decoder features = target_vars_len

        # the ouptu decoder input sequence is just positional encoding
        # the output decoder output sequence is the target_vars_len

        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=8,
            num_encoder_layers=3,
            num_decoder_layers=3,
            dim_feedforward=hidden_size*2,
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )

        self.encoder_input_projection = nn.Linear(input_vars_len, hidden_size)
        self.decoder_input_projection = nn.Linear(1, hidden_size)

        self.decoder_output_projection = nn.Linear(hidden_size, target_vars_len)


    def forward(self, input_data, input_idx, output_idx):
        projected_input_data = self.encoder_input_projection(input_data)
        
        output_idx_reshaped = output_idx.unsqueeze(-1)
        projected_output_idx = self.decoder_input_projection(output_idx_reshaped)

        decoder_output = self.transformer(projected_input_data, projected_output_idx)
        
        projected_decoder_output = self.decoder_output_projection(decoder_output)
        return projected_decoder_output
            
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

def path_distance_error_location(pred, actual):
    distances = haversine(pred, actual)
    sum_distance = torch.sum(distances, dim=-1)
    average_error = torch.mean(sum_distance)
    return average_error

def path_distance_error_mae(pred, actual):
    # pred.shape = (batch_size, seq_len, 2)

    diff = torch.abs(pred-actual)
    # reduce last two dim using sum
    diff_sum = torch.sum(diff, dim=(-1, -2))
    # reduce first dim using mean
    diff_mean = torch.mean(diff_sum, dim=0)
    return diff_mean

def train_loop(dataloader, loss_fn, optimizer):
    model.train()

    num_batches = len(dataloader)
    train_loss = 0
    train_error_distance = 0

    for batch_idx, data in enumerate(dataloader):
        # Compute prediction and loss
        input_data, input_idx, output_idx = data["input"].to(device), data["input_time_idx"].to(device), data["output_time_idx"].to(device)
        output_data = data["output"].to(device)
        pred = model(input_data, input_idx, output_idx)
        loss = loss_fn(pred, output_data)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss_fn(pred, output_data).item()
        train_error_distance += haversine(pred, output_data, batch=True).mean().detach().item()

    train_loss /= num_batches
    train_error_distance /= num_batches
    print(f"Train Loss: {train_loss:>8f}")
    print(f"Train Error Distance: {train_error_distance:>8f}")
    print()


def test_loop(dataloader, loss_fn):
    model.eval()

    num_batches = len(dataloader)
    test_loss = 0
    test_error_distance = 0

    with torch.no_grad():
        for data in dataloader:
            input_data, input_idx, output_idx = data["input"].to(device), data["input_time_idx"].to(device), data["output_time_idx"].to(device)
            output_data = data["output"].to(device)
            pred = model(input_data, input_idx, output_idx)
            test_loss += loss_fn(pred, output_data).item()
            test_error_distance += haversine(pred, output_data, batch=True).mean().detach().item()

    test_loss /= num_batches
    test_error_distance /= num_batches
    print(f"Test Loss: {test_loss:>8f}")
    print(f"Test Error Distance: {test_error_distance:>8f}")
    print()

if __name__ == "__main__":
    ############
    # HW SETUP #
    ############

    device = torch.device(torch.cuda.current_device() if torch.cuda.is_available() else 'cpu')
    print(device)


    ################
    # DATA LOADING #
    ################

    data = pd.read_csv('./data/hurdat/hurdat2_processed.csv')

    # data = data[data['year'] >= 2010]


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


    ###############
    # MODEL SETUP #
    ###############

    model = HurricaneForcastTransformer(
        input_vars_len=len(input_vars),
        target_vars_len=len(output_vars),
        past_horizon=past_horizon,
        future_horizon=future_horizon
    ).to(device)

    # x_demo = next(iter(train_dataloader))
    # y_pred_demo = model(x_demo["input"].to(device), x_demo["input_time_idx"].to(device), x_demo["output_time_idx"].to(device))


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = path_distance_error_location
    # loss_fn = path_distance_error_mae

    epochs = 200
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, loss_fn, optimizer)
        test_loop(test_dataloader, loss_fn)
    print("Done!")