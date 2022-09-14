import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm
import networkx as nx

from pathlib import Path
from typing import Union
from pprint import pprint as pp

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam

import torch_geometric as pyg


class HurricaneGraph(pyg.data.Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "y":
            return None
        else:
            return super().__cat_dim__(key, value, *args, **kwargs)


class HURDATGraph(Dataset):
    def __init__(
        self,
        hurdat_table: Union[pd.DataFrame, str],
        input_vars: list[str],
        target_vars: list[str],
        grouping_var: str,
        time_idx: str,
        past_horizon: int = 1,
        future_horizon: int = 1,
    ):

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

        self.generated_graphs = self.generate_all_graph_samples(self.generated_samples)

    def generate_all_ts_samples(self) -> list[dict]:
        all_samples = []
        for atcf_code, hurricane_df in tqdm.tqdm(
            self.hurdat_table.groupby(self.grouping_var, sort=False)
        ):
            storm_samples = self.generate_storm_ts_samples(hurricane_df)
            all_samples.extend(storm_samples)
        return all_samples

    def generate_storm_ts_samples(self, storm_df: pd.DataFrame) -> list[dict]:
        data = []
        for w in storm_df.rolling(window=self.past_horizon + self.future_horizon):
            if w.shape[0] == self.past_horizon + self.future_horizon:
                data_window = {}

                data_window["input"] = torch.tensor(
                    w.head(self.past_horizon)[self.input_vars].values, dtype=torch.float
                )
                data_window["input_time_idx"] = torch.tensor(
                    w.head(self.past_horizon)[self.time_idx].values, dtype=torch.float
                )

                data_window["output"] = torch.tensor(
                    w.tail(self.future_horizon)[self.target_vars].values,
                    dtype=torch.float,
                )
                data_window["output_time_idx"] = torch.tensor(
                    w.tail(self.future_horizon)[self.time_idx].values, dtype=torch.float
                )

                data_window["window_time_idx"] = torch.tensor(
                    w[self.time_idx].values, dtype=torch.float
                )
                data_window["atcf_code"] = w["atcf_code"].iloc[0]

                data.append(data_window)
        return data

    def generate_all_graph_samples(
        self, ts_samples: list[dict[str, torch.Tensor]]
    ) -> list[HurricaneGraph]:
        all_graph_samples: list[pyg.data.Data] = []
        for ts_sample in tqdm.tqdm(ts_samples):
            graph_sample = self.ts_sample_to_graph_sample(ts_sample)
            all_graph_samples.append(graph_sample)
        return all_graph_samples

    def ts_sample_to_graph_sample(
        self, ts_sample: dict[str, torch.Tensor]
    ) -> HurricaneGraph:
        input_data = ts_sample["input"]
        output_data = ts_sample["output"]
        input_time_idx = ts_sample["input_time_idx"]
        output_time_idx = ts_sample["output_time_idx"]
        window_time_idx = ts_sample["window_time_idx"]
        atcf_code = ts_sample["atcf_code"]

        # input_graph = nx.complete_graph(input_data.shape[0], create_using=nx.DiGraph)
        input_graph = nx.circulant_graph(input_data.shape[0], [1, 2, 3], create_using=nx.DiGraph)
        input_graph_coo = torch.tensor(list(input_graph.edges), dtype=torch.long).T

        input_graph_pyg = HurricaneGraph(
            x=input_data,
            edge_index=input_graph_coo,
            y=output_data,
            input_time_idx=input_time_idx,
            output_time_idx=output_time_idx,
            window_time_idx=window_time_idx,
            atcf_code=atcf_code,
        )
        # pp(input_graph_pyg)
        # input("...")

        return input_graph_pyg

    def __len__(self):
        return len(self.generated_graphs)

    def __getitem__(self, idx):
        sample = self.generated_graphs[idx]
        # input_data = sample["input"]
        # output_data = sample["output"]
        return sample


########################
### DEMO STARTS HERE ###
########################


data = pd.read_csv("./data/hurdat/hurdat2_processed.csv")

# data = data[data["year"] == 2015]
data = data[data["year"] >= 2000]

input_vars = [
    "longitude",
    "latitude",
    "min_pressure",
    "max_sus_wind",
    "landfall",
    "hour",
    "jday",
    "time_idx",
    "delta_distance",
    "delta_distance_x",
    "delta_distance_y",
    "azimuth",
    "x",
    "y",
    "vpre",
]
output_vars = ["longitude", "latitude"]


past_horizon = 18
future_horizon = 12


hurdat_dataset = HURDATGraph(
    data,
    input_vars=input_vars,
    target_vars=output_vars,
    grouping_var="atcf_code",
    time_idx="time_idx",
    past_horizon=past_horizon,
    future_horizon=future_horizon,
)

print(hurdat_dataset[0])

test_percent = 0.3
test_count = int(len(hurdat_dataset) * 0.3)
train_count = len(hurdat_dataset) - test_count

print(f"# of training samples: {train_count}")
print(f"# of testing samples: {test_count}")

train_data, test_data = random_split(
    hurdat_dataset,
    [train_count, test_count],
    generator=torch.Generator().manual_seed(42),
)

batch_size = 32
train_dataloader = pyg.loader.DataLoader(train_data, batch_size=batch_size)
test_dataloader = pyg.loader.DataLoader(test_data, batch_size=batch_size)
test_dataloader_viewing = pyg.loader.DataLoader(test_data, batch_size=1)

# for step, data in enumerate(train_dataloader):
#     print(f"Step {step + 1}:")
#     print("=======")
#     print(f"Number of graphs in the current batch: {data.num_graphs}")
#     print(data)
#     print()


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

    alpha = (
        torch.sin((lat_pred - lat_actual) / 2) ** 2
        + torch.cos(lat_pred)
        * torch.cos(lat_actual)
        * torch.sin((lon_pred - lon_actual) / 2) ** 2
    )
    d = 2 * R * torch.arcsin(torch.sqrt(alpha))

    d = d * 0.539957

    d = d.unsqueeze(-1)
    return d


def path_distance_error_location(pred, actual):
    distances = haversine(pred, actual)
    sum_distance = torch.sum(distances, dim=-1)
    average_error = torch.mean(sum_distance)
    return average_error


class MLP(nn.Module):
    def __init__(self, input_size=16, output_size=16, hidden_size=64, n_layers=4):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        for i in range(n_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            elif i == n_layers - 1:
                self.layers.append(nn.Linear(hidden_size, output_size))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, x):
        for i in range(self.n_layers):
            x = F.relu(self.layers[i](x))
        return x


class GraphModel(nn.Module):
    def __init__(
        self,
        input_vars_len: int,
        target_vars_len: int,
        past_horizon: int = 1,
        future_horizon: int = 1,
        hidden_size=32,
    ):
        super(GraphModel, self).__init__()

        self.input_vars_len = input_vars_len
        self.target_vars_len = target_vars_len
        self.past_horizon = past_horizon
        self.future_horizon = future_horizon

        self.hidden_size = hidden_size

        self.conv1 = pyg.nn.GATConv(self.input_vars_len, self.hidden_size)
        self.conv2 = pyg.nn.GATConv(self.hidden_size, self.hidden_size)
        self.conv3 = pyg.nn.GATConv(self.hidden_size, self.hidden_size)

        self.lin_reduce = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.mlp = MLP(
            input_size=self.hidden_size,
            output_size=self.hidden_size,
            hidden_size=self.hidden_size,
            n_layers=2,
        )

        self.lin = nn.Linear(hidden_size, target_vars_len * future_horizon)
        self.output_reshape = nn.Unflatten(
            1, torch.Size([future_horizon, target_vars_len])
        )

    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = F.gelu(x)
        x = self.conv2(x, edge_index)
        x = F.gelu(x)
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x_mean = pyg.nn.global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x_max = pyg.nn.global_max_pool(x, batch)  # [batch_size, hidden_channels]

        x = torch.cat([x_mean, x_max], dim=1)
        x = self.lin_reduce(x)


        # 3. Apply a final classifier
        # x = F.dropout(x, p=0.2, training=self.training)
        x = self.mlp(x)
        x = self.lin(x)
        # print(x.shape)
        x = self.output_reshape(x)
        # print(x.shape)

        return x


model = GraphModel(
    input_vars_len=len(input_vars),
    target_vars_len=len(output_vars),
    past_horizon=past_horizon,
    future_horizon=future_horizon,
    hidden_size=64,
)
print(model)


optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
loss_fn = nn.L1Loss()


def train(loader):
    model.train()

    loss_total = 0
    error_distance = 0

    for data in loader:  # Iterate in batches over the training dataset.
        out = model(
            data.x, data.edge_index, data.batch
        )  # Perform a single forward pass.
        loss = loss_fn(out, data.y)  # Compute the loss.
        loss_total += loss.item()  # Add the loss to the total.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        error_distance += haversine(out, data.y, batch=True).sum().detach().item()

    loss_avg = loss_total / len(loader)
    error_distance_avg = error_distance / len(loader)
    return loss_avg, error_distance_avg


def test(loader):
    model.eval()

    loss_total = 0
    error_distance = 0

    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        loss = loss_fn(out, data.y)  # Compute the loss.
        loss_total += loss.item()  # Add the loss to the total.
        error_distance += haversine(out, data.y, batch=True).sum().detach().item()

    loss_avg = loss_total / len(loader)  # Compute the average loss.
    error_distance_avg = error_distance / len(loader)
    return loss_avg, error_distance_avg


for epoch in range(1, 50):
    train_loss, train_error_distance = train(train_dataloader)
    test_loss, test_error_distance = test(test_dataloader)
    print(f"Epoch: {epoch:03d}")
    print(f"-----------------------")
    print(f"Train loss: {train_loss:.4f}")
    print(f"Train error distance: {train_error_distance:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test error distance: {test_error_distance:.4f}")
    print()
