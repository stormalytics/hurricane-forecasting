import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tqdm

from pathlib import Path
from typing import List
from pprint import pprint as pp

from torch.utils.data import Dataset, DataLoader
import torch


from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, PolynomialFeatures

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet,\
    Lars, BayesianRidge, HuberRegressor, RANSACRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.metrics import mean_absolute_error, make_scorer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split

import xgboost as xgb


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


########################
### DEMO STARTS HERE ###
########################


data = pd.read_csv('./data/hurdat/hurdat2_processed.csv')

# data = data[data['year'] == 2015]

input_vars = ["longitude", "latitude", "min_pressure", "max_sus_wind", "landfall", "hour",
              "jday", "time_idx", "delta_distance", "delta_distance_x", "delta_distance_y", "azimuth",
              "x", "y", "vpre"]
output_vars = ["longitude", "latitude"]


past_horizon = 12
future_horizon = 12
hurdat_dataset = HURDAT(data, input_vars=input_vars, target_vars=output_vars,
                        grouping_var="atcf_code", time_idx="time_idx",
                        past_horizon=past_horizon, future_horizon=future_horizon)


hurdat_dataset_dataloader = DataLoader(hurdat_dataset, batch_size=len(hurdat_dataset))

data = next(iter(hurdat_dataset_dataloader))

x = data["input"].cpu().detach().numpy()
y = data["output"].cpu().detach().numpy()


print(x.shape)
print(y.shape)

x = x.reshape((-1, x.shape[-1]*x.shape[-2]))
y = y.reshape((-1, y.shape[-1]*y.shape[-2]))

print(x.shape)
print(y.shape)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


def haversine(pred, actual, batch=True):
    R = 6371

    pred = pred.reshape((-1, future_horizon, 2))
    actual = actual.reshape((-1, future_horizon, 2))

    if batch:
        lon_actual, lon_pred = actual[:, :, 0], pred[:, :, 0]
        lat_actual, lat_pred = actual[:, :, 1], pred[:, :, 1]
    else:
        lon_actual, lon_pred = actual[:, 0], pred[:, 0]
        lat_actual, lat_pred = actual[:, 1], pred[:, 1]

    lon_actual, lon_pred = np.deg2rad(lon_actual), np.deg2rad(lon_pred)
    lat_actual, lat_pred = np.deg2rad(lat_actual), np.deg2rad(lat_pred)

    alpha = np.sin((lat_pred-lat_actual)/2)**2 + np.cos(lat_pred) * \
        np.cos(lat_actual)*np.sin((lon_pred-lon_actual)/2)**2
    d = 2*R*np.arcsin(np.sqrt(alpha))

    d = d*0.539957

    return d


def haversine_loss(pred, actual, batch=True):

    R = 6371

    pred = pred.reshape((-1, future_horizon, 2))
    actual = actual.reshape((-1, future_horizon, 2))

    if batch:
        lon_actual, lon_pred = actual[:, :, 0], pred[:, :, 0]
        lat_actual, lat_pred = actual[:, :, 1], pred[:, :, 1]
    else:
        lon_actual, lon_pred = actual[:, 0], pred[:, 0]
        lat_actual, lat_pred = actual[:, 1], pred[:, 1]

    lon_actual, lon_pred = np.deg2rad(lon_actual), np.deg2rad(lon_pred)
    lat_actual, lat_pred = np.deg2rad(lat_actual), np.deg2rad(lat_pred)

    alpha = np.sin((lat_pred-lat_actual)/2)**2 + np.cos(lat_pred) * \
        np.cos(lat_actual)*np.sin((lon_pred-lon_actual)/2)**2
    d = 2*R*np.arcsin(np.sqrt(alpha))

    d = d*0.539957

    d_path_sum = np.sum(d, axis=1)
    d_avg = np.mean(d_path_sum)
    # d_avg = np.mean(d)

    return d_avg


### GRID SEARCH EXPERIMENT

# pipe = Pipeline([
#     ('scale', RobustScaler()),
#     ('reg',  MultiOutputRegressor(LinearRegression()))
# ])

# param_grid = [{'reg__estimator': [SVR(tol=1e-3)],
#                'reg__estimator__C': [1, 10, 100],
#                'reg__estimator__gamma': [1, 0.1, 0.001],
#                'reg__estimator__degree':[2, 3, 4],
#                'reg__estimator__kernel': ['linear', 'rbf', 'poly']}]

# param_grid = [{'reg__estimator': [LinearSVR(max_iter=5000)],
#                'reg__estimator__C': [1, 10, 100]}]

# param_grid = [{'reg__estimator': [LinearRegression()]},
#               {'reg__estimator': [Ridge()],
#                'reg__estimator__alpha': [0.1, 1.0, 10.0]},
#               {'reg__estimator': [Lasso(max_iter=10000)],
#                'reg__estimator__alpha': [0.1, 1.0, 10.0]},
#               {'reg__estimator': [ElasticNet(max_iter=10000)],
#                'reg__estimator__alpha': [0.1, 1.0, 10.0],
#                'reg__estimator__l1_ratio': [0.2, 0.4, 0.6, 0.8]}]

# param_grid = [{'reg__estimator': [DecisionTreeRegressor()],
#                'reg__estimator__min_samples_split': [2, 5, 10],
#                'reg__estimator__min_samples_leaf': [2, 3, 4, 5, 6, 7, 8]}]

# param_grid = [{'reg__estimator': [GradientBoostingRegressor()],
#                'reg__estimator__loss': ['quantile'],
#                'reg__estimator__learning_rate': [1, 0.1, 0.01, 0.001],
#                'reg__estimator__n_estimators': [10, 20, 50, 100, 200, 500],
#                'reg__estimator__criterion': ['squared_error', 'absolute_error'],
#                'reg__estimator__max_depth':[3, 4, 5, 6, 7, 8, 9, 10]},
#               ]



# search = GridSearchCV(pipe, param_grid, cv=5, n_jobs=11, verbose=2,
#                       scoring=make_scorer(haversine_loss, greater_is_better=False))
# print(search.best_params_)

# search_results_df = pd.DataFrame({'param': search.cv_results_["params"], 'acc': search.cv_results_["mean_test_score"]})
# search_results_df.to_csv("grid_search_results.csv", index=False)


# LINEAR MODEL TEST

# linear_model = Pipeline([
#     ('scale', RobustScaler()),
#     ('reg',  MultiOutputRegressor(Ridge(alpha=1.0)))
# ])

# linear_model.fit(x_train, y_train)

# y_pred = linear_model.predict(x_test)
# error_distances = haversine(y_pred, y_test)

# error_distances_df = pd.DataFrame(error_distances)
# columns = [f'future_horizon_{(i+1)*6}' for i in range(future_horizon)]
# error_distances_df.columns = columns
# print(error_distances_df.describe())
# error_distances_df.to_csv("./results/track_error_ridge.csv", index=False)

# sns.displot(data=error_distances_df, kind="kde")
# plt.xlim(0, 400)
# plt.gca().set_xlabel('Forcast Error (n mi)')
# plt.show()


# DECISION TREE TEST

# decision_tree_model = Pipeline([
#     ('scale', RobustScaler()),
#     ('reg',  MultiOutputRegressor(DecisionTreeRegressor(min_samples_leaf=8, min_samples_split=10)))
# ])

# decision_tree_model.fit(x_train, y_train)

# y_pred = decision_tree_model.predict(x_test)
# error_distances = haversine(y_pred, y_test)

# error_distances_df = pd.DataFrame(error_distances)
# columns = [f'future_horizon_{(i+1)*6}' for i in range(future_horizon)]
# error_distances_df.columns = columns
# print(error_distances_df.describe())
# error_distances_df.to_csv("./results/track_error_decision_tree.csv", index=False)

# sns.displot(data=error_distances_df, kind="kde")
# plt.xlim(0, 400)
# plt.gca().set_xlabel('Forcast Error (n mi)')
# plt.show()


# BOOSTING TREE

# reg = MultiOutputRegressor(xgb.XGBRegressor(verbosity=2, n_jobs=12, tree_method="hist"))
# reg.fit(x_train, y_train)

# y_pred = reg.predict(x_test)
# print(y_pred.shape)

# error_distances = haversine(y_pred, y_test)
# print(error_distances.shape)

# error_distances_df = pd.DataFrame(error_distances)
# columns = [f'future_horizon_{(i+1)*6}' for i in range(future_horizon)]
# error_distances_df.columns = columns
# print(error_distances_df.describe())
# error_distances_df.to_csv("./results/track_error_xgboost.csv", index=False)


# sns.displot(data=error_distances_df, kind="kde")
# plt.xlim(0, 400)
# plt.gca().set_xlabel('Forcast Error (n mi)')
# plt.show()
