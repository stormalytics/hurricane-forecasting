import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import datetime
from pprint import pprint
import os

file_paths = glob.glob("../../data/cygnss/raw_data/*.nc")
file_paths = sorted(file_paths)

file_names = [Path(fp).name for fp in file_paths]


def extract_dt(file_name):
    year = int(file_name.split(".")[2][1:5])
    month = int(file_name.split(".")[2][5:7])
    day = int(file_name.split(".")[2][7:9])
    dt = datetime.datetime(year=year, month=month, day=day)
    return dt


datetimes = list(map(extract_dt, file_names))


# Latitude indexes and values
# 251 -> 10.1
# 400 (last) -> 39.9

# Longitude indexes and values
# 1300 -> 259.9 (-100.1)
# 1725 -> 344.9 (-15.1)

os.makedirs("../../data/cygnss/processed_data/", exist_ok=True)
processed_file_paths = []
processed_file_names = []


for fp, fn in zip(file_paths, file_names):
    print(fn)
    nc = Dataset(fp, "r")
    mss_min = nc["mean_square_slope"][:].filled(0).min(axis=0)
    mss_max = nc["mean_square_slope"][:].filled(0).max(axis=0)
    mss_mean = nc["mean_square_slope"][:].filled(0).mean(axis=0)
    mss = [mss_min, mss_max, mss_mean]

    mss_u_min = nc["mean_square_slope_uncertainty"][:].filled(0).min(axis=0)
    mss_u_max = nc["mean_square_slope_uncertainty"][:].filled(0).max(axis=0)
    mss_u_mean = nc["mean_square_slope_uncertainty"][:].filled(0).mean(axis=0)
    mss_u = [mss_u_min, mss_u_max, mss_u_mean]

    mss_s_sum = nc["num_mss_samples"][:].filled(0).sum(axis=0)
    mss_s = [mss_s_sum]

    ws_min = nc["wind_speed"][:].filled(0).min(axis=0)
    ws_max = nc["wind_speed"][:].filled(0).max(axis=0)
    ws_mean = nc["wind_speed"][:].filled(0).mean(axis=0)
    ws = [ws_min, ws_max, ws_mean]

    ws_u_min = nc["wind_speed_uncertainty"][:].filled(0).min(axis=0)
    ws_u_max = nc["wind_speed_uncertainty"][:].filled(0).max(axis=0)
    ws_u_mean = nc["wind_speed_uncertainty"][:].filled(0).mean(axis=0)
    ws_u = [ws_u_min, ws_u_max, ws_u_mean]

    ws_s_sum = nc["num_wind_speed_samples"][:].filled(0).sum(axis=0)
    ws_s = [ws_s_sum]

    ws_yslf_min = nc["yslf_wind_speed"][:].filled(0).min(axis=0)
    ws_yslf_max = nc["yslf_wind_speed"][:].filled(0).max(axis=0)
    ws_yslf_mean = nc["yslf_wind_speed"][:].filled(0).mean(axis=0)
    ws_yslf = [ws_yslf_min, ws_yslf_max, ws_yslf_mean]

    ws_yslf_u_min = nc["yslf_wind_speed_uncertainty"][:].filled(0).min(axis=0)
    ws_yslf_u_max = nc["yslf_wind_speed_uncertainty"][:].filled(0).max(axis=0)
    ws_yslf_u_mean = nc["yslf_wind_speed_uncertainty"][:].filled(0).mean(axis=0)
    ws_yslf_u = [ws_yslf_u_min, ws_yslf_u_max, ws_yslf_u_mean]

    ws_yslf_s_sum = nc["num_yslf_wind_speed_samples"][:].filled(0).sum(axis=0)
    ws_yslf_s = [ws_yslf_s_sum]

    layers = mss + mss_u + mss_s + ws + ws_u + ws_s + ws_yslf + ws_yslf_u + ws_yslf_s
    layers_stacked = np.stack(layers, axis=-1)
    layers_cropped = layers_stacked[251:400, 1300:1725, :]
    # print(layers_cropped.shape)

    # n = 10

    # fig, ax= plt.subplots(1,n)
    # for i in range(n):
    #     ax[i].pcolormesh(layers_cropped[:,:,i])
    #     # ax[i].pcolormesh(layers_cropped[:,:,i])
    #     # ax[i].pcolormesh(layers_cropped[:,:,i])
    # plt.show()

    fn_processed = fn.replace(".nc", ".npy")
    fp_processed = "../../data/cygnss/processed_data/" + fn_processed
    np.save(fp_processed, layers_cropped)

    processed_file_paths.append(fp_processed)
    processed_file_names.append(fn_processed)

    nc.close()

    # input("...")


inital_data = {"file_path": file_paths, "file_name": file_names, "datetime": datetimes, "processed_file_path": processed_file_paths, "processed_file_name": processed_file_names}

df = pd.DataFrame(inital_data)
print(df)
df.to_csv("../../data/cygnss/index.csv", index=False)