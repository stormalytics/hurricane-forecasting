import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pprint import pprint

file_paths = glob.glob("../../data/cygnss/raw_data/*.nc")
file_paths = sorted(file_paths)





for file_path in file_paths:
    nc = Dataset(file_path,'r')
    mss = nc['mean_square_slope'][:].max(axis = 0)
    mss_u = nc['mean_square_slope_uncertainty'][:].max(axis = 0)
    mss_s = nc['num_mss_samples'][:].max(axis = 0)
    # ws = nc['wind_speed'][:].max(axis = 0)
    # ws_u = nc['wind_speed_uncertainty'][:].max(axis = 0)
    # ws_s = nc['num_wind_speed_samples'][:].max(axis = 0)
    # ws_yslf = nc['yslf_wind_speed'][:].max(axis = 0)
    # ws_yslf_u = nc['yslf_wind_speed_uncertainty'][:].max(axis = 0)
    # ws_yslf_s = nc['num_yslf_wind_speed_samples'][:].max(axis = 0)
    # print(data.shape)
    fig, ax= plt.subplots(1,3)
    ax[0].pcolormesh(mss[251:400, 1300:1725])
    ax[1].pcolormesh(mss_u[251:400, 1300:1725])
    ax[2].pcolormesh(mss_s[251:400, 1300:1725])
    # ax[3].pcolormesh(ws)
    # ax[4].pcolormesh(ws_u)
    # ax[5].pcolormesh(ws_s)
    # ax[6].pcolormesh(ws_yslf)
    # ax[7].pcolormesh(ws_yslf_u)
    # ax[8].pcolormesh(ws_yslf_s)
    plt.show()
    input("...")