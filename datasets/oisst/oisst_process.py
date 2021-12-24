import glob
from netCDF4 import Dataset
import numpy as np
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import datetime
from pprint import pprint as pp
import os
from joblib import Parallel, delayed
import tqdm


DATA_DIR = "./data/"
RAW_DATA_DIR = DATA_DIR + "oisst/raw_data/"
PROCESSED_DATA_DIR = DATA_DIR + "oisst/processed_data/"

file_paths = glob.glob(RAW_DATA_DIR + "*.nc")
file_paths = sorted(file_paths)

file_names = [Path(fp).name for fp in file_paths]


def extract_dt(file_name):
    year = int(file_name.split(".")[1][0:4])
    month = int(file_name.split(".")[1][4:6])
    day = int(file_name.split(".")[1][6:8])
    dt = datetime.datetime(year=year, month=month, day=day)
    return dt


datetimes = list(map(extract_dt, file_names))

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
processed_file_paths = []
processed_file_names = []


def process_single_file(fp, fn):
    # print(fn)
    nc = Dataset(fp, "r")

    sst = nc["sst"][0, 0].filled(0)
    anom = nc["anom"][0, 0].filled(0)
    err = nc["err"][0, 0].filled(0)

    mask = np.ma.getmaskarray(nc["sst"][0, 0]).astype(int)

    layers = [sst, anom, err, mask]
    layers_stacked = np.stack(layers, axis=-1)


    layers_cropped = layers_stacked[381:541, 1041:1381, :]

    fn_processed = fn.replace(".nc", ".npy")
    fp_processed = PROCESSED_DATA_DIR + fn_processed
    np.save(fp_processed, layers_cropped)

    # processed_file_paths.append(fp_processed)
    # processed_file_names.append(fn_processed)

    nc.close()


Parallel(n_jobs=-1, verbose=0)(delayed(process_single_file)(fp, fn)
                               for fp, fn in tqdm.tqdm(zip(file_paths, file_names), total=len(file_paths)))

for fp, fn in zip(file_paths, file_names):
    fn_processed = fn.replace(".nc", ".npy")
    fp_processed = PROCESSED_DATA_DIR + fn_processed
    processed_file_paths.append(fp_processed)
    processed_file_names.append(fn_processed)

inital_data = {"file_path": file_paths, "file_name": file_names, "datetime": datetimes,
               "processed_file_path": processed_file_paths, "processed_file_name": processed_file_names}

df = pd.DataFrame(inital_data)
print(df)
df.to_csv(DATA_DIR + "oisst/index.csv", index=False)
