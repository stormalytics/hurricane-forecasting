import datetime
import os
import urllib.request
from pandas.tseries.offsets import Hour
import tqdm
from joblib import Parallel, delayed
import pandas as pd
import numpy as np

EXAMPLE = "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/1980/GRIDSAT-B1.1980.01.01.00.v02r01.nc"
BASE_URL = "https://www.ncei.noaa.gov/data/geostationary-ir-channel-brightness-temperature-gridsat-b1/access/"

DATA_DIR = "./data/"
HURDATE_PROCESSED_FP = DATA_DIR + "hurdat/hurdat2_processed.csv"
RAW_DATA_DIR = DATA_DIR + "gridsat_b1/raw_data/"


def download_gridsat_b1_single(t, raw_data_dir):
    file_url = BASE_URL + t.strftime("%Y") + "/GRIDSAT-B1." + t.strftime("%Y.%m.%d.%H") + ".v02r01.nc"
    file_name = "GRIDSAT-B1." + t.strftime("%Y.%m.%d.%H") + ".v02r01.nc"

    # print(file_url)
    # print(file_name)
    print(f"Downloading: {file_name}")

    try:
        urllib.request.urlretrieve(file_url, raw_data_dir+file_name)
    except Exception:
        print(f"Failed to download file: {file_name}")


def download_gridsat_b1(hurdat_processed_fp, raw_data_dir):
    hurdat_processed_df = pd.read_csv(hurdat_processed_fp)
    # print(hurdat_processed_df)

    dates = []
    for index, row in hurdat_processed_df.iterrows():
        year = int(row["year"])
        month = int(row["month"])
        day = int(row["day"])
        hour = int(row["hour"])
        # print(year,month,day,hour)
        dates.append(datetime.datetime(year,month,day,hour))

    raw_data_dir = "./data/gridsat_b1/raw_data/"
    os.makedirs(raw_data_dir, exist_ok=True)

    Parallel(n_jobs=-1, verbose=11)(delayed(download_gridsat_b1_single)(d, raw_data_dir) for d in dates)
    # for date in dates:
    #     download_gridsat_b1_single(date, raw_data_dir)


if __name__ == "__main__":
    download_gridsat_b1(HURDATE_PROCESSED_FP, RAW_DATA_DIR)
