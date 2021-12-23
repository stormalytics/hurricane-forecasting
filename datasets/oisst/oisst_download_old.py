import datetime
import os
import urllib.request
import tqdm
from joblib import Parallel, delayed
import pandas as pd


EXAMPLE = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/198201/oisst-avhrr-v02r01.19820101.nc"
BASE_URL = "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/"

DATA_DIR = "./data/"
HURDATE_PROCESSED_FP = DATA_DIR + "hurdat/hurdat2_processed.csv"
RAW_DATA_DIR = DATA_DIR + "oisst/raw_data/"


def download_oisst_single(t, raw_data_dir):
    file_url = BASE_URL + t.strftime("%Y%m") + "/oisst-avhrr-v02r01." + t.strftime("%Y%m%d") + ".nc"
    file_name = "oisst-avhrr-v02r01." + t.strftime("%Y%m%d") + ".nc"

    # print(file_url)
    # print(file_name)
    # print(f"Downloading: {file_name}")
    
    try:
        urllib.request.urlretrieve(file_url, raw_data_dir+file_name)
    except Exception:
        print(f"Failed to download file: {file_name}")


def download_oisst(hurdat_processed_fp, raw_data_dir):
    hurdat_processed_df = pd.read_csv(hurdat_processed_fp)
    # print(hurdat_processed_df)

    dates = []
    for index, row in hurdat_processed_df.iterrows():
        year = int(row["year"])
        month = int(row["month"])
        day = int(row["day"])
        # print(year,month,day,hour)
        date = datetime.datetime(year, month, day)
        if date not in dates:
            dates.append(date)

    raw_data_dir = "./data/oisst/raw_data/"
    os.makedirs(raw_data_dir, exist_ok=True)


    Parallel(n_jobs=-1,verbose=0)(delayed(download_oisst_single)(d, raw_data_dir) for d in tqdm.tqdm(dates))
    # for date in dates:
    #     download_oisst_single(date, raw_data_dir)
        



if __name__ == "__main__":
    download_oisst(HURDATE_PROCESSED_FP, RAW_DATA_DIR)