import datetime
import os
import urllib.request
import tqdm
from joblib import Parallel, delayed




EXAMPLE = "https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/AVHRR/198109/avhrr-only-v2.19810901.nc"
BASE_URL = "https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/AVHRR/"

START_DATE = datetime.datetime(year=1981, month=9, day=1)
END_DATE = datetime.datetime(year=2018, month=12, day=31)


def download_oisst_single(t, raw_data_dir):
    file_url = BASE_URL+ t.strftime("%Y%m") + "/avhrr-only-v2." + t.strftime("%Y%m%d") + ".nc"
    file_name = "avhrr-only-v2." + t.strftime("%Y%m%d") + ".nc"
    # print(file_url)
    # print(file_name)

    # print("Downloading " + file_name)
    try:
        urllib.request.urlretrieve(file_url, raw_data_dir+file_name)
    except Exception:
        print("Failed to download file: " + file_name)


def download_oisst():
    delta = END_DATE - START_DATE
    dates = []
    for i in range(delta.days + 1):
        dates.append(START_DATE + datetime.timedelta(days=i))
    raw_data_dir = "./raw_data/"
    os.makedirs(raw_data_dir, exist_ok=True)


    Parallel(n_jobs=-1,verbose=11)(delayed(download_oisst_single)(d, raw_data_dir) for d in dates)

    # for date in dates:
    #     download_oisst_single(date, raw_data_dir)
        



if __name__ == "__main__":
    download_oisst()