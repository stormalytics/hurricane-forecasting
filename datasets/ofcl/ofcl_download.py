import urllib.request
import os
from tqdm import tqdm


BASE_URL = r"https://www.nhc.noaa.gov/verification/errors/1970-present_OFCL_v_BCD5_ind_ATL_TI_errors_noTDs.txt"
FILE_NAME = r"ofcl.txt"


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_ofcl(data_dir):
    print("Downloading " + FILE_NAME)
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
            os.makedirs(data_dir+"ofcl/", exist_ok=True)
            urllib.request.urlretrieve(BASE_URL, data_dir+"ofcl/"+FILE_NAME, reporthook=t.update_to)
            print(data_dir+"ofcl/"+FILE_NAME)
    except Exception:
        print("Failed to download file")


if __name__ == "__main__":
    DATA_DIR = "./data/"
    download_ofcl(DATA_DIR)
