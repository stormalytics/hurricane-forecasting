import urllib.request
import os
from tqdm import tqdm


BASE_URL = r"https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2019-052520.txt"
FILE_NAME = r"hurdat2.txt"



class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_hurdat(data_dir):
    print("Downloading " + FILE_NAME)
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1) as t:
            os.makedirs(data_dir+"hurdat/", exist_ok=True) 
            urllib.request.urlretrieve(BASE_URL, data_dir+"hurdat/"+FILE_NAME, reporthook=t.update_to)
    except Exception:
        print("Failed to download file")


if __name__ == "__main__":
    DATA_DIR = "../../data/"
    download_hurdat(DATA_DIR)