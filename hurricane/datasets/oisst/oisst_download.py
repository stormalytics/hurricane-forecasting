import os
import tqdm
from joblib import Parallel, delayed
import requests 
from lxml import etree
import urllib.request
from pprint import pprint


CATALOG_URL = "https://www.ncei.noaa.gov/thredds/catalog/OisstBase/NetCDF/V2.1/AVHRR/"
DOWNLOAD_URL = "https://www.ncei.noaa.gov/thredds/fileServer/OisstBase/NetCDF/V2.1/AVHRR/"

def download_cygnss(data_dir, year=2019):
    print(f"Downloading OISST data: {year}")
    os.makedirs(data_dir+"oisst/raw_data/", exist_ok=True)

    files_to_download = []
    request_year_month_xml = requests.get(CATALOG_URL+f"/catalog.xml")
    # print(request_year_month_xml.text.encode())
    year_month_xml = etree.fromstring(request_year_month_xml.text.encode())
    # print(year_month_xml)
    for year_month_element in year_month_xml.findall(r'.//{*}catalogRef'):
        year_month = year_month_element.attrib['{http://www.w3.org/1999/xlink}title']
        if int(year_month[:4])== year:
            # print(year_month)
            request_day_xml = requests.get(CATALOG_URL+f"{year_month}/catalog.xml")
            day_xml = etree.fromstring(request_day_xml.text.encode())
            for e_file in day_xml.findall(".//{*}dataset"):
                if e_file.attrib['name'].split('.')[-1] == 'nc':
                    nc_file_name = e_file.attrib['name']
                    full_download_path = DOWNLOAD_URL+f"{year_month}/" + nc_file_name
                    files_to_download.append((full_download_path, data_dir+"oisst/raw_data/"+nc_file_name))
    pprint(files_to_download)
    download_single = lambda r, l: urllib.request.urlretrieve(r, l)
    Parallel(n_jobs=-1,verbose=11)(delayed(download_single)(r, l) for r, l in files_to_download)

if __name__ == "__main__":
    DATA_DIR = "../../data/"
    download_cygnss(DATA_DIR)