import os
import tqdm
from joblib import Parallel, delayed
import requests 
from lxml import etree
import urllib.request


DOWNLOAD_URL = "https://podaac-opendap.jpl.nasa.gov/opendap/allData/cygnss/L3/v3.0/"

def download_cygnss(data_dir, year=2019):
    print(f"Downloading CYGNSS data: {year}")
    os.makedirs(data_dir+"cygnss/raw_data/", exist_ok=True)
    files_to_download = []
    request_year_xml = requests.get(DOWNLOAD_URL+f"{year}/catalog.xml")
    # root_year = ET.fromstring(r_year.text)
    year_xml = etree.fromstring(request_year_xml.text.encode())
    print(year_xml)
    for day_element in year_xml.findall(r'.//{*}catalogRef'):
        day = day_element.attrib['name']
        # print(day)
        request_day_xml = requests.get(DOWNLOAD_URL+f"{year}/{day}/catalog.xml")
        day_xml = etree.fromstring(request_day_xml.text.encode())
        nc_file_name = None
        for e_file in day_xml.findall(".//{*}dataset"):
            if e_file.attrib['name'].split('.')[-1] == 'nc':
                nc_file_name = e_file.attrib['name']
        full_download_path = DOWNLOAD_URL+f"{year}/{day}/" + nc_file_name
        files_to_download.append((full_download_path, data_dir+"cygnss/raw_data/"+nc_file_name))
        print(nc_file_name)
    download_single = lambda r, l: urllib.request.urlretrieve(r, l)
    Parallel(n_jobs=-1,verbose=11)(delayed(download_single)(r, l) for r, l in files_to_download)



if __name__ == "__main__":
    DATA_DIR = "../../data/"
    download_cygnss(DATA_DIR, year=2019)
    download_cygnss(DATA_DIR, year=2020)