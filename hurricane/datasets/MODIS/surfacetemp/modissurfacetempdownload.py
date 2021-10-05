import urllib.request


ORIG_URL = 'https://podaac-opendap.jpl.nasa.gov/opendap/allData/ghrsst/data/GDS2/L2P/GOES16/STAR/v2.70/2021/' + '002/' + '20210102000000-STAR-L2P_GHRSST-SSTsubskin-ABI_G16-ACSPO_V2.70-v02.0-fv01.0.nc'


def download_file(url):
    local_filename = url.split('/')[-1]
    urllib.request.urlretrieve(url, './data/MODIS/surfacetempdata/' + local_filename)
    

download_file(ORIG_URL)
