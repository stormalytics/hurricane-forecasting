import joblib
import urllib.request
import os
from datetime import timedelta, date
import ftplib
from joblib import Parallel, delayed
import itertools
import csv


def daterange(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)


    

def download_gfs_data(dates, data_dir):
    os.makedirs(data_dir+'gfs/grb2_data/', exist_ok=True)
    download_data = []
    for date in dates:
        for hour in ["0000","0600","1200","1800"]:
            # print(date)
            year_month = date.strftime("%Y%m")
            year_month_day = date.strftime("%Y%m%d")
            file_url = "https://nomads.ncdc.noaa.gov/data/gfsanl/"+year_month+"/"+year_month_day+"/gfsanl_3_"+year_month_day+"_"+hour+"_000.grb2"
            file_name = "gfsanl_3_"+year_month_day+"_"+hour+"_000.grb2" 
            data = {"file_url": file_url,"file_name":file_name}
            download_data.append(data)
    # for d in download_data:
    #     urllib.request.urlretrieve(d['file_url'], data_dir+'gfs/grb2_data/'+d['file_name'])
    Parallel(n_jobs=-1, verbose=11)(delayed(urllib.request.urlretrieve)(d['file_url'], data_dir+'gfs/grb2_data/'+d['file_name']) for d in download_data)
    # print(download_data)

def convert_to_net_cdf(data_dir):
    file_list = []
    os.makedirs(data_dir+'gfs/netcdf_data/', exist_ok=True)
    for file in os.listdir(data_dir+'gfs/grb2_data/'):
        if file.endswith(".grb2"):
            file_list.append(os.path.join(data_dir+'gfs/grb2_data/', file))
    # print(file_list)
    for file in file_list:
        out_file_name = os.path.basename(file)
        out_file_name = out_file_name.replace(".grb2", ".nc")
        print(out_file_name)
        os.system("java -classpath ./toolsUI-5.2.0.jar ucar.nc2.dataset.NetcdfDataset -in " + file + " -out "+ data_dir+'gfs/netcdf_data/'+out_file_name)

def index_gfs_data(data_dir):
    file_list = []
    index_data = []
    for file in os.listdir(data_dir+'gfs/netcdf_data/'):
        if file.endswith(".nc"):
            file_list.append(os.path.join(data_dir+'gfs/netcdf_data/', file))
    for file in file_list:
        data = {}
        file_name = os.path.basename(file)

        split_file_name, file_extension = file_name.split(".")
        split_file_name = split_file_name.split("_")
        # print(file_extension)
        data['data_product'] = split_file_name[0]
        data['resoultion_type'] = split_file_name[1]
        data['year'] = int(split_file_name[2][0:4])
        data['month'] = int(split_file_name[2][4:6])
        data['day'] = int(split_file_name[2][6:8])
        data['hour'] = int(split_file_name[3][0:2])
        data['minute'] = int(split_file_name[3][2:4])
        data['forcast_time_delta_hour'] = int(split_file_name[4])
        data['file_path'] = file.replace(data_dir, "")
        data['file_extension'] = file_extension
        index_data.append(data)
    # print(index_data)

    keys = index_data[0].keys()
    with open(data_dir+'gfs/data_index.csv', 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(index_data)

if __name__ == "__main__":

    data_dir = "../../data/"

    start_date = date(2020, 1, 1)
    end_date = date(2020, 1, 31)
    dates = list(daterange(start_date, end_date))

    download_gfs_data(dates, data_dir)
    convert_to_net_cdf(data_dir)
    index_gfs_data(data_dir)
