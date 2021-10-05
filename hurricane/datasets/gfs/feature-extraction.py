import matplotlib.pyplot as plt
import netCDF4
import numpy as np
import pandas as pd
from collections import OrderedDict
import dateutil.parser
import glob
import csv
import os
import csv
import glob





def extract_scan_features(filename):

    feature_data = OrderedDict()

    f = netCDF4.Dataset(filename, 'r')
    f.set_auto_maskandscale(True)
    

    min_lat = 1
    max_lat = 5
    min_lon = 1
    max_lon = 5
    
    # this function gets the first sigma layer only
    def get_relative_humidity_sigma_layer(min_lat=0, max_lat=180, min_lon=0, max_lon=359):
        v_comp = f.variables['Relative_humidity_sigma_layer'][0][0]
        return v_comp[min_lat:max_lat + 1,:][:,min_lon:max_lon + 1]

    
    def get_v_component_of_wind_sigma(min_lat=0, max_lat=180, min_lon=0, max_lon=359):
        v_comp = f.variables['v-component_of_wind_sigma'][0][0]
        return v_comp[min_lat:max_lat + 1,:][:,min_lon:max_lon + 1]

    def get_u_component_of_wind_sigma(min_lat=0, max_lat=180, min_lon=0, max_lon=359):
        v_comp = f.variables['u-component_of_wind_sigma'][0][0]
        return v_comp[min_lat:max_lat + 1,:][:,min_lon:max_lon + 1]

    def get_time():
        hours_since = f.variables['time'][0]
        time = hours_since
        return time



    feature_data['u-component_of_wind_sigma'] = get_u_component_of_wind_sigma(min_lat,max_lat,min_lon,max_lon)
    feature_data['v-component_of_wind_sigma'] = get_v_component_of_wind_sigma(min_lat,max_lat,min_lon,max_lon)
    feature_data['Relative_humidity_sigma_layer'] = get_relative_humidity_sigma_layer(min_lat,max_lat,min_lon,max_lon)
    feature_data['time'] = get_time()


    return feature_data





if __name__ == "__main__":


    myOrderedDict = extract_scan_features('netcdf.nc')
    print(myOrderedDict['time'])







