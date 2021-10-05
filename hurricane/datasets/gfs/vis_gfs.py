import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib
from skimage import measure
import geojsoncontour
import geojson
from geojson import Polygon, Feature, FeatureCollection



import netCDF4


# get the path of the file. It can be found in the repo data directory.
fname = "../../data/gfs/netcdf_data/gfsanl_3_20170904_0600_000.nc"

dataset = netCDF4.Dataset(fname)
dataset.set_auto_scale(True)
dataset.set_auto_mask(False)
# print(dataset)

lats = dataset.variables['lat'][:]
lons = dataset.variables['lon'][:]
# print(lons)
# for i in range(len(lons)):
#     if lons[i]>180:
#         lons[i] = 180 - lons[i]
# print(lons)



X = dataset.variables['Cloud_mixing_ratio_isobaric'][0, :, :, :]

X = np.moveaxis(X, 0, -1)
slices = X.shape[-1]
# print(slices)
# print(np.min(X))
# print(np.max(X))

contourf = plt.contourf(lons,lats,X[:,:,5])
geojsoncontour.contourf_to_geojson(
    contourf=contourf,
    geojson_filepath='data.geojson',
    ndigits=6
)