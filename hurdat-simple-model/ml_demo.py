import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, HuberRegressor, BayesianRidge, Lasso, ElasticNet, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, Matern, RationalQuadratic, ExpSineSquared


from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error

from pyproj import Geod
from geopy.distance import geodesic
import sys, os

sys.path.append(os.path.abspath('../utils'))
from geo_calculations import vincenty_inverse, haversine

import matplotlib.pyplot as plt
from matplotlib import collections as mc
import seaborn as sns


import datetime


pd.set_option('display.max_columns', 500)


df = pd.read_csv('../data/hurdat/hurdat2_processed.csv')
df = df[df['year'] > 1979]
df = df[(df['system_status'] == 'HU') | (df['system_status'] == 'TS')]
df = df[df['wind_radii_34_NE'] != -999]





df = pd.concat([df,pd.get_dummies(df['system_status-12'], prefix='system_status-12')],axis=1)
df = pd.concat([df,pd.get_dummies(df['system_status-6'], prefix='system_status-6')],axis=1)
df = pd.concat([df,pd.get_dummies(df['system_status'], prefix='system_status')],axis=1)
df = pd.concat([df,pd.get_dummies(df['system_status+6'], prefix='system_status+24')],axis=1)
df = pd.concat([df,pd.get_dummies(df['system_status+24'], prefix='system_status+24')],axis=1)


for col in df.columns: 
    print(col) 

input_features = []

input_features_static = ['year', 'month', 'day', 'hour', 'day_of_year','aday']
input_features += input_features_static

input_features_time_shifted = ['longitude', 'latitude',
							   'x','y',
							   'max_sus_wind', 'min_pressure',
							   'delta_distance', 'azimuth',
							   'vpre', 'landfall']

input_features += input_features_time_shifted
for t in ['-6', '-12', '-18', '-24', '-30', '-36', '-42', '-48']:
	input_features += [s + t for s in input_features_time_shifted]



# input_features = ['year', 'month', 'day', 'hour',
# 				  'longitude', 'latitude',
# 				  'x','y',
# 				  'max_sus_wind', 'min_pressure',
# 				  'delta_distance', 'azimuth',
# 				  'longitude-6', 'latitude-6',
# 				  'x-6','y-6',
# 				  'max_sus_wind-6', 'min_pressure-6',
# 				  'delta_distance-6', 'azimuth-6',
# 				  'longitude-12', 'latitude-12',
# 				  'x-12','y-12',
# 				  'max_sus_wind-12', 'min_pressure-12',
# 				  'delta_distance-12', 'azimuth-12',
# 				  'day_of_year','aday',
# 				  'vpre','vpre-6','vpre-12',
# 				  'landfall','landfall-6','landfall-12']

for wind_speed in ['34', '50', '64']:
	for direction in ['NE', 'SE', 'SW', 'NW']:
		wind_radii_column_name = 'wind_radii_' + wind_speed + '_' + direction
		input_features.append(wind_radii_column_name)

output_features = ['latitude+24', 'longitude+24', 'max_sus_wind+24', 'min_pressure+24']

for col in df.columns: 
    print(col) 

print(df[input_features].head())

x = np.array(df[input_features].values.tolist())
y = np.array(df[output_features].values.tolist())
storm_ids = np.array(df['atcf_code'].values.tolist())

scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()

x_scaled = scaler_x.fit_transform(x)
y_scaled = scaler_y.fit_transform(y)

x_train, x_test, y_train, y_test, storm_ids_train, storm_ids_test = train_test_split(x_scaled, y_scaled, storm_ids, test_size=0.3, random_state=42)



linear = LinearRegression()
lasso = Lasso()
ridge = Ridge()
elastic = ElasticNet()
b_ridge = BayesianRidge(verbose=True)
huber = HuberRegressor()
svr = SVR(verbose=True, tol=0.001, kernel ='rbf')
gb = GradientBoostingRegressor(n_estimators = 500, max_depth = 8, verbose = 1)
mlp_reg = MLPRegressor((64,64,64,64,64,1), activation='tanh',
					   verbose=True,max_iter=500,
					   tol=0.000000001,
					   learning_rate='adaptive')

g_kernel = DotProduct() + Matern() + RationalQuadratic()
g_reg = GaussianProcessRegressor(kernel=g_kernel, random_state=0, normalize_y=True)

er = VotingRegressor([('lr', linear)], n_jobs=-1)

multi = MultiOutputRegressor(gb)

multi.fit(x_train, y_train)
y_pred = multi.predict(x_test)

y_test = scaler_y.inverse_transform(y_test)
y_pred = scaler_y.inverse_transform(y_pred)

# print(y_test)
# print(y_pred)

results = np.hstack((y_test,y_pred))
results_df = pd.DataFrame(results)
results_df.columns = ['lat_test', 'lon_test', 'max_sus_wind_test', 'min_pressure_test',
					  'lat_pred', 'lon_pred', 'max_sus_wind_pred', 'min_pressure_pred']


print(results_df.describe())

def delta_distance_azimuth(lat1,lon1,lat2,lon2):
	pos_1 = np.hstack((lat1,lon1))
	pos_2 = np.hstack((lat2,lon2))
	# dist, a1, a2 = vincenty_inverse(pos_1,pos_2)
	dist = haversine(pos_1,pos_2)
	# dist, a1, a2 = wgs84_geod.inv(lon1,lat1,lon2,lat2)
	# dist = [x / 1000.0 for x in dist]
	dist_km = dist/1000
	return dist_km

results_df['storm_id'] = storm_ids_test
results_df['error_distance'] = delta_distance_azimuth(results_df['lat_test'].tolist(),results_df['lon_test'].tolist(),results_df['lat_pred'].tolist(),results_df['lon_pred'].tolist())
results_df['error_wind'] = results_df['max_sus_wind_pred']-results_df['max_sus_wind_test']
results_df['error_pressure'] = results_df['min_pressure_pred']-results_df['min_pressure_test']

# print(results_df[results_df['error_distance'] > 10000])

# results_df[results_df['error_distance'] > 10000].to_csv('big_error.csv')

data = []
for i in results_df.values.tolist()[:]:
    data.append( [ (i[1], i[0]), (i[5], i[4]) ] )

lc = mc.LineCollection(data, colors='b', linewidths=1)
fig, ax = plt.subplots()
ax.add_collection(lc)
ax.autoscale()
ax.margins(0.1)
plt.show()

sns.distplot(results_df['error_distance'])
plt.show()


results_df.boxplot(column=['error_distance'])
plt.show()

results_df.boxplot(column=['error_wind'])
plt.show()

results_df.boxplot(column=['error_pressure'])
plt.show()

# corr_df = df[input_features]
# corr_df['latitude+24'] = df['latitude+24'].tolist()
# corr_df['longitude+24'] = df['longitude+24'].tolist()

# plt.imshow(corr_df.corr(), cmap='RdBu_r')
# plt.colorbar()
# tick_marks = [i for i in range(len(corr_df.columns))]
# plt.xticks(tick_marks, corr_df.columns, rotation='vertical')
# plt.yticks(tick_marks, corr_df.columns)
# plt.show()