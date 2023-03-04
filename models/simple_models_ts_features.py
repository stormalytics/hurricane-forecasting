import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

import tsfresh




if __name__ == '__main__':
    data = pd.read_csv('./data/hurdat/hurdat2_processed.csv')

    # data = data[data['year'] > 2015]

    input_vars = ["longitude", "latitude", "min_pressure", "max_sus_wind", "landfall", "hour",
                "jday", "time_idx", "delta_distance", "delta_distance_x", "delta_distance_y", "azimuth",
                "x", "y", "vpre"]
    output_vars = ["longitude", "latitude"]