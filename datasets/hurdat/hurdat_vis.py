import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.dates as mdates




df = pd.read_csv('./data/hurdat/hurdat2_processed.csv')
# df = df[df['wind_radii_34_NE'] != -999]
df = df.sort_values(by=['year', 'atcf_code', 'month', 'day', 'hour'])

for atcf_code, hurricane_df in df.groupby('atcf_code', sort=False):
    x = hurricane_df["longitude"].to_numpy()
    y = hurricane_df["latitude"].to_numpy()

    year = hurricane_df["year"].tolist()
    month = hurricane_df["month"].tolist()
    day = hurricane_df["day"].tolist()
    hour = hurricane_df["hour"].tolist()

    dts = list(map(lambda x: datetime.datetime(*x), zip(year, month, day, hour)))
    dts = np.array(dts)
    

    pres = hurricane_df["min_pressure"].to_numpy()
    wind = hurricane_df["max_sus_wind"].to_numpy()

    delta_distance = hurricane_df['delta_distance'].to_numpy()
    azimuth = hurricane_df['azimuth'].to_numpy()
    delta_distance_x = hurricane_df['delta_distance_x'].to_numpy()
    delta_distance_y = hurricane_df['delta_distance_y'].to_numpy()

    vpre = hurricane_df["vpre"].to_numpy()
    vpre_inverse_scaled = hurricane_df["vpre_inverse_scaled"].to_numpy()

    landfall = hurricane_df["landfall"].to_numpy()

    plot = True
    if plot:
        fig = plt.figure()
        gs = fig.add_gridspec(nrows=4, ncols=2, height_ratios=[3,1,1,1])

        ax_map = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
        ax_map.set_extent((-100, -10, 5, 50), crs=ccrs.PlateCarree())
        ax_map.add_feature(cfeature.COASTLINE)
        ax_map.add_feature(cfeature.BORDERS, linestyle=':')
        ax_map.set_aspect('auto')
        ax_map.plot(x, y, linestyle='-', marker='x', markersize=2)
        ax_map.set_title(hurricane_df["atcf_code"].iloc[0])


        ax_pres = fig.add_subplot(gs[1, 0])
        ax_pres.plot(dts, pres, linestyle='-', marker='o', markersize=2)
        ax_pres.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_pres.set_title("min_pressure")

        ax_wind = fig.add_subplot(gs[2, 0])
        ax_wind.plot(dts, wind, linestyle='-', marker='o', markersize=2)
        ax_wind.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_wind.set_title("max_sus_wind")

        ax_vpre_inverse_scaled = fig.add_subplot(gs[3, 0])
        ax_vpre_inverse_scaled.plot(dts, vpre_inverse_scaled, linestyle='-', marker='o', markersize=2)
        ax_vpre_inverse_scaled.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_vpre_inverse_scaled.set_title("vpre_inverse_scaled")

        ax_delta_distance = fig.add_subplot(gs[1, 1])
        ax_delta_distance.plot(dts, delta_distance, linestyle='-', marker='o', markersize=2)
        ax_delta_distance.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_delta_distance.set_title("delta_distance")

        ax_azimuth = fig.add_subplot(gs[2, 1])
        ax_azimuth.plot(dts, azimuth, linestyle='-', marker='o', markersize=2)
        ax_azimuth.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_azimuth.set_title("azimuth")

        ax_landfall = fig.add_subplot(gs[3, 1])
        ax_landfall.plot(dts, landfall, linestyle='-', marker='o', markersize=2)
        ax_landfall.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
        ax_landfall.set_title("landfall")

        plt.tight_layout()
        plt.show()
