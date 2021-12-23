import numpy as np
import pandas as pd

from pprint import pprint as pp

import matplotlib.pyplot as plt
import seaborn as sns


ofcl_df = pd.read_csv('./data/ofcl/ofcl_processed.csv')
ofcl_df = ofcl_df.sort_values(by=['year', 'atcf_code', 'month', 'day', 'hour'])

ofcl_df = ofcl_df[(ofcl_df["year"] >= 2015) & (ofcl_df["year"] <= 2019)]

intensity_columns = [c for c in ofcl_df.columns if "I_" in c]
ofcl_df[intensity_columns] = np.abs(ofcl_df[intensity_columns])


def weighted_average(df, hour, var):
    hour_str = str(hour).zfill(3)
    weights = df['F'+hour_str]
    values = df[hour_str+'h'+var]
    values_masked = np.ma.MaskedArray(values, mask=np.isnan(values))
    return np.average(values_masked, weights=weights)

x = [12,  24,  36, 48, 72, 96, 120]

y_t_ofcl = [weighted_average(ofcl_df, t, 'T_ofcl') for t in x]
y_t_bcd5 = [weighted_average(ofcl_df, t, 'T_bcd5') for t in x]
y_t_ofcl_skill = (1-np.array(y_t_ofcl)/np.array(y_t_bcd5))*100

y_i_ofcl = [weighted_average(ofcl_df, t, 'I_ofcl') for t in x]
y_i_bcd5 = [weighted_average(ofcl_df, t, 'I_bcd5') for t in x]
y_i_ofcl_skill = (1-np.array(y_i_ofcl)/np.array(y_i_bcd5))*100

fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(x, y_t_ofcl)
axs[0, 0].plot(x, y_t_bcd5)
axs[0, 0].set_xticks(x)
axs[0, 0].grid(axis="x")
axs[0, 0].set_xlim(0, 120)
axs[0, 0].set_xlabel("Forcast Hour")
axs[0, 0].set_ylabel("Forcast Track Error (n mi)")

axs[1, 0].plot(x, y_t_ofcl_skill)
axs[1, 0].set_xticks(x)
axs[1, 0].grid(axis="x")
axs[1, 0].set_xlim(0, 120)
axs[1, 0].set_ylim(0, 100)
axs[1, 0].set_xlabel("Forcast Hour")
axs[1, 0].set_ylabel("Track Forcast Skill % (realtive to BCD5)")

axs[0, 1].plot(x, y_i_ofcl)
axs[0, 1].plot(x, y_i_bcd5)
axs[0, 1].set_xticks(x)
axs[0, 1].grid(axis="x")
axs[0, 1].set_xlim(0, 120)
axs[0, 1].set_xlabel("Forcast Hour")
axs[0, 1].set_ylabel("Forcast Intensity Error (kt)")

axs[1, 1].plot(x, y_i_ofcl_skill)
axs[1, 1].set_xticks(x)
axs[1, 1].grid(axis="x")
axs[1, 1].set_xlim(0, 120)
axs[1, 1].set_ylim(0, 100)
axs[1, 1].set_xlabel("Forcast Hour")
axs[1, 1].set_ylabel("Intensity Forcast Skill % (realtive to BCD5)")

plt.show()
