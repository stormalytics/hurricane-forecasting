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


def ofcl_weighted_average(df, hour, var):
    hour_str = str(hour).zfill(3)
    weights = df['F'+hour_str]
    values = df[hour_str+'h'+var]
    values_masked = np.ma.MaskedArray(values, mask=np.isnan(values))
    return np.average(values_masked, weights=weights)


x = np.array([12,  24,  36, 48, 72])

y_t_ofcl = np.array([ofcl_weighted_average(ofcl_df, t, 'T_ofcl') for t in x])
y_t_bcd5 = np.array([ofcl_weighted_average(ofcl_df, t, 'T_bcd5') for t in x])
y_t_ofcl_skill = (1-(y_t_ofcl/y_t_bcd5))*100

y_i_ofcl = np.array([ofcl_weighted_average(ofcl_df, t, 'I_ofcl') for t in x])
y_i_bcd5 = np.array([ofcl_weighted_average(ofcl_df, t, 'I_bcd5') for t in x])
y_i_ofcl_skill = (1-(y_i_ofcl/y_i_bcd5))*100

x_new_models = np.array([6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72])
align_index = np.where(np.in1d(x_new_models, x))[0]
align_index = align_index.astype(int)
x_aligned = x_new_models[align_index]
print(align_index)

xgboost_t_df = pd.read_csv('./results/track_error_xgboost.csv')
y_t_xgboost = np.array([np.mean(xgboost_t_df[f'future_horizon_{str(t)}']) for t in x_new_models])
y_t_xgboost_aligned = y_t_xgboost[align_index]
y_t_xgboost_skill = (1-(y_t_xgboost_aligned/y_t_bcd5))*100

ridge_t_df = pd.read_csv('./results/track_error_ridge.csv')
y_t_ridge = np.array([np.mean(ridge_t_df[f'future_horizon_{str(t)}']) for t in x_new_models])
y_t_ridge_aligned = y_t_ridge[align_index]
y_t_ridge_skill = (1-(y_t_ridge_aligned/y_t_bcd5))*100

decision_tree_t_df = pd.read_csv('./results/track_error_decision_tree.csv')
y_t_decision_tree = np.array([np.mean(decision_tree_t_df[f'future_horizon_{str(t)}']) for t in x_new_models])
y_t_decision_tree_aligned = y_t_decision_tree[align_index]
y_t_decision_tree_skill = (1-(y_t_decision_tree_aligned/y_t_bcd5))*100

tcn_t_df = pd.read_csv('./results/track_error_tcn.csv')
y_t_tcn = np.array([np.mean(tcn_t_df[f'future_horizon_{str(t)}']) for t in x_new_models])
y_t_tcn_aligned = y_t_tcn[align_index]
y_t_tcn_skill = (1-(y_t_tcn_aligned/y_t_bcd5))*100


fig, axs = plt.subplots(2, 2)

axs[0, 0].plot(x, y_t_ofcl, label='OFCL')
axs[0, 0].plot(x, y_t_bcd5, label='BCD5')
axs[0, 0].plot(x_aligned, y_t_xgboost_aligned, label='Boosted Tree')
axs[0, 0].plot(x_aligned, y_t_ridge_aligned, label='Ridge Regression')
axs[0, 0].plot(x_aligned, y_t_decision_tree_aligned, label='Decision Tree')
axs[0, 0].plot(x_aligned, y_t_tcn_aligned, label='TCN')
axs[0, 0].set_xticks(x)
axs[0, 0].grid(axis="x")
axs[0, 0].set_xlim(0, 72)
axs[0, 0].set_xlabel("Forcast Hour")
axs[0, 0].set_ylabel("Forcast Track Error (n mi)")
axs[0, 0].legend()

axs[1, 0].plot(x, y_t_ofcl_skill, label='OFCL')
axs[1, 0].plot(x_aligned, y_t_xgboost_skill, label='Boosted Tree')
axs[1, 0].plot(x_aligned, y_t_ridge_skill, label='Ridge Regression')
axs[1, 0].plot(x_aligned, y_t_decision_tree_skill, label='Decision Tree')
axs[1, 0].plot(x_aligned, y_t_tcn_skill, label='TCN')
axs[1, 0].set_xticks(x)
axs[1, 0].grid(axis="x")
axs[1, 0].set_xlim(0, 72)
axs[1, 0].set_ylim(-100, 100)
axs[1, 0].set_xlabel("Forcast Hour")
axs[1, 0].set_ylabel("Track Forcast Skill % (realtive to BCD5)")
axs[1, 0].legend()

axs[0, 1].plot(x, y_i_ofcl, label='OFCL')
axs[0, 1].plot(x, y_i_bcd5, label='BCD5')
axs[0, 1].set_xticks(x)
axs[0, 1].grid(axis="x")
axs[0, 1].set_xlim(0, 72)
axs[0, 1].set_xlabel("Forcast Hour")
axs[0, 1].set_ylabel("Forcast Intensity Error (kt)")
axs[0, 1].legend()


axs[1, 1].plot(x, y_i_ofcl_skill, label='OFCL')
axs[1, 1].set_xticks(x)
axs[1, 1].grid(axis="x")
axs[1, 1].set_xlim(0, 72)
axs[1, 1].set_ylim(-100, 100)
axs[1, 1].set_xlabel("Forcast Hour")
axs[1, 1].set_ylabel("Intensity Forcast Skill % (realtive to BCD5)")
axs[1, 1].legend()

plt.show()
