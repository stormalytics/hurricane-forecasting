import numpy as np
import pandas as pd

from pprint import pprint as pp


def process_ofcl(data_dir):
    ofcl_df = pd.read_csv(data_dir+'ofcl/ofcl.txt', skiprows=6, delim_whitespace=True)

    ofcl_df = ofcl_df.replace(-9999.0, np.nan)
    ofcl_df["dt"] = pd.to_datetime(ofcl_df["Date/Time"], format='%d-%m-%Y/%H:%M:%S')
    ofcl_df["year"] = ofcl_df["dt"].dt.year
    ofcl_df["month"] = ofcl_df["dt"].dt.month
    ofcl_df["day"] = ofcl_df["dt"].dt.day
    ofcl_df["hour"] = ofcl_df["dt"].dt.hour

    ofcl_df = ofcl_df.drop(columns=["Date/Time", "dt"])

    ofcl_df.columns = ofcl_df.columns.astype(str).str.replace("T01", "T_ofcl")
    ofcl_df.columns = ofcl_df.columns.astype(str).str.replace("I01", "I_ofcl")
    ofcl_df.columns = ofcl_df.columns.astype(str).str.replace("T02", "T_bcd5")
    ofcl_df.columns = ofcl_df.columns.astype(str).str.replace("I02", "I_bcd5")

    ofcl_df = ofcl_df.rename(columns={'STMID': 'atcf_code',
                                      'Lat': 'latitude',
                                      'Lon': 'longitude',
                                      'WS': 'max_sus_wind'})
    
    print("#### Saving processed data to file ####")
    ofcl_df.to_csv(data_dir+'ofcl/ofcl_processed.csv', index=False)
    print("Done")


if __name__ == "__main__":
    DATA_DIR = "./data/"
    process_ofcl(DATA_DIR)
