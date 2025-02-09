import multiprocessing as mp
mp.set_start_method("fork", force=True)  # necessary for parmap

import glob
from pathlib import Path

import pandas as pd
import parmap


def parse_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        if len(df) == 0: return pd.DataFrame()
        else:
            return df.assign(ticker=file_path.split('/')[-1].split('.')[0])
    except:
        return pd.DataFrame()

csv_list = glob.glob('/Users/sjkdan/desk/cryscript/research/stooq/*.csv') 
df_list = parmap.map(parse_csv, csv_list, pm_pbar=True, pm_processes=4)
df = pd.concat(df_list)

print(df.ticker.nunique())
