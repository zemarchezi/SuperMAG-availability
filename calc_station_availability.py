#%%
import numpy as np
import pandas as pd
import glob
from netCDF4 import Dataset
from tqdm import tqdm
#%%

AUXDAT_PATH = "./auxData"
OUTPUT_PATH = "./outputData"
SUPERMAG_PATH = "/data/supermag/baseline-netcdf"# %%

supermagStations = pd.read_csv(f"{AUXDAT_PATH}/20230525-19-09-supermag-stations.csv")
supermagStations.drop(['OPERATOR-NUM', 'OPERATORS', 'Op2', 'Op3', 'Op4', 'Op5'], axis=1, inplace=True)
#%%
stationDataYears = {}
for station in tqdm(supermagStations["IAGA"].values):
    supermagFiles = glob.glob(f"{SUPERMAG_PATH}/{station}/*.csv")
    listYears = []
    stationDataYears[station] = {}
    for file in sorted(supermagFiles):
        year = file.split("-")[2]
        tempData = pd.read_csv(file)
        componentAvailable = []
        for comp in ['dbn_nez', 'dbe_nez', 'dbz_nez', 'dbn_geo', 'dbe_geo', 'dbz_geo']:
            componentAvailable.append(int(np.round(((tempData.shape[0] - tempData[comp].isna().sum()) / tempData.shape[0]) * 100)))
        stationDataYears[station][year] = int(np.mean(componentAvailable))
# %%

dfAvailability = pd.DataFrame.from_dict(stationDataYears, orient="index").sort_index(axis=1).fillna(0).astype(int)
#%%
finalAvailable = supermagStations.join(dfAvailability, on="IAGA", how="left")
# %%
finalAvailable.to_csv(f"{OUTPUT_PATH}/supermag_available.csv")
# %%
