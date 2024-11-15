# %%
import numpy as np
import pandas as pd
import pyproj as proj
import Loaderfunctions
import generation
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import rasterio
from generation import (
    OnshoreWindModel2000,
    OnshoreWindModel2500,
    OnshoreWindModel3000,
    OnshoreWindModel3500,
)
from rasterio.plot import show
from rasterio.warp import transform_bounds
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.features import geometry_window
import datetime
import scipy.stats as stats

sitedatalocation = "/Users/matt/code/grosstonet/"

onshoredata = pd.read_excel(sitedatalocation + "onshorewithsite.xlsx")


# %%

winddatapath = "/Volumes/macdrive/merraupdated/"
onshoredata["site"] = onshoredata["site"].astype(int)

sitebiaseddict = {}

with open(f"{winddatapath}site_locs_biased_median.csv", "r") as file:
    data = file.read().strip().split("\n")
    for row in data[1:]:
        sitebiaseddict[int(row.split(",")[0])] = float(row.split(",")[3])

src = rasterio.open("/Users/matt/SCORESdata/GBR_wind-speed_50m.tif")

srca = rasterio.open("/Users/matt/SCORESdata/GBR_combined-Weibull-A_50m.tif")
srck = rasterio.open("/Users/matt/SCORESdata/GBR_combined-Weibull-k_50m.tif")


def biasfinder(src, lat, lon, sitebiaseddict, site, margin=0.05):
    cellaverage = sitebiaseddict[site]
    cell = box(lon - margin, lat - margin, lon + margin, lat + margin)
    window = geometry_window(src, [cell], pad_x=0, pad_y=0, pixel_precision=1)
    # Read data within this window
    data = src.read(1, window=window, masked=True)

    # Calculate the mean, ignoring NoData values
    median_value = np.median(data.compressed())

    return median_value / cellaverage


def weibull_loader(srca, srck, lat, lon, margin=0.05):
    cell = box(lon - margin, lat - margin, lon + margin, lat + margin)
    window = geometry_window(srca, [cell], pad_x=0, pad_y=0, pixel_precision=1)
    # Read data within this window
    dataa = srca.read(1, window=window, masked=True)
    datak = srck.read(1, window=window, masked=True)

    scale = np.median(dataa.compressed())
    shape = np.median(datak.compressed())
    # return the median a and k values
    return [shape, scale]


onshoredata["Scaling Factors"] = onshoredata.apply(
    lambda x: biasfinder(src, x["Latitude"], x["Longitude"], sitebiaseddict, x["site"]),
    axis=1,
)
# overwrite the original excel document
onshoredata.to_excel(sitedatalocation + "onshorewithsite.xlsx")

yearmin = 2023
yearmax = 2023
datestamplist = []
starttime = datetime.datetime(yearmin, 1, 1)
currenttime = starttime
while currenttime.year <= yearmax:
    datestamplist.append(currenttime)
    currenttime += datetime.timedelta(hours=1)

powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"

gendict = {
    2: OnshoreWindModel2000,
    2.5: OnshoreWindModel2500,
    3: OnshoreWindModel3000,
    3.5: OnshoreWindModel3500,
}

datastartdate = datetime.datetime(1980, 1, 1)
dataofintereststart = datetime.datetime(2023, 1, 1)
# work out the hours between the start of the data and the start of the data of interest
hours = (dataofintereststart - datastartdate).days * 24

for index, row in onshoredata.iterrows():
    site = row["site"]
    name = row["Site Name"]
    sitedata = np.loadtxt(
        f"{winddatapath}{site}.csv", delimiter=",", skiprows=1, usecols=[2]
    )[hours:]
    shape, loc, scale = stats.weibull_min.fit(sitedata, floc=0)
    cell_weibull = [shape, scale]
    pixel_weibull = weibull_loader(srca, srck, row["Latitude"], row["Longitude"])
    turbinesize = row["Turbine Capacity (MW)"]
    turbinesize = round(turbinesize * 2) / 2
    if turbinesize - round(turbinesize) == 0:
        turbinesize = int(turbinesize)
    generator = gendict[turbinesize]
    powercurve = np.loadtxt(
        powercurvelocation + f"{turbinesize}_MW.csv", delimiter=",", skiprows=1
    )

    genobject = generator(
        sites=[site],
        year_min=yearmin,
        year_max=yearmax,
        data_path=winddatapath,
        n_turbine=[1],
        force_run=True,
        power_curve=powercurve,
        cell_weibull=cell_weibull,
        pixel_weibull=pixel_weibull,
    )

    power_out_array = genobject.power_out_array
    power_out_array = power_out_array / turbinesize
    with open(
        f"/Users/matt/code/grosstonet/onshore simmed weibull/{name}.csv", "w"
    ) as file:
        file.write("datetime,powerout\n")
        for i in range(len(datestamplist)):
            file.write(f"{datestamplist[i].isoformat()},{power_out_array[i]}\n")
