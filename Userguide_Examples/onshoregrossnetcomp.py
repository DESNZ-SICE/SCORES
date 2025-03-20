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
import datetime
from shapely.geometry import Point
from rasterio.plot import show
from rasterio.warp import transform_bounds
from shapely.geometry import box
from rasterio.windows import Window
from rasterio.features import geometry_window
import rasterio
import fiona

sitedatalocation = (
    "/Users/matt/code/grosstonet/"  # location of site data provided by CFD team
)

# read in the data
onshoredata = pd.read_excel(sitedatalocation + "onshore.xlsx")

# we need to convert the BNG (aka X and Y) coordinates to lat long
# first we need to create a pyproj transformer


transformer = proj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
# now we can convert the coordinates
print("Transforming coordinates")
onshoredata["Longitude"], onshoredata["Latitude"] = transformer.transform(
    onshoredata["X-coordinate"].values, onshoredata["Y-coordinate"].values
)

# we have added the lat and long to the dataframes

# %%
# now we load in the list of locations of the wind data
winddatapath = "/Users/matt/SCORESdata/era52020-2023/"
windsitelocs = np.loadtxt(f"{winddatapath}site_locs.csv", skiprows=1, delimiter=",")
# meeansitelocs = np.loadtxt(
#     f"{winddatapath}site_locs_withmean.csv", skiprows=1, delimiter=","
# )
print("Finding closest sites")
# we use loaderfunctions to find the closest site for each onshore and onshore site.
# the function returns the site number and a boolean indicating whether the site is within 100km:
# all sites should be, this is just a check
onshoredata["site"], onshoredata["Within 100Km"] = Loaderfunctions.latlongtosite(
    onshoredata["Latitude"],
    onshoredata["Longitude"],
    windsitelocs,
)

# the site numbers returned are floats, but they should be integers
onshoredata["site"] = onshoredata["site"].astype(int)
# save onshoredata to an excel file, in the same location as the original file
# onshoredata.to_excel(sitedatalocation + "onshorewithsite.xlsx")
yearmin = 2020
yearmax = 2023
# make a list with all the hours of 2023, for when we need to write to file

currenttime = datetime.datetime(yearmin, 1, 1, 0, 0, 0)
datetimelist = []
while currenttime.year != yearmax + 1:
    datetimelist.append(currenttime)
    currenttime += datetime.timedelta(hours=1)

src = rasterio.open("/Users/matt/SCORESdata/GBR_wind-speed_50m.tif")


powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"

# we have a big list of sites from the CFD team, but our data has a limited spatial resolution
# There's no point simulating using the same site multiple times, so we'll get the unique sites


# we're going to plot the location of the simulated sites on a map of the UK for posterity

# as noted before, one site can have multiple wind farms, so we'll store the data in a dictionary and extract it after

for index, row in onshoredata.iterrows():
    site = row["site"]
    sitename = row["Site Name"]
    # if index != 3:
    #     continue
    print(f"Simulating {sitename}, row {index} of {len(onshoredata)}")
    gensize = float(row["Turbine Capacity (MW)"])

    sitelat = row["Latitude"]
    sitelong = row["Longitude"]
    # site_mean = meeansitelocs[site][3]
    sitemargin = 0.025
    # load the geojson called acreloch.geojson
    # with fiona.open("stonelairg/stonelairg.shp", "r") as shapefile:
    #     shapes = [feature["geometry"] for feature in shapefile]

    # cell = box(
    #     sitelong - sitemargin,
    #     sitelat - sitemargin,
    #     sitelong + sitemargin,
    #     sitelat + sitemargin,
    # )
    # # use the acrloch geojson to get the window

    # # window the src with the shapefile

    # # window = geometry_window(src, shapes, pad_x=0, pad_y=0, pixel_precision=1)
    # window = geometry_window(src, [cell], pad_x=0, pad_y=0, pixel_precision=1)

    # data = src.read(1, window=window)

    # mean_value = data.mean()
    # scaling_factor = mean_value / site_mean

    # round the gensize to the nearest 0.5
    gensize = round(gensize)
    gensize = int(gensize)
    numberofturbines = 1

    # powercurve = np.loadtxt(
    #     "c:/Users/SA0011/Documents/data/genericpowercurve.csv",
    #     delimiter=",",
    #     skiprows=1,
    # )
    powercurve = np.loadtxt(
        f"/Users/matt/code/Processing-toolkit/genericonshorepowercurve.csv",
        delimiter=",",
        skiprows=1,
    )
    powercurve[:, 1] = powercurve[:, 1] * gensize
    # get the generator object
    genobject = generation.generatordictionaries().onshore[gensize]
    # run the generator
    datarun = genobject(
        sites=[site],
        year_min=yearmin,
        year_max=yearmax,
        data_path=winddatapath,
        n_turbine=[numberofturbines],
        force_run=True,
        power_curve=powercurve,
    )
    # get the load factor
    powerout = datarun.power_out_array
    powerout = powerout / gensize
    site_speeds = datarun.speeds
    with open(
        f"{sitedatalocation}onshore simmed era longer net/{sitename}.csv", "w"
    ) as file:
        file.write("datetime,powerout,speed(m/s)\n")
        for i in range(len(powerout)):
            file.write(
                f"{datetimelist[i]},{round(powerout[i],2)},{round(site_speeds[i],2)}\n"
            )

# now we're going to iterate through each row of the onshore data, and add the load factors to the dataframe. We'll also add a column with the site number

# now we're going to do the same for onshore sites
