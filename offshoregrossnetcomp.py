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

sitedatalocation = (
    "/Users/matt/code/bathtub curve/"  # location of site data provided by CFD team
)

# read in the data
offshoredata = pd.read_excel(sitedatalocation + "bathtub.xlsx")

# we need to convert the BNG (aka X and Y) coordinates to lat long
# first we need to create a pyproj transformer


transformer = proj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
# now we can convert the coordinates
print("Transforming coordinates")
offshoredata["Longitude"], offshoredata["Latitude"] = transformer.transform(
    offshoredata["X-coordinate"].values, offshoredata["Y-coordinate"].values
)

# we have added the lat and long to the dataframes

# %%
# now we load in the list of locations of the wind data
winddatapath = "/Users/matt/SCORESdata/era5/"
windsitelocs = np.loadtxt(f"{winddatapath}site_locs.csv", skiprows=1, delimiter=",")
print("Finding closest sites")
# we use loaderfunctions to find the closest site for each offshore and offshore site.
# the function returns the site number and a boolean indicating whether the site is within 100km:
# all sites should be, this is just a check
offshoredata["site"], offshoredata["Within 100Km"] = Loaderfunctions.latlongtosite(
    offshoredata["Latitude"],
    offshoredata["Longitude"],
    windsitelocs,
)

# the site numbers returned are floats, but they should be integers
offshoredata["site"] = offshoredata["site"].astype(int)
# save offshoredata to an excel file, in the same location as the original file
offshoredata.to_excel(sitedatalocation + "offshorewithsite.xlsx")
yearmin = 2006
yearmax = 2024
# make a list with all the hours of 2023, for when we need to write to file

currenttime = datetime.datetime(yearmin, 1, 1, 0, 0, 0)
datetimelist = []
while currenttime.year != yearmax + 1:
    datetimelist.append(currenttime)
    currenttime += datetime.timedelta(hours=1)


powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"

# we have a big list of sites from the CFD team, but our data has a limited spatial resolution
# There's no point simulating using the same site multiple times, so we'll get the unique sites


# we're going to plot the location of the simulated sites on a map of the UK for posterity

# as noted before, one site can have multiple wind farms, so we'll store the data in a dictionary and extract it after

for index, row in offshoredata.iterrows():
    site = row["site"]
    sitename = row["Site Name"]
    print(f"Simulating {sitename}, row {index} of {len(offshoredata)}")
    gensize = float(row["Turbine Capacity (MW)"])

    # round the gensize to the nearest 0.5
    gensize = round(gensize)
    gensize = int(gensize)
    numberofturbines = 1

    # powercurve = np.loadtxt(
    #     "/Users/matt/code/Processing-toolkit/genericoffshorepowercurve.csv",
    #     delimiter=",",
    #     skiprows=1,
    # )
    powercurve = np.loadtxt(
        f"{powercurvelocation}/{gensize}_MW.csv",
        delimiter=",",
        skiprows=1,
    )
    # powercurve[:, 1] = powercurve[:, 1] * gensize
    # get the generator object
    genobject = generation.generatordictionaries().offshore[gensize]
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
    with open(f"{sitedatalocation}/REGOsimmed/{sitename}.csv", "w") as file:
        file.write("datetime,powerout,speed(m/s)\n")
        for i in range(len(powerout)):
            file.write(
                f"{datetimelist[i]},{round(powerout[i],2)},{round(site_speeds[i],2)}\n"
            )

# now we're going to iterate through each row of the offshore data, and add the load factors to the dataframe. We'll also add a column with the site number

# now we're going to do the same for offshore sites
