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

sitedatalocation = (
    "/Users/matt/SCORESdata/cfdlocs/"  # location of site data provided by CFD team
)
offshoredata = pd.read_excel(sitedatalocation + "Offshore_Sites.xlsx")
offshoredata = offshoredata[offshoredata["Currently Operational"] == "no"]
# read in the data
# offshoredata = pd.read_excel("/Users/matt/code/temp/2023sites.xlsx")
# we need to convert the BNG (aka X and Y) coordinates to lat long
# first we need to create a pyproj transformer


# we have added the lat and long to the dataframes

# %%
# now we load in the list of locations of the wind data
winddatapath = "/Users/matt/SCORESdata/era5/"
windsitelocs = np.loadtxt(f"{winddatapath}site_locs.csv", skiprows=1, delimiter=",")
# we use loaderfunctions to find the closest site for each onshore and offshore site.
# the function returns the site number and a boolean indicating whether the site is within 100km:
# all sites should be, this is just a check
offshoredata["site"], offshoredata["Within 100Km"] = Loaderfunctions.latlongtosite(
    offshoredata["Latitude"],
    offshoredata["Longitude"],
    windsitelocs,
)
# now we do the same for offshore sites


# the site numbers returned are floats, but they should be integers
offshoredata["site"] = offshoredata["site"].astype(int)


yearmin = 2000
yearmax = 2023

powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"

oldsiteloadfactors = []
simsites = offshoredata["site"].values
installedcaps = offshoredata["InstalledCapacity (MW)"].values
# installedcaps = np.ones(len(simsites))
for index, site in enumerate(simsites):
    # we'll make a dictionary for this site which will store the load factors for each turbine size
    sitedict = {}
    numberofturbines = 1
    # iterate through the turbine sizes
    gensize = 15

    powercurve = np.loadtxt(
        powercurvelocation + f"{gensize}_MW.csv", delimiter=",", skiprows=1
    )

    genobject = generation.generatordictionaries().offshore[gensize]
    # run the generator
    print(site)
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
    loadfactor = datarun.get_load_factor()
    oldsiteloadfactors.append(loadfactor)

print(np.average(oldsiteloadfactors, weights=installedcaps))
