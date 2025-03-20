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

# read in the data
onshoredata = pd.read_excel(sitedatalocation + "Onshore_Sites.xlsx")
offshoredata = pd.read_excel(sitedatalocation + "Offshore_Sites.xlsx")

# we need to convert the BNG (aka X and Y) coordinates to lat long
# first we need to create a pyproj transformer


transformer = proj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
# now we can convert the coordinates
print("Transforming coordinates")
onshoredata["Longitude"], onshoredata["Latitude"] = transformer.transform(
    onshoredata["X-Coordinate"].values, onshoredata["Y-Coordinate"].values
)
offshoredata["Longitude"], offshoredata["Latitude"] = transformer.transform(
    offshoredata["X-Coordinate"].values, offshoredata["Y-Coordinate"].values
)
# we have added the lat and long to the dataframes

# %%
# now we load in the list of locations of the wind data
winddatapath = "/Users/matt/SCORESdata/era5/"
windsitelocs = np.loadtxt(f"{winddatapath}site_locs.csv", skiprows=1, delimiter=",")
print("Finding closest sites")
# we use loaderfunctions to find the closest site for each onshore and offshore site.
# the function returns the site number and a boolean indicating whether the site is within 100km:
# all sites should be, this is just a check
onshoredata["site"], onshoredata["Within 100Km"] = Loaderfunctions.latlongtosite(
    onshoredata["Latitude"],
    onshoredata["Longitude"],
    windsitelocs,
)
# now we do the same for offshore sites

offshoredata["site"], offshoredata["Within 100Km"] = Loaderfunctions.latlongtosite(
    offshoredata["Latitude"],
    offshoredata["Longitude"],
    windsitelocs,
)
# the site numbers returned are floats, but they should be integers
onshoredata["site"] = onshoredata["site"].astype(int)
offshoredata["site"] = offshoredata["site"].astype(int)

# the CFD team requested particular turbine sizes as below
onshoreturbinesizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
offshoreturbinesizes = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
]

# the year range of 2000-2023 gives a good range of years to simulate. We could simulate from 1980, but
# wind speeds have declined since 1980, so from the year 2000 seems more relevant. This decision is fairly
# arbitrary, but the CFD team have been informed of it so can raise any objections
yearmin = 2000
yearmax = 2023

powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"

# we have a big list of sites from the CFD team, but our data has a limited spatial resolution
# There's no point simulating using the same site multiple times, so we'll get the unique sites
uniqueonshoresites = onshoredata["site"].unique()
uniqueoffshoresites = offshoredata["site"].unique()


# we're going to plot the location of the simulated sites on a map of the UK for posterity
mapfilelocs = "/Users/matt/code/mapplot/"

# shapefile = mapfilelocs + "GBR_adm/GBR_adm0.shp"
# shapefileireland = mapfilelocs + "IRL_adm/IRL_adm0.shp"
# gdf = gpd.read_file(shapefile)
# # now we need to remove northern ireland from the shapefile
# # print the names of all entities in the gdf
# gdfireland = gpd.read_file(shapefileireland)
# britishisles = gpd.pd.concat([gdf, gdfireland], ignore_index=True)

# # we want to plot which grid cells are being used on the map. I know how many point are in sitelocs, so I'll make an imshow of that size

# imshowwidth = 20
# imshowheight = 23
# sitelocimshow = np.zeros((imshowheight, imshowwidth))
# # any square where a site is located will be set to 1
# for site in uniqueonshoresites:
#     sitelocimshow[int(site / imshowwidth), site % imshowwidth] = 1

# sitelocimshowzeros = np.where(sitelocimshow == 0)
# alphaarray = np.ones_like(sitelocimshow) * 0.7
# alphaarray[sitelocimshowzeros] = 0
# britishisles.plot(facecolor="darkolivegreen", edgecolor="grey")
# plt.imshow(
#     sitelocimshow,
#     alpha=alphaarray,
#     extent=[
#         np.min(windsitelocs[:, 2]) - 0.625 / 2,
#         np.max(windsitelocs[:, 2]) + 0.625 / 2,
#         np.min(windsitelocs[:, 1]) - 0.5 / 2,
#         np.max(windsitelocs[:, 1]) + 0.5 / 2,
#     ],
#     origin="lower",
#     zorder=1000,
#     cmap="Blues",
# )
# plt.title("Onshore sites")
# plt.show()


# # now we're going to do the same for offshore sites
# imshowwidth = 20
# imshowheight = 23
# sitelocimshow = np.zeros((imshowheight, imshowwidth))
# for site in uniqueoffshoresites:
#     sitelocimshow[int(site / imshowwidth), site % imshowwidth] = 1
# sitelocimshowzeros = np.where(sitelocimshow == 0)
# alphaarray = np.ones_like(sitelocimshow) * 0.7
# alphaarray[sitelocimshowzeros] = 0


# britishisles.plot(facecolor="darkolivegreen", edgecolor="grey")

# plt.imshow(
#     sitelocimshow,
#     alpha=alphaarray,
#     extent=[
#         np.min(windsitelocs[:, 2]) - 0.625 / 2,
#         np.max(windsitelocs[:, 2]) + 0.625 / 2,
#         np.min(windsitelocs[:, 1]) - 0.5 / 2,
#         np.max(windsitelocs[:, 1]) + 0.5 / 2,
#     ],  # extent is the x and y limits of the image
#     origin="lower",
#     zorder=1000,
#     cmap="Blues",
# )

# plt.title("Offshore sites")
# plt.show()

# as noted before, one site can have multiple wind farms, so we'll store the data in a dictionary and extract it after
offshoreloadfactordict = {}

for index, site in enumerate(uniqueoffshoresites):
    print(f"Simulating site {index+1} of {len(uniqueoffshoresites)}")
    # we'll make a dictionary for this site which will store the load factors for each turbine size
    sitedict = {}
    numberofturbines = 1
    # iterate through the turbine sizes
    for gensize in offshoreturbinesizes:
        # load the power curve
        powercurve = np.loadtxt(
            powercurvelocation + f"{gensize}_MW.csv", delimiter=",", skiprows=1
        )
        # powercurve = np.loadtxt(
        #     "/Users/matt/code/Processing-toolkit/genericoffshorepowercurve.csv",
        #     delimiter=",",
        #     skiprows=1,
        # )
        # powercurve *= gensize
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
        loadfactor = datarun.get_load_factor()
        sitedict[gensize] = loadfactor
    offshoreloadfactordict[site] = copy.deepcopy(
        sitedict
    )  # we need to deepcopy the dictionary to avoid overwriting the data

# now we're going to iterate through each row of the offshore data, and add the load factors to the dataframe. We'll also add a column with the site number

for turbsize in offshoreturbinesizes:
    # this function will put the site number into the lambda function, so we can use it to access the load factor dictionary
    # .apply will use the site as the argument for the lambda function, and the result will be stored in the column
    offshoredata[f"{turbsize}MW Load Factor"] = offshoredata["site"].apply(
        lambda x: offshoreloadfactordict[x][turbsize]
    )

    # converting to a float now
    offshoredata[f"{turbsize}MW Load Factor"] = offshoredata[
        f"{turbsize}MW Load Factor"
    ].astype(float)

# now save offshore data to an excel file
# print the mean load factor for each turb size
for turbsize in offshoreturbinesizes:
    print(
        f"{turbsize}MW mean load factor: {offshoredata[f'{turbsize}MW Load Factor'].mean()}"
    )
offshoredata.to_excel(sitedatalocation + "Offshore_Sites_simulated.xlsx")
# same again for onshore

# iterate through our unique onshore sites
onshoreloadfactordict = {}
print("Simulating onshore sites")
for index, site in enumerate(uniqueonshoresites):
    print(f"Simulating site {index+1} of {len(uniqueonshoresites)}")
    sitedict = {}
    # get the data for this site

    # get the number of turbines
    numberofturbines = 1
    # iterate through the turbine sizes
    for gensize in onshoreturbinesizes:
        # load the power curve
        # powercurve = np.loadtxt(
        #     "/Users/matt/code/Processing-toolkit/optonshorepowercurve.csv",
        #     delimiter=",",
        #     skiprows=1,
        # )
        # powercurve *= gensize
        # get the generator object
        powercurve = np.loadtxt(
            powercurvelocation + f"{gensize}_MW.csv", delimiter=",", skiprows=1
        )
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
        loadfactor = datarun.get_load_factor()
        sitedict[gensize] = loadfactor
    onshoreloadfactordict[site] = copy.deepcopy(sitedict)

# now we're going to iterate through each row of the onshore data, and add the load factors to the dataframe. We'll also add a column with the site number

for turbsize in onshoreturbinesizes:
    onshoredata[f"{turbsize}MW Load Factor"] = onshoredata["site"].apply(
        lambda x: onshoreloadfactordict[x][turbsize]
    )
    onshoredata[f"{turbsize}MW Load Factor"] = onshoredata[
        f"{turbsize}MW Load Factor"
    ].astype(float)

# print the mean load factor for each turb size
for turbsize in onshoreturbinesizes:
    print(
        f"{turbsize}MW mean load factor: {onshoredata[f'{turbsize}MW Load Factor'].mean()}"
    )

# now save onshore data to an excel file
onshoredata.to_excel(sitedatalocation + "Onshore_Sites_simulated.xlsx")

# now we're going to do the same for offshore sites
