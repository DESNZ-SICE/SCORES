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

sitedatalocation = "/Users/matt/SCORESdata/cfdlocs/"

onshoredata = pd.read_excel(sitedatalocation + "Onshore Sites.xlsx")
offshoredata = pd.read_excel(sitedatalocation + "Offshore Sites.xlsx")

# we need to convert the BNG coordinates to lat long
# first we need to define the projection
# this is the projection for BNG


transformer = proj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
# now we can convert the coordinates
print("Transforming coordinates")
onshoredata["Longitude"], onshoredata["Latitude"] = transformer.transform(
    onshoredata["X-Coordinate"].values, onshoredata["Y-Coordinate"].values
)
offshoredata["Longitude"], offshoredata["Latitude"] = transformer.transform(
    offshoredata["X-Coordinate"].values, offshoredata["Y-Coordinate"].values
)

# %%

winddatapath = "/Volumes/macdrive/merraupdated/"
windsitelocs = np.loadtxt(f"{winddatapath}site_locs.csv", skiprows=1, delimiter=",")
print("Finding closest sites")
onshoredata["site"], onshoredata["Within 100Km"] = Loaderfunctions.latlongtosite(
    onshoredata["Latitude"],
    onshoredata["Longitude"],
    windsitelocs,
)
offshoredata["site"], offshoredata["Within 100Km"] = Loaderfunctions.latlongtosite(
    offshoredata["Latitude"],
    offshoredata["Longitude"],
    windsitelocs,
)

onshoredata["site"] = onshoredata["site"].astype(int)
offshoredata["site"] = offshoredata["site"].astype(int)

onshoreturbinesizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
offshoreturbinesizes = [2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

yearmin = 2000
yearmax = 2023
powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"

uniqueonshoresites = onshoredata["site"].unique()
uniqueoffshoresites = offshoredata["site"].unique()


mapfilelocs = "/Users/matt/code/mapplot/"

shapefile = mapfilelocs + "GBR_adm/GBR_adm0.shp"
shapefileireland = mapfilelocs + "IRL_adm/IRL_adm0.shp"
gdf = gpd.read_file(shapefile)
# now we need to remove northern ireland from the shapefile
# print the names of all entities in the gdf
gdfireland = gpd.read_file(shapefileireland)
britishisles = gpd.pd.concat([gdf, gdfireland], ignore_index=True)

# we want to plot which grid cells are being used on the map. I know how many point are in sitelocs, so I'll make an imshow of that size

imshowwidth = 20
imshowheight = 23
sitelocimshow = np.zeros((imshowheight, imshowwidth))
for site in uniqueonshoresites:
    sitelocimshow[int(site / imshowwidth), site % imshowwidth] = 1

sitelocimshowzeros = np.where(sitelocimshow == 0)
alphaarray = np.ones_like(sitelocimshow) * 0.7
alphaarray[sitelocimshowzeros] = 0
britishisles.plot(facecolor="darkolivegreen", edgecolor="grey")
plt.imshow(
    sitelocimshow,
    alpha=alphaarray,
    extent=[
        np.min(windsitelocs[:, 2]) - 0.625 / 2,
        np.max(windsitelocs[:, 2]) + 0.625 / 2,
        np.min(windsitelocs[:, 1]) - 0.5 / 2,
        np.max(windsitelocs[:, 1]) + 0.5 / 2,
    ],
    origin="lower",
    zorder=1000,
    cmap="Blues",
)
plt.title("Onshore sites")
plt.show()


# now we're going to do the same for offshore sites
imshowwidth = 20
imshowheight = 23
sitelocimshow = np.zeros((imshowheight, imshowwidth))
for site in uniqueoffshoresites:
    sitelocimshow[int(site / imshowwidth), site % imshowwidth] = 1
sitelocimshowzeros = np.where(sitelocimshow == 0)
alphaarray = np.ones_like(sitelocimshow) * 0.7
alphaarray[sitelocimshowzeros] = 0

# iterate through all the offshore rows
# for index, row in offshoredata.iterrows():
#     britishisles.plot(facecolor="darkolivegreen", edgecolor="grey")
#     plt.title(row["Site Name"])
#     plt.scatter(row["Longitude"], row["Latitude"], color="red")
#     plt.show()
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

britishisles.plot(facecolor="darkolivegreen", edgecolor="grey")

plt.imshow(
    sitelocimshow,
    alpha=alphaarray,
    extent=[
        np.min(windsitelocs[:, 2]) - 0.625 / 2,
        np.max(windsitelocs[:, 2]) + 0.625 / 2,
        np.min(windsitelocs[:, 1]) - 0.5 / 2,
        np.max(windsitelocs[:, 1]) + 0.5 / 2,
    ],  # extent is the x and y limits of the image
    origin="lower",
    zorder=1000,
    cmap="Blues",
)

plt.title("Offshore sites")
plt.show()


offshoreloadfactordict = {}

for index, site in enumerate(uniqueoffshoresites):
    print(f"Simulating site {index+1} of {len(uniqueoffshoresites)}")
    sitedict = {}
    # get the data for this site

    # get the number of turbines
    numberofturbines = 1
    # iterate through the turbine sizes
    for gensize in offshoreturbinesizes:
        # load the power curve
        powercurve = np.loadtxt(
            powercurvelocation + f"{gensize}_MW.csv", delimiter=",", skiprows=1
        )
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
        oldloadfactor = datarun.get_load_factor()
        sitedict[gensize] = oldloadfactor
    offshoreloadfactordict[site] = copy.deepcopy(sitedict)

# now we're going to iterate through each row of the offshore data, and add the load factors to the dataframe. We'll also add a column with the site number

for turbsize in offshoreturbinesizes:
    offshoredata[f"{turbsize}MW Load Factor"] = offshoredata["site"].apply(
        lambda x: offshoreloadfactordict[x][turbsize]
    )
    offshoredata[f"{turbsize}MW Load Factor"] = offshoredata[
        f"{turbsize}MW Load Factor"
    ].astype(float)

# now save offshore data to an excel file

offshoredata.to_excel(sitedatalocation + "Offshore_Sites_simulated.xlsx")


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
        powercurve = np.loadtxt(
            powercurvelocation + f"{gensize}_MW.csv", delimiter=",", skiprows=1
        )
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
        oldloadfactor = datarun.get_load_factor()
        sitedict[gensize] = oldloadfactor
    onshoreloadfactordict[site] = copy.deepcopy(sitedict)

# now we're going to iterate through each row of the onshore data, and add the load factors to the dataframe. We'll also add a column with the site number

for turbsize in onshoreturbinesizes:
    onshoredata[f"{turbsize}MW Load Factor"] = onshoredata["site"].apply(
        lambda x: onshoreloadfactordict[x][turbsize]
    )
    onshoredata[f"{turbsize}MW Load Factor"] = onshoredata[
        f"{turbsize}MW Load Factor"
    ].astype(float)

# now save onshore data to an excel file
onshoredata.to_excel(sitedatalocation + "Onshore_Sites_simulated.xlsx")

# now we're going to do the same for offshore sites
