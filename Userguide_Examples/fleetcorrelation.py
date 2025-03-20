# %%
import numpy as np
import seaborn as sns

# %%
import pyproj as proj
import pandas as pd
import Loaderfunctions
from generation import OffshoreWindModel15000
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
from scipy.spatial import cKDTree

trimmedrepd = pd.read_excel("/Users/matt/SCORESdata/repd-q3-oct-2024-trimmed.xlsx")

# print the column names
# print the first row
print(trimmedrepd.head())

print(trimmedrepd.columns)
# we only want the onshore and offshore wind sites
trimmedrepd = trimmedrepd[
    trimmedrepd["Technology Type"].isin(["Wind Onshore", "Wind Offshore"])
]

# some of the rows have empty strings for the installed capacity, we want to remove these
trimmedrepd = trimmedrepd[
    trimmedrepd["Development Status (short)"].isin(
        ["Operational", "Under Construction", "Awaiting Construction"]
    )
]
# remove any row where the country is Northern Ireland
trimmedrepd = trimmedrepd[trimmedrepd["Country"] != "Northern Ireland"]
# iterate through the rows and remove the rows with empty strings
for index, row in trimmedrepd.iterrows():
    try:
        int(row["Installed Capacity (MWelec)"])
    except:
        trimmedrepd.drop(index, inplace=True)

trimmedrepd["Installed Capacity (MWelec)"] = trimmedrepd[
    "Installed Capacity (MWelec)"
].astype(float)

trimmedrepd = trimmedrepd[trimmedrepd["Installed Capacity (MWelec)"] > 20]

mapfilelocs = "/Users/matt/code/mapplot/"


# %%
transformer = proj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

trimmedrepd["Longitude"], trimmedrepd["Latitude"] = transformer.transform(
    trimmedrepd["X-coordinate"].values, trimmedrepd["Y-coordinate"].values
)


shapefile = mapfilelocs + "GBR_adm/GBR_adm0.shp"
shapefileireland = mapfilelocs + "IRL_adm/IRL_adm0.shp"
gdf = gpd.read_file(shapefile)
# now we need to remove northern ireland from the shapefile
# print the names of all entities in the gdf
# gdfireland = gpd.read_file(shapefileireland)
# britishisles = gpd.pd.concat([gdf, gdfireland], ignore_index=True)
britishisles = gdf
britishisles.plot(facecolor="darkolivegreen", edgecolor="grey", linewidth=0.5)


operational = trimmedrepd[trimmedrepd["Development Status (short)"] == "Operational"]
notyetoperational = trimmedrepd[
    trimmedrepd["Development Status (short)"] != "Operational"
]


# Define a function to combine points that are close to each other
# def combine_close_points(lons, lats, capacities, threshold=0.5):
#     points = np.vstack((lons, lats)).T
#     tree = cKDTree(points)
#     clusters = tree.query_ball_tree(tree, threshold)

#     combined_lons = []
#     combined_lats = []
#     combined_capacities = []
#     visited = set()

#     for cluster in clusters:
#         if cluster[0] in visited:
#             continue
#         visited.update(cluster)
#         combined_lons.append(np.mean(lons[cluster]))
#         combined_lats.append(np.mean(lats[cluster]))
#         combined_capacities.append(np.sum(capacities[cluster]))

#     return (
#         np.array(combined_lons),
#         np.array(combined_lats),
#         np.array(combined_capacities),
#     )


# Combine close points for operational and not yet operational sites
# operational_lons, operational_lats, operational_capacities = combine_close_points(
#     operational["Longitude"].values,
#     operational["Latitude"].values,
#     operational["Installed Capacity (MWelec)"].values,
# )

# notyetoperational_lons, notyetoperational_lats, notyetoperational_capacities = (
#     combine_close_points(
#         notyetoperational["Longitude"].values,
#         notyetoperational["Latitude"].values,
#         notyetoperational["Installed Capacity (MWelec)"].values,
#     )
# )

# Plot the combined points
# plt.scatter(
#     operational_lons,
#     operational_lats,
#     s=operational_capacities / 50,
#     label="Operational",
# )
# plt.scatter(
#     notyetoperational_lons,
#     notyetoperational_lats,
#     s=notyetoperational_capacities / 50,
#     label="Not Operational",
# )

plt.scatter(
    operational["Longitude"],
    operational["Latitude"],
    s=operational["Installed Capacity (MWelec)"] / 50,
    label="Operational",
)
plt.scatter(
    notyetoperational["Longitude"],
    notyetoperational["Latitude"],
    s=notyetoperational["Installed Capacity (MWelec)"] / 50,
    label="Not Operational",
)
plt.legend()
# remove the axes
plt.axis("off")
# save with a high dpi
plt.savefig("/Users/matt/Pictures/correlation2/operationalnotoperational.png", dpi=700)
plt.show()


# %%
winddatafolder = "/Users/matt/SCORESdata/era5/"
windsitelocs = np.loadtxt(winddatafolder + "site_locs.csv", skiprows=1, delimiter=",")
maskedwindsitelocs = np.loadtxt(
    winddatafolder + "site_locs_mask.csv", skiprows=1, delimiter=","
)
(
    trimmedrepd["site"],
    trimmedrepd["Within 100Km"],
) = Loaderfunctions.latlongtosite(
    trimmedrepd["Latitude"],
    trimmedrepd["Longitude"],
    windsitelocs,
)

trimmedrepd["site"] = trimmedrepd["site"].astype(int)
# %%
referenceprice = pd.read_excel(
    "/Users/matt/code/referencepriceinvestigator/intermittent_reference_price.xlsx"
)
# extract the column named price and turn it into a numpy array
referenceprice = referenceprice["Price"].to_numpy()

year_min = 2000
year_max = 2023
demanddata = np.loadtxt("demand.csv", usecols=2, delimiter=",", skiprows=1)

operational = trimmedrepd[trimmedrepd["Development Status (short)"] == "Operational"]
# add a row to operational
print("Finding the average generation profile")
uniquesites = trimmedrepd["site"].unique()
siteinstalledcaps = []
genprofiles = []
for site in uniquesites:
    siteinstalledcaps.append(
        trimmedrepd[trimmedrepd["site"] == site]["Installed Capacity (MWelec)"].sum()
    )
    genobject = OffshoreWindModel15000(
        sites=[site],
        year_min=year_min,
        year_max=year_max,
        data_path=winddatafolder,
        n_turbine=[1],
        force_run=True,
    )
    genprofiles.append(genobject.power_out_array)
# %%
genprofiles = np.vstack(genprofiles)
averagegenprofile = np.average(genprofiles, axis=0, weights=siteinstalledcaps)
# %%


onshoreonly = trimmedrepd[trimmedrepd["Technology Type"] == "Wind Onshore"]
offshoreonly = trimmedrepd[trimmedrepd["Technology Type"] == "Wind Offshore"]

onshoreinstalledcaps = []
offshoreinstalledcaps = []
onshoregenprofiles = []
offshoregenprofiles = []
print(f"Finding the average generation profile for onshore and offshore")
for site in onshoreonly["site"].unique():
    onshoreinstalledcaps.append(
        onshoreonly[onshoreonly["site"] == site]["Installed Capacity (MWelec)"].sum()
    )
    genobject = OffshoreWindModel15000(
        sites=[site],
        year_min=year_min,
        year_max=year_max,
        data_path=winddatafolder,
        n_turbine=[1],
        force_run=True,
    )
    onshoregenprofiles.append(genobject.power_out_array)

for site in offshoreonly["site"].unique():
    offshoreinstalledcaps.append(
        offshoreonly[offshoreonly["site"] == site]["Installed Capacity (MWelec)"].sum()
    )
    genobject = OffshoreWindModel15000(
        sites=[site],
        year_min=year_min,
        year_max=year_max,
        data_path=winddatafolder,
        n_turbine=[1],
        force_run=True,
    )
    offshoregenprofiles.append(genobject.power_out_array)

onshoregenprofiles = np.vstack(onshoregenprofiles)
offshoregenprofiles = np.vstack(offshoregenprofiles)

onshoreaveragegenprofile = np.average(
    onshoregenprofiles, axis=0, weights=onshoreinstalledcaps
)
offshoreaveragegenprofile = np.average(
    offshoregenprofiles, axis=0, weights=offshoreinstalledcaps
)

siteinstalledcaps = np.array(siteinstalledcaps)
reducedsiteinstalledcaps = siteinstalledcaps / np.min(siteinstalledcaps)
intsiteinstalledcaps = reducedsiteinstalledcaps.astype(int)


repeatedmeanarray = []
correlationarray = []

sitelocsshape = windsitelocs.shape

# %%
# we need to find how many latitudes and longitudes we have
startinglat = windsitelocs[0, 1]
numberoflons = 1
while True:
    if windsitelocs[numberoflons, 1] == startinglat:
        numberoflons += 1
    else:
        break
# %%
lonres = windsitelocs[1, 2] - windsitelocs[0, 2]
latres = windsitelocs[numberoflons, 1] - windsitelocs[0, 1]
numberoflats = int(sitelocsshape[0] / numberoflons)
# %%
print(numberoflats)
print(windsitelocs[0])
print(windsitelocs[-1])


fleetcorrelationarray = np.zeros((numberoflats, numberoflons))
onshorecorrelationarray = np.zeros((numberoflats, numberoflons))
offshorecorrelationarray = np.zeros((numberoflats, numberoflons))
hourlydemandcorrelationarray = np.zeros((numberoflats, numberoflons))
generatorrevenuearray = np.zeros((numberoflats, numberoflons))
weeklydemandcorrelationarray = np.zeros((numberoflats, numberoflons))
loadfactorarray = np.zeros((numberoflats, numberoflons))
startdatetime = datetime.datetime(2000, 1, 1)


demanddatastartdatetime = datetime.datetime(2009, 1, 1)
demanddataenddatetime = datetime.datetime(2020, 1, 1)

demandstartindex = int(
    (demanddatastartdatetime - startdatetime).total_seconds() // (3600)
)
demandendindex = int((demanddataenddatetime - startdatetime).total_seconds() // (3600))

imrpdatastarttime = datetime.datetime(2020, 1, 1)
imrpdataendtime = datetime.datetime(2024, 1, 1)

imrpdatastartindex = int((imrpdatastarttime - startdatetime).total_seconds() // (3600))
imrpdataendindex = int((imrpdataendtime - startdatetime).total_seconds() // (3600))
print("Simulating each cell")
for i in range(len(windsitelocs)):
    print(f"{100*i/len(windsitelocs)}%")
    sitenum = windsitelocs[i, 0]
    sitegenprofile = OffshoreWindModel15000(
        sites=[str(int(sitenum))],
        year_min=year_min,
        year_max=year_max,
        data_path=winddatafolder,
        n_turbine=[1],
        force_run=True,
    )

    sitepowerout = sitegenprofile.power_out_array

    demandpowerout = sitepowerout[demandstartindex:demandendindex]
    loadfactorarray[i // numberoflons, i % numberoflons] = (
        sitegenprofile.get_load_factor()
    )
    imrppowerout = sitepowerout[imrpdatastartindex:imrpdataendindex]
    # multiply the power output by the reference price
    generatorrevenuearray[i // numberoflons, i % numberoflons] = (
        np.sum(imrppowerout * referenceprice) / (4 * 15)
    ) / 10**6

    hourlydemandcorrelation = np.corrcoef(demandpowerout, demanddata)
    hourlydemandcorrelationarray[i // numberoflons, i % numberoflons] = (
        hourlydemandcorrelation[0, 1]
    )

    numberofweeks = len(demandpowerout) // (24 * 7)
    demandpoweroutsubset = demandpowerout[0 : numberofweeks * 24 * 7].reshape(
        numberofweeks, 24 * 7
    )
    demanddatasubset = demanddata[0 : numberofweeks * 24 * 7].reshape(
        numberofweeks, 24 * 7
    )

    weeklypowerout = np.mean(demandpoweroutsubset, axis=1)
    weeklydemand = np.mean(demanddatasubset, axis=1)

    weeklydemandcorrelation = np.corrcoef(weeklypowerout, weeklydemand)
    weeklydemandcorrelationarray[i // numberoflons, i % numberoflons] = (
        weeklydemandcorrelation[0, 1]
    )
    thisfleetcorrelation = np.corrcoef(
        sitegenprofile.power_out_array, averagegenprofile
    )
    fleetcorrelationarray[i // numberoflons, i % numberoflons] = thisfleetcorrelation[
        0, 1
    ]

    thisonshorecorrelation = np.corrcoef(
        sitegenprofile.power_out_array, onshoreaveragegenprofile
    )
    onshorecorrelationarray[i // numberoflons, i % numberoflons] = (
        thisonshorecorrelation[0, 1]
    )

    thisoffshorecorrelation = np.corrcoef(
        sitegenprofile.power_out_array, offshoreaveragegenprofile
    )
    offshorecorrelationarray[i // numberoflons, i % numberoflons] = (
        thisoffshorecorrelation[0, 1]
    )

# %%

maskarray = np.zeros((numberoflats, numberoflons))
for i in range(len(maskedwindsitelocs)):
    maskarray[i // numberoflons, i % numberoflons] = maskedwindsitelocs[i, 3]

onshoremask = np.where(maskarray == 1)
offshoremask = np.where(maskarray == 0)

onshoreloadfactorarray = loadfactorarray.copy()
offshoreloadfactorarray = loadfactorarray.copy()

onshoreloadfactorarray[offshoremask] = 0
offshoreloadfactorarray[onshoremask] = 0


# find the lowest value in any of the correlation arrays
minval = np.min(
    [
        np.min(fleetcorrelationarray),
        np.min(onshorecorrelationarray),
        np.min(offshorecorrelationarray),
        np.min(hourlydemandcorrelationarray),
        np.min(weeklydemandcorrelationarray),
    ]
)
maxval = np.max(
    [
        np.max(fleetcorrelationarray),
        np.max(onshorecorrelationarray),
        np.max(offshorecorrelationarray),
        np.max(hourlydemandcorrelationarray),
        np.max(weeklydemandcorrelationarray),
    ]
)

sns.set_theme()

shapefile = mapfilelocs + "GBR_adm/GBR_adm0.shp"
shapefileireland = mapfilelocs + "IRL_adm/IRL_adm0.shp"
guernseylocation = mapfilelocs + "guernsey/geoBoundaries-GGY-ADM0.shp"
gdf = gpd.read_file(shapefile)
isleofman = gpd.read_file(mapfilelocs + "IMN_adm/IMN_adm0.shp")
# now we need to remove northern ireland from the shapefile
# print the names of all entities in the gdf
gdfireland = gpd.read_file(shapefileireland)
guernsey = gpd.read_file(guernseylocation)
britishisles = gdf
guernsey = guernsey.to_crs("EPSG:4326")
isleofman = isleofman.to_crs("EPSG:4326")
britishisles = gpd.pd.concat([britishisles, guernsey, isleofman], ignore_index=True)
britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    generatorrevenuearray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    cmap="YlOrBr",
)
plt.title("Revenue per MW per year")
plt.axis("off")

plt.colorbar(label="Revenue (£m)")
plt.show()

britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
minonshoreloadfactor = np.min(onshoreloadfactorarray[onshoremask])
maxonshoreloadfactor = np.max(onshoreloadfactorarray[onshoremask])
onshorealpha = np.zeros(onshoreloadfactorarray.shape)
onshorealpha[onshoremask] = 1
plt.imshow(
    onshoreloadfactorarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    vmin=minonshoreloadfactor,
    vmax=maxonshoreloadfactor,
    cmap="viridis",
    alpha=onshorealpha,
)
plt.axis("off")
plt.colorbar(label="Load Factor")
plt.show()

britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
minoffshoreloadfactor = np.min(offshoreloadfactorarray[offshoremask])
maxoffshoreloadfactor = np.max(offshoreloadfactorarray[offshoremask])
offshorealpha = np.zeros(offshoreloadfactorarray.shape)
offshorealpha[offshoremask] = 1
plt.imshow(
    offshoreloadfactorarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    vmin=minoffshoreloadfactor,
    vmax=maxoffshoreloadfactor,
    cmap="viridis",
    alpha=offshorealpha,
)
plt.axis("off")
plt.colorbar(label="Gross Load Factor")
plt.show()


onshorerevenue = generatorrevenuearray.copy()
onshorerevenue[offshoremask] = 0
minonshorerevenue = np.min(onshorerevenue[onshoremask])
maxonshorerevenue = np.max(onshorerevenue[onshoremask])
onshorealpha = np.zeros(onshorerevenue.shape)
onshorealpha[onshoremask] = 1


britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    onshorerevenue,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    cmap="YlOrBr",
    alpha=onshorealpha,
    vmin=minonshorerevenue,
    vmax=maxonshorerevenue,
)
plt.axis("off")
plt.colorbar(label="Revenue (£m)")
plt.title("Onshore Revenue per MW per year")
plt.show()

offshorerevenue = generatorrevenuearray.copy()
offshorerevenue[onshoremask] = 0
minoffshorerevenue = np.min(offshorerevenue[offshoremask])
maxoffshorerevenue = np.max(offshorerevenue[offshoremask])
offshorealpha = np.zeros(offshorerevenue.shape)
offshorealpha[offshoremask] = 1

britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    offshorerevenue,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    cmap="YlOrBr",
    alpha=offshorealpha,
    vmin=minoffshorerevenue,
    vmax=maxoffshorerevenue,
)
plt.axis("off")
plt.colorbar(label="Revenue (£m)")
plt.title("Offshore Revenue per MW per year")
plt.show()
# %%
britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    onshorecorrelationarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    vmin=minval,
    vmax=maxval,
    cmap="viridis",
)
plt.axis("off")


plt.title("Correlation with Onshore Fleet Average")
plt.colorbar(label="Correlation Coefficient")
plt.show()

britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    offshorecorrelationarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    vmin=minval,
    vmax=maxval,
    cmap="viridis",
)
plt.title("Correlation with Offshore Fleet Average")
plt.colorbar(label="Correlation Coefficient")
plt.axis("off")

plt.show()


britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    fleetcorrelationarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    vmin=minval,
    vmax=maxval,
    cmap="viridis",
)
plt.title("Correlation with Fleet Average")
plt.colorbar(label="Correlation Coefficient")
plt.axis("off")

plt.show()

britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    hourlydemandcorrelationarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    vmin=minval,
    vmax=maxval,
    cmap="viridis",
)
plt.title("Correlation with hourly demand")
plt.colorbar(label="Correlation Coefficient")
plt.axis("off")

plt.show()


britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    weeklydemandcorrelationarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    vmin=minval,
    vmax=maxval,
    cmap="viridis",
)
plt.title("Correlation with weekly demand")
plt.colorbar(label="Correlation Coefficient")
plt.axis("off")

plt.show()


britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    hourlydemandcorrelationarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    cmap="viridis",
)
plt.title("Correlation with hourly demand")
plt.colorbar(label="Correlation Coefficient")
plt.axis("off")

plt.show()


britishisles.plot(facecolor="none", edgecolor="black", linewidth=0.5)
plt.imshow(
    weeklydemandcorrelationarray,
    origin="lower",
    extent=[
        np.min(windsitelocs[:, 2]) - lonres / 2,
        np.max(windsitelocs[:, 2]) + lonres / 2,
        np.min(windsitelocs[:, 1]) - latres / 2,
        np.max(windsitelocs[:, 1]) + latres / 2,
    ],
    cmap="viridis",
)
plt.title("Correlation with weekly demand")
plt.colorbar(label="Correlation Coefficient")
plt.axis("off")

plt.show()

# %%
# for i in range(genprofiles.shape[1]):
#     if i % 1000 == 0:
#         print(f"{100*i/genprofiles.shape[1]}%")
#     for j in range(genprofiles.shape[0]):
#         for k in range(intsiteinstalledcaps[j]):
#             correlationarray.append(genprofiles[j, i])
#             repeatedmeanarray.append(averagegenprofile[i])

# sitecorrelationcoefficients = []
# for i in range(genprofiles.shape[0]):
#     # remove row i from genprofiles
#     thisgenprofile = genprofiles[i, :]
#     genprofilesminusi = np.delete(genprofiles, i, axis=0)
#     siteinstallcapsminusi = np.delete(siteinstalledcaps, i, axis=0)
#     averagegenprofileminusi = np.average(
#         genprofilesminusi, axis=0, weights=siteinstallcapsminusi
#     )

#     thiscorrelation = np.corrcoef(thisgenprofile, averagegenprofile)
#     sitecorrelationcoefficients.append(thiscorrelation[0, 1])


mostcorrelatedindex = np.argmax(sitecorrelationcoefficients)
leastcorrelatedindex = np.argmin(sitecorrelationcoefficients)

mostcorrelatedsitenum = uniquesites[mostcorrelatedindex]
leastcorrelatedsitenum = uniquesites[leastcorrelatedindex]
mostcorrelatedsite = windsitelocs[mostcorrelatedsitenum]
leastcorrelatedsite = windsitelocs[leastcorrelatedsitenum]
# %%


# get the lat and long of the most and least correlated sites

# there may be more than one site with the same index, so we need to take the first one

plt.scatter(
    mostcorrelatedsite[2],
    mostcorrelatedsite[1],
    label="Most Correlated",
    s=100,
    marker="X",
)
plt.scatter(
    leastcorrelatedsite[2],
    leastcorrelatedsite[1],
    label="Least Correlated",
    s=100,
)
plt.legend()
plt.show()
# %%
loadfactors = []
for i in range(genprofiles.shape[0]):
    loadfactors.append(np.mean(genprofiles[i, :]))
print(np.mean(loadfactors))
print(np.max(loadfactors))
print(np.min(loadfactors))
print(loadfactors[0:10])
# %%
repeatedmeanarray = []
correlationarray = []


for i in range(genprofiles.shape[1]):
    if i % 1000 == 0:
        print(f"{100*i/genprofiles.shape[1]}%")
    for j in range(genprofiles.shape[0]):
        for k in range(intsiteinstalledcaps[j]):
            correlationarray.append(genprofiles[j, i])
            repeatedmeanarray.append(averagegenprofile[i])

# %%
correlationarray = np.array(correlationarray)
repeatedmeanarray = np.array(repeatedmeanarray)
correlation = np.corrcoef(correlationarray, repeatedmeanarray)
print(f"Current sites correlation: {correlation}")

# %%
print("Looking at all sites")
year_min = 2000
year_max = 2023
uniquesites = trimmedrepd["site"].unique()
siteinstalledcaps = []
genprofiles = []
for site in uniquesites:
    siteinstalledcaps.append(
        trimmedrepd[trimmedrepd["site"] == site]["Installed Capacity (MWelec)"].sum()
    )
    genobject = OffshoreWindModel15000(
        sites=[site],
        year_min=year_min,
        year_max=year_max,
        data_path=winddatafolder,
        n_turbine=[1],
        force_run=True,
    )
    genprofiles.append(genobject.power_out_array)
# %%
genprofiles = np.vstack(genprofiles)
averagegenprofile = np.average(genprofiles, axis=0, weights=siteinstalledcaps)
# %%
siteinstalledcaps = np.array(siteinstalledcaps)
reducedsiteinstalledcaps = siteinstalledcaps / np.min(siteinstalledcaps)
intsiteinstalledcaps = reducedsiteinstalledcaps.astype(int)

# %%
repeatedmeanarray = []
correlationarray = []

# for i in range(genprofiles.shape[1]):
#     if i % 1000 == 0:
#         print(f"{100*i/genprofiles.shape[1]}%")
#     for j in range(genprofiles.shape[0]):
#         for k in range(intsiteinstalledcaps[j]):
#             correlationarray.append(genprofiles[j, i])
#             repeatedmeanarray.append(averagegenprofile[i])

sitecorrelationcoefficients = []
for i in range(genprofiles.shape[0]):
    # remove row i from genprofiles
    genprofilesminusi = np.delete(genprofiles, i, 1)
    siteinstallcapsminusi = np.delete(siteinstalledcaps, i)
    sitecorrelationcoefficients.append(
        np.corrcoef(genprofilesminusi, averagegenprofile)[0, 1]
    )


# we
# %%
correlationarray = np.array(correlationarray)
repeatedmeanarray = np.array(repeatedmeanarray)
correlation = np.corrcoef(correlationarray, repeatedmeanarray)
print(f"Future sites correlation: {correlation}")

# %%
