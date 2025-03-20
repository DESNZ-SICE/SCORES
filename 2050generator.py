# %%
import numpy as np
import pyproj as proj
import pandas as pd
import Loaderfunctions
from generation import OffshoreWindModel15000, OnshoreWindModel7000, SolarModel
from storage import BatteryStorageModel
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime
from system import ElectricitySystem

trimmedrepd = pd.read_excel("/Users/matt/SCORESdata/repd-q3-oct-2024-trimmed.xlsx")

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


# shapefile = mapfilelocs + "GBR_adm/GBR_adm0.shp"
# shapefileireland = mapfilelocs + "IRL_adm/IRL_adm0.shp"
# gdf = gpd.read_file(shapefile)
# # now we need to remove northern ireland from the shapefile
# # print the names of all entities in the gdf
# gdfireland = gpd.read_file(shapefileireland)
# britishisles = gpd.pd.concat([gdf, gdfireland], ignore_index=True)
# britishisles.plot(facecolor="darkolivegreen", edgecolor="grey")


operational = trimmedrepd[trimmedrepd["Development Status (short)"] == "Operational"]
notyetoperational = trimmedrepd[
    trimmedrepd["Development Status (short)"] != "Operational"
]
# plt.scatter(
#     operational["Longitude"],
#     operational["Latitude"],
#     s=operational["Installed Capacity (MWelec)"] / 50,
#     label="Operational",
# )
# plt.scatter(
#     notyetoperational["Longitude"],
#     notyetoperational["Latitude"],
#     s=notyetoperational["Installed Capacity (MWelec)"] / 50,
#     label="Not Operational",
# )
# plt.legend()
# plt.show()


# %%
winddatafolder = "/Users/matt/SCORESdata/merraupdated/"
windsitelocs = np.loadtxt(winddatafolder + "site_locs.csv", skiprows=1, delimiter=",")

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

year_min = 2009
year_max = 2016
demand = np.loadtxt("demand.csv", usecols=2, delimiter=",", skiprows=1)

demandstarttime = datetime.datetime(2009, 1, 1)


starttime = datetime.datetime(year_min, 1, 1)
endtime = datetime.datetime(year_max + 1, 1, 1)

startindex = int((starttime - demandstarttime).total_seconds() / 3600)
endindex = int((endtime - demandstarttime).total_seconds() / 3600)

numberofpoints = int((endtime - starttime).total_seconds() / 3600)
demand = demand[startindex:endindex]
numberofyears = year_max - year_min + 1
# scale demand so the total demand per year is 570TWh. Demand is currently in MWh

yearlydemand = 550
nucleargenamount = 22
currentyear = year_min

for year in range(numberofyears):
    endofyear = datetime.datetime(year_min + year + 1, 1, 1) - datetime.timedelta(
        hours=1
    )
    startindex = int(
        (datetime.datetime(currentyear, 1, 1) - starttime).total_seconds() / 3600
    )
    endindex = int((endofyear - starttime).total_seconds() / 3600)
    yearlysum = np.sum(demand[startindex:endindex])
    scalingfactor = yearlydemand * 10**6 / yearlysum
    demand[startindex:endindex] *= scalingfactor
    currentyear += 1

demand -= nucleargenamount * 1000 * 0.9


operational = trimmedrepd[trimmedrepd["Development Status (short)"] == "Operational"]
# add a row to operational

onshorefarms = operational[operational["Technology Type"] == "Wind Onshore"]
offshorefarms = operational[operational["Technology Type"] == "Wind Offshore"]

onshoresites = onshorefarms["site"].unique()
onshorecapacities = np.array(
    [
        sum(onshorefarms[onshorefarms["site"] == site]["Installed Capacity (MWelec)"])
        for site in onshoresites
    ]
)


offshoresites = offshorefarms["site"].unique()
offshorecapacities = np.array(
    [
        sum(offshorefarms[offshorefarms["site"] == site]["Installed Capacity (MWelec)"])
        for site in offshoresites
    ]
)

currentonshorecapacity = np.sum(onshorecapacities)
currentoffshorecapacity = np.sum(offshorecapacities)

onshorecapacity2050 = 42
offshorecapacity2050 = 96
solarcapacity2050 = 88

solardatapath = "/Volumes/macdrive/updatedsolarcomb/"


solardata = pd.read_excel(
    "/Users/matt/code/transform/toptengenerators/top10solar_modified.xlsx"
)

solardata["site"], solardata["Within 100Km"] = Loaderfunctions.latlongtosite(
    solardata["Latitude"],
    solardata["Longitude"],
    np.loadtxt(f"{solardatapath}site_locs.csv", skiprows=1, delimiter=","),
)

solardata["site"] = solardata["site"].astype(int)
solarsites = solardata["site"].unique()

solarcaps = [solarcapacity2050 * 10**3 / len(solarsites)] * len(solarsites)

solarobject = SolarModel(
    sites=solarsites,
    year_min=year_min,
    year_max=year_max,
    data_path=solardatapath,
    plant_capacities=solarcaps,
    force_run=True,
)

onshore_turb_num = [
    int((i / np.sum(onshorecapacities) * onshorecapacity2050) / 7)
    for i in onshorecapacities
]
offshore_turb_num = [
    int((i / np.sum(offshorecapacities) * offshorecapacity2050) / 15)
    for i in offshorecapacities
]

onshoreobject = OnshoreWindModel7000(
    sites=onshoresites,
    year_min=year_min,
    year_max=year_max,
    data_path=winddatafolder,
    n_turbine=onshore_turb_num,
)

offshoreobject = OffshoreWindModel15000(
    sites=offshoresites,
    year_min=year_min,
    year_max=year_max,
    data_path=winddatafolder,
    n_turbine=offshore_turb_num,
)

genlist = [solarobject, onshoreobject, offshoreobject]

# %%


Battstore = BatteryStorageModel(capacity=120 * 10**3)
storlist = [Battstore]
system = ElectricitySystem(genlist, storlist, demand)
system.update_surplus()
reliability = system.get_reliability()
