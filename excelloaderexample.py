"""
Written by Matt RT

This file demonstrates loading a spreadsheet containing offshore wind turbines, and 
loading them in as generator objects.

The file must have the following headings:
Installed Capacity (MWelec)
Turbine Capacity (MW) 
No. of Turbines 
Latitude 
Longitude
Country
Operational

Operational should be the date the facility went online, in format dd/mm/yyyy


The terminology throughout can be confusing. Sites refers to the positions of each wind measurement,
which are not coincident with the turbines. The closest site to each turbine is found

"""
# %%
import pandas as pd
import numpy as np
import generation
import loaderfunctions
import datetime
import time

folder = "C:/Users/SA0011/Documents/data/"  # general folder with all data in it
offshorewinddatafolder = (
    folder + "meerabiasedwind/"
)  # subfolder with offshore wind site data
filename = "Offshore_wind_operational_July_2023.xlsx"
loadeddata = pd.read_excel(folder + filename)
loadeddata = loadeddata[loadeddata["Country"] == "Scotland"]

# %%
generatordict = generation.generatordictionaries().offshore
generatorkeys = np.array(list(generatordict.keys()))
# makes the turbne sizes into an array

windsitedata = np.loadtxt(
    offshorewinddatafolder + "site_locs.csv", skiprows=1, delimiter=","
)

(
    loadeddata["site"],
    loadeddata["Within 100Km"],
) = loaderfunctions.latlongtosite(
    loadeddata["Latitude"],
    loadeddata["Longitude"],
    windsitedata,
)
tiledgens = np.tile(generatorkeys, (len(loadeddata), 1))
tiledcaps = np.tile(loadeddata["Turbine Capacity (MW)"], (len(generatorkeys), 1)).T

# these lines tile the generator sizes and the turbine capacities so that we can compare them
# to find the closest available generator size for each row

# heres a low dimension example to make this clearer:
# if gen keys is : [1,2,3,4]
# and caps is [3,1,2]
# then tiledgens is:
# [[1,2,3,4],
#  [1,2,3,4],
#  [1,2,3,4]]

# and tiledcaps is:
# [[3,3,3,3],
#  [1,1,1,1],
#  [2,2,2,2]]

# then tiledcaps-tiledgens is:
# [[2,1,0,-1],
#  [0,-1,-2,-3],
#  [1,0,-1,-2]]

# then np.argmin(abs(tiledcaps-tiledgens), axis=1) is:
# [2,0,1]

# We can then use these indices to find the closest generator size for each row
# %%
# # %%
loadeddata["Closest Turbine Size"] = [
    generatorkeys[i] for i in np.argmin(abs(tiledcaps - tiledgens), axis=1)
]

# Several sites may have the same size generators. A generator object can take a list of sites,
# and a list of the number of turbines at each site, so we need to group the sites by generator size

loadeddata["site"] = loadeddata["site"].astype(int)
# %%
loadeddata["OperationalDatetime"] = pd.to_datetime(
    loadeddata["Operational"], format="%d/%m/%Y"
)
# %%
differentgensizes = loadeddata["Closest Turbine Size"].unique()

allgenerators = []  # makes an empty list to store the generator objects in

starttime = time.time()
for gensize in differentgensizes:
    subset = loadeddata[loadeddata["Closest Turbine Size"] == gensize]
    sites = subset["site"].to_list()
    nturbines = subset["No. of Turbines"].to_list()
    datetimeobjects = subset["OperationalDatetime"].to_list()
    years = [i.year for i in datetimeobjects]
    months = [i.month for i in datetimeobjects]
    selectedgenerator = generatordict[gensize]
    allgenerators.append(
        selectedgenerator(
            year_min=2021,
            year_max=2021,
            sites=sites,
            n_turbine=nturbines,
            data_path=offshorewinddatafolder,
            year_online=years,
            month_online=months,
            force_run=True,
        )
    )

total1 = 0
for entry in allgenerators:
    averageyearlypowergenerated = np.sum(entry.power_out)
    total1 += averageyearlypowergenerated
print(f"Time elaped: {time.time()-starttime}")
# %%
import matplotlib.pyplot as plt

turbinsize = [i.turbine_size for i in allgenerators]
loadfactors = [i.get_load_factor() for i in allgenerators]
plt.plot(turbinsize, loadfactors, "o")
plt.xlabel("Turbine Size (MW)")
plt.ylabel("Load Factor")
plt.show()
for i in allgenerators:
    print(f"{i.turbine_size} MW turbine")
    print(i.get_load_factor())
    print("----------")

# %%
