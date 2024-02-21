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
import matplotlib.pyplot as plt
import seaborn as sns

folder = "C:/Users/SA0011/Documents/data/"  # general folder with all data in it
onshorewinddatafolder = (
    folder + "meerabiasedwind/"
)  # subfolder with offshore wind site data
filename = "Onshore_wind_operational_July_2023.xlsx"
loadeddata = pd.read_excel(folder + filename)
loadeddata = loadeddata[loadeddata["Country"] == "Scotland"]


# outputfilename = "Onshore_bias_corrected_moreturbines"
# %%
generatordict = generation.generatordictionaries().onshore
generatorkeys = np.array(list(generatordict.keys()))
# makes the turbne sizes into an array

windsitedata = np.loadtxt(
    onshorewinddatafolder + "site_locs.csv", skiprows=1, delimiter=","
)

(
    loadeddata["site"],
    loadeddata["Within 100Km"],
) = loaderfunctions.latlongtosite(
    loadeddata["Latitude"],
    loadeddata["Longitude"],
    windsitedata,
)
# plots histograme of the turbine capacities
# plt.hist(loadeddata["Turbine Capacity (MW)"], bins=20)
# plt.xlabel("Turbine Capacity (MW)")
# plt.ylabel("Number of turbines")
# plt.show()

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
# sorts the differentgenerator sizes into ascending order
differentgensizes.sort()
allgenerators = []  # makes an empty list to store the generator objects in
print("Turbine sizes found")

totalcapacities = []

# %%
starttime = time.time()
for i, gensize in enumerate(differentgensizes):
    print(f"{100*i/len(differentgensizes)}% complete")

    subset = loadeddata[loadeddata["Closest Turbine Size"] == gensize]
    sites = subset["site"].to_list()
    print(len(sites))
    nturbines = subset["No. of Turbines"].to_list()
    totalcapacities.append(np.sum(nturbines) * gensize)
    datetimeobjects = subset["OperationalDatetime"].to_list()
    years = [i.year for i in datetimeobjects]
    months = [i.month for i in datetimeobjects]
    selectedgenerator = generatordict[gensize]
    allgenerators.append(
        selectedgenerator(
            year_min=2022,
            year_max=2022,
            sites=sites,
            n_turbine=nturbines,
            data_path=onshorewinddatafolder,
            year_online=years,
            month_online=months,
            force_run=True,
        )
    )

print(f"time taken: {time.time()-starttime}")
total1 = 0
poweroutlist = []
for entry in allgenerators:
    averageyearlypowergenerated = np.sum(entry.power_out)
    poweroutlist.append(np.sum(entry.power_out))
    total1 += averageyearlypowergenerated

# %%
# plots pie chart of total capacity
sns.set_theme()
totalcapacities = [i / 1e3 for i in totalcapacities]
genlabels = [f"{round(i, 1)} MW" for i in differentgensizes]
# plt.pie(totalcapacities, labels=genlabels, autopct="%1.1f%%")
# plt.pie(totalcapacities, labels=genlabels)
# # plt.bar(genlabels, totalcapacities)

# plt.title("Scotland share of installed capacity (GW) ")
# plt.show()

# # plots pie chart of total power generated
# poweroutlist = [np.sum(i) / 1e6 for i in poweroutlist]
# # plt.pie(poweroutlist, labels=genlabels, autopct="%1.1f%%")
# plt.pie(poweroutlist, labels=genlabels)
# plt.title("Scotland share of power generated, 2021 (TWh)")
# plt.show()
print(f"Time elaped: {time.time()-starttime}")

print(f"Total power generated: {total1/1e6} TWh")


# %%

turbinsize = [i.turbine_size for i in allgenerators]
loadfactors = [i.get_load_factor() for i in allgenerators]
plt.plot(turbinsize, loadfactors, "o")
plt.xlabel("Turbine Size (MW)")
plt.ylabel("Load Factor")
plt.show()
# for i in allgenerators:
#     print(f"{i.turbine_size} MW turbine")
#     print(i.get_load_factor())
#     print("----------")


# with open(folder + "outputs/" + outputfilename + "2021.txt", "w") as f:
#     f.write(f"Total output\t{total1/1e6}\n")
#     f.write(f"Load factors\n")
#     for i in allgenerators:
#         f.write(f"{i.turbine_size}\t")
#         f.write(f"{i.get_load_factor()}\n")


# %%
