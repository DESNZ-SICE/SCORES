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
    folder + "updatedwind/"
)  # subfolder with offshore wind site data
filename = "Offshore_wind_operational_July_2023.xlsx"
loadeddata = pd.read_excel(folder + filename)
# selects only the data from England and Scotland
loadeddata = loadeddata[
    (loadeddata["Country"] == "England") | (loadeddata["Country"] == "Scotland")
]

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


# Several sites may have the same size generators. A generator object can take a list of sites,
# and a list of the number of turbines at each site, so we need to group the sites by generator size

loadeddata["site"] = loadeddata["site"].astype(int)
uniquesites = loadeddata["site"].unique()
sitelist = list(uniquesites)
# %%
nums = [1] * len(sitelist)
# %%
# %%


totalwind = len(sitelist) * 5
totalsolar = totalwind / 2
solardatafolder = folder + "adjustedsolar/"

solarsitedata = np.loadtxt(solardatafolder + "site_locs.csv", skiprows=1, delimiter=",")
solarlistfilename = "Solarpv_operational_July_2023.xlsx"


solarlist = pd.read_excel(folder + solarlistfilename)
# extracts the 10 largest solar sites
solarlist = solarlist.nlargest(10, "Installed Capacity (MWelec)")
solarlist["site"], solarlist["Within 100Km"] = loaderfunctions.latlongtosite(
    solarlist["Latitude"], solarlist["Longitude"], windsitedata
)


solarsites = solarlist["site"].unique().astype(int).tolist()

solarcapacities = [totalsolar / len(solarsites)] * len(solarsites)


allgenerators = []  # makes an empty list to store the generator objects in
totalcapcity = 0
starttime = time.time()

selectedgenerator = generation.OffshoreWindModel5000

windgenerator = selectedgenerator(
    year_min=1980,
    year_max=2022,
    sites=sitelist,
    n_turbine=nums,
    data_path=offshorewinddatafolder,
    force_run=True,
)


solargenerator = generation.SolarModel(
    year_min=1980,
    year_max=2022,
    sites=solarsites,
    plant_capacities=solarcapacities,
    data_path=solardatafolder,
)


# %%

#####add solar section here


windpowerout = np.array(windgenerator.power_out)
solarpowerout = np.array(solargenerator.power_out)

summedpowerout = windpowerout + solarpowerout


# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

datetimelist = []
currentdatetime = datetime.datetime(1980, 1, 1, 0, 0)
index = 0
currentyear = 1980
finalyear = 2022
yearindexdict = {}
startindex = 0
while True:
    if currentdatetime.year != currentyear:
        yearindexdict[currentyear] = [startindex, index]
        currentyear += 1
        startindex = index
        if currentyear == finalyear:
            break
    currentdatetime += datetime.timedelta(hours=1)
    index += 1


runyears = yearindexdict.keys()

datayears = {}


yearlist = []
sumlist = []
windgendict = {}
solargendict = {}
for year in runyears:
    indeces = yearindexdict[year]
    datayears[year] = summedpowerout[indeces[0] : indeces[1]]
    windgendict[year] = windpowerout[indeces[0] : indeces[1]]
    solargendict[year] = solarpowerout[indeces[0] : indeces[1]]
    yearlist.append(year)
    sumlist.append(np.sum(summedpowerout[indeces[0] : indeces[1]]))


sortedindeces = np.argsort(sumlist)

lowesttenyearsindeces = sortedindeces[0:10]

lowestdict = {}

for index in lowesttenyearsindeces:
    lowestdict[yearlist[index]] = datayears[yearlist[index]]


# %%

dataoutfile = folder + "droughtreport/droughtreport.txt"

with open(dataoutfile, "w") as file:
    file.write("Drought report\n\n\n")

for year in lowestdict.keys():
    yeardata = lowestdict[year]
    yearsum = np.sum(yeardata)
    weeksums = [np.sum(yeardata[i * 7 * 24 : (i + 1) * 7 * 24]) for i in range(52)]
    windweeksums = [
        np.sum(windgendict[year][i * 7 * 24 : (i + 1) * 7 * 24]) for i in range(52)
    ]
    solarweeksums = [
        np.sum(solargendict[year][i * 7 * 24 : (i + 1) * 7 * 24]) for i in range(52)
    ]
    weekmean = np.mean(weeksums)
    weekmeanstandarddeviation = np.std(weeksums)
    weekmin = np.min(weeksums)
    weekmax = np.max(weeksums)
    totalwind = np.sum(windgendict[year])
    totalsolar = np.sum(solargendict[year])

    with open(dataoutfile, "a") as file:
        file.write("--------------\n\n")
        file.write(
            f"year:{year}\nTotal Power Generated:{np.round(yearsum,2)} MWh\nMean weekly power generated:{np.round(weekmean,2)} MWh\nStandard deviation of weekly power generated:{np.round(weekmeanstandarddeviation,2)} MWh\nMinimum weekly power generated:{np.round(weekmin,2)} MWh\nMaximum weekly power generated:{np.round(weekmax,2)} MWh\n\nTotal wind power generated:{np.round(totalwind,2)} MWh\nTotal solar power generated:{np.round(totalsolar,2)} MWh\n"
        )

    weeknumbers = [i for i in range(1, 53)]

    print(len(weeknumbers), len(weeksums))
    plt.figure(figsize=(10, 5))
    # triples the font size
    plt.rcParams.update({"font.size": 30})
    plt.plot(weeknumbers, weeksums, label="Total power generated")
    plt.plot(weeknumbers, windweeksums, label="Wind power generated")
    plt.plot(weeknumbers, solarweeksums, label="Solar power generated")
    plt.legend()
    plt.xlabel("Week number")
    plt.ylabel("Power generated (MWh)")
    plt.ylim(0, 14000)
    plt.title(f"Power generated in {year}")
    plt.savefig(folder + "droughtreport/" + str(year) + ".png", dpi=300)
    plt.show()


# %%


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
