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
import Loaderfunctions
import datetime
import time

folder = "/Users/matt/SCORESdata/"  # general folder with all data in it
offshorewinddatafolder = (
    folder + "2022offshore/"
)  # subfolder with offshore wind site data
filename = "Offshore_wind_operational_July_2023.xlsx"
loadeddata = pd.read_excel(folder + filename)
#selects only the data from England and Scotland
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
) = Loaderfunctions.latlongtosite(
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
uniquesites=loadeddata["site"].unique()
# %%
# %%

allgenerators = []  # makes an empty list to store the generator objects in
totalcapcity = 0
starttime = time.time()
for site in uniquesites:
    selectedgenerator=generation.OffshoreWindModel5000
    sites=[site]
    nturbines = [1]
    allgenerators.append(
        selectedgenerator(
            year_min=1980,
            year_max=2022,
            sites=sites,
            n_turbine=nturbines,
            data_path=offshorewinddatafolder,
            year_online=None,
            month_online=None,
            force_run=True,
        )
    )

#####add solar section here
    

total1 = 0
for entry in allgenerators:
    averageyearlypowergenerated = np.sum(entry.power_out)
    total1 += averageyearlypowergenerated


#sum data along axis here
print(f"Time elaped: {time.time()-starttime}")
print(f"Total power generated: {total1/1e6} TWh")
print(f"Total capacity: {totalcapcity/1e3} GW")
# %%
summedpowerout=np.sum(generation) #this isnt right, fix later

datetimelist=[]
currentdatetime=datetime.datetime(1980,1,1,0,0)
index=0
currentyear=1980
finalyear=2023
yearindexdict={}
startindex=0
while True:
    if currentdatetime.year!=currentyear:
        yearindexdict[currentyear]=[startindex, index]
        currentyear+=1
        startindex=index
        if currentyear==finalyear:
            break
    currentdatetime+=datetime.timedelta(hours=1)
    index+=1


runyears=yearindexdict.keys()

datayears={}


yearlist=[]
sumlist=[]
for year in runyears:
    indeces=yearindexdict[year]
    datayears[year]=summedpowerout[indeces[0]:indeces[1]]
    yearlist.append(year)
    sumlist.append(np.sum(summedpowerout[indeces[0]:indeces[1]]))


sortedindeces=np.argsort(sumlist)

lowesttenyearsindeces=sortedindeces[0:10]

lowestdict={}

for index in lowesttenyearsindeces:
    lowestdict[yearlist[index]]=datayears[yearlist[index]]



dataoutfile="droughtreport.txt"

for year in lowesttenyearsindeces.keys:
    yeardata=lowesttenyearsindeces[year]
    yearsum=np.sum(yeardata)
    weeksums=[yeardata[i*7*24:(i+1)*7*24] for i in range(52)]
    weekmean=np.mean(weeksums)
    weekmeanstandarddeviation=np.sdev(weeksums)
    weekmin=np.min(weeksums)
    weekmax=np.max(weeksums)


    with open(dataoutfile, 'a') as file:
        file.write("--------------\n\n")
        file.write("year:{year}\n")





    
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
