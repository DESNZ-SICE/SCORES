"""
This script gives a simple example of forming the Linear program and solving it.
"""

# %%
from generation import (
    SolarModel,
    OnshoreWindModel4000,
    OffshoreWindModel15000,
    NuclearModel,
    DispatchableGenerator,
)
import aggregatedEVs as aggEV
from opt_con_class import System_LinProg_Model
from storage import BatteryStorageModel, HydrogenStorageModel, MultipleStorageAssets
import numpy as np
from fns import get_GB_demand
import pandas as pd
import Loaderfunctions
import matplotlib.pyplot as plt
import datetime


yearmin = 2014
yearmax = 2019
modelledyear = 2040
demand = np.loadtxt("demand.csv", usecols=2, delimiter=",", skiprows=1)


totaldemand = 524

demandstarttime = datetime.datetime(2009, 1, 1)


starttime = datetime.datetime(yearmin, 1, 1)
endtime = datetime.datetime(yearmax + 1, 1, 1)

startindex = int((starttime - demandstarttime).total_seconds() / 3600)
endindex = int((endtime - demandstarttime).total_seconds() / 3600)

numberofpoints = int((endtime - starttime).total_seconds() / 3600)
demand = demand[startindex:endindex]
numberofyears = yearmax - yearmin + 1
# scale demand so the total demand per year is 570TWh. Demand is currently in MWh

yearlydemand = totaldemand

# each year needs to be scaled seperately
currentyear = yearmin
for year in range(numberofyears):
    endofyear = datetime.datetime(yearmin + year + 1, 1, 1) - datetime.timedelta(
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


# Define the generators
generatorlocationfolder = "/Users/matt/code/transform/toptengenerators/"

# we want to use the real locations for the generators

winddatafolder = "/Volumes/macdrive/merraupdated/"
solardatafolder = "/Volumes/macdrive/updatedsolarcomb/"

windsitelocs = np.loadtxt(winddatafolder + "site_locs.csv", delimiter=",", skiprows=1)
solarsitelocs = np.loadtxt(solardatafolder + "site_locs.csv", delimiter=",", skiprows=1)

offshoredata = pd.read_excel(generatorlocationfolder + "top10offshore_modified.xlsx")
onshoredata = pd.read_excel(generatorlocationfolder + "top10onshore_modified.xlsx")
solardata = pd.read_excel(generatorlocationfolder + "top10solar_modified.xlsx")


offshoredata["site"], offshoredata["Within 100km"] = Loaderfunctions.latlongtosite(
    offshoredata["Latitude"], offshoredata["Longitude"], windsitelocs
)
onshoredata["site"], onshoredata["Within 100km"] = Loaderfunctions.latlongtosite(
    onshoredata["Latitude"], onshoredata["Longitude"], windsitelocs
)
solardata["site"], solardata["Within 100km"] = Loaderfunctions.latlongtosite(
    solardata["Latitude"], solardata["Longitude"], solarsitelocs
)

offshore_sites = offshoredata["site"].unique().astype(int).tolist()
onshore_sites = onshoredata["site"].unique().astype(int).tolist()
solar_sites = solardata["site"].unique().astype(int).tolist()
nuclear_generator = NuclearModel(
    sites=[1],
    year_min=2013,
    year_max=2019,
    data_path=winddatafolder,
    capacities=[1],
)

print(np.sum(nuclear_generator.power_out))
print("Calculating offshore output")

powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"


offshorepowercurve = np.loadtxt(
    powercurvelocation + "15_MW.csv", delimiter=",", skiprows=1
)
onshorepowercurve = np.loadtxt(
    powercurvelocation + "4_MW.csv", delimiter=",", skiprows=1
)
offshoregenerator = OffshoreWindModel15000(
    sites=offshore_sites,
    year_min=yearmin,
    year_max=yearmax,
    data_path=winddatafolder,
    n_turbine=[1] * len(offshore_sites),
    force_run=True,
    power_curve=offshorepowercurve,
    limits=[0, 100000],
)
print("Calculating onshore output")
onshoregenerator = OnshoreWindModel4000(
    sites=onshore_sites,
    year_min=yearmin,
    year_max=yearmax,
    data_path=winddatafolder,
    n_turbine=[1] * len(onshore_sites),
    force_run=True,
    power_curve=onshorepowercurve,
    limits=[0, 100000],
)
print("Calculating solar output")
solar_generator = SolarModel(
    sites=solar_sites,
    year_min=yearmin,
    year_max=yearmax,
    data_path=solardatafolder,
    plant_capacities=[1] * len(solar_sites),
    force_run=True,
    limits=[0, 120000],
)
print(solar_generator.get_load_factor())


Gasgen = DispatchableGenerator(
    gentype="Gas",
    year_min=yearmin,
    year_max=yearmax,
    capacities=[1],
    capex=1 * 10**3,
    opex=1,
    fuel_cost=1,
    carbon_cost=5,
    variable_opex=5,
)
# Gasgen = DispatchableGenerator(
#     gentype="Gas",
#     year_min=yearmin,
#     year_max=yearmax,
#     capacities=[1],
#     capex=1400 * 10**3,
#     opex=25800,
#     fuel_cost=47,
#     carbon_cost=5,
#     variable_opex=5,
# )
# %%
generators = [offshoregenerator, onshoregenerator, solar_generator]

# Define the Storage

B = BatteryStorageModel()
H = HydrogenStorageModel()
storage = [H, B]


# Define Demand
# Initialise LinProg Model
x = System_LinProg_Model(
    surplus=-demand,
    fossilLimit=0.01,
    Mult_Stor=MultipleStorageAssets(storage),
    Mult_aggEV=aggEV.MultipleAggregatedEVs([]),
    gen_list=generators,
    YearRange=[yearmin, yearmax],
    dispatchable_list=[Gasgen],
)

# get the wind time series
# Form the Linear Program Model
x.Form_Model()

# Solve the Linear Program
x.Run_Sizing(solver="gurobi")

# %%
# %%


# Store Results
x.df_capacity.to_csv("log/mixedsystemcapacitieshighh2cost.csv", index=False)
x.df_costs.to_csv("log/mixedsystemcosts.csv", index=False)
# %%


# %%
