"""
This script gives a simple example of forming the Linear program and solving it.
"""

# %%
from generation import (
    SolarModel,
    OnshoreWindModel3600,
    OffshoreWindModel10000,
    NuclearModel,
)
import aggregatedEVs as aggEV
from opt_con_class import System_LinProg_Model
from storage import BatteryStorageModel, HydrogenStorageModel, MultipleStorageAssets
import numpy as np
from fns import get_GB_demand
import pandas as pd
import loaderfunctions

ymin = 2013
ymax = 2019

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


offshoredata["site"], offshoredata["Within 100km"] = loaderfunctions.latlongtosite(
    offshoredata["Latitude"], offshoredata["Longitude"], windsitelocs
)
onshoredata["site"], onshoredata["Within 100km"] = loaderfunctions.latlongtosite(
    onshoredata["Latitude"], onshoredata["Longitude"], windsitelocs
)
solardata["site"], solardata["Within 100km"] = loaderfunctions.latlongtosite(
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
offshoregenerator = OffshoreWindModel10000(
    sites=offshore_sites,
    year_min=2013,
    year_max=2019,
    data_path=winddatafolder,
    n_turbine=[1] * len(offshore_sites),
    force_run=True,
)
print("Calculating onshore output")
onshoregenerator = OnshoreWindModel3600(
    sites=onshore_sites,
    year_min=2013,
    year_max=2019,
    data_path=winddatafolder,
    n_turbine=[1] * len(onshore_sites),
    force_run=True,
)
print("Calculating solar output")
solar_generator = SolarModel(
    sites=solar_sites,
    year_min=2013,
    year_max=2019,
    data_path=solardatafolder,
    plant_capacities=[1] * len(solar_sites),
)
print(solar_generator.get_load_factor())

# %%
generators = [offshoregenerator, onshoregenerator, solar_generator, nuclear_generator]

# Define the Storage

B = BatteryStorageModel()
H = HydrogenStorageModel()
storage = [H, B]


# Define Demand
demand = np.loadtxt(
    "/Users/matt/code/demand-ninja/2013-2019demand.csv",
    delimiter=",",
    skiprows=1,
    usecols=1,
)
# Initialise LinProg Model
x = System_LinProg_Model(
    surplus=-demand,
    fossilLimit=0.01,
    Mult_Stor=MultipleStorageAssets(storage),
    Mult_aggEV=aggEV.MultipleAggregatedEVs([]),
    gen_list=generators,
    YearRange=[ymin, ymax],
)

# Form the Linear Program Model
x.Form_Model()

# Solve the Linear Program
x.Run_Sizing(solver="gurobi")

# Plot Results
start = 4000
end = 4500
x.PlotSurplus(start, end)
B.plot_timeseries(start, end)
H.plot_timeseries(start, end)

# Store Results
x.df_capital.to_csv("log/capitalnuclear.csv", index=False)
x.df_costs.to_csv("log/costsnuclear.csv", index=False)


# ##### This is an extension to show how parameters can be used to run sensitivity analysis efficiently ####
# #Sensitivity to amount of Fossil Fuels
# FosLim=[0.04,0.02,0.0]
# Cap_Record = []
# Cost_Record = []

# for b in range(len(FosLim)):
#     x.model.foss_lim_param = FosLim[b] * sum(demand)
#     x.Run_Sizing()
#     Cap_Record.append(x.df_capital)
#     Cost_Record.append(x.df_costs)

# Cap_Record1 = pd.concat(Cap_Record,ignore_index=True)
# Cost_Record1 = pd.concat(Cost_Record,ignore_index=True)

# Cost_Record1.to_csv('log/Cost_Rec_1.csv', index=False)
# Cap_Record1.to_csv('log/Cap_Rec_1.csv', index=False)

# #Sensitivity to Limit Solar
# Max_Solar = [75000,50000,25000]
# for s in range(len(Max_Solar)):
#     x.model.Gen_Limit_Param_Upper[0] = Max_Solar[s]
#     x.Run_Sizing()
#     Cap_Record.append(x.df_capital)
#     Cost_Record.append(x.df_costs)

#     FosLim.append(0.0)

# Cap_Record2 = pd.concat(Cap_Record,ignore_index=True)
# Cost_Record2 = pd.concat(Cost_Record,ignore_index=True)

# Cap_Record2['Fos Lmit (%)'] = FosLim
# Cap_Record2['Max Solar (GW)'] = [400,400,400,75,50,25]

# Cost_Record2['Fos Lmit (%)'] = FosLim
# Cost_Record2['Max Solar (GW)'] = [400,400,400,75,50,25]

# Cost_Record2.to_csv('log/Cost_Rec_2.csv', index=False)
# Cap_Record2.to_csv('log/Cap_Rec_2.csv', index=False)
