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
yearmax = 2018
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

print(f"Peak demand is {np.max(demand)}")
# Define the generators
generatorlocationfolder = "/Users/matt/code/transform/toptengenerators/"

# we want to use the real locations for the generators

winddatafolder = "/Users/matt/SCORESdata/merraupdated/"
solardatafolder = "/Users/matt/SCORESdata/adjustedsolar/"

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
    limits=[7600, 24000],
    loadfactor=0.9,
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


hydrogencosts = {
    "low": {
        "chargeCapex": 427.32 * 10**3,
        "chargeFixedOpex": 31.61 * 10**3,
        "chargeVarOpex": 0.0017 * 10**3,
        "eff_in": 83,
    },
    "medium": {
        "chargeCapex": 530 * 10**3,
        "chargeFixedOpex": 31.36 * 10**3,
        "chargeVarOpex": 0.002926 * 10**3,
        "eff_in": 77,
    },
    "high": {
        "chargeCapex": 1610 * 10**3,
        "chargeFixedOpex": 50.13 * 10**3,
        "chargeVarOpex": 0.002977 * 10**3,
        "eff_in": 77,
    },
    "veryhigh": {
        "chargeCapex": 2396 * 10**3,
        "chargeFixedOpex": 72.24 * 10**3,
        "chargeVarOpex": 0.008 * 10**3,
        "eff_in": 70,
    },
}

gascosts = {
    "low": {
        "capex": 1400 * 10**3,
        "opex": 25.8 * 10**3,
        "fuel_cost": 44,
        "carbon_cost": 4.15,
        "variable_opex": 5,
    },
    "medium": {
        "capex": 1879 * 10**3,
        "opex": 19.68 * 10**3,
        "fuel_cost": 44,
        "carbon_cost": 4.15,
        "variable_opex": 4.01,
    },
    "high": {
        "capex": 2687 * 10**3,
        "opex": 27.33 * 10**3,
        "fuel_cost": 44,
        "carbon_cost": 4.15,
        "variable_opex": 5.8,
    },
}


hydrogenranges = ["low", "medium", "high", "veryhigh"]
gasranges = ["low", "medium", "high"]
for hydrogensensitivity in hydrogenranges:
    for gassenstivity in gasranges:
        runname = f"hydrogen{hydrogensensitivity}gas{gassenstivity}"
        offshoregenerator = OffshoreWindModel15000(
            sites=offshore_sites,
            year_min=yearmin,
            year_max=yearmax,
            data_path=winddatafolder,
            n_turbine=[1] * len(offshore_sites),
            force_run=True,
            power_curve=offshorepowercurve,
            limits=[0, 80000],
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
            limits=[0, 70000],
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
            capex=gascosts[gassenstivity]["capex"],
            opex=gascosts[gassenstivity]["opex"],
            fuel_cost=44,
            carbon_cost=4.15,
            variable_opex=gascosts[gassenstivity]["variable_opex"],
        )

        Unabatedgasgen = DispatchableGenerator(
            gentype="Unabated_Gas",
            year_min=yearmin,
            year_max=yearmax,
            capacities=[1],
            capex=600 * 10**3,
            opex=14 * 10**3,
            fuel_cost=44,
            carbon_cost=0,
            variable_opex=2,
        )

        # %%
        generators = [
            onshoregenerator,
            offshoregenerator,
            solar_generator,
            nuclear_generator,
        ]

        # Define the Storage

        B = BatteryStorageModel()
        H = HydrogenStorageModel(
            chargeCapex=hydrogencosts[hydrogensensitivity]["chargeCapex"],
            chargeFixedOpex=hydrogencosts[hydrogensensitivity]["chargeFixedOpex"],
            chargeVarOpex=hydrogencosts[hydrogensensitivity]["chargeVarOpex"],
            hurdleRate=0.065,
            eff_in=hydrogencosts[hydrogensensitivity]["eff_in"],
            max_c_rate=0.6,
            max_d_rate=1.2,
        )
        storage = [B, H]

        # Define Demand
        # Initialise LinProg Model
        print(f"Running model: {runname}")
        x = System_LinProg_Model(
            surplus=-demand,
            fossilLimit=0.00001,
            Mult_Stor=MultipleStorageAssets(storage),
            Mult_aggEV=aggEV.MultipleAggregatedEVs([]),
            gen_list=generators,
            YearRange=[yearmin, yearmax],
            dispatchable_list=[Unabatedgasgen],
            dispatchable_energy_limits=[0.05],
        )

        # get the wind time series
        # Form the Linear Program Model
        x.Form_Model()
        # print the time sizing started at
        print(datetime.datetime.now())
        # Solve the Linear Program
        x.Run_Sizing(solver="gurobi", timelimit=1000)

        # %%
        # %%

        # Store Results
        x.df_capacity.to_csv(f"log/unabatedgas/{runname}Capacities.csv", index=False)
        x.df_costs.to_csv(f"log/unabatedgas/{runname}Costs.csv", index=False)
# %%


# %%
