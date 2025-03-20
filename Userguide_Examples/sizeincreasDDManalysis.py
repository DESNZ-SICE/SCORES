# %%
import numpy as np
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import Loaderfunctions
import pyproj as proj
import generation
import storage
from system import ElectricitySystem
import seaborn as sns

sns.set_theme()

demand = np.loadtxt("demand.csv", usecols=2, delimiter=",", skiprows=1)


totaldemand = 352

# powercurvechoice = "Gross"
# powercurvechoice="Net"
powercurvechoice = "WorseNet"

demandstarttime = datetime.datetime(2009, 1, 1)
yearmin = 2009
yearmax = 2019


varcapacitylists = [[19 + 2 * i, 19 + 4 * i] for i in range(15)]

curtailplotlist = []
gaspoweroutplotlist = []

for onshorecapacity, offshorecapacity in varcapacitylists:

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

    print(np.max(demand))

    existingdata = pd.read_excel("/Users/matt/SCORESdata/repd-q3-oct-2024-trimmed.xlsx")
    existingonshore = existingdata[
        existingdata["Technology Type"] == "Wind Onshore"
    ].copy()
    existingonshore = existingonshore[
        existingonshore["Development Status (short)"] == "Operational"
    ]
    transformer = proj.Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

    existingonshore["Longitude"], existingonshore["Latitude"] = transformer.transform(
        existingonshore["X-coordinate"].values, existingonshore["Y-coordinate"].values
    )
    offshoredata = pd.read_excel("/Users/matt/SCORESdata/offshorewindpipeline.xlsx")

    offshoredata["Longitude"], offshoredata["Latitude"] = transformer.transform(
        offshoredata["X-coordinate"].values, offshoredata["Y-coordinate"].values
    )

    winddatafolder = "/Users/matt/SCORESdata/merraupdated/"
    windsitelocs = np.loadtxt(
        winddatafolder + "site_locs.csv", skiprows=1, delimiter=","
    )

    (
        existingonshore["site"],
        existingonshore["Within 100Km"],
    ) = Loaderfunctions.latlongtosite(
        existingonshore["Latitude"],
        existingonshore["Longitude"],
        windsitelocs,
    )

    existingonshore["site"] = existingonshore["site"].astype(int)

    (
        offshoredata["site"],
        offshoredata["Within 100Km"],
    ) = Loaderfunctions.latlongtosite(
        offshoredata["Latitude"],
        offshoredata["Longitude"],
        windsitelocs,
    )

    # round the turbine size of offshore to the nearest 1MW
    offshoredata["Comb turbine size"] = np.round(offshoredata["Comb turbine size"])

    existingoffshore = offshoredata[
        offshoredata["Estimated operational year"] == "Operational"
    ].copy()

    existingonshoresites = existingonshore["site"].unique().tolist()
    onshoresitecapacities = np.zeros(len(existingonshoresites))
    for i, site in enumerate(existingonshoresites):
        onshoresitecapacities[i] = np.sum(
            existingonshore[existingonshore["site"] == site][
                "Installed Capacity (MWelec)"
            ]
        )
    existingonshorenumberofturbines = onshoresitecapacities / 4
    existingonshorenumberofturbines = [int(i) for i in existingonshorenumberofturbines]

    powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"
    if powercurvechoice == "Gross":
        existingonshorepowercurve = np.loadtxt(
            powercurvelocation + "4_MW.csv", delimiter=",", skiprows=1
        )
    else:
        existingonshorepowercurve = 4 * np.loadtxt(
            "/Users/matt/SCORESdata/genericonshorepowercurve.csv",
            delimiter=",",
            skiprows=1,
        )

    totalinstalledcapacity = np.sum(onshoresitecapacities)
    print(totalinstalledcapacity)
    existingonshoregenerator = generation.OnshoreWindModel4000(
        sites=existingonshoresites,
        year_min=yearmin,
        year_max=yearmax,
        data_path=winddatafolder,
        n_turbine=existingonshorenumberofturbines,
        force_run=True,
        power_curve=existingonshorepowercurve,
    )

    futureonshoreinstall = onshorecapacity * 10**3 - totalinstalledcapacity
    print(futureonshoreinstall)

    scalesize = futureonshoreinstall / totalinstalledcapacity
    scaledcapacities = onshoresitecapacities * scalesize
    scalednumberofturbines = scaledcapacities / 7
    scalednumberofturbines = [int(i) for i in scalednumberofturbines]

    if powercurvechoice == "Gross":
        futureonshorepowercurve = np.loadtxt(
            powercurvelocation + "7_MW.csv", delimiter=",", skiprows=1
        )
    else:
        futureonshorepowercurve = 7 * np.loadtxt(
            "/Users/matt/SCORESdata/genericonshorepowercurve.csv",
            delimiter=",",
            skiprows=1,
        )

    futureonshoregenerator = generation.OnshoreWindModel7000(
        sites=existingonshoresites,
        year_min=yearmin,
        year_max=yearmax,
        data_path=winddatafolder,
        n_turbine=scalednumberofturbines,
        force_run=True,
        power_curve=futureonshorepowercurve,
    )

    generatordict = generation.generatordictionaries().offshore

    existingoffshoresizes = existingoffshore["Comb turbine size"].unique().tolist()
    existingoffshoregenerators = []
    for turbsize in existingoffshoresizes:
        thisturbinsizedata = existingoffshore[
            existingoffshore["Comb turbine size"] == turbsize
        ]
        sites = thisturbinsizedata["site"].unique().tolist()
        sites = [int(i) for i in sites]
        capacities = np.zeros(len(sites))
        for i, site in enumerate(sites):
            capacities[i] = np.sum(
                thisturbinsizedata[thisturbinsizedata["site"] == site][
                    "Installed Capacity (MWelec)"
                ]
            )
        numberofturbines = capacities / turbsize
        numberofturbines = [int(i) for i in numberofturbines]

        if powercurvechoice == "Gross":
            powercurve = np.loadtxt(
                powercurvelocation + f"{int(turbsize)}_MW.csv",
                delimiter=",",
                skiprows=1,
            )
        elif powercurvechoice == "Net":
            powercurve = turbsize * np.loadtxt(
                "/Users/matt/SCORESdata/genericoffshorepowercurve.csv",
                delimiter=",",
                skiprows=1,
            )
        elif powercurvechoice == "WorseNet":
            powercurve = turbsize * np.loadtxt(
                "/Users/matt/SCORESdata/worsegenericoffshorepowercurve.csv",
                delimiter=",",
                skiprows=1,
            )

        generatorobject = generatordict[turbsize]
        thissizegenerator = generatorobject(
            sites=sites,
            year_min=yearmin,
            year_max=yearmax,
            data_path=winddatafolder,
            n_turbine=numberofturbines,
            force_run=True,
            power_curve=powercurve,
        )
        existingoffshoregenerators.append(thissizegenerator)

    futureoffshore = offshoredata[
        offshoredata["Estimated operational year"] != "Operational"
    ].copy()

    existingoffshorecapacity = np.sum(
        [i.total_installed_capacity for i in existingoffshoregenerators]
    )

    futurecapacity = np.sum(futureoffshore["Installed Capacity (MWelec)"])
    requiredfuturecapacity = offshorecapacity * 10**3 - existingoffshorecapacity
    scalesize = requiredfuturecapacity / futurecapacity

    futureoffshoresizes = futureoffshore["Comb turbine size"].unique().tolist()
    futureoffshoregenerators = []

    for turbsize in futureoffshoresizes:
        thisturbinsizedata = futureoffshore[
            futureoffshore["Comb turbine size"] == turbsize
        ]
        sites = thisturbinsizedata["site"].unique().tolist()
        sites = [int(i) for i in sites]

        capacities = np.zeros(len(sites))
        for i, site in enumerate(sites):
            capacities[i] = np.sum(
                thisturbinsizedata[thisturbinsizedata["site"] == site][
                    "Installed Capacity (MWelec)"
                ]
            )
        capacities *= scalesize
        numberofturbines = capacities / turbsize
        numberofturbines = [int(i) for i in numberofturbines]

        if powercurvechoice == "Gross":
            powercurve = np.loadtxt(
                powercurvelocation + f"{int(turbsize)}_MW.csv",
                delimiter=",",
                skiprows=1,
            )
        elif powercurvechoice == "Net":
            powercurve = turbsize * np.loadtxt(
                "/Users/matt/SCORESdata/genericoffshorepowercurve.csv",
                delimiter=",",
                skiprows=1,
            )
        elif powercurvechoice == "WorseNet":
            powercurve = turbsize * np.loadtxt(
                "/Users/matt/SCORESdata/worsegenericoffshorepowercurve.csv",
                delimiter=",",
                skiprows=1,
            )

        generatorobject = generatordict[turbsize]
        thissizegenerator = generatorobject(
            sites=sites,
            year_min=yearmin,
            year_max=yearmax,
            data_path=winddatafolder,
            n_turbine=numberofturbines,
            force_run=True,
            power_curve=powercurve,
        )
        futureoffshoregenerators.append(thissizegenerator)

    solardatapath = "/Users/matt/SCORESdata/adjustedsolar/"

    solardata = pd.read_excel(
        "/Users/matt/code/transform/toptengenerators/top10solar_modified.xlsx"
    )

    solardata["site"], solardata["Within 100Km"] = Loaderfunctions.latlongtosite(
        solardata["Latitude"],
        solardata["Longitude"],
        np.loadtxt(f"{solardatapath}site_locs.csv", skiprows=1, delimiter=","),
    )

    solarcapacity = 45
    solardata["site"] = solardata["site"].astype(int)
    solarsites = solardata["site"].unique()

    solarcaps = [solarcapacity * 10**3 / len(solarsites)] * len(solarsites)

    solargenerator = generation.SolarModel(
        sites=solarsites,
        year_min=yearmin,
        year_max=yearmax,
        data_path=solardatapath,
        plant_capacities=solarcaps,
        force_run=True,
    )

    generatorlist = (
        [existingonshoregenerator, futureonshoregenerator]
        + existingoffshoregenerators
        + futureoffshoregenerators
        + [solargenerator]
    )

    nuclearinstalled = 3
    GasCCUSinstalled = 2
    H2Pinstalled = 0.1
    UnabatedGasinstalled = 32
    Biomass = 2.5
    BECCS = 0.5
    Interconnectors = 12

    lithiumstorage = storage.BatteryStorageModel(capacity=120 * 10**3)

    CCSDispatchable = generation.DispatchableGenerator(
        sites=[1],
        year_min=yearmin,
        year_max=yearmax,
        capacities=[GasCCUSinstalled * 10**3],
        gentype="GasCCS",
    )
    H2PDispatchable = generation.DispatchableGenerator(
        sites=[1],
        year_min=yearmin,
        year_max=yearmax,
        capacities=[H2Pinstalled * 10**3],
        gentype="H2P",
    )
    UnabatedGasDispatchable = generation.DispatchableGenerator(
        sites=[1],
        year_min=yearmin,
        year_max=yearmax,
        capacities=[UnabatedGasinstalled * 10**3],
        gentype="Gas",
    )
    BiomassDispatchable = generation.DispatchableGenerator(
        sites=[1],
        year_min=yearmin,
        year_max=yearmax,
        capacities=[Biomass * 10**3],
        gentype="Biomass",
    )
    BECCSDispatchable = generation.DispatchableGenerator(
        sites=[1],
        year_min=yearmin,
        year_max=yearmax,
        capacities=[BECCS * 10**3],
        gentype="BECCS",
    )
    InterconnectorDispatchable = generation.Interconnector(
        sites=[1],
        year_min=yearmin,
        year_max=yearmax,
        capacities=[Interconnectors * 10**3],
        gentype="Interconnector",
    )

    DispatchableAssetList = [
        BECCSDispatchable,
        BiomassDispatchable,
        H2PDispatchable,
        CCSDispatchable,
        InterconnectorDispatchable,
        UnabatedGasDispatchable,
    ]

    for i in range(len(futureoffshoregenerators)):
        print(
            f"Gensize:{futureoffshoregenerators[i].turbine_size}\tLoadFactor:{futureoffshoregenerators[i].get_load_factor()}"
        )
    nuclearloadfactor = 0.83
    netdemand = demand - nuclearloadfactor * nuclearinstalled * 1000
    system = ElectricitySystem(
        generatorlist,
        [lithiumstorage],
        netdemand,
        DispatchableAssetList=DispatchableAssetList,
        Interconnector=InterconnectorDispatchable,
    )
    system.update_surplus()
    reliability = system.get_reliability()
    nyears = yearmax - yearmin + 1

    unabatedgaspercent = (
        100 * np.sum(UnabatedGasDispatchable.power_out_array) / np.sum(demand)
    )
    gaspoweroutplotlist.append((unabatedgaspercent / 100) * 352)
    yearlycurtailement = system.storage.analyse_usage()[2] / (nyears * 10**6)
    curtailplotlist.append(yearlycurtailement)

totalwindcapacities = [sum(i) for i in varcapacitylists]
# %%
ymin = -2
ymax = np.max([np.max(curtailplotlist), np.max(gaspoweroutplotlist)]) + 5
sns.set_theme()
# get a seaborn colour
palette = sns.color_palette()
color = palette[0]
vlabelscolor = palette[4]
plt.plot(totalwindcapacities, curtailplotlist, label="Annual curtailment", color=color)

plt.xlabel("Total wind capacity (GW)")
plt.ylabel("Energy (TWh)")
plt.ylim([ymin, ymax])

# insert a vertical line at 90GW labeled "2030 Theoretical limit"
plt.axvline(x=90.5, color=vlabelscolor, linestyle="--")
plt.text(
    91,
    ymax - 40,
    "2030 Theoretical limit",
    rotation=90,
    verticalalignment="center",
    color=vlabelscolor,
)

plt.axvline(x=75, color=vlabelscolor, linestyle="--")
plt.text(
    76,
    ymax - 40,
    "2030 Reference Case",
    rotation=90,
    verticalalignment="center",
    color=vlabelscolor,
)

color = palette[1]
plt.plot(
    totalwindcapacities,
    gaspoweroutplotlist,
    label="Annual gas",
    color=color,
)
# add a hoirzontal line at 17.6 labelled "CP2030 target"
plt.axhline(y=17.6, color=color, linestyle="--")
plt.text(45, 18, "CP2030 target", horizontalalignment="center", color=color)
plt.legend()
plt.show()

# %%
# make a figure with two rows and one column
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
# get a seaborn colour
palette = sns.color_palette()
color = palette[0]
ax[0].plot(
    totalwindcapacities, curtailplotlist, label="Annual curtailment", color=color
)
ax[0].set_ylabel("Annual curtailment (TWh)")
ax[0].tick_params(axis="y", labelcolor=color)

color = palette[1]
ax[1].plot(
    totalwindcapacities,
    gaspoweroutplotlist,
    label="Annual gas power output",
    color=color,
)
ax[1].set_ylabel("Annual gas power output (TWh)")
ax[1].set_xlabel("Total wind capacity (GW)")
ax[1].tick_params(axis="y", labelcolor=color)
ax[0].tick_params(axis="x", labelcolor="none")
plt.show()
# %%
