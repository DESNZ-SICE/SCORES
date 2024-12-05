from generation import (
    SolarModel,
    OffshoreWindModel15000,
    OnshoreWindModel8000,
    NuclearModel,
)
import datetime
from Loaderfunctions import latlongtosite
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from storage import HydrogenStorageModel, MultipleStorageAssets, BatteryStorageModel
from system import ElectricitySystem

# iport scipy packages for linear fit
from scipy import stats

# %%
# %%
# %%

solardata = pd.read_excel(
    "/Users/matt/code/transform/toptengenerators/top10solar_modified.xlsx"
)
onshoredata = pd.read_excel(
    "/Users/matt/code/transform/toptengenerators/top10onshore_modified.xlsx"
)
offshoredata = pd.read_excel(
    "/Users/matt/code/transform/toptengenerators/top10offshore_modified.xlsx"
)

solardatapath = "/Volumes/macdrive/updatedsolarcomb/"
winddatapath = "/Volumes/macdrive/merraupdated/"

solardata["site"], solardata["Within 100Km"] = latlongtosite(
    solardata["Latitude"],
    solardata["Longitude"],
    np.loadtxt(f"{solardatapath}site_locs.csv", skiprows=1, delimiter=","),
)
onshoredata["site"], onshoredata["Within 100Km"] = latlongtosite(
    onshoredata["Latitude"],
    onshoredata["Longitude"],
    np.loadtxt(f"{winddatapath}site_locs.csv", skiprows=1, delimiter=","),
)
offshoredata["site"], offshoredata["Within 100Km"] = latlongtosite(
    offshoredata["Latitude"],
    offshoredata["Longitude"],
    np.loadtxt(f"{winddatapath}site_locs.csv", skiprows=1, delimiter=","),
)

solardata["site"] = solardata["site"].astype(int)
onshoredata["site"] = onshoredata["site"].astype(int)
offshoredata["site"] = offshoredata["site"].astype(int)


# %%


powercurvelocation = "/Users/matt/SCORESdata/DNV power curves/CSV/"
onshorepowercurve = np.loadtxt(
    powercurvelocation + f"8_MW.csv", delimiter=",", skiprows=1
)

offshorepowercurve = np.loadtxt(
    powercurvelocation + f"15_MW.csv", delimiter=",", skiprows=1
)

onshoreloadfactors = []
offshoreloadfactors = []
solarloadfactors = []

startyear = 1980
endyear = 2023

yearmin = 1980
yearmax = 2023

solobj = SolarModel(
    sites=solardata["site"].values.tolist(),
    year_min=yearmin,
    year_max=yearmax,
    data_path=solardatapath,
    plant_capacities=len(solardata) * [1],
    force_run=True,
)
solarcapacities = sum(len(solardata) * [1])
solarpoweroutput = solobj.power_out_array
onshoreobj = OnshoreWindModel8000(
    sites=onshoredata["site"].values.tolist(),
    year_min=yearmin,
    year_max=yearmax,
    data_path=winddatapath,
    n_turbine=[1] * len(onshoredata),
    force_run=True,
    power_curve=onshorepowercurve,
)
onshorecapcities = len(onshoredata) * 8
onshorepoweroutput = onshoreobj.power_out_array
offshoreobj = OffshoreWindModel15000(
    sites=offshoredata["site"].values.tolist(),
    year_min=yearmin,
    year_max=yearmax,
    data_path=winddatapath,
    n_turbine=[1] * len(offshoredata),
    force_run=True,
    power_curve=offshorepowercurve,
)
offshorecapacities = len(offshoredata) * 15
offshorepoweroutput = offshoreobj.power_out_array


# %%
incrementsize = [i + 1 for i in range(0, 15)]
startpoint = int(365.25 * 7.5 * 24 + 1)
studiedyears = 28
onshoreincrementstds = []
offshoreincrementstds = []
solarincrementstds = []
for increment in incrementsize:
    solaraveraged = []
    onshoreaveraged = []
    offshoreaveraged = []
    for i in range(0, studiedyears):
        solaraveraged.append(
            np.mean(
                solarpoweroutput[
                    int(
                        (startpoint + i * 365.25 * 24) - (365.24 * 24 * increment / 2)
                    ) : int(
                        startpoint + i * 365.25 * 24 + (365.24 * 24 * increment / 2)
                    )
                ]
                / solarcapacities
            )
        )
        onshoreaveraged.append(
            np.mean(
                onshorepoweroutput[
                    int(
                        (startpoint + i * 365.25 * 24) - (365.24 * 24 * increment / 2)
                    ) : int(
                        startpoint + i * 365.25 * 24 + (365.24 * 24 * increment / 2)
                    )
                ]
                / onshorecapcities
            )
        )
        offshoreaveraged.append(
            np.mean(
                offshorepoweroutput[
                    int(
                        (startpoint + i * 365.25 * 24) - (365.24 * 24 * increment / 2)
                    ) : int(
                        startpoint + i * 365.25 * 24 + (365.24 * 24 * increment / 2)
                    )
                ]
                / offshorecapacities
            )
        )

    solaraveraged = np.array(solaraveraged)
    onshoreaveraged = np.array(onshoreaveraged)
    offshoreaveraged = np.array(offshoreaveraged)

    solaraveraged = solaraveraged / np.mean(solaraveraged)
    onshoreaveraged = onshoreaveraged / np.mean(onshoreaveraged)
    offshoreaveraged = offshoreaveraged / np.mean(offshoreaveraged)

    solarincrementstds.append(np.std(solaraveraged))
    onshoreincrementstds.append(np.std(onshoreaveraged))
    offshoreincrementstds.append(np.std(offshoreaveraged))

with open("variableaveraging.csv", "w") as file:
    file.write(",Years over which generation data is averaged\n")
    file.write(f"Technology,{','.join([str(i) for i in incrementsize])}\n")
    file.write(f"Onshore,{','.join([str(round(i,3)) for i in onshoreincrementstds])}\n")
    file.write(
        f"Offshore,{','.join([str(round(i,3)) for i in offshoreincrementstds])}\n"
    )
    file.write(f"Solar,{','.join([str(round(i,3)) for i in solarincrementstds])}\n")
plt.plot(incrementsize, onshoreincrementstds, label="Onshore")
plt.plot(incrementsize, offshoreincrementstds, label="Offshore")
plt.plot(incrementsize, solarincrementstds, label="Solar")
plt.xlabel("Years averaged over")
plt.ylabel("Standard Deviation")
plt.legend()
plt.show()
# %%


# %%
meanonshore = np.mean(onshoreloadfactors)
meannormedonshore = [x / meanonshore for x in onshoreloadfactors]

meanoffshore = np.mean(offshoreloadfactors)
meannormedoffshore = [x / meanoffshore for x in offshoreloadfactors]

meansolar = np.mean(solarloadfactors)
meannormedsolar = [x / meansolar for x in solarloadfactors]

# work out the standard deviation of the normed load factors for each technology
# %%
stdonshore = np.std(meannormedonshore)
stdoffshore = np.std(meannormedoffshore)
stdsolar = np.std(meannormedsolar)

# round the standard deviations to 2 decimal places
stdonshore = round(stdonshore, 2)
stdoffshore = round(stdoffshore, 2)
stdsolar = round(stdsolar, 2)

# do a linear fit for each of the load factors
slopeonshore, interceptonshore, r_valueonshore, p_valueonshore, std_erronshore = (
    stats.linregress(range(startyear, endyear), meannormedonshore)
)
slopeoffshore, interceptoffshore, r_valueoffshore, p_valueoffshore, std_erroffshore = (
    stats.linregress(range(startyear, endyear), meannormedoffshore)
)
slopesolar, interceptsolar, r_valuesolar, p_valuesolar, std_errsolar = stats.linregress(
    range(startyear, endyear), meannormedsolar
)


# %%
# create 3 plots side by side
fig, axs = plt.subplots(1, 3, figsize=(10, 5))
axs[0].scatter(
    range(startyear, endyear), meannormedonshore, label="Onshore", color="teal"
)
# set the title of the first plot
# plot the fit line
axs[0].plot(
    range(startyear, endyear),
    [slopeonshore * x + interceptonshore for x in range(startyear, endyear)],
    color="slategray",
    linestyle="--",
)
axs[0].set_title("Onshore Wind: std dev = " + str(stdonshore))
axs[0].set_ylim(0.80, 1.16)
# add a y label
axs[0].set_ylabel("Normalised Load Factor: 1 = mean")
# add text to the bottom left of the plot with the r value rounded to 2 decimal places
axs[0].text(
    0.05,
    0.05,
    "R Value = " + str(round(r_valueonshore, 2)),
    transform=axs[0].transAxes,
)

axs[1].scatter(
    range(startyear, endyear), meannormedoffshore, label="Offshore", color="coral"
)
axs[1].set_title("Offshore Wind: std dev = " + str(stdoffshore))
axs[1].set_ylim(0.80, 1.16)

# plot the fit line
axs[1].plot(
    range(startyear, endyear),
    [slopeoffshore * x + interceptoffshore for x in range(startyear, endyear)],
    color="slategray",
    linestyle="--",
)
# add text to the bottom left of the plot with the r squared value rounded to 2 decimal places
axs[1].text(
    0.05,
    0.05,
    "R Value = " + str(round(r_valueoffshore, 2)),
    transform=axs[1].transAxes,
)

axs[2].scatter(range(startyear, endyear), meannormedsolar, label="Solar", color="gold")
axs[2].set_title("Solar: std dev = " + str(stdsolar))
axs[2].set_ylim(0.80, 1.16)
# plot the fit line
axs[2].plot(
    range(startyear, endyear),
    [slopesolar * x + interceptsolar for x in range(startyear, endyear)],
    color="slategray",
    linestyle="--",
)
# add text to the bottom left of the plot with the r squared value rounded to 2 decimal places
axs[2].text(
    0.05,
    0.05,
    "R Value = " + str(round(r_valuesolar, 2)),
    transform=axs[2].transAxes,
)


plt.show()
# %%
