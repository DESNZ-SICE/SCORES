import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


dataname = "nogashydrogen10years"
capacityfile = f"log/gashydrogenrange99.999/{dataname}Capacities.csv"
costsfile = f"log/gashydrogenrange99.999/{dataname}Costs.csv"
# make a figure with 4 subplots arranged in a grid
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

# read in the data
capacities = pd.read_csv(capacityfile)
costs = pd.read_csv(costsfile)

# on the top left figure plot the capacity of all of the generators. The generator names are:
# Onshore Wind, Offshore Wind, Solar, Nuclear, Dispatchable_Gas
# the first 4 columns will be called "Gen Onshore Wind Cap (GW)", with onshore wind replaced with the other generator names
# theres only 1 row, so we can just plot the first row of the dataframe
gen_names = [
    "Gen Onshore Wind",
    "Gen Offshore Wind",
    "Gen Solar",
    "Gen Nuclear",
]

shortgen_names = ["Onshore", "Offshore", "Solar", "Nuclear"]
capacitylist = [capacities[gen + " Cap (GW)"].iloc[0] for gen in gen_names]

colors = sns.color_palette()
axs[0, 0].bar(gen_names, capacitylist, color=colors)
axs[0, 0].set_title("Generator Capacities")
axs[0, 0].set_ylabel("Capacity (GW)")
axs[0, 0].set_ylim([0, 140])
# use the short names for the x axis
axs[0, 0].set_xticklabels(shortgen_names)

# make the top right plot into 2 subplots
# the first subplot will be the size of the hydrogen store
# the second subplot will be the size of the battery store
# the columns are called "H2 Storage Cap (GWh)" and "Batt Storage Cap (GWh)"
# so we can just plot the first row of the dataframe

h2storcap = capacities["Stor Hydrogen Storage Cap (GWh)"].iloc[0] / 1000
h2discharge = capacities["Stor Hydrogen Storage Discharge (TWh)"].iloc[0]
lithiumstorcap = capacities["Stor Li-Ion Battery Cap (GWh)"].iloc[0]
lithiumdischarge = capacities["Stor Li-Ion Battery Discharge (TWh)"].iloc[0]
# on the middle left  figure plot the h2storcap and text showing the total discharge
h2limit = 100
axs[1, 0].bar(["H2 Storage"], [h2storcap])
axs[1, 0].text(0, h2limit / 2, f"Total Discharge: {round(h2discharge)} TWh")
axs[1, 0].set_title("Hydrogen Capacities")
axs[1, 0].set_ylabel("Capacity (TWh)")

axs[1, 0].set_ylim([0, h2limit])
# on the middle right figure plot the lithiumstorcap and text showing the total discharge
# lithiumlimit = 300
# axs[1, 1].bar(["Li-Ion Battery"], [lithiumstorcap], color=colors[1])
# axs[1, 1].text(0, lithiumlimit / 2, f"Total Discharge: {round(lithiumdischarge)} TWh")
# axs[1, 1].set_title("Battery Capacities")
# axs[1, 1].set_ylabel("Capacity (GWh)")

# axs[1, 1].set_ylim([0, lithiumlimit])

hydrogenlabels = ["Electrolysers", "Turbines"]
hydrogencapacities = [
    capacities["Stor Hydrogen Storage Max Charge Rate (GW)"].iloc[0],
    capacities["Stor Hydrogen Storage Max Discharge Rate (GW)"].iloc[0],
]

axs[1, 1].bar(hydrogenlabels, hydrogencapacities, color=colors)
axs[1, 1].set_title("Hydrogen Storage Rates")
axs[1, 1].set_ylabel("Rate (GW)")
axs[1, 1].set_ylim([0, 120])

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
numberofyears = 10
# on the top right figure, plot the total energy demand and the total curtailment
totaldemand = capacities["Total Demand (GWh)"].iloc[0] / (1000 * numberofyears)
totalcurtailment = capacities["Total Curtailment (GWh)"].iloc[0] / (
    1000 * numberofyears
)

labels = ["Demand", "Curtailment"]
values = [totaldemand, totalcurtailment]
colors = sns.color_palette("husl", len(labels))
axs[0, 1].bar(labels, values, color=colors)
axs[0, 1].set_title("Yearly average Demand and Curtailment")
axs[0, 1].set_ylabel("Energy (TWh)/yr")
axs[0, 1].set_ylim([0, 550])


# on the bottom left plot the costs of the system. The costs are in two components, Capital costs and operation costs. The columns are called "Gen name Capital (£m/yr)" and
# "Gen name Operation (£m/yr)" where name is the generator name. Plot a stacked bar chart with the capital costs on the bottom and the operation costs on top
# the x axis should be the generator names

stornames = [
    "Stor Hydrogen Storage Storage",
    "Stor Hydrogen Storage Charge",
    "Stor Hydrogen Storage Discharge",
]
storshortnames = ["H2 Caverns", "Electrolysers", "Turbines"]

gencapitalcosts = [costs[gen + " Capital (£m/yr)"].iloc[0] / 1000 for gen in gen_names]
genopcosts = [costs[gen + " Operation (£m/yr)"].iloc[0] / 1000 for gen in gen_names]

storcapitalcosts = [
    costs[stor + " Capital (£m/yr)"].iloc[0] / 1000 for stor in stornames
]
storopcosts = [costs[stor + " Operation (£m/yr)"].iloc[0] / 1000 for stor in stornames]

combinedcapitalcosts = gencapitalcosts + storcapitalcosts
combinedopcosts = genopcosts + storopcosts

combinednames = gen_names + stornames

combinedshortnames = shortgen_names + storshortnames

axs[2, 0].bar(
    combinednames, combinedcapitalcosts, color=colors[0], label="Capital Costs"
)
axs[2, 0].bar(
    combinednames,
    combinedopcosts,
    color=colors[1],
    label="Operation Costs",
    bottom=combinedcapitalcosts,
)
axs[2, 0].set_title("All Costs")
axs[2, 0].set_ylabel("Cost (£bn/yr)")
# use the short names for the x axis
axs[2, 0].set_xticklabels(combinedshortnames)
axs[2, 0].legend()
axs[2, 0].set_ylim([0, 20])
# the labels cant be read, so we will rotate them
axs[2, 0].tick_params(axis="x", rotation=45)
# on the bottom right plot the total costs of the system
totalcapitalcost = costs["Total Capital (£m/yr)"].iloc[0] / 1000
totalopcost = costs["Total Operation (£m/yr)"].iloc[0] / 1000
# use a stacked bar

axs[2, 1].bar(
    ["Total Cost"], [totalcapitalcost], color=colors[0], label="Capital Costs"
)
axs[2, 1].bar(
    ["Total Cost"],
    [totalopcost],
    color=colors[1],
    label="Operation Costs",
    bottom=totalcapitalcost,
)
axs[2, 1].set_title("Total Costs")
axs[2, 1].set_ylabel("Cost (£bn/yr)")
axs[2, 1].legend()
axs[2, 1].set_ylim([0, 75])
totalcost = totalcapitalcost + totalopcost
costpermwh = totalcost * 10**9 / (totaldemand * 10**6)
print(f"Cost per MWh: {costpermwh}")
costpermwh = round(costpermwh, 2)
fig.suptitle(f"{dataname}, £/MWh: {costpermwh}")

plt.show()
