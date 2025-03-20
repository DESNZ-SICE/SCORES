# %%
import pandas as pd
import numpy as np
import generation
import Loaderfunctions
import datetime
import time
import os


# %%
winddatafolder = "/Users/matt/SCORESdata/merraupdated/"
windsitedata = np.loadtxt(winddatafolder + "site_locs.csv", skiprows=1, delimiter=",")
outputfolder = "/Users/matt/SCORESdata/variablererun/"

years = [2010, 2012, 2017]

onshoregens = generation.generatordictionaries().onshore
offshoregens = generation.generatordictionaries().offshore

turbtypes = ["onshore", "offshore", "floating"]
# turbtypes = ["offshore", "floating"]

gens = {
    "onshore": onshoregens,
    "offshore": offshoregens,
    "floating": offshoregens,
}
powercurvelocation = "/Users/matt/SCORESdata/"
powercurves = {
    "onshore": powercurvelocation + "genericonshorepowercurve.csv",
    "offshore": powercurvelocation + "genericoffshorepowercurve.csv",
    "floating": powercurvelocation + "genericoffshorepowercurve.csv",
}

for turbtype in turbtypes:
    loadfactorturbdict = {}
    inputfile = f"/Users/matt/code/mapplot/{turbtype}sites.csv"
    gensites = pd.read_csv(inputfile)

    gensites["site"], gensites["Within 100Km"] = Loaderfunctions.latlongtosite(
        gensites["site lat"], gensites["site long"], windsitedata
    )
    # %%
    gensites["site"] = gensites["site"].astype(int)

    for index, row in gensites.iterrows():
        print(f'zone:{row["Area"]}')
        # if row["Area"] != 7 and row["Area"] != 8 and row["Area"] != 9:
        #     continue
        dataoutputfolder = outputfolder + f"{turbtype}/" + "zone_" + str(row["Area"])
        try:
            os.makedirs(dataoutputfolder)
        except FileExistsError:
            pass
        for gensize in gens[turbtype].keys():

            genfolder = dataoutputfolder + f"/{gensize}MW/"
            powercurve = np.loadtxt(powercurves[turbtype], delimiter=",", skiprows=1)
            powercurve[:, 1] *= gensize
            try:
                os.mkdir(genfolder)
            except FileExistsError:
                pass

            for year in years:
                genobject = gens[turbtype][gensize](
                    year_min=year,
                    year_max=year,
                    sites=[row["site"]],
                    n_turbine=[1],
                    data_path=winddatafolder,
                    force_run=True,
                    power_curve=powercurve,
                )

                powerout = genobject.power_out
                poweroutarray = np.array(powerout)
                capacityfactors = poweroutarray / float(gensize)
                # round to 3 decimal places
                capacityfactors = np.round(capacityfactors, 3)
                print(
                    f"Turbtype: {turbtype}, Year: {year}, Site: {row['site']}, Capacity: {gensize}MW, Capacity factor: {np.mean(capacityfactors)}"
                )
                yearhours = np.array([i for i in range(len(capacityfactors))])
                # combines the yearhours and capacityfactors into a single array
                capacityfactors = np.column_stack((yearhours, capacityfactors))
                print(f"{gensize}MW capacity factor: {np.mean(capacityfactors[:,1])}")
                aveloadfactor = np.mean(capacityfactors[:, 1])
                if gensize not in loadfactorturbdict.keys():
                    loadfactorturbdict[gensize] = []
                loadfactorturbdict[gensize].append(aveloadfactor)

                with open(genfolder + str(year) + ".csv", "w") as file:
                    file.write("Hour, Capacity Factor\n")
                    for line in capacityfactors:
                        file.write(str(int(line[0])) + "," + str(line[1]) + "\n")

    with open(f"{turbtype}loadfactors.csv", "w") as file:
        file.write("Size,Average Load Factor, Max load factor\n")
        for size in loadfactorturbdict.keys():
            file.write(
                f"{size},{np.mean(loadfactorturbdict[size])},{np.max(loadfactorturbdict[size])}\n"
            )
