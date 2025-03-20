import numpy as np

from generation import SolarModel
import datetime
import os

years = [2010, 2012, 2017]
alldatetimestrings = []
for year in years:
    datetimestrings = []
    startdatetime = datetime.datetime(year, 1, 1, 0, 0)
    currentdatetime = startdatetime
    while currentdatetime.year == year:
        datetimestrings.append(currentdatetime.strftime("%Y-%m-%d %H:%M:%S"))
        currentdatetime += datetime.timedelta(hours=1)
    alldatetimestrings.append(datetimestrings)

zonefolder = "/Users/matt/code/zonalsolar/"

outputfolder = "/Users/matt/SCORESdata/zonalfolder/"

zones = [f"zone{i}" for i in range(1, 13)]

for zone in zones:
    zonedata = np.loadtxt(zonefolder + zone + ".csv", delimiter=",", skiprows=1)
    # remove the rows with less than 0.01 MWe
    zonedata = zonedata[zonedata[:, 1] > 0.01]
    zonedata[:, 1] = zonedata[:, 1] / np.sum(zonedata[:, 1])
    selectedzones = [int(i) for i in zonedata[:, 0]]
    installecapacities = [float(i) for i in zonedata[:, 1]]
    # make a directory called zone{i}
    try:
        os.mkdir(outputfolder + zone)
    except FileExistsError:
        pass
    for index, year in enumerate(years):
        solarmodel = SolarModel(
            year_min=year,
            year_max=year,
            sites=selectedzones,
            plant_capacities=installecapacities,
            data_path="/Volumes/macdrive/updatedsolarcomb/",
            force_run=True,
        )
        print(f"Year {year} zone {zone} load factor is {solarmodel.get_load_factor()}")
        powerout = solarmodel.power_out
        powerout = np.array(powerout)
        powerout = np.round(powerout, 4)
        print(np.max(powerout))
        with open(outputfolder + zone + f"/{year}.csv", "w") as f:
            f.write("datetime, Capacity Factor\n")
            for j, power in enumerate(powerout):
                f.write(f"{alldatetimestrings[index][j]},{power}\n")
