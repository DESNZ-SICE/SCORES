import numpy as np

import generation
import Loaderfunctions
import datetime
import os

years = [2010, 2012, 2017]
onshoreoutputfolder = "/Users/matt/SCORESdata/bid3hourlyloadfactors/onshore/"
optchoices = {"base": "generic", "optimistic": "opt"}
zones = [f"zone{i}" for i in range(1, 13)]
zonaldatafolder = "/Users/matt/code/zonalwind/"
offshoreturbinesizes = [
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
]
onshoreturbinesizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
bid3sites = np.loadtxt(
    "/Users/matt/code/mapplot/bid3selectedsites.csv",
    delimiter=",",
    skiprows=1,
    usecols=(5, 6),
)
for optchoice in optchoices.keys():
    print(f"Running {optchoice}")
    try:
        os.mkdir(onshoreoutputfolder + optchoice)
    except FileExistsError:
        pass
    parentfolderpath = onshoreoutputfolder + optchoice
    for year in years:
        print(f"Running year {year}")
        try:
            os.mkdir(parentfolderpath + f"/{year}")
        except FileExistsError:
            pass
        yearfolderpath = parentfolderpath + f"/{year}"
        yeardatetimestrings = []
        startdatetime = datetime.datetime(year, 1, 1, 0, 0)
        currentdatetime = startdatetime
        while currentdatetime.year == year:
            yeardatetimestrings.append(currentdatetime.strftime("%Y-%m-%d %H:%M:%S"))
            currentdatetime += datetime.timedelta(hours=1)

        winddatafolder = f"/Users/matt/SCORESdata/era{year}/"
        for zone in zones:
            try:
                os.mkdir(yearfolderpath + f"/{zone}")
            except FileExistsError:
                pass
            zonefolderpath = yearfolderpath + f"/{zone}"
            zonedata = np.loadtxt(
                f"{zonaldatafolder}{zone}.csv", delimiter=",", skiprows=1
            )
            mininstalled = np.min(zonedata[:, 1])
            nturbines = zonedata[:, 1] / mininstalled

            nturbines = [int(i) for i in nturbines]
            sites = [int(i) for i in zonedata[:, 0]]

            for gensize in onshoreturbinesizes:
                # load the power curve
                powercurve = np.loadtxt(
                    f"/Users/matt/code/Processing-toolkit/{optchoices[optchoice]}onshorepowercurve.csv",
                    delimiter=",",
                    skiprows=1,
                )
                powercurve *= gensize
                # get the generator object
                genobject = generation.generatordictionaries().onshore[gensize]
                # run the generator
                datarun = genobject(
                    sites=sites,
                    year_min=year,
                    year_max=year,
                    data_path=winddatafolder,
                    n_turbine=nturbines,
                    force_run=True,
                    power_curve=powercurve,
                )
                powerout = datarun.power_out_array
                loadfactorarray = powerout / (np.sum(nturbines) * gensize)

                with open(zonefolderpath + f"/{gensize}_MW.csv", "w") as f:
                    f.write("datetime,loadfactorarray\n")
                    for j, power in enumerate(loadfactorarray):
                        f.write(f"{yeardatetimestrings[j]},{round(power,2)}\n")

offshoreoutputfolder = "/Users/matt/SCORESdata/bid3hourlyloadfactors/offshore/"
print("Running offshore")
for optchoice in optchoices.keys():
    print(f"Running {optchoice}")
    try:
        os.mkdir(offshoreoutputfolder + optchoice)
    except FileExistsError:
        pass
    parentfolderpath = offshoreoutputfolder + optchoice
    for year in years:
        print(f"Running year {year}")
        try:
            os.mkdir(parentfolderpath + f"/{year}")
        except FileExistsError:
            pass
        yearfolderpath = parentfolderpath + f"/{year}"
        yeardatetimestrings = []
        startdatetime = datetime.datetime(year, 1, 1, 0, 0)
        currentdatetime = startdatetime
        while currentdatetime.year == year:
            yeardatetimestrings.append(currentdatetime.strftime("%Y-%m-%d %H:%M:%S"))
            currentdatetime += datetime.timedelta(hours=1)

        winddatafolder = f"/Users/matt/SCORESdata/era{year}/"
        windsitelocs = np.loadtxt(
            f"{winddatafolder}site_locs.csv", delimiter=",", skiprows=1
        )
        for index, zone in enumerate(zones):
            print(f"Running zone {zone}")
            try:
                os.mkdir(yearfolderpath + f"/{zone}")
            except FileExistsError:
                pass
            zonefolderpath = yearfolderpath + f"/{zone}"
            lat, lon = bid3sites[index]
            site, within100km = Loaderfunctions.latlongtosite(
                [lat], [lon], windsitelocs
            )
            sites = [int(site[0])]
            nturbines = [1]

            for gensize in offshoreturbinesizes:
                # load the power curve
                powercurve = np.loadtxt(
                    f"/Users/matt/code/Processing-toolkit/{optchoices[optchoice]}offshorepowercurve.csv",
                    delimiter=",",
                    skiprows=1,
                )
                powercurve *= gensize
                # get the generator object
                genobject = generation.generatordictionaries().offshore[gensize]
                # run the generator
                datarun = genobject(
                    sites=sites,
                    year_min=year,
                    year_max=year,
                    data_path=winddatafolder,
                    n_turbine=nturbines,
                    force_run=True,
                    power_curve=powercurve,
                )
                powerout = datarun.power_out_array
                loadfactorarray = powerout / (np.sum(nturbines) * gensize)

                with open(zonefolderpath + f"/{gensize}_MW.csv", "w") as f:
                    f.write("datetime,loadfactorarray\n")
                    for j, power in enumerate(loadfactorarray):
                        f.write(f"{yeardatetimestrings[j]},{round(power,2)}\n")

floatingoutputfolder = "/Users/matt/SCORESdata/bid3hourlyloadfactors/floating/"
print("Running floating")
floatingzones = ["1", "2", "3"]
floatingsites = np.loadtxt(
    "/Users/matt/code/mapplot/floatingsites.csv",
    delimiter=",",
    skiprows=1,
    usecols=(1, 2),
)
for optchoice in optchoices.keys():
    print(f"Running {optchoice}")
    try:
        os.mkdir(floatingoutputfolder + optchoice)
    except FileExistsError:
        pass
    parentfolderpath = floatingoutputfolder + optchoice
    for year in years:
        print(f"Running year {year}")
        try:
            os.mkdir(parentfolderpath + f"/{year}")
        except FileExistsError:
            pass
        yearfolderpath = parentfolderpath + f"/{year}"
        yeardatetimestrings = []
        startdatetime = datetime.datetime(year, 1, 1, 0, 0)
        currentdatetime = startdatetime
        while currentdatetime.year == year:
            yeardatetimestrings.append(currentdatetime.strftime("%Y-%m-%d %H:%M:%S"))
            currentdatetime += datetime.timedelta(hours=1)

        winddatafolder = f"/Users/matt/SCORESdata/era{year}/"
        windsitelocs = np.loadtxt(
            f"{winddatafolder}site_locs.csv", delimiter=",", skiprows=1
        )
        for index, zone in enumerate(floatingzones):
            print(f"Running zone {zone}")
            try:
                os.mkdir(yearfolderpath + f"/{zone}")
            except FileExistsError:
                pass
            zonefolderpath = yearfolderpath + f"/{zone}"
            lat, lon = floatingsites[index]
            site, within100km = Loaderfunctions.latlongtosite(
                [lat], [lon], windsitelocs
            )
            sites = [int(site[0])]
            nturbines = [1]

            for gensize in offshoreturbinesizes:
                # load the power curve
                powercurve = np.loadtxt(
                    f"/Users/matt/code/Processing-toolkit/{optchoices[optchoice]}offshorepowercurve.csv",
                    delimiter=",",
                    skiprows=1,
                )
                powercurve *= gensize
                # get the generator object
                genobject = generation.generatordictionaries().offshore[gensize]
                # run the generator
                datarun = genobject(
                    sites=sites,
                    year_min=year,
                    year_max=year,
                    data_path=winddatafolder,
                    n_turbine=nturbines,
                    force_run=True,
                    power_curve=powercurve,
                )
                powerout = datarun.power_out_array
                loadfactorarray = powerout / (np.sum(nturbines) * gensize)

                with open(zonefolderpath + f"/{gensize}_MW.csv", "w") as f:
                    f.write("datetime,loadfactorarray\n")
                    for j, power in enumerate(loadfactorarray):
                        f.write(f"{yeardatetimestrings[j]},{round(power,2)}\n")
