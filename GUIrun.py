import tkinter.filedialog
import tkinter as tk
import glob
import generation
import loaderfunctions
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import os


class GUIRun:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("SCORES run")
        self.solardatafolder = "C:/Users/SA0011/Documents/data/adjustedsolar/"
        self.winddatafolder = "C:/Users/SA0011/Documents/data/merraupdated/"
        self.yearoptions = [1980 + i for i in range(44)]
        self.inputfiles = glob.glob("*.xlsx")
        self.outputfolder = "output/"
        self.create_widgets()

    def create_widgets(self):
        fileframe = tk.Frame(self.window)
        fileframe.pack()

        inputframe = tk.Frame(fileframe)
        inputframe.pack(side="left")
        inputfilelabel = tk.Label(inputframe, text="Select input file")
        inputfilelabel.pack()
        self.inputfile = tk.StringVar(inputframe)
        self.inputfile.set(self.inputfiles[0])
        inputfilemenu = tk.OptionMenu(inputframe, self.inputfile, *self.inputfiles)
        inputfilemenu.pack()

        outputframe = tk.Frame(fileframe)
        outputframe.pack(side="right")
        # outputfolderlabel = tk.Label(outputframe, text="Type output folder name")
        # outputfolderlabel.pack()
        # self.outputfolder = tk.StringVar(outputframe)
        # self.outputfolder.set("output_folder")
        # outputfolderentry = tk.Entry(outputframe, textvariable=self.outputfolder)
        # outputfolderentry.pack()

        # use tkinter.filedialog to ask the user to select the output directory
        outputfolderlabel = tk.Label(outputframe, text="Select output folder")
        outputfolderlabel.pack()
        self.outputfolder = tk.StringVar(outputframe)
        outputfolderbutton = tk.Button(
            outputframe, text="Click to select", command=self.select_output_folder
        )
        outputfolderbutton.pack()

        # displays the current value of self.outputfolder
        self.outputfolderdisplay = tk.Text(outputframe, height=1, width=30)
        self.outputfolderdisplay.pack()

        startyearlabel = tk.Label(self.window, text="Start year")
        startyearlabel.pack()
        self.startyear = tk.StringVar(self.window)
        self.startyear.set(self.yearoptions[0])
        startyearmenu = tk.OptionMenu(self.window, self.startyear, *self.yearoptions)
        startyearmenu.pack()

        endyearlabel = tk.Label(self.window, text="End year")
        endyearlabel.pack()
        self.endyear = tk.StringVar(self.window)
        self.endyear.set(self.yearoptions[-1])
        endyearmenu = tk.OptionMenu(self.window, self.endyear, *self.yearoptions)
        endyearmenu.pack()

        outputlabel = tk.Label(self.window, text="Select your outputs")
        outputlabel.pack()

        self.outputs = [
            "Generation Time Series",
            "Load Factors",
            "Report",
            "Generation Figures",
        ]
        self.outputvars = [tk.IntVar() for _ in range(len(self.outputs))]
        self.outputcheckboxes = [
            tk.Checkbutton(
                self.window, text=self.outputs[i], variable=self.outputvars[i]
            )
            for i in range(len(self.outputs))
        ]
        for i in range(len(self.outputs)):
            self.outputcheckboxes[i].pack()

        buttonframe = tk.Frame(self.window)
        buttonframe.pack()

        runbutton = tk.Button(
            buttonframe,
            text="Run",
            command=self.run,
            width=int(self.window.winfo_width() / 2),
        )
        runbutton.pack(side="left")

        quitbutton = tk.Button(
            buttonframe,
            text="Quit",
            command=self.window.quit,
            width=int(self.window.winfo_width() / 2),
        )
        quitbutton.pack(side="right")

    def select_output_folder(self):
        self.outputfolder = tk.filedialog.askdirectory()
        # updates the text of the label to display the new value of self.outputfolder
        self.outputfolderdisplay.delete(1.0, tk.END)
        self.outputfolderdisplay.insert(tk.END, self.outputfolder)

    def run(self):
        print("Running")
        # get inputs
        inputfile = self.inputfile.get()
        print(inputfile)
        outputfolder = self.outputfolder
        # create output folder
        if not os.path.exists(outputfolder):
            os.makedirs(outputfolder)
        startyear = int(self.startyear.get())
        endyear = int(self.endyear.get())
        outputs = [self.outputvars[i].get() for i in range(len(self.outputs))]
        Gentimeseries = bool(outputs[0])
        Loadfactors = bool(outputs[1])
        Report = bool(outputs[2])
        Genfigures = bool(outputs[3])

        if startyear > endyear:
            raise ValueError("Start year must be less than end year")

        inputdata = pd.read_excel(inputfile)
        inputdata["OperationalDatetime"] = pd.to_datetime(
            inputdata["Operational"], format="%d/%m/%Y"
        )
        solarentries = inputdata[inputdata["Generation Type"] == "Solar"].copy()
        offshoreentries = inputdata[inputdata["Generation Type"] == "Offshore"].copy()
        onshoreentries = inputdata[inputdata["Generation Type"] == "Onshore"].copy()
        generationobjects = {}
        if len(solarentries) > 0:
            print(f"Running {len(solarentries)} solar sites")
            solarsitelocs = np.loadtxt(
                self.solardatafolder + "site_locs.csv", skiprows=1, delimiter=","
            )

            solarentries["site"], solarentries["Within 100Km"] = (
                loaderfunctions.latlongtosite(
                    solarentries["Latitude"], solarentries["Longitude"], solarsitelocs
                )
            )
            solarentries["site"] = solarentries["site"].astype(int)
            # gets a list of the solar sites
            solarsitelist = solarentries["site"].tolist()
            solardatetimes = solarentries["OperationalDatetime"].tolist()
            solarcapacities = solarentries["Installed Capacity (MW)"].tolist()
            years_online = [i.year for i in solardatetimes]
            months_online = [i.month for i in solardatetimes]

            generationobjects["Solar"] = generation.SolarModel(
                year_min=startyear,
                year_max=endyear,
                sites=solarsitelist,
                plant_capacities=solarcapacities,
                year_online=years_online,
                month_online=months_online,
                data_path=self.solardatafolder,
            )

        if len(offshoreentries) > 0:
            print(f"Running {len(offshoreentries)} offshore sites")
            gendict = generation.generatordictionaries().offshore
            generatorkeys = np.array(list(gendict.keys()))
            winddata = np.loadtxt(
                self.winddatafolder + "site_locs.csv", delimiter=",", skiprows=1
            )
            offshoreentries["site"], offshoreentries["Within 100Km"] = (
                loaderfunctions.latlongtosite(
                    offshoreentries["Latitude"], offshoreentries["Longitude"], winddata
                )
            )
            offshoreentries["site"] = offshoreentries["site"].astype(int)

            offshoreentries["Closest Turbine Size"] = loaderfunctions.generationtiles(
                generatorkeys, offshoreentries["Turbine Size (MW)"]
            )

            differentgensizes = offshoreentries["Closest Turbine Size"].unique()
            offshoregenobjects = []
            for gensize in differentgensizes:
                gensizeentries = offshoreentries[
                    offshoreentries["Closest Turbine Size"] == gensize
                ]
                gensitelist = gensizeentries["site"].tolist()
                genyears = gensizeentries["OperationalDatetime"].tolist()
                turbinenums = gensizeentries["Number of Turbines"].astype(int).tolist()
                years_online = [i.year for i in genyears]
                months_online = [i.month for i in genyears]

                selectedgenerator = gendict[gensize]
                selectedgenobject = selectedgenerator(
                    year_min=startyear,
                    year_max=endyear,
                    sites=gensitelist,
                    n_turbine=turbinenums,
                    year_online=years_online,
                    month_online=months_online,
                    data_path=self.winddatafolder,
                    force_run=True,
                )
                offshoregenobjects.append(selectedgenobject)

            generationobjects["Offshore"] = offshoregenobjects

        if len(onshoreentries) > 0:
            print(f"Running {len(onshoreentries)} onshore sites")
            gendict = generation.generatordictionaries().onshore
            generatorkeys = np.array(list(gendict.keys()))
            winddata = np.loadtxt(
                self.winddatafolder + "site_locs.csv", delimiter=",", skiprows=1
            )
            onshoreentries["site"], onshoreentries["Within 100Km"] = (
                loaderfunctions.latlongtosite(
                    onshoreentries["Latitude"], onshoreentries["Longitude"], winddata
                )
            )
            onshoreentries["site"] = onshoreentries["site"].astype(int)

            onshoreentries["Closest Turbine Size"] = loaderfunctions.generationtiles(
                generatorkeys, onshoreentries["Turbine Size (MW)"]
            )

            differentgensizes = onshoreentries["Closest Turbine Size"].unique()
            onshoregenobjects = []
            for gensize in differentgensizes:
                gensizeentries = onshoreentries[
                    onshoreentries["Closest Turbine Size"] == gensize
                ]
                gensitelist = gensizeentries["site"].tolist()
                genyears = gensizeentries["OperationalDatetime"].tolist()
                turbinenums = gensizeentries["Number of Turbines"].astype(int).tolist()
                years_online = [i.year for i in genyears]
                months_online = [i.month for i in genyears]

                selectedgenerator = gendict[gensize]
                selectedgenobject = selectedgenerator(
                    year_min=startyear,
                    year_max=endyear,
                    sites=gensitelist,
                    n_turbine=turbinenums,
                    year_online=years_online,
                    month_online=months_online,
                    data_path=self.winddatafolder,
                    force_run=True,
                )
                onshoregenobjects.append(selectedgenobject)

            generationobjects["Onshore"] = onshoregenobjects

        if Gentimeseries:
            solarpowerout = [generationobjects["Solar"].power_out]
            offshorepowerouts = [i.power_out for i in generationobjects["Offshore"]]
            onshorepowerouts = [i.power_out for i in generationobjects["Onshore"]]

            combinedpoweroutput = solarpowerout + offshorepowerouts + onshorepowerouts
            stackedpoweroutputs = np.vstack(combinedpoweroutput)
            # sums the power outputs of all the generators
            totalpowerout = np.sum(stackedpoweroutputs, axis=0)
            currenttimestamp = datetime.datetime(startyear, 1, 1)

            with open(outputfolder + "/GenerationTimeSeries.csv", "w") as f:
                f.write("Timestamp,Power Output (MW)\n")
                for i in range(len(totalpowerout)):
                    f.write(f"{currenttimestamp},{totalpowerout[i]}\n")
                    currenttimestamp = currenttimestamp + datetime.timedelta(hours=1)


if __name__ == "__main__":
    gui_run = GUIRun()
    gui_run.window.mainloop()


# when run button is clicked, the function run is called
