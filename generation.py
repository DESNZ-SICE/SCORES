"""
Created: 08/04/2020 by C.CROZIER, updated by C Quarton and C O'Malley

File description:
"""

import datetime
import csv
import copy
import numpy as np
import matplotlib.pyplot as plt
import time
from fns import lambda_i, c_p, get_filename


class GenerationModel:
    # Base class for a generation model
    def __init__(
        self,
        sites,
        year_min,
        year_max,
        months,
        capex,
        opex,
        variable_cost,
        name,
        data_path,
        save_path,
        limits=[0, 1000000],
        year_online=None,
        month_online=None,
        lifetime=25,
        hurdlerate=0.052,
    ):
        """
        == description ==
        This function initialises the class, builds empty arrays to store the
        generated power and the number of good data points for each hour, and
        creates a dictionary which maps date to the date index.

        == parameters ==
        sites: (Array<int>) List of site indexes to be used
        year_min: (int) earliest year in simulation
        year_max: (int) latest year in simulation
        months: (Array<int>) list of months to be included in the simulation
        capex: (float) cost per MW of installation in GBP
        opex: (float) yearly cost per MW of installation in GBP
        variable_cost: (float) cost incurred per MWh of generation in GBP
        limits: (array<float>) used for .full_optimise to define the max and min installed generation in MWh ([min,max])
        year_online: list(int) year the generation unit was installed, at each site
        month_online: list(int) month the generation unit was installed, at each site
        lifetime: (int) lifetime of the generation unit in years
        hurdlerate: (float) hurdle rate for the generation unit
        == returns ==
        None
        """
        self.sites = sites
        self.year_min = year_min
        self.year_max = year_max
        self.months = months
        self.fixed_cost = self.calculate_fixed_costs(lifetime, capex, opex, hurdlerate)
        self.variable_cost = variable_cost
        self.name = name
        self.data_path = data_path
        self.save_path = save_path
        self.limits = limits
        self.max_possible_output = 0  # keeps track of max possible output: this ensures the load factor accounts for the year and month the generator came online
        # if no online date is given, assume the generator is online from the start
        if year_online is None:
            self.year_online = [year_min] * len(sites)
        else:
            self.year_online = year_online

        if month_online is None:
            self.month_online = [months[0]] * len(sites)
        else:
            self.month_online = month_online

        if months != (list(range(1, 13))) and year_max > year_min:
            raise (
                "The model can only be run for continous period: months cannot be subsampled over multiple years.\nTo return generation profiles for non contingous months, either run the model for each month separately or run it for the entire years and subsequently subsample the results."
            )

        self.firstdatadatetime = False  # this is used to check if the start date of the weather data has been found
        self.loadindex = 0  # this shows us which row of the weather data corresponds to the start date of the simulation. For now it is set to 0, but this may be changed later

        self.date_map = {}
        n = 0

        # first get date range which fits within the bounds
        d = datetime.datetime(self.year_min, min(self.months), 1)
        self.startdatetime = d

        while d.year <= self.year_max:
            if d.month in self.months:
                self.date_map[d] = n
                n += 1
            d += datetime.timedelta(1)

        self.operationaldatetime = [
            datetime.datetime(self.year_online[i], self.month_online[i], 1)
            for i in range(len(self.year_online))
        ]
        # this creates datetime objects to represent the date at which each generator site went online for the first time.

        # we need to find how many hours the simulations takes place over. This is the number of hours between the start date and the end date
        if max(self.months) == 12:
            enddatetime = datetime.datetime(self.year_max + 1, 1, 1)
        else:
            enddatetime = datetime.datetime(self.year_max, max(self.months) + 1, 1)
        numberofpoints = int(
            (enddatetime - self.startdatetime).total_seconds() // 3600
        )  # each hour gets a datapoint
        self.power_out = [0.0] * numberofpoints
        self.power_out_scaled = [0.0] * len(self.power_out)

        self.power_out_array = np.array(self.power_out)
        self.n_good_points = [0] * numberofpoints

    def scale_output(self, installed_capacity, scale=False):
        """
        == description ==
        This function sets the parameter power_out_scaled to be a linearly
        scaled version of power_out, whose maximum value is the input parameter


        == parameters ==
        installed_capacity: (float) the installed generation capacity in MW?

        == returns ==
        None
        """
        if scale:
            sf = installed_capacity / self.total_installed_capacity
        else:
            sf = 1

        self.power_out_scaled = sf * np.array(self.power_out)
        self.scaled_installed_capacity = installed_capacity

        return np.array(self.power_out_scaled)

    def scale_output_energy(self, energy):
        sf = energy / sum(self.power_out)
        mp = max(self.n_good_points)
        for t in range(len(self.power_out)):
            # If no data is available forward fill
            if self.n_good_points[t] == 0:
                self.power_out_scaled[t] = self.power_out_scaled[t - 1]
                continue
            # Otherwise scale by percentage of available data
            self.power_out_scaled[t] = (
                self.power_out[t] * sf * mp / self.n_good_points[t]
            )
        self.scaled_installed_capacity = max(self.power_out_scaled)

        return np.array(self.power_out_scaled)

    def check_for_saved_run(self, path):
        """
        == description ==
        This function checks to see whether this simulation has been previously
        run, and if so sets power_out to the stored values.

        == parameters ==
        path: (str) location the csv file would stored if it exists

        == returns ==
        True if a previous run has been recovered
        False otherwise
        """
        try:
            with open(path, "r") as csvfile:
                reader = csv.reader(csvfile)
                t = 0
                for row in reader:
                    self.power_out[t] = float(row[0])
                    self.n_good_points[t] = 1
                    t += 1
            self.power_out_scaled = copy.deepcopy(self.power_out)
            self.total_installed_capacity = 1
            self.scaled_installed_capacity = 1
            return True
        except:
            return False

    def save_run(self, path):
        """
        == description ==
        This function stores the results from a simulation run into a csv file
        at the stated path.

        == parameters ==
        path: (str) location the csv file would stored if it exists

        == returns ==
        None
        """
        # adjust for missing points
        # self.scale_output(self.total_installed_capacity)

        with open(path, "w") as csvfile:
            writer = csv.writer(csvfile)
            for t in range(len(self.power_out_scaled)):
                writer.writerow(
                    [self.power_out_scaled[t] / self.total_installed_capacity]
                )

        return None

    def get_load_factor(self):
        """
        == description ==
        This function returns the average load factor over the simulated period

        == parameters ==
        None

        == returns ==
        (float) load factor in percent (0-100)
        """
        # maximum possible output is computed in run_model, based on the year and month the generator came online

        return 100 * sum(self.power_out) / (self.max_possible_output)

    def get_cost(self):
        """
        == description ==
        This function calculates the cost per year in GBP of the power produced
        by the generation unit.

        == parameters ==
        None

        == returns ==
        (float) cost per year in GBP
        """
        return self.fixed_cost * self.scaled_installed_capacity + (
            self.variable_cost
            * sum(self.power_out_scaled)
            / (self.year_max + 1 - self.year_min)
        )

    def get_diurnal_profile(self):
        # this reshapes the powerout array into a 2d array with 24 columns (one for each hour of the day), and then takes the mean of each column
        p = self.power_out_scaled.reshape(-1, 24).mean(axis=0)
        return p

    def calculate_fixed_costs(self, lifetime, totalcapex, yearlyopex, hurdlerate):
        """
        == description ==
        This function calculates the yearly fixed costs of the generation input

        == parameters ==
        lifetime: (int) lifetime of the generation unit in years
        totalcapex: (float) total capital expenditure in GBP
        yearlyopex: (float) yearly operational expenditure in GBP
        hurdlerate: (float) hurdle rate for the generation unit
        == returns ==
        (float) cost per year in GBP
        """
        yearlyreturncost = (
            hurdlerate / (1 - (1 + hurdlerate) ** -lifetime)
        ) * totalcapex
        fixed_cost = yearlyreturncost + yearlyopex
        return fixed_cost


class NuclearModel(GenerationModel):
    def __init__(
        self,
        sites=[0],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        capex=4000000,
        opex=50000,
        variable_cost=2,
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        year_online=None,
        month_online=None,
        capacities=[1000],
        limits=[0, 1000000],
        lifetime=40,
        hurdlerate=0.1,
        loadfactor=0.77,
    ):
        """
        == description ==
        Initialises a Nuclear object. Searches for a saved result at
        save_path, otherwise generates a power curve and calculates the
        aggregated power output from turbines at the locations contained in
        sites.

        == parameters ==
        sites: (Array<int>) List of site indexes to be used. The site indexes here mean little, but
        are used for consistency. The length of the site index must match the length of the capacities
        year_min: (int) earliest year in sumlation
        year_max: (int) latest year in simulation
        months: (Array<int>) list of months to be included in the simulation
        Capex: (float) cost incurred per MW of installation in GBP
        opex: (float) yearly cost per MW of installation in GBP
        variable_cost: (float) cost incurred per MWh of generation in GBP
        data_path: (str) path to file containing raw data
        save_path: (str) path to file where output will be saved
        save: (bool) determines whether to save the results of the run
        capacity: (Array <float>) installed capacity of each site in MW
        limits: (Array<float>) used to define the max and min installed generation in MW ([min,max])
        lifetime: (int) lifetime of the generation unit in years
        hurdlerate: (float) hurdle rate for the generation unit, between 0 and 1
        loadfactor: (float) load factor of the generation unit
        == returns ==
        None
        """
        # raises an error if the number of sites and the number of capacities are not the same
        if len(capacities) != len(sites):
            raise Exception(
                "The number of sites and the number of capacities must be the same"
            )

        super().__init__(
            sites,
            year_min,
            year_max,
            months,
            capex,
            opex,
            variable_cost,
            "Nuclear",
            data_path,
            save_path,
            year_online=year_online,
            month_online=month_online,
            limits=limits,
            lifetime=lifetime,
            hurdlerate=hurdlerate,
        )
        self.power_out = np.array(self.power_out)
        self.total_installed_capacity = sum(capacities)
        self.plant_capacities = capacities
        self.loadfactor = loadfactor
        self.run_model()

    def __str__(self):
        return f"Nuclear Generator, total capacity: {self.total_installed_capacity} MW"

    def run_model(self):
        """
        == description ==
        Generates power output. This is assumed as constant for Nuclear plants, so is only
        affected by the installed capacity and the operational date.
        """

        for sitenum in range(len(self.sites)):
            operationaltime = self.operationaldatetime[sitenum]
            timedelta = self.startdatetime - operationaltime
            timedeltahours = timedelta.days * 24 + timedelta.seconds / 3600
            timedeltahours = int(timedeltahours)
            self.power_out[timedeltahours:] += (
                self.plant_capacities[sitenum] * self.loadfactor
            )
        self.scale_output(self.total_installed_capacity)


class GeothermalModel(GenerationModel):
    def __init__(
        self,
        sites=[0],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        capex=2000000,
        opex=2000000 * 0.05,
        variable_cost=0,
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        year_online=None,
        month_online=None,
        capacities=[1000],
        limits=[0, 1000000],
        lifetime=40,
        hurdlerate=0.1,
        loadfactor=0.90,
    ):
        """
        == description ==
        Initialises a Geothermal object.
        == parameters ==
        sites: (Array<int>) List of site indexes to be used. The site indexes here mean little, but
        are used for consistency. The length of the site index must match the length of the capacities
        year_min: (int) earliest year in simlation
        year_max: (int) latest year in simulation
        months: (Array<int>) list of months to be included in the simulation
        fixed_cost: (float) cost incurred per MW of installation in GBP
        variable_cost: (float) cost incurred per MWh of generation in GBP
        data_path: (str) path to file containing raw data
        save_path: (str) path to file where output will be saved
        save: (boo) determines whether to save the results of the run
        capacity: (Array <float>) installed capacity of each site in MW
        limits: (Array<float>) used to define the max and min installed generation in MW ([min,max])
        lifetime: (int) lifetime of the generation unit in years
        hurdlerate: (float) hurdle rate for the generation unit (between 0 and 1)
        loadfactor: (float) load factor of the generation unit, between 0 and 1
        == returns ==
        None
        """
        # raises an error if the number of sites and the number of capacities are not the same
        if len(capacities) != len(sites):
            raise Exception(
                "The number of sites and the number of capacities must be the same"
            )

        super().__init__(
            sites,
            year_min,
            year_max,
            months,
            capex,
            opex,
            variable_cost,
            "Geothermal",
            data_path,
            save_path,
            year_online=year_online,
            month_online=month_online,
            limits=limits,
            lifetime=lifetime,
            hurdlerate=hurdlerate,
        )
        self.power_out = np.array(self.power_out)
        self.total_installed_capacity = sum(capacities)
        self.plant_capacities = capacities
        self.loadfactor = loadfactor
        self.run_model()

    def __str__(self):
        return (
            f"Geothermal Generator, total capacity: {self.total_installed_capacity} MW"
        )

    def run_model(self):
        """
        == description ==
        Generates power output. This is assumed as constant for Geothermal plants, so is only
        affected by the installed capacity and the operational date.
        """

        for sitenum in range(len(self.sites)):
            operationaltime = self.operationaldatetime[sitenum]
            timedelta = self.startdatetime - operationaltime
            timedeltahours = timedelta.days * 24 + timedelta.seconds / 3600
            timedeltahours = int(timedeltahours)
            self.power_out[timedeltahours:] += (
                self.plant_capacities[sitenum] * self.loadfactor
            )

        self.scale_output(self.total_installed_capacity)


class TidalStreamTurbineModel(GenerationModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        fixed_cost=445400,
        variable_cost=0,
        water_density=1027.0,
        rotor_diameter=20,
        rated_water_speed=2.91,
        v_cut_in=0.88,
        Cp=0.37,
        v_cut_out=10,
        n_turbine=None,
        turbine_size=1.47,
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        """
        == description ==
        Initialises an OffshoreWindModel object. Searches for a saved result at
        save_path, otherwise generates a power curve and calculates the
        aggregated power output from turbines at the locations contained in
        sites.

        == parameters ==
        sites: (Array<int>) List of site indexes to be used
        year_min: (int) earliest year in sumlation
        year_max: (int) latest year in simulation
        months: (Array<int>) list of months to be included in the simulation
        fixed_cost: (float) cost incurred per MW of installation in GBP
        variable_cost: (float) cost incurred per MWh of generation in GBP
        water_density: (float) density of water in kg/m3
        rotor_diameter: (float) rotor diameter in m
        rated_wind_speed: (float) rated wind speed in m/s
        v_cut_in: (float) cut in wind speed in m/s
        Cp: (float) power coefficient, assumed constant over flow speeds
        v_cut_out: (float) cut out wind speed in m/s
        n_turbine: (Array<int>) number of turbines installed at each site
        turbine_size: (float) size of each turbine in MW
        data_path: (str) path to file containing raw data
        save_path: (str) path to file where output will be saved
        save: (boo) determines whether to save the results of the run

        == returns ==
        None
        """
        super().__init__(
            sites,
            year_min,
            year_max,
            months,
            fixed_cost,
            variable_cost,
            "Tidal Stream",
            data_path,
            save_path,
        )

        self.water_density = water_density
        self.rotor_diameter = rotor_diameter
        self.rated_water_speed = rated_water_speed
        self.v_cut_in = v_cut_in
        self.v_cut_out = v_cut_out
        self.n_turbine = n_turbine
        self.turbine_size = turbine_size
        self.Cp = Cp

        file_name = get_filename(sites, "tidal", year_min, year_max, months)
        if file_name == "":
            save = False

        if self.check_for_saved_run(self.save_path + file_name) is False:
            self.run_model()
            if save is True:
                self.save_run(self.save_path + file_name)

    def run_model(self):
        """
        == description ==
        Generates power curve and runs model from historic data

        == parameters ==
        None

        == returns ==
        None
        """

        if self.data_path == "":
            raise Exception("model can not be run without a data path")
        if self.sites[0] == "all":
            sites = []
            with open(self.data_path + "site_locs.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    sites.append(int(row[0]))
            self.sites = sites

        elif self.sites[:2] == "lf":
            sites = []
            lwst = str(sites[2:])
            locs = []
            # with open(self.save_path+'s_load_factors.csv','r') as csvfile:
            with open(self.save_path + "tidal_load_factors.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if float(row[2]) * 100 > lwst:
                        locs.append([row[0] + row[1]])
            with open(self.data_path + "site_locs.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if row[1] + row[2] in locs:
                        sites.apend(int(row[0]))
            self.sites = sites

        # If no values given assume an equl distribution of turbines over sites
        if self.n_turbine is None:
            self.n_turbine = [1] * len(self.sites)

        self.total_installed_capacity = sum(self.n_turbine) * self.turbine_size

        area = np.pi * self.rotor_diameter * self.rotor_diameter / 4

        # create the power curve at intervals of 0.1
        v = np.arange(0, self.v_cut_out, 0.1)  # wind speeds (m/s)
        P = [0.0] * len(v)  # power output (MW)

        # assume a fixed Cp - calculate this value using the turbine's rated wind speed and rated power
        Cp = self.Cp

        for i in range(len(v)):
            if v[i] < self.v_cut_in:
                continue

            P[i] = (
                0.5 * Cp * self.water_density * area * np.power(v[i], 3)
            )  # new power equation using fixed Cp
            P[i] = P[i] / 1e6  # W to MW

            if P[i] > self.turbine_size:
                P[i] = self.turbine_size

        # Next get the tidal data
        for si in range(len(self.sites)):
            site = self.sites[si]
            with open(self.data_path + str(site) + ".csv", "rU") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    d = datetime.datetime(int(row[0]), int(row[1]), int(row[2]))
                    if d not in self.date_map:
                        continue
                    dn = self.date_map[d]  # day number (int)
                    hr = int(row[3]) - 1  # hour (int) 0-23

                    # skip missing data
                    if float(row[6]) >= 0:
                        speed = float(row[6])
                    else:
                        continue

                    # prevent overload
                    if speed > v[-1]:
                        speed = v[-1]
                    self.max_possible_output += self.turbine_size * self.n_turbine[si]
                    # interpolate the closest values from the power curve
                    p1 = int(speed / 0.1)
                    p2 = p1 + 1
                    if p2 == len(P):
                        p2 = p1
                    f = (speed % 0.1) / 0.1
                    self.power_out[dn * 24 + hr] += (
                        f * P[p2] + (1 - f) * P[p1]
                    ) * self.n_turbine[si]
                    # self.n_good_points[dn* 24 + hr] += 1
                    # I believe this should be =1 not +=1 (Matt)
                    self.n_good_points[dn * 24 + hr] = 1
        # the power values have been generated for each point. However, points with missing data are
        # still zero. The power scaled values, which are initalised at zero. Running self.scale_ouput sorts
        # this out. As we dont want to increase the capacity at this point, we just run scale_output with the
        # currently installed capacity: the values should not
        self.scale_output(self.total_installed_capacity)


class OffshoreWindModel(GenerationModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        capex=1189000,
        opex=29.2 * 1000,
        variable_cost=1,
        tilt=5,
        air_density=1.23,
        rotor_diameter=164,
        rated_rotor_rpm=7,
        rated_wind_speed=11,
        v_cut_in=3,
        v_cut_out=25,
        n_turbine=None,
        turbine_size=9.5,
        data_path="",
        hub_height=122,
        data_height=50,
        alpha=0.143,  # this line added by CQ
        save_path="stored_model_runs/",
        save=True,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
        power_curve=None,
        lifetime=25,
        hurdlerate=0.052,
    ):
        """
        == description ==
        Initialises an OffshoreWindModel object. Searches for a saved result at
        save_path, otherwise generates a power curve and calculates the
        aggregated power output from turbines at the locations contained in
        sites.

        == parameters ==
        sites: (Array<int>) List of site indexes to be used
        year_min: (int) earliest year in sumlation
        year_max: (int) latest year in simulation
        months: (Array<int>) list of months to be included in the simulation
        Capex: (float) cost incurred per MW of installation in GBP
        opex: (float) yearly cost per MW of installation in GBP
        variable_cost: (float) cost incurred per MWh of generation in GBP
        tilt: (float) blade tilt in degrees
        air_density: (float) density of air in kg/m3
        rotor_diameter: (float) rotor diameter in m
        rated_rotor_rpm: (float) rated rotation speed in rpm
        rated_wind_speed: (float) rated wind speed in m/s
        v_cut_in: (float) cut in wind speed in m/s
        v_cut_out: (float) cut out wind speed in m/s
        n_turbine: (Array<int>) number of turbines installed at each site
        turbine_size: (float) size of each turbine in MW
        data_path: (str) path to file containing raw data
        save_path: (str) path to file where output will be saved
        save: (bool) determines whether to save the results of the run
        year_online: list(int) year the generation unit was installed, at each site
        month_online: list(int) month the generation unit was installed, at each site
        force_run: (bool) determines whether to rerun the model
        limits: (Array<float>) used to define the max and min installed generation in MW ([min,max])
        power_curve: (Array<float>) power curve for the turbine
        == returns ==
        None
        """
        self.startdate = datetime.datetime(year_min, months[0], 1)
        self.enddate = datetime.datetime(year_max + 1, months[0], 1)
        super().__init__(
            sites,
            year_min,
            year_max,
            months,
            capex,
            opex,
            variable_cost,
            "Offshore Wind",
            data_path,
            save_path,
            year_online=year_online,
            month_online=month_online,
            limits=limits,
        )

        self.tilt = tilt
        self.air_density = air_density
        self.rotor_diameter = rotor_diameter
        self.rated_rotor_rpm = rated_rotor_rpm
        self.rated_wind_speed = rated_wind_speed
        self.v_cut_in = v_cut_in
        self.v_cut_out = v_cut_out
        self.n_turbine = n_turbine
        self.turbine_size = turbine_size
        self.hub_height = hub_height  # added by CQ
        self.data_height = data_height  # added by CQ
        self.alpha = alpha  # added by CQ
        self.power_curve = power_curve
        file_name = get_filename(
            sites, "osw_" + str(turbine_size), year_min, year_max, months
        )
        if file_name == "":
            save = False

        if (
            self.check_for_saved_run(self.save_path + file_name) is False
            or force_run is True
        ):
            self.run_model()
            if save is True:
                self.save_run(self.save_path + file_name)

    def __str__(self):
        return f"Offshore wind model\nNumber of Turbines:{sum(self.n_turbine)}\t Turbine Power:\
            {self.turbine_size} MW\nTotal power:{round(sum(self.n_turbine)*self.turbine_size)}MW"

    def run_model(self):
        """
        == description ==
        Generates power curve and runs model from historic data

        == parameters ==
        None

        == returns ==
        None
        """
        stepstartime = time.time()
        if self.data_path == "":
            raise Exception("model can not be run without a data path")
        if self.sites[0] == "all":
            sites = []
            with open(self.data_path + "site_locs.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    sites.append(int(row[0]))
            self.sites = sites

        elif self.sites[:2] == "lf":
            sites = []
            lwst = str(sites[2:])
            locs = []
            with open(self.save_path + "s_load_factors.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if float(row[2]) * 100 > lwst:
                        locs.append([row[0] + row[1]])
            with open(self.data_path + "site_locs.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if row[1] + row[2] in locs:
                        sites.apend(int(row[0]))
            self.sites = sites

        # If no values given assume an equl distribution of turbines over sites
        if self.n_turbine is None:
            self.n_turbine = [1] * len(self.sites)

        self.total_installed_capacity = sum(self.n_turbine) * self.turbine_size
        # tip speed ratio
        tsr = (
            self.rated_rotor_rpm
            * self.rotor_diameter
            / (2 * 9.549 * self.rated_wind_speed)
        )
        area = np.pi * self.rotor_diameter * self.rotor_diameter / 4
        b = self.tilt
        if self.power_curve is None:
            # create the power curve at intervals of 0.1
            v = np.linspace(
                0, self.v_cut_out, self.v_cut_out * 10 + 1
            )  # wind speeds (m/s)
            P = [0.0] * len(v)  # power output (MW)
            # assume a fixed Cp - calculate this value using the turbine's rated wind speed and rated power
            Cp = (
                self.turbine_size
                * 1e6
                / (0.5 * self.air_density * area * np.power(self.rated_wind_speed, 3))
            )

            for i in range(len(v)):
                if v[i] < self.v_cut_in:
                    continue

                # P[i] = 0.5*c_p(tsr, b)* self.air_density*area*np.power(v[i], 3)
                P[i] = 0.5 * Cp * self.air_density * area * np.power(v[i], 3)
                P[i] = P[i] / 1e6  # W to MW

                if P[i] > self.turbine_size:
                    P[i] = self.turbine_size

            Parray = np.array(P)
        else:
            # check that the power curve tops out at the right wind speed
            if self.power_curve[-1][0] < self.v_cut_out:
                raise Exception("Power curve does not extend to cut out wind speed")
            Parray = np.array([x[1] for x in self.power_curve])
        # Next get the wind data

        for si in range(len(self.sites)):
            site = self.sites[si]
            site_speeds = []

            if self.firstdatadatetime == False:
                with open(self.data_path + str(site) + ".csv", "r") as file:
                    loadeddata = file.readlines()
                    firstrow = loadeddata[1].split(",")
                    #
                    # MERRA 2 format
                    #
                    firstdatadatetime = datetime.datetime.strptime(
                        firstrow[0], "%d/%m/%Y %H:%M"
                    )
                    #
                    # Met office format (uncomment to use)
                    #
                    # firstdatadatetime = datetime.datetime(int(firstrow[0]), int(firstrow[1]), int(firstrow[2]), int(firstrow[3]))

                    self.firstdatadatetime = firstdatadatetime
                    # find the number of hours between the first date in the data and the first date in the simulation
                    firstdatadatehours = (
                        self.startdatetime - firstdatadatetime
                    ).total_seconds() / 3600
                    self.loadindex = int(firstdatadatehours)

            operationalindex = int(
                (self.operationaldatetime[si] - self.firstdatadatetime).total_seconds()
                / 3600
            )

            # the range selector index allows us to remove the portion of data within the range, but where the generator was not
            # operational. It also allows us to handle the case where the operational date is before the first date in the data.
            if operationalindex < self.loadindex:
                rangeselectorindex = self.loadindex
            else:
                rangeselectorindex = operationalindex

            #
            # Merra format:
            #
            site_speeds = np.loadtxt(
                self.data_path + str(site) + ".csv",
                delimiter=",",
                skiprows=1,
                usecols=(2),
            )
            #
            # Met office format, uncomment to use: it uses a different column to the MERRA format
            #
            # site_speeds = np.loadtxt(
            #     self.data_path + str(site) + ".csv",
            #     delimiter=",",
            #     skiprows=1,
            #     usecols=(6),
            # )

            site_speeds = site_speeds[
                rangeselectorindex : self.loadindex + len(self.n_good_points)
            ]

            # the approach hass been changed to vectorise the calculation of power output
            # this is done by loading all the wind speeds into an array and then calculating
            # the power output for each point in the array. This is much faster than the previous
            # approach of calculating each point individually
            site_speeds = np.array(site_speeds)
            site_speeds = site_speeds.astype(float)
            site_speeds[site_speeds < 0] = 0

            # adjusts the wind speeds to hub height
            site_speeds = site_speeds * np.power(
                self.hub_height / self.data_height, self.alpha
            )

            site_speeds[site_speeds >= self.v_cut_out] = (
                self.v_cut_out
            )  # prevents overload
            p1s = np.floor(site_speeds / 0.1).astype(
                int
            )  # gets the index of the lower bound of the interpolation
            p2s = p1s + 1  # gets the index of the upper bound of the interpolation
            p2s[p2s == len(Parray)] = p1s[p2s == len(Parray)]
            fs = (site_speeds % 0.1) / 0.1
            poweroutvals = (fs * Parray[p2s] + (1 - fs) * Parray[p1s]) * self.n_turbine[
                si
            ]  # interpolates the power output for the entire array
            self.power_out_array[
                rangeselectorindex - self.loadindex :
            ] += poweroutvals  # adds the power output to the total power output array
            self.max_possible_output += (
                self.turbine_size * len(poweroutvals) * self.n_turbine[si]
            )

        # the power values have been generated for each point. However, points with missing data are
        # still zero. The power scaled values, which are initalised at zero. Running self.scale_ouput sorts
        # this out. As we dont want to increase the capacity at this point, we just run scale_output with the
        # currently installed capacity: the values should not

        self.power_out = self.power_out_array.tolist()
        self.scale_output(self.total_installed_capacity)


class SolarModel(GenerationModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        capex=450000,
        opex=9300,
        variable_cost=0,
        orient=0,
        tilt=22,
        efficiency=0.17,
        performance_ratio=0.85,
        plant_capacities=[1],
        area_factor=5.84,
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        year_online=None,
        month_online=None,
        limits=[0, 1000000],
        force_run=False,
        hurdlerate=0.05,
        lifetime=35,
    ):
        """
        == description ==
        Initialises an OffshoreWindModel object. Searches for a saved result at
        save_path, otherwise generates a power curve and calculates the
        aggregated power output from turbines at the locations contained in
        sites.

        == parameters ==
        sites: (Array<int>) List of site indexes to be used
        year_min: (int) earliest year in sumlation
        year_max: (int) latest year in simulation
        months: (Array<int>) list of months to be included in the simulation
        fixed_cost: (float) cost incurred per MW of installation in GBP
        variable_cost: (float) cost incurred per MWh of generation in GBP
        orient: (float) surface azimuth angle in degrees
        tilt: (float) panel tilt in degrees
        efficiency: (float) panel efficiency 0-1
        performance_ratio: (float) panel perfromace ratio 0-1
        plant_capacities: (Array<float>) installed capacity of each site in MW
        area_factor: (float) panel area per installed kW in m2/kW
        data_path: (str) path to file containing raw data
        save_path: (str) path to file where output will be saved
        save: (boo) determines whether to save the results of the run
        force_run: (bool) determines whether to force the model to run
        == returns ==
        None
        """
        super().__init__(
            sites,
            year_min,
            year_max,
            months,
            capex,
            opex,
            variable_cost,
            "Solar",
            data_path,
            save_path,
            year_online=year_online,
            month_online=month_online,
            limits=limits,
            hurdlerate=hurdlerate,
            lifetime=lifetime,
        )

        self.orient = np.deg2rad(orient)  # deg -> rad
        self.tilt = np.deg2rad(tilt)  # deg -> rad
        self.efficiency = efficiency
        self.performance_ratio = performance_ratio
        self.plant_capacities = plant_capacities
        self.area_factor = area_factor

        file_name = get_filename(sites, "s", year_min, year_max, months)
        if file_name == "":
            save = False

        if (
            self.check_for_saved_run(self.save_path + file_name) is False
            or force_run is True
        ):
            self.run_model()
            if save is True:
                self.save_run(self.save_path + file_name)
        else:
            print("Loading saved run")

    def run_model(self):
        """
        == description ==
        Generates power curve and runs model from historic data

        == parameters ==
        None

        == returns ==
        None
        """

        if self.data_path == "":
            raise Exception("model can not be run without a data path")
        if self.sites[0] == "all":
            sites = []
            with open(self.data_path + "site_locs.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    sites.append(int(row[0]))
            self.sites = sites
            if self.plant_capacities == []:
                print("No plant capacities given")
                self.plant_capacities = [1] * len(sites)

        # elif self.sites[:2] == "lf":
        #     sites = []
        #     lwst = str(sites[2:])
        #     locs = []
        #     with open(self.save_path + "s_load_factors.csv", "r") as csvfile:
        #         reader = csv.reader(csvfile)
        #         next(reader)
        #         for row in reader:
        #             if float(row[2]) * 100 > lwst:
        #                 locs.append([row[0] + row[1]])
        #     with open(self.data_path + "site_locs.csv", "r") as csvfile:
        #         reader = csv.reader(csvfile)
        #         next(reader)
        #         for row in reader:
        #             if row[1] + row[2] in locs:
        #                 sites.apend(int(row[0]))
        #     self.sites = sites

        self.total_installed_capacity = sum(self.plant_capacities)

        # Need to get the site latitutudes
        site_lat = {}
        with open(self.data_path + "site_locs.csv", "rU") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                site_lat[int(row[0])] = np.deg2rad(float(row[1]))

        plant_area = [i * self.area_factor * 10**3 for i in self.plant_capacities]
        solar_constant = 1367  # W/m2

        # hourly angles
        day_hr_angle_deg = np.arange(-172.5, 187.5, 15)
        day_hr_angle = np.deg2rad(day_hr_angle_deg)
        day_hr_angle = day_hr_angle.tolist()
        day_diff_hr_angle = np.sin(np.deg2rad(day_hr_angle_deg + 7.5)) - np.sin(
            np.deg2rad(day_hr_angle_deg - 7.5)
        )
        day_diff_hr_angle = day_diff_hr_angle.tolist()

        # this list will contain the difference in sin(angle) between the
        # start and end of the hour

        # Get the solar data
        for index, site in enumerate(self.sites):
            if self.firstdatadatetime == False:
                with open(self.data_path + str(site) + ".csv", "r") as file:
                    loadeddata = file.readlines()
                    firstrow = loadeddata[1].split(",")

                    firstdatadatetime = datetime.datetime.strptime(
                        firstrow[0], "%d/%m/%Y %H:%M"
                    )

                    self.firstdatadatetime = firstdatadatetime
                    # find the number of hours between the first date in the data and the first date in the simulation
                    firstdatadatehours = (
                        self.startdatetime - firstdatadatetime
                    ).total_seconds() / 3600
                    self.loadindex = int(firstdatadatehours)

            operationalindex = int(
                (
                    self.operationaldatetime[index] - self.firstdatadatetime
                ).total_seconds()
                / 3600
            )

            if operationalindex < self.loadindex:
                rangeselectorindex = self.loadindex
            else:
                rangeselectorindex = operationalindex

            irradiances = np.loadtxt(
                self.data_path + str(site) + ".csv",
                delimiter=",",
                skiprows=1,
                usecols=(2),
            )
            irradiances = irradiances[
                rangeselectorindex : self.loadindex + len(self.n_good_points)
            ]
            irradiances = np.array(irradiances)
            powerout = np.zeros_like(
                irradiances
            )  # creates zeros array to hold the power
            irradiances /= 3.6  # kJ -> Wh
            irradiances /= 1.051  # merra2 overestimates
            hours = [
                i % 24 for i in range(len(irradiances))
            ]  # hours of the day, repeating for each day
            hourarray = np.array(hours)
            diy = [
                (self.operationaldatetime[index] + datetime.timedelta(hours=i))
                .timetuple()
                .tm_yday
                for i in range(len(irradiances))
            ]
            diy = np.array(diy)
            hr_angles = day_hr_angle * int(
                len(irradiances) / 24
            )  # repeats the hr_angles for each day
            hr_angles = np.array(hr_angles)

            diff_hr_angle = day_diff_hr_angle * int(len(irradiances) / 24)
            diff_hr_angle = np.array(diff_hr_angle)
            decl = 23.45 * np.sin(np.deg2rad(360 * (284 + diy) / 365))
            decl = np.deg2rad(decl)
            lat = site_lat[site]

            c_incident = (
                np.sin(decl) * np.sin(lat) * np.cos(self.tilt)
                - (np.sin(decl) * np.cos(lat) * np.sin(self.tilt) * np.cos(self.orient))
                + (np.cos(decl) * np.cos(lat) * np.cos(self.tilt) * np.cos(hr_angles))
                + (
                    np.cos(decl)
                    * np.sin(lat)
                    * np.sin(self.tilt)
                    * np.cos(self.orient)
                    * np.cos(hr_angles)
                )
                + (
                    np.cos(decl)
                    * np.sin(self.tilt)
                    * np.sin(self.orient)
                    * np.sin(hr_angles)
                )
            )

            incident = np.arccos(c_incident)
            # this is the angle between the sun and the panel
            sunbehindpanel = incident > np.pi / 2
            # if the sun is behind the panel, the incident angle is greater than 90 degrees

            # slight concerns that this is just for horizontal panels?
            c_zenith = np.cos(lat) * np.cos(decl) * np.cos(hr_angles) + np.sin(
                lat
            ) * np.sin(decl)
            zenith = np.arccos(c_zenith)
            zentihtoohigh = (
                zenith > np.pi / 2
            )  # if the zenith angle is greater than 90, the sun is below the horizon
            zenithtoolow = (
                c_zenith < 0
            )  # if c_zenith is less than zero, the sun is below the horizon

            c_zenith[zenithtoolow] = 1  # need to set these to 1 to avoid divide by zero
            geometric_factor = c_incident / c_zenith

            sunwrong = (
                zenithtoolow | sunbehindpanel | zentihtoohigh
            )  # combines the three conditions
            geometric_factor[sunwrong] = (
                0  # set the geometric factor to zero for these values
            )

            # extraterrestial radiation incident on the normal
            g_on = solar_constant * (1 + 0.033 * np.cos(np.deg2rad(360 * diy / 365)))
            irradiation0 = (
                (12 / np.pi)
                * g_on
                * (
                    np.cos(lat) * np.cos(decl) * diff_hr_angle
                    + (np.pi * 15 / 180) * np.sin(lat) * np.sin(decl)
                )
            )

            # we don't want to divide by zero, but sometimes the irradiation0 is zero
            # in that case, we will find the index where this happens, set irradiation0 to 1,
            # and then subsequently set the power out to zero
            noirradiance = irradiation0 < 0
            irradiation0[noirradiance] = 1

            clearness_index = irradiances / irradiation0
            # finds indeces where the clearness index is greater than 1
            clearness1mask = np.where(clearness_index > 1)
            irradiances[clearness1mask] = irradiation0[clearness1mask]

            erbs_ratio = np.ones_like(clearness_index) * 0.165
            erbs_ratio[clearness_index <= 0.22] = (
                1 - 0.09 * clearness_index[clearness_index <= 0.22]
            )
            midclearness = (clearness_index > 0.22) & (clearness_index <= 0.8)

            erbs_ratio[midclearness] = (
                0.9511
                - 0.1604 * clearness_index[midclearness]
                + 4.388 * np.power(clearness_index[midclearness], 2)
                - 16.638 * np.power(clearness_index[midclearness], 3)
                + 12.336 * np.power(clearness_index[midclearness], 4)
            )

            D_beam = (
                irradiances - erbs_ratio * irradiances
            ) * geometric_factor  # Wh/m2
            D_dhi = irradiances * erbs_ratio * (1 + np.cos(self.tilt)) / 2
            D = D_beam + D_dhi

            poweroutvals = (
                D * plant_area[index] * self.efficiency * self.performance_ratio * 1e-6
            )

            poweroutvals[sunwrong] = 0
            poweroutvals[noirradiance] = 0

            # quit()
            # the previous code smoothed the ramp up and down rates. We will do the same, using the same technique
            # to smooth the data, we find the first and last times in each day where the power is non-zero
            # The power 2 hours after and before these times is then used as a baseline
            # the sunrise and sunset times are then set to 10% of the power 2 hours after and before these times
            # the times in between are then set to 33% of the power 2 hours after and before these times
            sunrises = []
            sunsets = []
            night = True
            for j in range(len(poweroutvals)):
                power = poweroutvals[j]
                if power != 0 and night:
                    sunrises.append(j)
                    night = False
                if power == 0 and not night:
                    sunsets.append(j - 1)
                    night = True

            sunrisearray = np.array(sunrises)
            sunsetarray = np.array(sunsets)

            sunrisingselector = sunrisearray + 1
            sunrisenselectors = sunrisearray + 2
            sunsettingselector = sunsetarray - 1
            sunsetselectors = sunsetarray - 2
            poweroutvals[sunrisearray] = 0.1 * poweroutvals[sunrisenselectors]
            poweroutvals[sunrisingselector] = 0.33 * poweroutvals[sunrisenselectors]
            poweroutvals[sunsetarray] = 0.1 * poweroutvals[sunsetselectors]
            poweroutvals[sunsettingselector] = 0.33 * poweroutvals[sunsetselectors]
            self.power_out_array[rangeselectorindex - self.loadindex :] += poweroutvals
            self.power_out = self.power_out_array.tolist()

            # for i in range(24 * 4):
            #     print(f"{i} {poweroutvals[i]}")
            self.max_possible_output += self.plant_capacities[index] * len(poweroutvals)
            # this needs checking

            # # somewhere here I need to do the smoothing fix on final output
            # for d in self.date_map:
            #     if d < self.operationaldatetime[index]:
            #         continue
            #     self.max_possible_output += self.plant_capacities[index] * 24
            #     dn = self.date_map[d]
            #     t = 0
            #     if sum(site_power[dn * 24 : (dn + 1) * 24]) == 0:
            #         continue
            #     while site_power[dn * 24 + t] == 0:
            #         t += 1
            #     # sunrise
            #     site_power[dn * 24 + t] = 0.1 * site_power[dn * 24 + t + 2]
            #     site_power[dn * 24 + t + 1] = 0.33 * site_power[dn * 24 + t + 2]

            #     t = 23
            #     while site_power[dn * 24 + t] == 0:
            #         t -= 1
            #     # sunset
            #     site_power[dn * 24 + t] = 0.1 * site_power[dn * 24 + t - 2]
            #     site_power[dn * 24 + t - 1] = (
            #         0.33 * site_power[dn * 24 + t - 2]
            #     )  # CQ correction to typo

            # for t in range(len(site_power)):
            #     self.power_out[t] += site_power[t]
        # the power values have been generated for each point. However, points with missing data are
        # still zero. The power scaled values, which are initalised at zero. Running self.scale_ouput sorts
        # this out. As we dont want to increase the capacity at this point, we just run scale_output with the
        # currently installed capacity: the values should not change

        self.scale_output(self.total_installed_capacity)


class OnshoreWindModel(GenerationModel):
    # need to adjust the cost!

    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        capex=1230000,
        opex=30700,
        variable_cost=6,
        tilt=5,
        air_density=1.23,
        rotor_diameter=120,
        rated_rotor_rpm=13,
        rated_wind_speed=12.5,
        v_cut_in=3,
        v_cut_out=25,
        n_turbine=None,
        turbine_size=3.6,
        hub_height=90,
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        data_height=50,
        alpha=0.143,  # this row added by CQ to calculate wind shear
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
        lifetime=25,
        hurdlerate=0.052,
    ):  # this added by CQ so that a power curve can optionally be imported
        """
        == description ==
        Initialises an OnshoreWindModel object. Searches for a saved result at
        save_path, otherwise generates a power curve and calculates the
        aggregated power output from turbines at the locations contained in
        sites.

        == parameters ==
        sites: (Array<int>) List of site indexes to be used
        year_min: (int) earliest year in sumlation
        year_max: (int) latest year in simulation
        months: (Array<int>) list of months to be included in the simulation
        fixed_cost: (float) cost incurred per MW of installation in GBP
        variable_cost: (float) cost incurred per MWh of generation in GBP
        tilt: (float) blade tilt in degrees
        air_density: (float) density of air in kg/m3
        rotor_diameter: (float) rotor diameter in m
        rated_rotor_rpm: (float) rated rotation speed in rpm
        rated_wind_speed: (float) rated wind speed in m/s
        v_cut_in: (float) cut in wind speed in m/s
        v_cut_out: (float) cut out wind speed in m/s
        n_turbine: (Array<int>) number of turbines installed at each site
        turbine_size: (float) size of each turbine in MW
        hub_height: (float) hub height in m
        data_path: (str) path to file containing raw data
        save_path: (str) path to file where output will be saved
        save: (boo) determines whether to save the results of the run
        data_height: (float) height at which wind speed data applies   # added by CQ
        alpha: (float) wind shear coefficient                          # added by CQ
        power_curve: (Array<float>) optional power curve - power outputs that correspond to v array spaced at 0.1m/s
        == returns ==
        None
        """
        super().__init__(
            sites,
            year_min,
            year_max,
            months,
            capex,
            opex,
            variable_cost,
            "Onshore Wind",
            data_path,
            save_path,
            year_online=year_online,
            month_online=month_online,
            limits=limits,
            hurdlerate=hurdlerate,
            lifetime=lifetime,
        )

        # If no values given assume an equl distribution of turbines over sites
        self.tilt = tilt
        self.air_density = air_density
        self.rotor_diameter = rotor_diameter
        self.rated_rotor_rpm = rated_rotor_rpm
        self.rated_wind_speed = rated_wind_speed
        self.v_cut_in = v_cut_in
        self.v_cut_out = v_cut_out
        self.n_turbine = n_turbine
        self.turbine_size = turbine_size
        self.hub_height = hub_height
        self.data_height = data_height  # added by CQ
        self.alpha = alpha  # added by CQ
        self.power_curve = power_curve

        file_name = get_filename(
            sites, "w" + str(turbine_size), year_min, year_max, months
        )
        if file_name == "":
            save = False

        if (
            self.check_for_saved_run(self.save_path + file_name) is False
            or force_run is True
        ):
            self.run_model()
            if save is True:
                self.save_run(self.save_path + file_name)

    def __str__(self):
        return f"Onshore wind model\nNumber of Turbines:{sum(self.n_turbine)}\t Turbine Power:\
            {self.turbine_size} MW\nTotal power:{round(sum(self.n_turbine)*self.turbine_size)}Mw"

    def run_model(self):
        """
        == description ==
        Generates power curve and runs model from historic data

        == parameters ==
        None

        == returns ==
        None
        """

        if self.data_path == "":
            raise Exception("model can not be run without a data path")
        if self.sites[0] == "all":
            sites = []
            with open(self.data_path + "site_locs.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    sites.append(int(row[0]))
            self.sites = sites

        elif self.sites[:2] == "lf":
            sites = []
            lwst = str(sites[2:])
            locs = []
            with open(
                self.save_path + "w" + str(self.turbine_size) + "_load_factors.csv", "r"
            ) as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if float(row[2]) * 100 > lwst:
                        locs.append([row[0] + row[1]])
            with open(self.data_path + "site_locs.csv", "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    if row[1] + row[2] in locs:
                        sites.apend(int(row[0]))
            self.sites = sites

        if self.n_turbine is None:
            self.n_turbine = [1] * len(self.sites)
        self.total_installed_capacity = sum(self.n_turbine) * self.turbine_size

        # tip speed ratio
        tsr = (
            self.rated_rotor_rpm
            * self.rotor_diameter
            / (2 * 9.549 * self.rated_wind_speed)
        )
        area = np.pi * self.rotor_diameter * self.rotor_diameter / 4
        b = self.tilt

        # create the power curve at intervals of 0.1
        v = np.arange(0, self.v_cut_out, 0.1)  # wind speeds (m/s)

        # CQ added two power_curve options: either calculate, or import
        if self.power_curve is None:
            P = [0.0] * len(v)  # power output (MW)

            # the following is a CQ edit - new Cp calculation
            Cp = (
                self.turbine_size
                * 1e6
                / (0.5 * self.air_density * area * np.power(self.rated_wind_speed, 3))
            )

            for i in range(len(v)):
                if v[i] < self.v_cut_in:
                    continue

                # P[i] = (0.5 * c_p(tsr, b) * self.air_density * area * np.power(v[i], 3))
                P[i] = 0.5 * Cp * self.air_density * area * np.power(v[i], 3)  # CQ edit
                P[i] = P[i] / 1e6  # W to MW

                if P[i] > self.turbine_size:
                    P[i] = self.turbine_size
        else:  # this is the new bit: import a power curve
            P = self.power_curve

        Parray = np.array(P)
        loadtimes = []
        for si in range(len(self.sites)):
            site = self.sites[si]
            site_speeds = []
            # since extending the data back to 1980, loading has become a real bottleneck. Previous, the data is loaded in, and then
            # stepped through in series, to check when the datetime matches the first year in the time series, and the first operational
            # date for the wind turbine. However, assuming we can work out these two numbers, we can just load the data in, and then
            # slice it to the correct time period. This is much faster.
            # We assume here that each datafile starts on the same date, which should be the case.

            if self.firstdatadatetime == False:
                with open(self.data_path + str(site) + ".csv", "r") as file:
                    loadeddata = file.readlines()
                    firstrow = loadeddata[1].split(",")
                    firstdatadatetime = datetime.datetime.strptime(
                        firstrow[0], "%d/%m/%Y %H:%M"
                    )
                    self.firstdatadatetime = firstdatadatetime
                    # find the number of hours between the first date in the data and the first date in the simulation
                    firstdatadatehours = (
                        self.startdatetime - firstdatadatetime
                    ).total_seconds() / 3600
                    self.loadindex = int(firstdatadatehours)

            operationalindex = int(
                (self.operationaldatetime[si] - self.firstdatadatetime).total_seconds()
                / 3600
            )

            # the range selector index allows us to remove the portion of data within the range, but where the generator was not
            # operational. It also allows us to handle the case where the operational date is before the first date in the data.
            if operationalindex < self.loadindex:
                rangeselectorindex = self.loadindex
            else:
                rangeselectorindex = operationalindex

            site_speeds = np.loadtxt(
                self.data_path + str(site) + ".csv",
                delimiter=",",
                skiprows=1,
                usecols=(2),
            )
            site_speeds = site_speeds[
                rangeselectorindex : self.loadindex + len(self.n_good_points)
            ]
            # neededdata=splitdata[self.loadindex:self.loadindex+len(self.n_good_points)]
            # neededdata=splitdata[operationalindex:self.loadindex+len(self.n_good_points)]

            site_speeds = site_speeds.astype(float)
            site_speeds[site_speeds < 0] = 0

            # adjusts the wind speeds to hub height
            site_speeds = site_speeds * np.power(
                self.hub_height / self.data_height, self.alpha
            )
            site_speeds[site_speeds > v[-1]] = v[-1]  # prevents overload
            p1s = np.floor(site_speeds / 0.1).astype(
                int
            )  # gets the index of the lower bound of the interpolation
            p2s = p1s + 1  # gets the index of the upper bound of the interpolation
            p2s[p2s == len(P)] = p1s[p2s == len(P)]
            fs = (site_speeds % 0.1) / 0.1
            poweroutvals = (fs * Parray[p2s] + (1 - fs) * Parray[p1s]) * self.n_turbine[
                si
            ]  # interpolates the power output for the entire array

            self.power_out_array[
                rangeselectorindex - self.loadindex :
            ] += poweroutvals  # adds the power output to the total power output array
            self.max_possible_output += (
                self.turbine_size * len(poweroutvals) * self.n_turbine[si]
            )
        # the power values have been generated for each point. However, points with missing data are
        # still zero. The power scaled values, which are initalised at zero. Running self.scale_ouput sorts
        # this out. As we dont want to increase the capacity at this point, we just run scale_output with the
        # currently installed capacity: the values should not
        self.power_out = self.power_out_array.tolist()
        self.scale_output(self.total_installed_capacity)


class TidalStreamTurbineModel_P1(TidalStreamTurbineModel):
    # 1MW turbine
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=24,
            rated_water_speed=2.19,
            v_cut_in=0.66,
            Cp=0.41,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=1.0,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class TidalStreamTurbineModel_P2(TidalStreamTurbineModel):
    # 1.5MW turbine
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=24,
            rated_water_speed=2.50,
            v_cut_in=0.75,
            Cp=0.41,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=1.5,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class TidalStreamTurbineModel_P3(TidalStreamTurbineModel):
    # 2MW turbine
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=24,
            rated_water_speed=2.75,
            v_cut_in=0.83,
            Cp=0.41,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=2.0,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class OnshoreWindModel500(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=39,
            rated_rotor_rpm=30.0,
            rated_wind_speed=15,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=0.5,
            hub_height=53,
            data_path=data_path,  #
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )
        # based on Vestas V39


class OnshoreWindModel1000(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=54.2,
            rated_rotor_rpm=22.0,
            rated_wind_speed=15,
            v_cut_in=3,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=1,
            hub_height=60,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )
        # based on AN Bonus 1000/54


class OnshoreWindModel1500(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=82,
            rated_rotor_rpm=14.4,
            rated_wind_speed=13,
            v_cut_in=3,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=1.5,
            hub_height=94,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )
        # based on Vestas V82


class OnshoreWindModel2000(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=90,
            rated_rotor_rpm=14.9,
            rated_wind_speed=13,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=2.0,
            hub_height=80,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


# class OnshoreWindModel2300(OnshoreWindModel):
#     def __init__(
#         self,
#         sites=["all"],
#         year_min=2013,
#         year_max=2019,
#         months=list(range(1, 13)),
#         data_path="",
#         save_path="stored_model_runs/",
#         save=True,
#         n_turbine=None,
#         year_online=None,
#         month_online=None,
#         force_run=False,
# limits = ([0, 1000000],)
#     ):
#         super().__init__(
#             sites=sites,
#             year_min=year_min,
#             year_max=year_max,
#             months=months,
#             tilt=5,
#             air_density=1.23,
#             rotor_diameter=108,
#             rated_rotor_rpm=16,
#             rated_wind_speed=11.5,
#             v_cut_in=3,
#             v_cut_out=25,
#             n_turbine=n_turbine,
#             turbine_size=2.3,
#             hub_height=80,
#             data_path=data_path,
#             save_path=save_path,
#             save=save,
#             year_online=year_online,
#             month_online=month_online,
#             force_run=force_run,
# limits = limits
#         )
#         # based on the Siemens SWT-2.3-108


class OnshoreWindModel2500(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=100,
            rated_rotor_rpm=14,
            rated_wind_speed=13,
            v_cut_in=3,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=2.5,
            hub_height=85,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )
        # based on GE 2.5-100


class OnshoreWindModel3000(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=113,
            rated_rotor_rpm=15.5,
            rated_wind_speed=12.5,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=3.0,
            hub_height=80,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OnshoreWindModel3600(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=123,
            rated_rotor_rpm=13,
            rated_wind_speed=12,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=3.6,
            hub_height=80,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OnshoreWindModel4000(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=130,
            rated_rotor_rpm=13,
            rated_wind_speed=12,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=4,
            hub_height=90,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OnshoreWindModel5000(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=145,
            rated_rotor_rpm=11.5,
            rated_wind_speed=10.5,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=5,
            hub_height=100,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OnshoreWindModel6000(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=164,
            rated_rotor_rpm=11,
            rated_wind_speed=10,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=6.0,
            hub_height=100,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OnshoreWindModel6600(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=170,
            rated_rotor_rpm=11,
            rated_wind_speed=10,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=6.6,
            hub_height=100,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OnshoreWindModel7000(OnshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=171.2,
            rated_rotor_rpm=10.5,
            rated_wind_speed=10,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=7.0,
            hub_height=110,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OffshoreWindModel2000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=80,
            rated_rotor_rpm=19,
            rated_wind_speed=14.5,
            v_cut_in=3.5,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=2,
            hub_height=80,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )

        # based on Vestas v80 2MW: https://en.wind-turbine-models.com/turbines/19-vestas-v80-2.0


class OffshoreWindModel3000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=90,
            rated_rotor_rpm=18.4,
            rated_wind_speed=13,
            v_cut_in=3,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=3,
            hub_height=80,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )

        # based on Vestas V90 3MW: https://en.wind-turbine-models.com/turbines/603-vestas-v90-3.0


class OffshoreWindModel5000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=126,
            rated_rotor_rpm=12,
            rated_wind_speed=14.5,
            v_cut_in=3.5,
            v_cut_out=30,
            n_turbine=n_turbine,
            turbine_size=5,
            hub_height=105,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )

        # Based on Repower 5M: https://www.thewindpower.net/turbine_en_14_repower_5m.php


class OffshoreWindModel6000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=154,
            rated_rotor_rpm=11,
            rated_wind_speed=13,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=6,
            hub_height=120,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )
        # based on Siemens SWT-6.0-154: https://en.wind-turbine-models.com/turbines/657-siemens-swt-6.0-154
        # hub height is an estimation, as it's site specific


class OffshoreWindModel7000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=154,
            rated_rotor_rpm=11,
            rated_wind_speed=13,
            v_cut_in=3,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=7,
            hub_height=120,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )
        # based on Siemens SWT-7.0-154: https://en.wind-turbine-models.com/turbines/1102-siemens-swt-7.0-154
        # hub height is an estimation, as it's site specific


class OffshoreWindModel8000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=164,
            rated_rotor_rpm=12,
            rated_wind_speed=13,
            v_cut_in=4,
            v_cut_out=25,
            n_turbine=n_turbine,
            turbine_size=8,
            hub_height=140,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )
        # based on Vestas V164-8MW: https://en.wind-turbine-models.com/turbines/318-vestas-v164-8.0


class OffshoreWindModel10000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=190,
            rated_rotor_rpm=11,
            rated_wind_speed=11.5,
            v_cut_in=4,
            v_cut_out=30,
            n_turbine=n_turbine,
            turbine_size=10,
            hub_height=135,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OffshoreWindModel12000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=220,
            rated_rotor_rpm=11,
            rated_wind_speed=11.5,
            v_cut_in=4,
            v_cut_out=30,
            n_turbine=n_turbine,
            turbine_size=12,
            hub_height=150,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OffshoreWindModel15000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=246,
            rated_rotor_rpm=11,
            rated_wind_speed=11.5,
            v_cut_in=4,
            v_cut_out=30,
            n_turbine=n_turbine,
            turbine_size=15,
            hub_height=150,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OffshoreWindModel17000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=262,
            rated_rotor_rpm=11,
            rated_wind_speed=11.5,
            v_cut_in=4,
            v_cut_out=30,
            n_turbine=n_turbine,
            turbine_size=17,
            hub_height=180,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class OffshoreWindModel20000(OffshoreWindModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
        n_turbine=None,
        power_curve=None,
        year_online=None,
        month_online=None,
        force_run=False,
        limits=[0, 1000000],
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            tilt=5,
            air_density=1.23,
            rotor_diameter=284,
            rated_rotor_rpm=11,
            rated_wind_speed=11.5,
            v_cut_in=3,
            v_cut_out=30,
            n_turbine=n_turbine,
            turbine_size=20,
            hub_height=180,
            data_path=data_path,
            save_path=save_path,
            save=save,
            year_online=year_online,
            month_online=month_online,
            force_run=force_run,
            limits=limits,
            power_curve=power_curve,
        )


class TidalStreamTurbine_VR_1_0(TidalStreamTurbineModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=20,
            rated_water_speed=1.0,
            v_cut_in=0.3,
            Cp=0.37,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=0.060,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class TidalStreamTurbine_VR_1_5(TidalStreamTurbineModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=20,
            rated_water_speed=1.5,
            v_cut_in=0.45,
            Cp=0.37,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=0.201,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class TidalStreamTurbine_VR_2_0(TidalStreamTurbineModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=20,
            rated_water_speed=2.0,
            v_cut_in=0.60,
            Cp=0.37,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=0.478,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class TidalStreamTurbine_VR_2_5(TidalStreamTurbineModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=20,
            rated_water_speed=2.5,
            v_cut_in=0.75,
            Cp=0.37,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=0.933,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class TidalStreamTurbine_VR_3_0(TidalStreamTurbineModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=20,
            rated_water_speed=3.0,
            v_cut_in=0.9,
            Cp=0.37,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=1.612,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class TidalStreamTurbine_VR_3_5(TidalStreamTurbineModel):
    def __init__(
        self,
        sites=["all"],
        year_min=2013,
        year_max=2019,
        months=list(range(1, 13)),
        data_path="",
        save_path="stored_model_runs/",
        save=True,
    ):
        super().__init__(
            sites=sites,
            year_min=year_min,
            year_max=year_max,
            months=months,
            water_density=1027.0,
            rotor_diameter=20,
            rated_water_speed=3.5,
            v_cut_in=1.05,
            Cp=0.37,
            v_cut_out=30,
            n_turbine=None,
            turbine_size=2.559,
            data_path=data_path,
            save_path=save_path,
            save=save,
        )


class generatordictionaries:
    """This class exists to hold generation dictionaries, which connect
    the turbine size to the correct generation model."""

    def __init__(self) -> None:
        self.offshore = {
            2: OffshoreWindModel2000,
            3: OffshoreWindModel3000,
            5: OffshoreWindModel5000,
            6: OffshoreWindModel6000,
            7: OffshoreWindModel7000,
            8: OffshoreWindModel8000,
            10: OffshoreWindModel10000,
            12: OffshoreWindModel12000,
            15: OffshoreWindModel15000,
            17: OffshoreWindModel17000,
            20: OffshoreWindModel20000,
        }
        self.onshore = {
            0.5: OnshoreWindModel500,
            1: OnshoreWindModel1000,
            1.5: OnshoreWindModel1500,
            2: OnshoreWindModel2000,
            # 2.3: OnshoreWindModel2300,
            2.5: OnshoreWindModel2500,
            3: OnshoreWindModel3000,
            4: OnshoreWindModel4000,
            5: OnshoreWindModel5000,
            6: OnshoreWindModel6000,
            6.6: OnshoreWindModel6600,
            7: OnshoreWindModel7000,
        }
