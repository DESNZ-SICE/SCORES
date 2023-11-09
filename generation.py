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
        fixed_cost,
        variable_cost,
        name,
        data_path,
        save_path,
        limits=[0, 1000000],
        year_online=None,
        month_online=None,
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
        fixed_cost: (float) cost incurred per MW-year of installation in GBP
        variable_cost: (float) cost incurred per MWh of generation in GBP
        limits: (array<float>) used for .full_optimise to define the max and min installed generation in MWh ([min,max])
        year_online: list(int) year the generation unit was installed
        month_online: list(int) month the generation unit was installed
        == returns ==
        None
        """
        self.sites = sites
        self.year_min = year_min
        self.year_max = year_max
        self.months = months
        self.fixed_cost = fixed_cost
        self.variable_cost = variable_cost
        self.name = name
        self.data_path = data_path
        self.save_path = save_path
        self.limits = limits
        if year_online is None:
            self.year_online = [year_min] * len(sites)
        else:
            self.year_online = year_online

        if month_online is None:
            self.month_online = [months[0]] * len(sites)
        else:
            self.month_online = month_online

        self.date_map = {}
        n = 0

        # first get date range which fits within the bounds
        d = datetime.datetime(self.year_min, min(self.months), 1)

        while d.year <= self.year_max:
            if d.month in self.months:
                self.date_map[d] = n
                n += 1
                d += datetime.timedelta(1)

        self.operationaldatetime = [
            datetime.datetime(self.year_online[i], self.month_online[i], 1)
            for i in range(len(self.year_online))
        ]

        self.power_out = [0.0] * len(self.date_map) * 24
        self.power_out_scaled = [0.0] * len(self.power_out)

        self.power_out_array = np.array(self.power_out)
        self.n_good_points = [0] * len(self.date_map) * 24

    def scale_output(self, installed_capacity):
        """
        == description ==
        This function sets the parameter power_out_scaled to be a linearly
        scaled version of power_out, whose maximum value is the input parameter


        == parameters ==
        installed_capacity: (float) the installed generation capacity in MW?

        == returns ==
        None
        """

        sf = installed_capacity / self.total_installed_capacity

        mp = max(self.n_good_points)
        for t in range(len(self.power_out)):
            # If no data is available forward fill
            if self.n_good_points[t] == 0:
                self.power_out_scaled[t] = self.power_out_scaled[t - 1]
                continue
            # Otherwise scale by percentage of available data
            # self.power_out_scaled[t] = (self.power_out[t] * sf
            #                             * mp / self.n_good_points[t])
            self.power_out_scaled[t] = self.power_out[t] * sf
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
        if sum(self.n_good_points) < 0.25 * len(self.n_good_points):
            raise Exception("not enough good data")

        if max(self.n_good_points) == 1:
            return (
                100
                * sum(self.power_out)
                / (self.total_installed_capacity * sum(self.n_good_points))
            )

        else:
            if max(self.power_out_scaled) == 0:
                self.scale_output(1)
            return 100 * np.mean(self.power_out_scaled) / self.scaled_installed_capacity

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
        p = [0.0] * 24
        sf = int(len(self.power_out_scaled) / 24)
        for t in range(len(self.power_out_scaled)):
            p[t % 24] += self.power_out_scaled[t] / sf

        return p


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
        fixed_cost=225618,
        variable_cost=3,
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
        data_height=150,
        alpha=0.143,  # this line added by CQ
        save_path="stored_model_runs/",
        save=True,
        year_online=None,
        month_online=None,
        force_run=False,
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
        save: (boo) determines whether to save the results of the run

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
            fixed_cost,
            variable_cost,
            "Offshore Wind",
            data_path,
            save_path,
            year_online=year_online,
            month_online=month_online,
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

        # create the power curve at intervals of 0.1
        v = np.arange(0, self.v_cut_out, 0.1)  # wind speeds (m/s)
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
        # Next get the wind data

        for si in range(len(self.sites)):
            site = self.sites[si]
            site_speeds = []

            rowcounter = 0
            with open(self.data_path + str(site) + ".csv", "rU") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    rowcounter += 1
                    stepstartime = time.time()
                    d = datetime.datetime(int(row[0]), int(row[1]), int(row[2]))
                    if d not in self.date_map:
                        continue
                    if d < self.operationaldatetime[si]:
                        continue
                    site_speeds.append(row[6])
                    dn = self.date_map[d]  # day number (int)
                    hr = int(row[3])  # hour (int) 0-23
                    self.n_good_points[dn * 24 + hr] += 1

                # the approach ahs been changed to vectorise the calculation of power output
                # this is done by loading all the wind speeds into an array and then calculating
                # the power output for each point in the array. This is much faster than the previous
                # approach of calculating each point individually
                # The loading code still could be improved, as it is now the bottleneck
                site_speeds = np.array(site_speeds)
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
                poweroutvals = (
                    fs * Parray[p2s] + (1 - fs) * Parray[p1s]
                ) * self.n_turbine[
                    si
                ]  # interpolates the power output for the entire array
                self.power_out_array += poweroutvals  # adds the power output to the total power output array

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
        fixed_cost=42000,
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
            "Solar",
            data_path,
            save_path,
            year_online,
            month_online,
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
            if self.plant_capacities == []:
                self.plant_capacities = [1] * len(sites)

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
        hr_angle_deg = np.arange(-172.5, 187.5, 15)
        hr_angle = np.deg2rad(hr_angle_deg)

        # this list will contain the difference in sin(angle) between the
        # start and end of the hour
        diff_hr_angle = []
        for t in range(24):
            diff_hr_angle.append(
                np.sin(np.deg2rad(hr_angle_deg[t] + 7.5))
                - np.sin(np.deg2rad(hr_angle_deg[t] - 7.5))
            )

        # Get the solar data
        for index, site in enumerate(self.sites):
            site_power = [0.0] * len(self.date_map) * 24
            # need to keep each site sepeate at first due to smoothing fix
            site_power = [0] * len(self.n_good_points)
            with open(self.data_path + str(site) + ".csv", "rU") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    d = datetime.datetime(
                        int(row[0][:4]), int(row[0][5:7]), int(row[0][8:10])
                    )

                    if d not in self.date_map:
                        continue
                    dn = self.date_map[d]  # day number (int)
                    hr = int(row[1]) - 1  # hour (int) 0-23
                    diy = d.timetuple().tm_yday  # day in year 1-365

                    try:
                        irradiation = float(row[2]) / 3.6  # kJ -> Wh
                        irradiation = irradiation / 1.051  # merra2 overestimates
                        # self.n_good_points[dn* 24 + hr] += 1
                        # I believe this should be =1 not +=1 (Matt)
                        self.n_good_points[dn * 24 + hr] = 1
                    except:
                        continue

                    decl = 23.45 * np.sin(np.deg2rad(360 * (284 + diy) / 365))
                    decl = np.deg2rad(decl)
                    lat = site_lat[site]

                    # get incident angle
                    c_incident = (
                        np.sin(decl) * np.sin(lat) * np.cos(self.tilt)
                        - (
                            np.sin(decl)
                            * np.cos(lat)
                            * np.sin(self.tilt)
                            * np.cos(self.orient)
                        )
                        + (
                            np.cos(decl)
                            * np.cos(lat)
                            * np.cos(self.tilt)
                            * np.cos(hr_angle[hr])
                        )
                        + (
                            np.cos(decl)
                            * np.sin(lat)
                            * np.sin(self.tilt)
                            * np.cos(self.orient)
                            * np.cos(hr_angle[hr])
                        )
                        + (
                            np.cos(decl)
                            * np.sin(self.tilt)
                            * np.sin(self.orient)
                            * np.sin(hr_angle[hr])
                        )
                    )
                    incident = np.arccos(c_incident)
                    if incident > np.pi / 2:
                        continue  # sun behind panel

                    # slight concerns that this is just for horizontal panels?
                    c_zenith = np.cos(lat) * np.cos(decl) * np.cos(
                        hr_angle[hr]
                    ) + np.sin(lat) * np.sin(decl)
                    zenith = np.arccos(c_zenith)
                    if zenith > np.pi / 2 or c_zenith < 0:
                        continue

                    geometric_factor = c_incident / c_zenith

                    # extraterrestial radiation incident on the normal
                    g_on = solar_constant * (
                        1 + 0.033 * np.cos(np.deg2rad(360 * diy / 365))
                    )
                    irradiation0 = (
                        (12 / np.pi)
                        * g_on
                        * (
                            np.cos(lat) * np.cos(decl) * diff_hr_angle[hr]
                            + (np.pi * 15 / 180) * np.sin(lat) * np.sin(decl)
                        )
                    )

                    if irradiation0 < 0:
                        continue

                    clearness_index = irradiation / irradiation0
                    if clearness_index > 1:
                        irradiation = irradiation0

                    if clearness_index <= 0.22:
                        erbs_ratio = 1 - 0.09 * clearness_index
                    elif clearness_index <= 0.8:
                        erbs_ratio = (
                            0.9511
                            - 0.1604 * clearness_index
                            + 4.388 * np.power(clearness_index, 2)
                            - 16.638 * np.power(clearness_index, 3)
                            + 12.336 * np.power(clearness_index, 4)
                        )
                    else:
                        erbs_ratio = 0.165

                    # got to here
                    D_beam = (
                        irradiation - erbs_ratio * irradiation
                    ) * geometric_factor  # Wh/m2
                    D_dhi = irradiation * erbs_ratio * (1 + np.cos(self.tilt)) / 2
                    D = D_beam + D_dhi

                    site_power[dn * 24 + hr] += (
                        D
                        * plant_area[index]
                        * self.efficiency
                        * self.performance_ratio
                        * 1e-6
                    )

            # somewhere here I need to do the smoothing fix on final output
            for d in self.date_map:
                if d < self.operationaldatetime[index]:
                    continue
                dn = self.date_map[d]
                t = 0
                if sum(site_power[dn * 24 : (dn + 1) * 24]) == 0:
                    continue
                while site_power[dn * 24 + t] == 0:
                    t += 1
                # sunrise
                site_power[dn * 24 + t] = 0.1 * site_power[dn * 24 + t + 2]
                site_power[dn * 24 + t + 1] = 0.33 * site_power[dn * 24 + t + 2]

                t = 23
                while site_power[dn * 24 + t] == 0:
                    t -= 1
                # sunset
                site_power[dn * 24 + t] = 0.1 * site_power[dn * 24 + t - 2]
                site_power[dn * 24 + t - 1] = (
                    0.33 * site_power[dn * 24 + t - 2]
                )  # CQ correction to typo

            for t in range(len(site_power)):
                self.power_out[t] += site_power[t]
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
        fixed_cost=115561,
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
            fixed_cost,
            variable_cost,
            "Onshore Wind",
            data_path,
            save_path,
            year_online=year_online,
            month_online=month_online,
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

        for si in range(len(self.sites)):
            site = self.sites[si]
            site_speeds = []
            with open(self.data_path + str(site) + ".csv", "rU") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                for row in reader:
                    d = datetime.datetime(
                        int(row[0][:4]), int(row[0][5:7]), int(row[0][8:10])
                    )
                    if d not in self.date_map:
                        continue
                    if d < self.operationaldatetime[si]:
                        continue
                    dn = self.date_map[d]  # day number (int)
                    hr = int(row[1]) - 1  # hour (int) 0-23
                    site_speeds.append(row[2])
                    self.n_good_points[dn * 24 + hr] = 1

                site_speeds = np.array(site_speeds)
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
                poweroutvals = (
                    fs * Parray[p2s] + (1 - fs) * Parray[p1s]
                ) * self.n_turbine[
                    si
                ]  # interpolates the power output for the entire array
                self.power_out_array += poweroutvals  # adds the power output to the total power output array
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        )


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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
            rated_wind_speed=15,
            v_cut_in=4,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
            rated_wind_speed=11,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
            rated_wind_speed=10.25,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
            rated_wind_speed=9.75,
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
        year_online=None,
        month_online=None,
        force_run=False,
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
            rated_wind_speed=9.55,
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
            2: OnshoreWindModel2000,
            3: OnshoreWindModel3000,
            4: OnshoreWindModel4000,
            5: OnshoreWindModel5000,
            6: OnshoreWindModel6000,
            6.6: OnshoreWindModel6600,
            7: OnshoreWindModel7000,
        }
