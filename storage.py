"""
Created: 28/07/2020 by C.CROZIER

File description: This file contains the classes for a single type of energy
storage and an aggregated portfolio of storage assets.

Notes: The storage base class is technology agnostic, but child classes are
icluded that are parameterised for Li-Ion, hydrogen, and thermal storage.
"""

import copy
import numpy as np
import matplotlib.pyplot as plt

# optimisation high level language, help found at https://www.ima.umn.edu/materials/2017-2018.2/W8.21-25.17/26326/3_PyomoFundamentals.pdf
import pyomo.environ as pyo
import aggregatedEVs as aggEV
from pandas import DataFrame
from opt_con_class import System_LinProg_Model, store_optimisation_results
import pandas as pd


class StorageModel:
    def __init__(
        self,
        cost_params_file="params/SCORES Cost assumptions.xlsx",
        technical_params_file="params/Storage_technical_assumptions.xlsx",
        cost_sensitivity="Medium",
        cost_year=2025,
        storage_param_entry="Li-Ion",
        charge_param_entry=None,
        discharge_param_entry=None,
        eff_in=None,
        eff_out=None,
        self_dis=None,
        storageCapex=None,
        storageFixedOpex=None,
        storagelifetime=None,
        storageVarOpex=None,
        chargeCapex=None,
        chargeFixedOpex=None,
        chargeVarOpex=None,
        chargeLifetime=None,
        dischargeCapex=None,
        dischargeFixedOpex=None,
        dischargeVarOpex=None,
        dischargeLifetime=None,
        hurdleRate=None,
        max_c_rate=None,
        max_d_rate=None,
        name="Li-Ion Battery",
        capacity=1,
        limits=[0, 1000000000],
        initial_charge=1.0,
    ):
        """
        == description ==
        .

        == parameters ==
        Cost_params_file: (str) the file path to the cost assumptions: these can be overwritten by keyword arguments
        Technical_params_file: (str) the file path to the technical assumptions: these can be overwritten by keyword arguments
        Cost_sensitivity: (str) the sensitivity of the cost assumptions
        Storage_param_entry: (str) the entry in the cost assumptions file for the storage medium
        Charge_param_entry: (str) the entry in the cost assumptions file for the charging equipment
        Discharge_param_entry: (str) the entry in the cost assumptions file for the discharging equipment
        eff_in: (float) charging efficiency in % (0-100)
        eff_out: (float) discharge efficiency in % (0-100)
        self_dis: (float) self discharge rate in % per month (0-100)
        storageCapex: (float) cost incurred per MWh of the storage medium in GBP
        storageFixedOpex: (float) yearly operational cost of the storage medium per MWh in GBP
        storagelifetime: (int) lifetime of the storage medium in years
        storageVarOpex: (float) cost incurred per MWh of throughput in GBP
        chargeCapex: (float) cost incurred per MW of charging equipment in GBP
        chargeFixedOpex: (float) yearly operational cost of charging equipment per MW in GBP
        chargeLifetime: (int) lifetime of the charging equipment in years
        chargeVarOpex: (float) cost incurred per MWh of throughput in GBP
        dischargeCapex: (float) cost incurred per MW of discharging equipment in GBP
        dischargeFixedOpex: (float) yearly operational cost of discharging equipment per MW in GBP
        dischargeLifetime: (int) lifetime of the discharging equipment in years
        variable_cost: (float) cost incurred per MWh of throughput in GBP
        max_c_rate: (float) the maximum charging rate (% per hour) (0-100). This class assumes that the max charge rate is set by the storage medium rather than the charging equipment.
        max_d_rate: (float) the maximum discharging rate (% per hour)(0-100). This class assumes that the max discharge rate is set by the storage medium rather than the discharging equipment.
        name: (str) the name of the asset - for use in graph plotting
        capacity: (float) MWh of storage installed
        limits: array[(float)] the [min,max] capacity in MWh
        initial_charge: (float) initial state of charge (0-1)

        NOTE: both max_c and max_d rate defined FROM THE GRID SIDE. I.E. the maximum energy into and out of the
        storage will be less and more than these respectively.

        == returns ==
        None
        """

        if cost_params_file != None:
            cost_params = pd.read_excel(cost_params_file, sheet_name=cost_sensitivity)
            cost_params = cost_params.set_index(["Technology", "Year"])
            if storage_param_entry == None:
                raise ValueError(
                    "Storage parameter entry not specified: the storage parameter entry must be set. The entries for charging and discharging equipment are optional."
                )
            storagerow = cost_params.loc[storage_param_entry, cost_year]
            loaded_storageCapex = storagerow["Capex-£/MWh"]
            loaded_storage_fixedopex = storagerow["Fixed Opex-£/MWh/year"]
            loaded_storage_variableopex = storagerow["Variable O&M-£/MWh"]
            loaded_storage_lifetime = storagerow["Operating lifetime-years"]
            loaded_hurdle_rate = storagerow["Hurdle Rate-%"]
            if charge_param_entry != None:
                loaded_charge_capex = (
                    cost_params.loc[charge_param_entry, cost_year]["Capex-£/kW"] * 1000
                )
                loaded_charge_fixedopex = cost_params.loc[
                    charge_param_entry, cost_year
                ]["Fixed Opex-£/MW/year"]
                loaded_charge_variableopex = cost_params.loc[
                    charge_param_entry, cost_year
                ]["Variable O&M-£/MWh"]
                loaded_charge_lifetime = cost_params.loc[charge_param_entry, cost_year][
                    "Operating lifetime-years"
                ]
            else:
                (
                    loaded_charge_capex,
                    loaded_charge_fixedopex,
                    loaded_charge_variableopex,
                ) = (0, 0, 0)
            if discharge_param_entry != None:
                loaded_discharge_capex = (
                    cost_params.loc[discharge_param_entry, cost_year]["Capex-£/kW"]
                    * 1000
                )
                loaded_discharge_fixedopex = cost_params.loc[
                    discharge_param_entry, cost_year
                ]["Fixed Opex-£/MW/year"]
                loaded_discharge_variableopex = cost_params.loc[
                    discharge_param_entry, cost_year
                ]["Variable O&M-£/MWh"]
                loaded_discharge_lifetime = cost_params.loc[
                    discharge_param_entry, cost_year
                ]["Operating lifetime-years"]

        storageCapex = storageCapex if storageCapex != None else loaded_storageCapex
        storageFixedOpex = (
            storageFixedOpex if storageFixedOpex != None else loaded_storage_fixedopex
        )
        storageVarOpex = (
            storageVarOpex if storageVarOpex != None else loaded_storage_variableopex
        )
        storagelifetime = (
            storagelifetime if storagelifetime != None else loaded_storage_lifetime
        )

        chargeCapex = chargeCapex if chargeCapex != None else loaded_charge_capex
        chargeFixedOpex = (
            chargeFixedOpex if chargeFixedOpex != None else loaded_charge_fixedopex
        )
        chargeVarOpex = (
            chargeVarOpex if chargeVarOpex != None else loaded_charge_variableopex
        )
        chargeLifetime = (
            chargeLifetime if chargeLifetime != None else loaded_charge_lifetime
        )

        dischargeCapex = (
            dischargeCapex if dischargeCapex != None else loaded_discharge_capex
        )
        dischargeFixedOpex = (
            dischargeFixedOpex
            if dischargeFixedOpex != None
            else loaded_discharge_fixedopex
        )
        dischargeVarOpex = (
            dischargeVarOpex
            if dischargeVarOpex != None
            else loaded_discharge_variableopex
        )
        dischargeLifetime = (
            dischargeLifetime
            if dischargeLifetime != None
            else loaded_discharge_lifetime
        )

        hurdleRate = hurdleRate if hurdleRate != None else loaded_hurdle_rate

        if technical_params_file != None:
            technical_params = pd.read_excel(technical_params_file)
            technical_params = technical_params.set_index(
                ["Technology", "Technology type"]
            )
            storagerow = technical_params.loc[storage_param_entry, "Storage"]
            if charge_param_entry != None:
                chargerow = technical_params.loc[charge_param_entry, "Charge"]
                loaded_eff_in = chargerow["Efficiency-%"] * 100
            if discharge_param_entry != None:
                dischargerow = technical_params.loc[discharge_param_entry, "Discharge"]
                loaded_eff_out = dischargerow["Efficiency-%"] * 100

            loaded_self_dis = storagerow["Self Discharge-%/month"] * 100
            loaded_max_c_rate = storagerow["Max C Rate-%/hour"] * 100
            loaded_max_d_rate = storagerow["Max D Rate-%/hour"] * 100

        eff_in = eff_in if eff_in != None else loaded_eff_in
        eff_out = eff_out if eff_out != None else loaded_eff_out
        self_dis = self_dis if self_dis != None else loaded_self_dis
        max_c_rate = max_c_rate if max_c_rate != None else loaded_max_c_rate
        max_d_rate = max_d_rate if max_d_rate != None else loaded_max_d_rate

        self.energy_shortfalls = 0
        self.actual_reliability = 0
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.self_dis = self_dis

        storagefixedcost = self.calculate_fixed_costs(
            storagelifetime, storageCapex, storageFixedOpex, hurdleRate
        )
        self.storage_fixed_cost = storagefixedcost
        self.storage_variable_cost = storageVarOpex
        # the storage fixed cost refers of storing the energy. For some technologies (such as li-ion batteries),
        # the storage cannot be separated from the charging and discharging equipment:
        # in this case, the charge and discharge costs should be set to zero.
        # however, for technologies such as hydrogen storage, it is benefical to be able to scale
        # the storage capacity independently of the charging and discharging equipment.

        chargefixedcost = self.calculate_fixed_costs(
            chargeLifetime, chargeCapex, chargeFixedOpex, hurdleRate
        )
        self.charge_fixed_cost = chargefixedcost
        self.charge_variable_cost = chargeVarOpex
        dischargefixedcost = self.calculate_fixed_costs(
            dischargeLifetime, dischargeCapex, dischargeFixedOpex, hurdleRate
        )
        self.discharge_fixed_cost = dischargefixedcost
        self.discharge_variable_cost = dischargeVarOpex
        print(f"Storage fixed cost: {storagefixedcost}")
        print(f"Charge fixed cost: {chargefixedcost*max_c_rate/100}")
        print(f"Discharge fixed cost: {dischargefixedcost*max_d_rate/100}")
        self.variable_cost = sum([storageVarOpex, chargeVarOpex, dischargeVarOpex])
        self.fixed_cost = (
            storagefixedcost
            + (max_c_rate / 100) * chargefixedcost
            + (max_d_rate / 100) * dischargefixedcost
        )
        self.max_c_rate = max_c_rate
        self.max_d_rate = max_d_rate
        self.capacity = capacity
        self.name = name
        self.limits = limits
        self.initial_charge = initial_charge
        self.R = 0
        self.dischargelosses = 0
        # These will be used to monitor storage usage
        self.en_in = 0  # total energy into storage (grid side)
        self.en_out = 0  # total energy out of storage (grid side)
        self.curt = 0  # total supply that could not be stored
        self.curtarray = []
        # from optimise setting only (added by Mac)
        self.discharge = np.empty([])  # timeseries of discharge rate (grid side) MW
        self.charge = 0  # store state of charge in MWh
        self.SOC = []  # timeseries of Storage State of Charge (SOC) MWh
        self.chargetimeseries = []

    def reset(self):
        """
        == description ==
        Resets the parameters recording the use of the storage assets.

        == parameters ==
        None

        == returns ==
        None
        """
        self.en_in = 0
        self.en_out = 0
        self.curt = 0
        self.efficiencylosses = 0

        self.discharge = np.empty([])
        self.charge = np.empty([])
        self.SOC = []

    def set_capacity(self, capacity):
        """
        == description ==
        Sets the installed  storage capacity to the specified value.

        == parameters ==
        capacity: (float) MWh of storage installed

        == returns ==
        None
        """
        self.capacity = capacity
        self.reset()

    def get_cost(self):
        """
        == description ==
        Gets the total cost of running the storage system.

        == parameters ==
        None

        == returns ==
        (float) cost in GBP/yr of the storage unit
        """
        if self.capacity == np.inf:
            return np.inf
        else:
            return (
                self.capacity * self.fixed_cost
                + self.en_out * self.variable_cost * 100 / (self.eff_out * self.n_years)
            )

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

    def plot_timeseries(self, start=0, end=-1):
        """
        == parameters ==
        start: (int) start time of plot
        end: (int) end time of plot
        """

        if self.discharge.shape == ():
            print(
                "Charging timeseries not avaialable, try running MultipleStorageAssets.optimise_storage()."
            )
        else:
            if end <= 0:
                timehorizon = self.discharge.size
            else:
                timehorizon = end
            plt.rc("font", size=12)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                range(int(start), int(timehorizon + 1)),
                self.SOC[int(start) : int(timehorizon + 1)],
                color="tab:red",
                label="SOC",
            )
            ax.plot(
                range(start, timehorizon),
                self.charge[start:timehorizon],
                color="tab:blue",
                label="Charge",
            )
            ax.plot(
                range(start, timehorizon),
                self.discharge[start:timehorizon],
                color="tab:orange",
                label="Discharge",
            )

            # Same as above
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Power (MW), Energy (MWh)")
            if self.capacity < 1000:
                ax.set_title(
                    self.name + "Timeseries. (" + str(int(self.capacity)) + "MWh)"
                )
            elif self.capacity < 1000000:
                ax.set_title(
                    self.name
                    + "Timeseries. ("
                    + str(int(self.capacity / 100) / 10)
                    + "GWh)"
                )
            else:
                ax.set_title(
                    self.name
                    + "Timeseries. ("
                    + str(int(self.capacity / 100000) / 10)
                    + "TWh)"
                )
            ax.grid(True)
            ax.legend(loc="upper left")

    def charge_sim(
        self, surplus, t_res=1, return_output=False, return_soc=False, start_up_time=0
    ):
        """
        == description ==
        Runs a simulation using opportunistic charging the storage asset.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        t_res: (float) the size of time intervals in hours
        return_output: (boo) whether the smoothed profile should be returned
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).

        == returns ==
        reliability: (float) the percentage of time without shortfalls (0-100)
        output: (Array<float>) the stabilised output profile in MW
        """
        self.reset()
        self.t_res = t_res
        self.start_up_time = start_up_time
        self.charge = 0.0  # intialise stosrage as empty
        self.remaining_surplus = [0] * len(surplus)
        self.curtarray = np.zeros(len(surplus))
        self.missed_demand = [0] * len(surplus)
        self.n_years = len(surplus) / (365.25 * 24 / t_res)
        self.energy_shortfalls = 0
        self.efficiencylosses = 0
        shortfalls = 0  # timesteps where demand could not be met
        self.SOC = []
        # for convenience, these are the maximum charge and discharge rates in MWh
        self.max_c = self.capacity * self.max_c_rate * t_res / 100
        self.max_d = self.capacity * self.max_d_rate * t_res / 100

        for t in range(len(surplus)):
            self.time_step(t, surplus[t])
            self.SOC.append(self.charge / self.capacity)

            if self.remaining_surplus[t] < 0:
                if t > start_up_time:
                    shortfalls += 1
                    self.energy_shortfalls += self.remaining_surplus[t] * -1

        reliability = 100 - ((shortfalls * 100) / (len(surplus) - self.start_up_time))
        self.actual_reliability = reliability
        if return_output is False and return_soc is False:
            return reliability
        elif return_soc is True:
            return [reliability, self.SOC]
        else:
            return [reliability, self.remaining_surplus]

    def self_discharge_timestep(self):
        """
        == description ==
        Reduces stored charge due to self-discharge over one time-step

        == parameters ==
        None

        == returns ==
        None
        """
        # conversion factors because self.dis measured in %/month not MWh/hr
        dischargeamount = (self.self_dis * self.capacity) * self.t_res / (100 * 24 * 30)
        self.charge -= dischargeamount
        self.dischargelosses += dischargeamount
        if self.charge < 0:
            self.charge = 0.0

    def charge_timestep(self, t, surplus):
        """
        == description ==
        Charges the asset for one timestep - either until all the surplus is
        used, the asset is full, or the charging rate limit is reached (which
        ever comes first)

        == parameters ==
        t: (int) the current timestep - so that the output vector can be updated
        suplus: (float) the excess available energy in MWh for that timestep

        == returns ==
        None
        """
        # amount required to fill storage
        to_fill = (self.capacity - self.charge) * 100 / self.eff_in
        if to_fill > self.max_c:
            # if the amount to be put in is greater than the maximum charge rate, set the largest possible input to the maximum charge rate
            largest_in = self.max_c
        else:
            largest_in = to_fill

        if surplus * self.t_res > largest_in:
            # not all surplus can be stored
            # add to the total energy in storage, but multiply by the efficiency to account for losses
            self.charge += largest_in * self.eff_in / 100
            # work out how much is lost to efficiency losses and add to the efficiencylosses counter
            self.efficiencylosses += largest_in * (100 - self.eff_in) / 100
            # track the energy in
            self.en_in += largest_in
            # track the energy that could not be stored
            self.curt += surplus * self.t_res - largest_in
            self.remaining_surplus[t] = surplus - largest_in / self.t_res

        else:
            # all of surplus transfterred to storage
            self.charge += surplus * self.t_res * self.eff_in / 100
            self.efficiencylosses += surplus * (100 - self.eff_in) / 100

            self.en_in += surplus * self.t_res
            self.remaining_surplus[t] = 0.0

    def discharge_timestep(self, t, deficit):
        """
        == description ==
        Discharges the asset for one timestep - either until all the deficit  is
        used, the asset is full, or the charging rate limit is reached (which
        ever comes first)

        == parameters ==
        t: (int) the current timestep - so that the output vector can be updated
        suplus: (float) the excess available energy in MWh for that timestep

        == returns ==
        None
        """
        # amount that can be extracted from storage
        to_empty = self.charge * self.eff_out / 100
        if to_empty > self.max_d:
            # if the maximum amount which can be discharged is less than the remaining amount in the store, set the largest possible output to the maximum discharge rate
            largest_out = self.max_d
        else:
            # otherwise set the largest possible output to the remaining amount in the store
            largest_out = to_empty

        # deficit in this instance is negative, the largest output is positive. Flip the sign of the deficit and see if the largest output possible is enough to meet the surplus
        if deficit * self.t_res * (-1) < largest_out:
            # sufficent storage can be discharged to meet shortfall
            self.charge += deficit * self.t_res * 100 / self.eff_out
            self.efficiencylosses += (
                (-1) * deficit * self.t_res * 100 / self.eff_out
            ) + deficit  # works out the losses due to the effeciency
            self.en_out -= (
                deficit * self.t_res
            )  # deficit is -ve so this effective adds it to the sum: this tracks the energy which has flown out of the store.
            self.remaining_surplus[t] = 0.0  # we've met the shortfall

        else:
            # there is insufficient storage to meet shortfall
            # work out how much demand is missed
            self.missed_demand[t] = deficit + largest_out * self.eff_out / (
                100 * self.t_res
            )
            # we're discharging the max amount possible
            self.en_out += largest_out

            self.remaining_surplus[t] = deficit + largest_out * self.eff_out / (
                100 * self.t_res
            )
            if t >= self.start_up_time:
                shortfall = True
                self.charge -= largest_out * 100 / self.eff_out
                self.efficiencylosses += largest_out * 100 / self.eff_out - largest_out

    def time_step(self, t, surplus):
        """
        == description ==
        This executes a timestep of the charge simulation. If the surplus is
        positive it charges storage and if it is negative it discharges.

        == parameters ==
        t: (int) the current timestep - so that the output vector can be updated
        suplus: (float) the excess available energy in MWh for that timestep

        == returns ==
        None
        """
        self.self_discharge_timestep()

        if surplus > 0:
            self.charge_timestep(t, surplus)
        elif surplus < 0:
            self.discharge_timestep(t, surplus)

    def analyse_usage(self):
        """
        == description ==
        Returns a few of the key metrics for the storage asset.

        == parameters ==
        None

        == returns ==
        en_in (float): the energy put into storage during the simulation (MWh)
        en_out (float): energy recovered from storage during simulation (MWh)
        curt (float): the energy curtailed during the simulation (MWh)
        """

        return [self.en_in, self.en_out, self.curt]

    def size_storage(
        self,
        surplus,
        reliability,
        initial_capacity=0,
        req_res=1e3,
        t_res=1,
        max_capacity=1e8,
        start_up_time=0,
    ):
        """
        == description ==
        Sizes storage or a required system reliability using bisection. Returns
        np.inf if the amount required is above max_storage.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        reliability: (float) required reliability in % (0-100)
        initial_capacity: (float) intital capacity to try in MWh
        req_res: (float) the required capacity resolution in MWh
        t_res: (float) the size of time intervals in hours
        max_storage: (float) the maximum size of storage in MWh to consider
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).

        == returns ==
        (float) required storage capacity (MWh)
        """
        lower = initial_capacity
        upper = max_capacity

        print(f"Desired reliability: {reliability}")

        self.set_capacity(upper)
        rel3 = self.charge_sim(surplus, t_res=t_res, start_up_time=start_up_time)
        if rel3 < reliability:
            self.capacity = np.inf
            return np.inf

        self.set_capacity(lower)
        rel1 = self.charge_sim(surplus, t_res=t_res, start_up_time=start_up_time)
        if rel1 > reliability:
            print("Initial capacity too high")
            if initial_capacity == 0:
                return 0.0
            else:
                self.size_storage(
                    surplus,
                    reliability,
                    initial_capacity=0,
                    req_res=req_res,
                    t_res=t_res,
                    max_capacity=max_capacity,
                    start_up_time=start_up_time,
                    strategy=strategy,
                )

        while upper - lower > req_res:
            mid = (lower + upper) / 2
            self.set_capacity(mid)
            rel2 = self.charge_sim(surplus, t_res=t_res, start_up_time=start_up_time)

            if rel2 < reliability:
                lower = mid
                rel1 = rel2
            else:
                upper = mid
                rel3 = rel2

        return (upper + lower) / 2


class BatteryStorageModel(StorageModel):
    def __init__(
        self,
        eff_in=95,
        eff_out=95,
        self_dis=2,
        storageCapex=391 * 10**3,
        storageFixedOpex=7 * 10**3,
        storagelifetime=15,
        storageVarOpex=0,
        chargeCapex=0,
        chargeFixedOpex=0,
        chargeVarOpex=0,
        chargeLifetime=15,
        dischargeCapex=0,
        dischargeFixedOpex=0,
        dischargeVarOpex=0,
        dischargeLifetime=15,
        max_c_rate=25,
        max_d_rate=25,
        capacity=1,
    ):
        super().__init__(
            cost_params_file=None,
            technical_params_file=None,
            eff_in=eff_in,
            eff_out=eff_out,
            self_dis=self_dis,
            storageCapex=storageCapex,
            storageFixedOpex=storageFixedOpex,
            storagelifetime=storagelifetime,
            storageVarOpex=storageVarOpex,
            chargeCapex=chargeCapex,
            chargeFixedOpex=chargeFixedOpex,
            chargeVarOpex=chargeVarOpex,
            chargeLifetime=chargeLifetime,
            dischargeCapex=dischargeCapex,
            dischargeFixedOpex=dischargeFixedOpex,
            dischargeVarOpex=dischargeVarOpex,
            dischargeLifetime=dischargeLifetime,
            hurdleRate=0.08,
            max_c_rate=max_c_rate,
            max_d_rate=max_d_rate,
            name="Li-Ion Battery",
            capacity=capacity,
            limits=[0, 1000000000],
        )


class HydrogenStorageModel(StorageModel):
    def __init__(
        self,
        cost_params_file="params/SCORES Cost assumptions.xlsx",
        technical_params_file="params/Storage_technical_assumptions.xlsx",
        cost_sensitivity="Medium",
        cost_year=2025,
        storage_param_entry="Salt Cavern",
        charge_param_entry="PEM",
        discharge_param_entry="Hydrogen CCGT",
        eff_in=None,
        eff_out=None,
        self_dis=None,
        storageCapex=None,
        storageFixedOpex=None,
        storagelifetime=None,
        storageVarOpex=None,
        chargeCapex=None,
        chargeFixedOpex=None,
        chargeVarOpex=None,
        chargeLifetime=None,
        dischargeCapex=None,
        dischargeFixedOpex=None,
        dischargeVarOpex=None,
        dischargeLifetime=None,
        max_c_rate=None,
        max_d_rate=None,
        capacity=1,
        initial_charge=1,
        hurdleRate=None,
        limits=[0, 1000000000],
    ):
        super().__init__(
            cost_params_file=cost_params_file,
            technical_params_file=technical_params_file,
            cost_sensitivity=cost_sensitivity,
            cost_year=cost_year,
            storage_param_entry=storage_param_entry,
            charge_param_entry=charge_param_entry,
            discharge_param_entry=discharge_param_entry,
            eff_in=eff_in,
            eff_out=eff_out,
            self_dis=self_dis,
            storageCapex=storageCapex,
            storageFixedOpex=storageFixedOpex,
            storagelifetime=storagelifetime,
            storageVarOpex=storageVarOpex,
            chargeCapex=chargeCapex,
            chargeFixedOpex=chargeFixedOpex,
            chargeVarOpex=chargeVarOpex,
            chargeLifetime=chargeLifetime,
            dischargeCapex=dischargeCapex,
            dischargeFixedOpex=dischargeFixedOpex,
            dischargeVarOpex=dischargeVarOpex,
            dischargeLifetime=dischargeLifetime,
            hurdleRate=hurdleRate,
            max_c_rate=max_c_rate,
            max_d_rate=max_d_rate,
            name="Hydrogen Storage",
            capacity=capacity,
            limits=limits,
            initial_charge=initial_charge,
        )


class ThermalStorageModel(StorageModel):
    def __init__(
        self,
        eff_in=80,
        eff_out=47,
        self_dis=9.66,
        variable_cost=331.6,
        fixed_cost=773.5,
        max_c_rate=8.56,
        max_d_rate=6.82,
        capacity=1,
    ):
        super().__init__(
            eff_in,
            eff_out,
            self_dis,
            variable_cost,
            fixed_cost,
            max_c_rate,
            max_d_rate,
            "Thermal",
            capacity=capacity,
        )


class MultipleStorageAssets:
    def __init__(
        self,
        assets,
        c_order=None,
        d_order=None,
        DispatchableAssetList=None,
        DispatchTimeHorizon=24,
        Interconnector=None,
    ):
        """
        == description ==
        Initialisation of a multiple storage object. Note that if charging or
        discharging orders are not specified the model defaults to discharge in
        the order of the asset list, and charge in the reverse.

        == parameters ==
        assets: (Array<StorageModel>) a list of storage model objects
        c_order: (Array<int>) a list of the order which assets should be
            prioritised for charging under 'ordered' operation
        d_order: (Array<int>) a list of the order which assets should be
            prioritised for discharging under 'ordered' operation
        DispatchableAssetList: (Array<DispatchableAsset>) a list of dispatchable assets, in merit order


        == returns ==
        None
        """
        self.energy_shortfalls = 0
        self.actual_reliability = 0
        self.assets = assets
        self.n_assets = len(assets)
        self.unit_capacity = [0.0] * len(assets)
        self.rel_capacity = [0.0] * len(assets)
        self.units = {}
        # added by cormac for plotting timeseries from optimisation
        self.surplus = np.empty([])  # the last surplus used as input for optimise
        self.Pfos = np.empty(
            []
        )  # the necessary fossil fuel generation timeseries from the last optimise run
        self.Shed = np.empty([])  # timeseries of surplus shedding
        self.DispatchableAssetList = DispatchableAssetList
        self.DispatchTimeHorizon = DispatchTimeHorizon
        self.Interconnector = Interconnector
        if DispatchableAssetList is not None:
            self.DispatchEnabled = True
        else:
            self.DispatchEnabled = False
        if c_order is None:
            c_order = list(range(self.n_assets))

        if d_order is None:
            d_order = list(range(self.n_assets))

        # if the order is not specified, default to the order of the assets in the list of assets

        self.c_order = c_order
        self.d_order = d_order

        for i in range(self.n_assets):
            # adding each asset as a "unit"
            self.units[i] = assets[i]

            self.unit_capacity[i] = assets[i].capacity

        total_capacity = sum(self.unit_capacity)
        self.capacity = total_capacity
        # work out the relative size of
        for i in range(self.n_assets):
            self.rel_capacity[i] = float(self.unit_capacity[i]) / total_capacity

    def reset(self):
        """
        == description ==
        Resets the measurement on all storage units.

        == parameters ==
        None

        == returns ==
        None
        """
        for i in range(self.n_assets):
            self.units[i].reset()

        self.surplus = np.empty([])
        self.Pfos = np.empty([])
        self.Shed = np.empty([])

    def set_capacity(self, capacity):
        """
        == description ==
        Scales the total installed capacity to the specified value, the
        relative capacity of the individual assets remains the same.

        == parameters ==
        capacity: (float) The total installed capacity in MWh

        == returns ==
        None
        """
        for i in range(self.n_assets):
            self.units[i].set_capacity(capacity * self.rel_capacity[i])
        self.capacity = capacity

    def self_discharge_timestep(self):
        """
        == description ==
        Self-discharge all assets for one timestep.

        == parameters ==
        None

        == returns ==
        None
        """
        for i in range(self.n_assets):
            self.units[i].self_discharge_timestep()

    def is_MultipleStorageAssets(self):
        """
        == description ==
        Returns True if it is a Multiple Storage Asset

        == parameters ==
        None

        == returns ==
        (float) True
        """
        return True

    def get_cost(self):
        """
        == description ==
        Gets the cumulative cost of all of the storage assets.

        == parameters ==
        None

        == returns ==
        (float) total cost of all storage units in GBP/yr
        """
        if self.capacity == np.inf:
            return np.inf
        else:
            total = 0.0
            for i in range(self.n_assets):
                total += self.units[i].get_cost()
            return total

    def charge_specfied_order(
        self,
        surplus,
        c_order,
        d_order,
        t_res=1,
        return_output=False,
        start_up_time=0,
        return_di_av=False,
    ):
        """
        == description ==
        Charges the storage assets in the order specified by c_order

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        c_order: (Array<int>) the order in which to charge the assets
        d_order: (Array<int>) the order in which to discharge the assets
        t_res: (float) the size of time intervals in hours
        return_output: (boo) whether the smoothed profile should be returned
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).
        return_di_av: (boo) whether the average discharge and charge profiles
            should be returned

        == returns ==
        None
        """
        if len(c_order) != self.n_assets:
            raise Exception("c_order wrong length")
        if len(d_order) != self.n_assets:
            raise Exception("d_order wrong length")

        shortfalls = 0
        self.energy_shortfalls = 0  # keeps track of total energy  shortfalls

        remaining_surplus = [0] * len(
            surplus
        )  # keeps track of the remaining surplus after each timestep
        self.curtarray = np.zeros(
            len(surplus)
        )  # keeps track of the total surplus that could not be stored
        soc = []
        for i in range(self.n_assets):
            soc.append([i])
        self.curt = 0.0
        di_profiles = (
            {}
        )  # keeps track of the average daily discharge and charge profiles
        T = int(24 / t_res)
        for i in range(len(c_order)):
            di_profiles[i] = {"c": [0.0] * T, "d": [0.0] * T}

        # initialise all storage units
        for i in range(self.n_assets):
            # set the maximum charge and discharge rates in units of MWh instead of %
            self.units[i].max_c = (
                self.units[i].capacity * self.units[i].max_c_rate * t_res / 100
            )
            self.units[i].max_d = (
                self.units[i].capacity * self.units[i].max_d_rate * t_res / 100
            )
            self.units[i].t_res = t_res
            self.units[i].start_up_time = start_up_time
            self.units[i].charge = self.units[i].capacity * self.units[i].initial_charge
            self.units[i].n_years = len(surplus) / (365.25 * 24 / t_res)
            self.units[i].remaining_surplus = [0] * len(surplus)
            self.units[i].missed_demand = [0] * len(surplus)
            self.units[i].energy_shortfalls = 0
            self.units[i].efficiencylosses = 0
            self.units[i].chargetimeseries = []
            self.units[i].SOC = []
        for t in range(len(surplus)):
            # this steps through every value in the surplus array, and performs the required steps: self discharge, and then charge or discharge
            # self discharge all assets
            self.self_discharge_timestep()

            t_surplus = copy.deepcopy(surplus[t])

            if t_surplus > 0:
                # if the surplus is positive, then we want to charge the storage assets
                for i in range(self.n_assets):
                    self.units[i].SOC.append(self.units[i].charge)
                    if t_surplus > 0:
                        self.units[c_order[i]].charge_timestep(t, t_surplus)
                        remaining_surplus[t] = self.units[c_order[i]].remaining_surplus[
                            t
                        ]
                        if t > start_up_time:
                            di_profiles[i]["c"][t % T] += (
                                remaining_surplus[t] - t_surplus
                            )
                        t_surplus = self.units[c_order[i]].remaining_surplus[t]
                if self.Interconnector != None:
                    t_surplus = self.Interconnector.export(t, t_surplus)
                    remaining_surplus[t] = t_surplus

                self.curt += remaining_surplus[t]
                self.curtarray[t] = remaining_surplus[t]

            elif t_surplus < 0:
                # if the surplus is negative, then we want to discharge the storage assets
                if self.DispatchEnabled:
                    # we want to see if the energy demand over the time horizon exceeds the energy available from the storage
                    # if it does we will need to dispatch the dispatchable asset

                    if t > len(surplus) - self.DispatchTimeHorizon:
                        lengthoftimehorizon = len(surplus) - t
                    else:
                        lengthoftimehorizon = self.DispatchTimeHorizon
                    # we now need to predict the storage levels for the timehorizon. We'll simulate the storage levels for the time horizon,
                    # and if the storage levels are zero at any point, we will dispatch the dispatchable asset
                    storelevels = [self.units[i].charge for i in range(self.n_assets)]
                    maxcapacity = [self.units[i].capacity for i in range(self.n_assets)]

                    for hourinfutureprediction in range(lengthoftimehorizon):
                        this_surplus = surplus[t + hourinfutureprediction]

                        if this_surplus >= 0:

                            for i in range(self.n_assets):
                                if this_surplus > 0:
                                    storelevels[c_order[i]] += (
                                        this_surplus
                                        * self.units[c_order[i]].eff_in
                                        / 100
                                    )

                                    if (
                                        storelevels[c_order[i]]
                                        > maxcapacity[c_order[i]]
                                    ):
                                        # if the store is full, then only part of the surplus can be stored
                                        this_surplus -= (
                                            (
                                                storelevels[c_order[i]]
                                                - maxcapacity[c_order[i]]
                                            )
                                            * self.units[c_order[i]].eff_in
                                            / 100
                                        )
                                        storelevels[c_order[i]] = maxcapacity[
                                            c_order[i]
                                        ]
                                    else:
                                        # if the store is not full, then the surplus is all stored
                                        this_surplus = 0
                        elif this_surplus < 0:
                            for i in range(self.n_assets):
                                if this_surplus < 0:
                                    storelevels[d_order[i]] += (
                                        this_surplus
                                        * 100
                                        / self.units[d_order[i]].eff_out
                                    )
                                    if storelevels[d_order[i]] < 0:
                                        # in this case, the store has been discharged more than it can be, so we'll set it to zero. We'll then reduce the surplus
                                        # by the amount that was discharged
                                        this_surplus += (
                                            storelevels[d_order[i]]
                                            * 100
                                            / self.units[d_order[i]].eff_out
                                        )
                                        storelevels[d_order[i]] = 0
                                    else:
                                        this_surplus = 0

                        summedstorelevels = sum(storelevels)
                        if summedstorelevels <= 0:
                            for DispatchableAsset in self.DispatchableAssetList:
                                t_surplus = DispatchableAsset.dispatch(t, t_surplus)
                                remaining_surplus[t] = t_surplus
                            break

                for i in range(self.n_assets):
                    self.units[i].SOC.append(self.units[i].charge)

                    if t_surplus < 0:
                        self.units[d_order[i]].discharge_timestep(t, t_surplus)
                        remaining_surplus[t] = self.units[d_order[i]].remaining_surplus[
                            t
                        ]
                        if t > start_up_time:
                            di_profiles[i]["d"][t % T] += (
                                remaining_surplus[t] - t_surplus
                            )
                        t_surplus = self.units[d_order[i]].remaining_surplus[t]

                if remaining_surplus[t] < 0:
                    if t > start_up_time:
                        shortfalls += 1
                        self.energy_shortfalls += remaining_surplus[t] * -1
            for i in range(self.n_assets):
                self.units[i].chargetimeseries.append(self.units[i].charge)

        reliability = 100 - ((shortfalls * 100) / (len(surplus) - start_up_time))

        if return_output is False and return_di_av is False:
            return reliability

        ret = [reliability]

        if return_output is True:
            ret += [remaining_surplus]

        if return_di_av is True:
            sf = (len(surplus) - start_up_time) / T
            for i in di_profiles:
                for t in range(T):
                    di_profiles[i]["c"][t] = float(di_profiles[i]["c"][t]) / sf
                    di_profiles[i]["d"][t] = float(di_profiles[i]["d"][t]) / sf
            ret += [di_profiles]

        return ret

    def charge_sim(
        self,
        surplus,
        t_res=1,
        return_output=False,
        start_up_time=0,
        strategy="ordered",
        return_di_av=False,
    ):
        """
        == description ==
        This simulates the charging of storage assets. This is a holder class to enable other function to be added when other charging strategies are developed

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        t_res: (float) the size of time intervals in hours
        return_output: (boo) whether the smoothed profile should be returned
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).
        strategy: (str) the strategy for operating the assets. Options:
                'ordered' - charges/discharges according to self.c_order/d_order
                'balanced' - ?

        == returns ==
        reliability: (float) the percentage of time without shortfalls (0-100)
        output: (Array<float>) the stabilised output profile in MW
        """

        if strategy == "ordered":
            res = self.charge_specfied_order(
                surplus,
                self.c_order,
                self.d_order,
                t_res=t_res,
                return_output=return_output,
                start_up_time=start_up_time,
                return_di_av=return_di_av,
            )
        self.actual_reliability = res
        return res

    def analyse_usage(self):
        """
        == description ==
        Get the usage of each storage asset following a simulation.

        == parameters ==
        None

        == returns ==
        en_in (Array<float>): the energy put into each storage asset during the
            simulation (MWh)
        en_out (Array<float>): energy recovered from each storage asset during
            the simulation (MWh)
        curt (float): the energy curtailed during the simulation (MWh)
        """
        stored = []
        recovered = []
        # formerly per year, now returned as net
        # for i in range(self.n_assets):
        #     stored.append(self.units[i].en_in/self.units[i].n_years)
        #     recovered.append(self.units[i].en_out/self.units[i].n_years)
        # curtailed = self.curt/self.units[i].n_years

        for i in range(self.n_assets):
            stored.append(self.units[i].en_in)
            recovered.append(self.units[i].en_out)
        curtailed = self.curt

        return [stored, recovered, curtailed]

    def causal_system_operation(
        self,
        demand,
        power,
        c_order,
        d_order,
        Mult_aggEV,
        start,
        end,
        IncludeEVLeapDay=True,
        t_res=1,
        start_up_time=24,
        plot_timeseries=False,
        V2G_discharge_threshold=0.0,
        initial_SOC=[0.5],
    ):
        """
        == description ==
        This function is similiar to charge specified order but with two key differences:
            1) It allows aggregated EVs to be operated also (the order of their discharge specified). To do this it
                it splits each EV fleet into two batteries, one for V2G and one for Smart, these are then operated seperately.
            2) It outputs two new outputs: system reliability based on amount of demand served by renewables
               rather than the old reliability metric based on time where renewables don't cover everything; and
               EV reliability which gives the % under delivery of power to the EVs.

        == parameters ==
        demand: array <floats> this is +ve values, a timeseries of the system passive demand (i.e. that not from EVs) (MW)
        power: array <float> generation profile of the renewables (MW), must be the same length as the demand
        Mult_aggEV: (MultipleAggregatedEVs) different fleets of EVs with defined chargertype ratios!
        start and end: <datetime> the start and end time of the simulation. These are needed to construct the correct EV connectivity timeseries.
        c_order: list <int>, of order of the charge with c_order[0] being charged first, c_order[1] charged second etc...,
                             the numbering refers to: 0:(n_stor_assets-1) refers to the storage units in order
                                                     n_stor_assets:(n_stor_assets + 2*n_aggEV_fleets -1) for EV fleets, where the number refer to the virtual batteries representing: V2G_fleet0, smart_fleet0, V2G_fleet1, smart_fleet1...
        start_up_time: <int>, number of hours before reliability results are calculated
        plot_timeseries: (bool), if true will plot the storage SOCs and charge/discharge, as well as the surplus before and after adjustement. The
        V2G_discharge_threshold: (float), The kWh limit for the EV batteries, below whcih V2G will not discharge. The state of charge can still drop below this due to driving energy, but V2G will not discharge when the SOC is less than this value.
        initial_SOC:  array<floats>, float value between 0:1, determines the start SOC of the EVs and batteries (i.e. 0.5 corresponds to them starting 50% fully charged)
                            if single float given, all storage + EVs start on it, if given as array, allows choosing of individual storage start SOCs, specified in order: [stor0,stor1…,Fleet0 V2G, Fleet0 Uni, Fleet1 V2G…]

        == returns ==
        dataframe <Causal Reliability,EV_Reliability>: Causal Reliability is the % total demand (EV demand + passive demand) that is met by renewable energy
                                                 EV_Reliability: is the % of driving energy met by renewable energy. Given in order [Fleet0 V2g, Fleet0 Unidirectional, Fleet1 V2G, ...]
                                                 For V2G this can be -ve, as when the EVs are plugged back in they can be discharged to zero again, thus they will need to be charged to 90% from zero rather than from about 30% as for the Unidirectional.
                                                 Thus the energy needed from fossil fuels is larger that the driving energy.
        """
        if np.asarray(power).size != np.asarray(demand).size:
            raise Exception("power and demand timeseries must be the same length")

        if sum(np.asarray(demand) < 0) != 0:
            raise Exception("demand timeseries must contain only +ve values")

        surplus = np.asarray(power) - np.asarray(demand)
        surplus = surplus.tolist()
        units = {}
        counter = 0
        for i in range(self.n_assets):
            units[i] = self.assets[i]
            counter = counter + 1

        # split the EV fleets into a battery for Smart and a Battery for non smart
        for k in range(Mult_aggEV.n_assets):
            for b in range(2):
                units[counter] = BatteryStorageModel()
                counter = counter + 1

        Num_units = len(units)

        if len(c_order) != Num_units:
            raise Exception(
                "c_order wrong length, need two entries for every agg fleet object"
            )
        if len(d_order) != Num_units:
            raise Exception(
                "d_order wrong length, need two entries for every agg fleet object"
            )

        power_deficit = 0.0  # this is the energy in MWh met by fossil fuels, including for EV driving demand!
        remaining_surplus = [0] * len(surplus)  # this is the surplus after charging!
        self.curt = 0.0
        di_profiles = {}
        T = int(24 / t_res)
        for i in range(len(c_order)):
            di_profiles[i] = {"c": [0.0] * len(surplus), "d": [0.0] * len(surplus)}

        # initialise all storage units (EVs updated at each timestep)
        counter = 0
        for i in range(self.n_assets):
            units[i].max_c = units[i].capacity * units[i].max_c_rate * t_res / 100
            units[i].max_d = units[i].capacity * units[i].max_d_rate * t_res / 100
            units[i].t_res = t_res

            # if initial_SOC is float, then uniform start SOC
            if len(initial_SOC) == 1:
                units[i].charge = initial_SOC[0] * units[i].capacity
            else:
                if len(initial_SOC) != self.n_assets + Mult_aggEV.n_assets * 2:
                    raise Exception(
                        "Error, Initial SOC must either be a float or list of length (self.n_assets + Mult_aggEV.n_assets*2)"
                    )
                units[i].charge = initial_SOC[i] * units[i].capacity
            counter = counter + 1

        for i in range(Num_units):
            units[i].start_up_time = 0
            units[i].n_years = len(surplus) / (365.25 * 24 / t_res)
            units[i].remaining_surplus = [0] * len(
                surplus
            )  # this is the left over defecit after the charge action on asset i
            units[i].t_res = t_res

        # Elongate the EV connectivity data if necessesary #
        Mult_aggEV.construct_connectivity_timeseries(start, end, IncludeEVLeapDay)

        # Begin simulating system #
        EV_Energy_Underserve = np.zeros(
            [Mult_aggEV.n_assets * 2]
        )  # this is the total energy for the EVs that needs to be supplied by fossil fuels
        Total_Driving_Energy = np.zeros(
            [Mult_aggEV.n_assets * 2]
        )  # this is the total desired plugout energy of the EVs
        charge_hist = np.zeros([Num_units, len(surplus)])
        V2G = True

        for t in range(len(surplus)):
            # self discharge all assets
            for i in range(Num_units):
                units[i].self_discharge_timestep()
                # Update State of the Aggregated EV batteries #
                if i >= self.n_assets:
                    if V2G:
                        k = int((i + 1 - self.n_assets) / 2)
                        b = 0
                    else:
                        b = 1

                    # work out the energy remaining after the EVs unplug
                    if t == 0:
                        if Mult_aggEV.assets[k].Eout != Mult_aggEV.assets[k].max_SOC:
                            raise Exception(
                                "The max SOC does not equal the plugout SOC. This leads to errors in the causal system operation. Make these the same or improve code."
                            )
                        N = Mult_aggEV.assets[k].N[t]
                        if len(initial_SOC) == 1:
                            units[i].charge = (
                                initial_SOC[0]
                                * N
                                * Mult_aggEV.assets[k].chargertype[b]
                                * Mult_aggEV.assets[k].number
                                * Mult_aggEV.assets[k].max_SOC
                                / 1000
                            )
                        else:
                            units[i].charge = (
                                initial_SOC[self.n_assets + 2 * k + b]
                                * N
                                * Mult_aggEV.assets[k].chargertype[b]
                                * Mult_aggEV.assets[k].number
                                * Mult_aggEV.assets[k].max_SOC
                                / 1000
                            )

                    Energy_Remaining = (
                        units[i].charge
                        - Mult_aggEV.assets[k].Nout[t]
                        * Mult_aggEV.assets[k].chargertype[b]
                        * Mult_aggEV.assets[k].number
                        * Mult_aggEV.assets[k].Eout
                        / 1000
                    )  # work out the energy remaining after the EVs have plugged out
                    if t >= start_up_time:
                        Total_Driving_Energy[k + b] += (
                            Mult_aggEV.assets[k].Nout[t]
                            * Mult_aggEV.assets[k].chargertype[b]
                            * Mult_aggEV.assets[k].number
                            * Mult_aggEV.assets[k].Eout
                            / 1000
                            - Mult_aggEV.assets[k].Nin[t]
                            * Mult_aggEV.assets[k].chargertype[b]
                            * Mult_aggEV.assets[k].number
                            * Mult_aggEV.assets[k].Ein
                            / 1000
                        )

                    # if there is sufficient for the driving, update the SOC and continue
                    if Energy_Remaining > 0:
                        units[i].charge = (
                            Energy_Remaining
                            + Mult_aggEV.assets[k].Nin[t]
                            * Mult_aggEV.assets[k].chargertype[b]
                            * Mult_aggEV.assets[k].number
                            * Mult_aggEV.assets[k].Ein
                            / 1000
                        )

                    # if there is not, set the charge to 0 and record the underserve
                    else:
                        units[i].charge = (
                            Mult_aggEV.assets[k].Nin[t]
                            * Mult_aggEV.assets[k].chargertype[b]
                            * Mult_aggEV.assets[k].number
                            * Mult_aggEV.assets[k].Ein
                            / 1000
                        )
                        EV_Energy_Underserve[k + b] += -Energy_Remaining
                        if t >= start_up_time:
                            power_deficit += -Energy_Remaining

                    # update the max charge limit
                    if V2G:
                        # N = N + Mult_aggEV.assets[k].Nin[t] - Mult_aggEV.assets[k].Nout[t]
                        N = Mult_aggEV.assets[k].N[t]
                        V2G = False
                        discharge_threshold = (
                            N
                            * Mult_aggEV.assets[k].chargertype[b]
                            * Mult_aggEV.assets[k].number
                            * V2G_discharge_threshold
                            / 1000
                        )

                        if units[i].charge <= discharge_threshold:
                            units[i].max_d = 0.0
                        else:
                            units[i].max_d = min(
                                N
                                * Mult_aggEV.assets[k].chargertype[b]
                                * Mult_aggEV.assets[k].number
                                * Mult_aggEV.assets[k].max_d_rate
                                / 1000
                                * t_res,
                                units[i].charge - discharge_threshold,
                            )
                    else:
                        V2G = True
                        units[i].max_d = 0.0

                    units[i].max_c = (
                        N
                        * Mult_aggEV.assets[k].chargertype[b]
                        * Mult_aggEV.assets[k].number
                        * Mult_aggEV.assets[k].max_c_rate
                        * t_res
                        / 1000
                    )
                    units[i].capacity = (
                        N
                        * Mult_aggEV.assets[k].chargertype[b]
                        * Mult_aggEV.assets[k].number
                        * Mult_aggEV.assets[k].max_SOC
                        / 1000
                    )

            t_surplus = copy.deepcopy(surplus[t])

            if t_surplus >= 0:
                for i in range(Num_units):
                    units[c_order[i]].charge_timestep(t, t_surplus)
                    remaining_surplus[t] = units[c_order[i]].remaining_surplus[t]
                    di_profiles[c_order[i]]["c"][t] = remaining_surplus[t] - t_surplus
                    t_surplus = units[c_order[i]].remaining_surplus[t]
                    charge_hist[c_order[i], t] = units[c_order[i]].charge
                self.curt += remaining_surplus[t]

            elif t_surplus < 0:
                for i in range(Num_units):
                    units[d_order[i]].discharge_timestep(t, t_surplus)
                    remaining_surplus[t] = units[d_order[i]].remaining_surplus[t]
                    di_profiles[d_order[i]]["d"][t] = remaining_surplus[t] - t_surplus
                    t_surplus = units[d_order[i]].remaining_surplus[t]

                    charge_hist[d_order[i], t] = units[d_order[i]].charge
                if t >= start_up_time:
                    power_deficit += -remaining_surplus[
                        t
                    ]  # this is the power that needs to be supplied by fossil fuels

        if plot_timeseries:
            timehorizon = len(surplus)
            plt.rc("font", size=12)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(range(timehorizon), surplus, color="k", label="Surplus")
            ax.plot(
                range(timehorizon),
                remaining_surplus,
                color="b",
                label="Surplus post Charging",
            )
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Power (MW)")
            ax.set_title("Surplus Timeseries")
            ax.legend(loc="upper left")

            for i in range(Num_units):
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(timehorizon), charge_hist[i, :], color="k", label="SOC")
                # print(i, max(di_profiles[i]['c'][:]))
                ax.plot(
                    range(timehorizon),
                    di_profiles[i]["c"][:],
                    color="r",
                    label="Charge",
                )
                ax.plot(
                    range(timehorizon),
                    di_profiles[i]["d"][:],
                    color="b",
                    label="Discharge",
                )
                if i < self.n_assets:
                    ax.set_title(self.assets[i].name + " (Unit " + str(i) + ")")
                else:
                    if (i - self.n_assets) % 2 == 0:
                        asset_no = int((i - self.n_assets) / 2)
                        ax.set_title(
                            Mult_aggEV.assets[asset_no].name
                            + " V2G (Unit "
                            + str(i)
                            + ")"
                        )
                    elif (i - self.n_assets) % 2 == 1:
                        ax.set_title(
                            Mult_aggEV.assets[asset_no].name
                            + " Unidirectional (Unit "
                            + str(i)
                            + ")"
                        )
                ax.set_xlabel("Time (h)")
                ax.set_ylabel("Power (MW), Energy (MWh)")
                ax.legend(loc="upper left")

        EV_Reliability = np.ones([Mult_aggEV.n_assets * 2]) * 100
        for i in range(Mult_aggEV.n_assets * 2):
            if Total_Driving_Energy[i] > 0:
                EV_Reliability[i] = (
                    (Total_Driving_Energy[i] - EV_Energy_Underserve[i])
                    / Total_Driving_Energy[i]
                ) * 100  # the % of driving energy met with renewables
        ret = [int(power_deficit), EV_Reliability]

        # Record Outputs #

        Causal_EV_Reliability = (
            []
        )  # The % of driving Energy Provided by Renewable energy (if no chargers of a certain type this is set to 100%)
        for x in range(2 * Mult_aggEV.n_assets):
            Causal_EV_Reliability.append([])

        Causal_Reliability = (
            1 - (ret[0] / (sum(demand[start_up_time:]) + sum(Total_Driving_Energy)))
        ) * 100
        # print('Power Def',ret[0],'Total Demand', sum(demand[start_up_time:]), 'Total Driving Energy', sum(Total_Driving_Energy))
        b = 0
        for x in range(2 * Mult_aggEV.n_assets):
            Causal_EV_Reliability[x].append(ret[1][b])
            b = 1 - b

        df = DataFrame({"Causal Reliability": [Causal_Reliability]})

        for x in range(Mult_aggEV.n_assets):
            df["Fleet " + str(x) + " V2G"] = Causal_EV_Reliability[x]
            df["Fleet " + str(x) + " Uni"] = Causal_EV_Reliability[x + 1]

        return df

    def non_causal_system_operation(
        self,
        demand,
        power,
        Mult_aggEV,
        start,
        end,
        start_up_time=24,
        includeleapdaysEVs=True,
        plot_timeseries=False,
        InitialSOC=[0.5],
        form_model=True,
    ):
        """
        == description ==
        This function non-causally operate the storage and EVs over the given year. To save time on repeated operations, the model can be specified weather it needs to be rebuilt or not.

        == parameters ==
        demand: array <floats> this is +ve values, a timeseries of the system passive demand (i.e. that not from EVs) (MW)
        power: array <float> generation profile of the renewables (MW), must be the same length as the demand
        Mult_aggEV: (MultipleAggregatedEVs) different fleets of EVs with defined chargertype ratios!
        plot_timeseries: (bool), if true will plot the storage SOCs and charge/discharge, as well as the surplus before and after adjustement. The
        initial_SOC:  array<floats>, float value between 0:1, determines the start SOC of the EVs and batteries (i.e. 0.5 corresponds to them starting 50% fully charged)
                            if single float given, all storage + EVs start on it, if given as array, allows choosing of individual storage start SOCs, specified in order: [stor0,stor1…,Fleet0 V2G, Fleet0 Uni, Fleet1 V2G…]
        form_model: (bool), when true the function will form the entire model, when false it will use the model previously created (this saves time during repeated simulations)
        start_up_time: <int>, number of hours before reliability results are calculated

        == returns ==
        Non Causal Reliability <float>: Non Causal Reliability is the % total demand (EV demand + passive demand) that is met by renewable energy. Unlike Non Causal Operation, EV reliability is always 100% as these are hard
                                            constraints within the optimisation. This may come at the cost of decreased total Causal reliability however.

        """

        # constrain the storage and EVs to have their set value
        # the storage and EV objects are copied to not overwrite teh orignals if this is being used within Run_then_opt
        sim_Mult_Stor = self
        for i in range(sim_Mult_Stor.n_assets):
            sim_Mult_Stor.assets[i].limits = [
                sim_Mult_Stor.assets[i].capacity,
                sim_Mult_Stor.assets[i].capacity,
            ]

        sim_Mult_aggEV = Mult_aggEV
        for k in range(sim_Mult_aggEV.n_assets):
            sim_Mult_aggEV.assets[k].limits = []
            for b in range(2):
                sim_Mult_aggEV.assets[k].limits.append(
                    sim_Mult_aggEV.assets[k].chargertype[b]
                    * sim_Mult_aggEV.assets[k].number
                )
                sim_Mult_aggEV.assets[k].limits.append(
                    sim_Mult_aggEV.assets[k].chargertype[b]
                    * sim_Mult_aggEV.assets[k].number
                )

        # for the non causal operation want to remove constraint on fossil fuel use, but heavily cost it so the optimiser will operate the system at lowest carbon. The built capacities are also fixed!
        if form_model:
            x2 = System_LinProg_Model(
                surplus=np.asarray(power - demand),
                fossilLimit=10000.0,
                Mult_Stor=sim_Mult_Stor,
                Mult_aggEV=sim_Mult_aggEV,
            )
            x2.Form_Model(
                start_EV=start,
                end_EV=end,
                SizingThenOperation=False,
                includeleapdays=includeleapdaysEVs,
                fossilfuelpenalty=10000000.0,
                StartSOCEqualsEndSOC=False,
                InitialSOC=InitialSOC,
            )
            self.non_causal_linprog = x2
        else:
            # update with correct gen data
            for t in self.non_causal_linprog.model.TimeIndex:
                self.non_causal_linprog.model.Demand[t] = power[t] - demand[t]

        self.non_causal_linprog.Run_Sizing()
        store_optimisation_results(
            self.non_causal_linprog.model, sim_Mult_aggEV, sim_Mult_Stor
        )

        # Plot the Timeseries of EV and Storage Charging if Required
        if plot_timeseries:
            timehorizon = len(demand)
            plt.rc("font", size=12)
            fig, ax = plt.subplots(figsize=(10, 6))
            surplus = power - demand
            ax.plot(range(timehorizon), surplus, color="k", label="Surplus")

            # work out the surplus post charging
            surplus_pc = surplus
            for k in range(sim_Mult_aggEV.n_assets):
                # subtract the Smart AND V2G Charging amounts
                surplus_pc += (
                    np.asarray(sim_Mult_aggEV.assets[k].discharge)
                    - np.asarray(sim_Mult_aggEV.assets[k].charge[:, 0])
                    - np.asarray(sim_Mult_aggEV.assets[k].charge[:, 1])
                )
                # print('discharge EV',np.asarray(sim_Mult_aggEV.assets[k].discharge))
            # print('charge EV',np.asarray(sim_Mult_aggEV.assets[k].charge[0:20]))

            for i in range(sim_Mult_Stor.n_assets):
                surplus_pc += -np.asarray(
                    sim_Mult_Stor.assets[i].discharge
                ) - np.asarray(sim_Mult_Stor.assets[i].charge)
                # print('discharge EV',np.asarray(sim_Mult_Stor.assets[k].discharge))
                # print('charge EV',np.asarray(sim_Mult_Stor.assets[k].charge[0:20]))

            ax.plot(
                range(timehorizon), surplus_pc, color="b", label="Surplus post Charging"
            )
            ax.set_title("Surplus Timeseries")
            ax.legend(loc="upper left")

            for i in range(sim_Mult_Stor.n_assets):
                sim_Mult_Stor.assets[i].plot_timeseries()

            for k in range(sim_Mult_aggEV.n_assets):
                sim_Mult_aggEV.assets[k].plot_timeseries(withSOClimits=False)

        # Record Outputs #

        # work out driving demand
        Total_Driving_Energy = np.zeros(
            [Mult_aggEV.n_assets * 2]
        )  # Fleet0 V2G, Fleet0 Smart, Fleet1 V2G, Fleet1 Smart...
        for t in range(len(demand)):
            for k in range(Mult_aggEV.n_assets):
                for b in range(2):
                    if t >= start_up_time:
                        Total_Driving_Energy[k + b] += (
                            Mult_aggEV.assets[k].Nout[t]
                            * Mult_aggEV.assets[k].chargertype[b]
                            * Mult_aggEV.assets[k].number
                            * Mult_aggEV.assets[k].Eout
                            / 1000
                            - Mult_aggEV.assets[k].Nin[t]
                            * Mult_aggEV.assets[k].chargertype[b]
                            * Mult_aggEV.assets[k].number
                            * Mult_aggEV.assets[k].Ein
                            / 1000
                        )

        Non_Causal_Reliability = (
            1
            - sum(
                pyo.value(self.non_causal_linprog.model.Pfos[:])[
                    start_up_time : len(demand) - 1
                ]
            )
            / (sum(Total_Driving_Energy) + sum(demand[start_up_time:]))
        ) * 100
        # print('Power Def',sum(pyo.value(self.non_causal_linprog.model.Pfos[:])[start_up_time:len(demand)-1]),'Total Demand', sum(demand[start_up_time:]), 'Total Driving Energy', sum(Total_Driving_Energy))
        return int(Non_Causal_Reliability * 10000) / 10000

    def plot_timeseries(self, start=0, end=-1):
        """
        == parameters ==
        start: (int) start time of plot
        end: (int) end time of plot

        """

        if self.Pfos.shape == ():
            print(
                "Charging timeseries not avaialable, try running MultipleStorageAssets.optimise_storage()."
            )
        else:
            if end <= 0:
                timehorizon = self.Pfos.size
            else:
                timehorizon = end
            plt.rc("font", size=12)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(
                range(start, timehorizon),
                self.Pfos[start:timehorizon],
                color="tab:red",
                label="FF Power",
            )
            ax.plot(
                range(start, timehorizon),
                self.Shed[start:timehorizon],
                color="tab:blue",
                label="Renewable Shed",
            )
            ax.plot(
                range(start, timehorizon),
                self.surplus[start:timehorizon],
                color="tab:orange",
                label="Surplus",
            )

            # Same as above
            ax.set_xlabel("Time (h)")
            ax.set_ylabel("Power (MW), Energy (MWh)")
            ax.set_title(" Power Timeseries")
            ax.grid(True)
            ax.legend(loc="upper left")

    def old_size_storage(
        self,
        surplus,
        reliability,
        initial_capacity=None,
        req_res=1e5,
        t_res=1,
        max_capacity=1e9,
        start_up_time=0,
        strategy="ordered",
    ):
        """
        == description ==
        For a fixed relative size of storage assets, this function finds the
        total storage required to meet a certain level of reliability.

        This has been depreciated but is maintained for

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        reliability: (float) required reliability in % (0-100)
        initial_capacity: (float) intital capacity to try in MWh
        req_res: (float) the required capacity resolution in MWh
        t_res: (float) the size of time intervals in hours
        max_storage: (float) the maximum size of storage in MWh to consider
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).

        == returns ==
        capacity: the required total storage capacity in MWh
        """
        if initial_capacity is None:
            initial_capacity = min(surplus) * -1

        lower = initial_capacity
        upper = max_capacity
        print(initial_capacity)
        print(f"desired reliability: {reliability}")
        self.set_capacity(upper)
        rel3 = self.charge_sim(
            surplus, t_res=t_res, start_up_time=start_up_time, strategy=strategy
        )
        if rel3 < reliability:
            self.capacity = np.inf
            return np.inf

        self.set_capacity(lower)
        rel1 = self.charge_sim(
            surplus, t_res=t_res, start_up_time=start_up_time, strategy=strategy
        )
        print(f"Debug: reliability1:{rel1}")
        if rel1 > reliability:
            print("Initial capacity too high")
            if initial_capacity == 0:
                return 0.0
            else:
                self.size_storage(
                    surplus,
                    reliability,
                    initial_capacity=0,
                    req_res=req_res,
                    t_res=t_res,
                    max_capacity=max_capacity,
                    start_up_time=start_up_time,
                    strategy=strategy,
                )

        while upper - lower > req_res:
            mid = (lower + upper) / 2
            self.set_capacity(mid)
            rel2 = self.charge_sim(
                surplus, t_res=t_res, start_up_time=start_up_time, strategy=strategy
            )
            if rel2 < reliability:
                lower = mid
                rel1 = rel2
            else:
                upper = mid
                rel3 = rel2

        return (upper + lower) / 2

    def size_storage(
        self,
        surplus,
        reliability,
        initial_capacity=None,
        req_res=1e4,
        t_res=1,
        max_capacity=1e7,
        start_up_time=0,
        strategy="ordered",
    ):
        """
        == description ==

        For a fixed relative size of storage assets, this function finds the
        total storage required to meet a certain level of reliability.

        This uses a crude binary search approach.

        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        reliability: (float) required reliability in % (0-100)
        initial_capacity: (float) intital capacity to try in MWh
        req_res: (float) the required capacity resolution in MWh
        t_res: (float) the size of time intervals in hours
        max_storage: (float) the maximum size of storage in MWh to consider
        start_up_time: (int) number of first time intervals to be ignored when
            calculating the % of met demand (to allow for start up effects).

        == returns ==
        capacity: the required total storage capacity in MWh
        """
        initial_capacity = sum([i.capacity for i in self.assets])

        lower = initial_capacity
        upper = max_capacity
        print(initial_capacity)
        print(f"desired reliability: {reliability}")
        self.set_capacity(upper)
        rel3 = self.charge_sim(
            surplus, t_res=t_res, start_up_time=start_up_time, strategy=strategy
        )
        if rel3 < reliability:
            self.capacity = upper
            print("reliabilty constraints not met")
            return np.inf

        self.set_capacity(lower)
        rel1 = self.charge_sim(
            surplus, t_res=t_res, start_up_time=start_up_time, strategy=strategy
        )
        print(f"Debug: reliability1:{rel1}")

        if rel1 > reliability:
            print("Initial capacity too high")
            if initial_capacity == 0:
                self.capacity = 0
                return 0.0
            else:
                self.size_storage(
                    surplus,
                    reliability,
                    initial_capacity=0,
                    req_res=req_res,
                    t_res=t_res,
                    max_capacity=max_capacity,
                    start_up_time=start_up_time,
                    strategy=strategy,
                )

        while upper - lower > req_res:
            mid = (lower + upper) / 2
            self.set_capacity(mid)
            rel2 = self.charge_sim(
                surplus, t_res=t_res, start_up_time=start_up_time, strategy=strategy
            )
            if rel2 < reliability:
                lower = mid
                rel1 = rel2
            else:
                upper = mid
                rel3 = rel2

        return (upper + lower) / 2
