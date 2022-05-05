'''
Created by Cormac O'Malley on 22/02/22

File description: This file contains the classes for a single set of aggregated EVs, as well as
the class for multiple such fleets.
'''

import copy
import numpy as np
import datetime
import optimise_configuration as opt_con
import storage as stor
import matplotlib.pyplot as plt
import datetime as dt

class AggregatedEVModel:
    def __init__(self, eff_in, eff_out, chargercost, 
                 max_c_rate, max_d_rate, min_SOC, max_SOC, number, initial_number, Ein, Eout, Nin, Nout, Nin_weekend, Nout_weekend, name,chargertype = [0.5,0.5], limits = []):
        '''
        == description ==
        Describes a fleet of EVs. Their charger costs/type can be specified or optimised. Behavioural plugin patterns are read in via timeseries.
        All EVs and chargers within the same fleet are homogeneous.

        == parameters ==
        eff_in: (float) charging efficiency in % (0-100)  : if max_c_rate is 10KW, and eff_in = 50%, then will remove 10KWh from grid to increase SOC by 5KWh.
        eff_out: (float) discharge efficiency in % (0-100)
        chargertype: array<(float)> the optimal ratio of different charger types (0: V2G, 1: Smart Uni, 2: Dumb)
        chargercost:  array [V2G charger cost, Smart Unidirectional, Dumb Unidirectional]
        max_c_rate: (float) the maximum charging rate (kW per Charger) from the grid side. (so energy into battery will be less)
        max_d_rate: (float) the maximum discharging rate (kW per Charger) from the grid side. (So the energy out of battery will be more)
        min/max_SOC: (float) min/max SOC of individual EV in kWh
        number: (float) Number of Chargers needed
        initial_number: (float) Proportion of chargers with EVs attached at the start of the simulation (0-1), (split evenly between charger types)
        Ein: (float) Energy stored in when plugged in EV (kWh)
        Eout: (float) Minimum Energy in EV when disconnected (kWh)
        Nin: (Array<float>) Normalised timeseries of EV connections (e.g. 0.1 for 1000 chargers says 100 EVs unplug at this timestep), 24hrs long for a weekday
        Nout: (Array<float>) Timeseries of EV disconnections (e.g. 0.1 for 1000 chargers says 100 EVs unplug at this timestep), 24hrs long for a weekday
        Nin_weekend: (Array<float>) Normalised timeseries of EV connections (e.g. 0.1 for 1000 chargers says 100 EVs unplug at this timestep), 24hrs long for a weekend
        Nout_weekend: (Array<float>) Timeseries of EV disconnections (e.g. 0.1 for 1000 chargers says 100 EVs unplug at this timestep), 24hrs long for a weekend
        name: (string) Name of the fleet (e.g. Domestic, Work, Commercial) Used for labelling plots
        chargercost: array<(float)> Cost of chargers (£ per Charger)/(years of lifetime),
        limits: array<float> Used in Full_optimise to limits the number of charger types built [MinV2G, MaxV2G, MinUnidirectional, MaxUnidirectional]
        
        == returns ==
        None
        '''
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.chargertype = chargertype
        self.chargercost=chargercost
        self.max_c_rate = max_c_rate
        self.max_d_rate = max_d_rate
        self.min_SOC = min_SOC
        self.max_SOC = max_SOC
        self.number = number
        self.initial_number =initial_number
        self.Ein = Ein
        self.Eout = Eout
        self.Nin = Nin
        self.Nout_weekend = Nout_weekend
        self.Nin_weekend = Nin_weekend
        self.Nout = Nout 
        self.name = name
        
        if(limits == []):
            self.limits = [0,self.number,0,self.number]
        else:
            self.limits = limits

        # These will be used to monitor storage usage
        self.V2G_en_in = 0 # total energy into storage (grid side)
        self.Smart_en_in = 0 # total energy into storage (grid side)
        self.V2G_en_out = 0 # total energy out of storage (grid side) (MWh)
        
        
        # from optimise setting only (added by Mac)
        self.discharge = np.empty([]) #timeseries of discharge rate (grid side) MW
        self.charge = np.empty([]) #timeseries of charge rate (grid side) MW
        self.SOC = np.empty([]) #timeseries of Storage State of Charge (SOC) MWh


        # timehorizon = 365*24*2
        if not self.Nin.size == 24 or not self.Nout.size == 24:
            print('Nin/Nout data for fleet ' + self.name + ' is not 24hrs long. Model not yet built to deal with longer timeseries.')
            return
        
        self.N = [] #this is the proportion of total EVs connected at a given hour, filled in with construct connectivity timeseries method.
        self.Nin_full = []
        self.Nout_full = []
        
        if(Eout != max_SOC ):
            print('Error, Eout must equal max_SOC for the Causal Simulation or the Optimisation Method to work.')

    def reset(self):
        '''
        == description ==
        Resets the parameters recording the use of the storage assets.

        == parameters ==
        None

        == returns ==
        None
        '''
        self.en_in = 0
        self.en_out = 0
        
        self.discharge = np.empty([]) 
        self.charge = np.empty([]) 
        self.SOC = np.empty([])

 
    def plot_timeseries(self,start=0,end=-1,withSOClimits=False):
            
        '''   
        == parameters ==
        start: (int) start time of plot
        end: (int) end time of plot
        withSOClimits: (bool) when true will plot the SOC limits imposed on the aggregate battery caused by EV driving patterns
        '''
        
        if(self.discharge.shape == ()):
            print('Charging timeseries not avaialable, try running MultipleAggregatedEVs.optimise_charger_type().')
        else:
            if(end<=0):
                timehorizon = self.discharge.size
            else:
                timehorizon = end
            
            for b in range(2):
                plt.rc('font', size=12)
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(start,timehorizon+1), self.SOC[start:timehorizon+1,b], color='tab:red', label='SOC')
                ax.plot(range(start,timehorizon), self.charge[start:timehorizon,b], color='tab:blue', label='Charge')
                if(b==0):
                    ax.plot(range(start,timehorizon), -self.discharge[start:timehorizon], color='tab:orange', label='Discharge')
                    ax.set_title(self.name+' V2G ('+str(int(self.number*self.chargertype[b]))+' chargers)')
                elif(b==1):
                    ax.set_title(self.name+' Smart ('+str(int(self.number*self.chargertype[b]))+' chargers)')
                    
                if(withSOClimits): #because SOC DV is the SOC at the END of the timestep, move these limits forwards by one!
                    if(start==0):
                        start=1
                        ax.plot(range(1,timehorizon), self.N[start-1:timehorizon-1]*self.max_SOC/1000 * self.chargertype[b] * self.number, 'c--', label='Max SOC Limit')
                    else:
                        ax.plot(range(start,timehorizon), self.N[start-1:timehorizon-1]*self.max_SOC/1000 * self.chargertype[b] * self.number, 'c--', label='Max SOC Limit')
    
                # Same as above
                ax.set_xlabel('Time (h)')
                ax.set_ylabel('Power (MW), Energy (MWh)')
                ax.grid(True)
                ax.legend(loc='upper left');
             
class MultipleAggregatedEVs:

    def __init__(self, assets):
        '''
        == description ==
        Initialisation of a multiple fleets object. 

        == parameters ==
        assets: (Array<AggregatedEVModel>) a list of aggregated EV model objects


        == returns ==
        None
        '''
        self.assets = assets
        self.n_assets = len(assets)
        
        #added by cormac for plotting timeseries from optimisation
        self.surplus = np.empty([]) #the last surplus used as input for optimise
        self.Pfos = np.empty([]) #the necessary fossil fuel generation timeseries from the last optimise run
        self.Shed = np.empty([]) #timeseries of surplus shedding
        self.driving_energy = 0 #total energy used for driving in MW
                
    def optimise_charger_type(self,surplus,fossilLimit,MultStor):
    
        '''
        == description ==
        For a given surplus, returns the cost optimal storage mix to meet the specified reliability. Charge order not relevant here.
    
        == parameters ==
        surplus: (Array<float>) the surplus generation to be smoothed in MW
        fossilLimit: (float) max acceptable amount of fossil fuel generated energy (MWh)
        MultStor: (MultipleStorageAssets)  if dont want to consider storage, simply input this: stor.MultipleStorageAssets([])
    
        == returns ==
        '''
        
        for i in range(0,self.n_assets):            
            # Check that the input timeseries are divisible by 24
            if not self.assets[i].Nin.size % 24 == 0:
                print('Nin/Nout data for fleet ' + self.assets[i].name + ' is not exactly divisible by 24hrs, could lead to unnatural periodicities.')
            
            #increase the length of the in/out plugin series to be longer than the simulation
            repeat_num = surplus.size // self.assets[i].Nin.size
            self.assets[i].Nin = np.tile(self.assets[i].Nin,repeat_num+1)
            repeat_num = surplus.size // self.assets[i].Nout.size
            self.assets[i].Nout = np.tile(self.assets[i].Nout,repeat_num+1)
            
        opt_con.optimise_configuration(surplus,fossilLimit,MultStor,self)
        
        
    def construct_connectivity_timeseries (self,start,end,includeleapdays=True):
        
        '''
        == description ==
        The EV connectivity is specified as 24hr profiles of normalised connection, this function constructs a timeseries of these of length timehorizon
        for use in causal operation or optimisation.
    
        == parameters ==
        start: <datetime> start time (inclusive)
        end: <datetime> end time (will run to the hour before this, so if want to do entire year make this 1st Jan Year 00:00)
        includeleapdays: <bool> when false, the timeseries will be constructed as normal and then the leap days will be removed, this keeps it inline with the demand and generation profiles used in Size_Then_Op_Function
        
        == returns ==
        '''
        duration = end-start
        timehorizon = duration.total_seconds()
        timehorizon = int(divmod(timehorizon, 3600*24)[0])
        
        #construct datetimeseries
        x = dt.timedelta(days = 1)
        N1 = start
            
        date_range = []
        for t in range(timehorizon):
            date_range.append(N1)
            N1 = N1+x
        
        #remove leap days if using for sizing then op
        if not includeleapdays:
            for t in range(timehorizon):
                if date_range[t].month == 2 and date_range[t].day == 29:
                    date_range.pop(t)
         
        N = np.empty([self.n_assets,timehorizon*24])
        Nin = np.empty([self.n_assets,timehorizon*24])
        Nout = np.empty([self.n_assets,timehorizon*24])         
        for k in range(self.n_assets):
            #now load in either weekday of weekend connectivity data
            for t in range(timehorizon):            
                if date_range[t].weekday() <= 4:
                    Nin[k,t*24:(t+1)*24] = self.assets[k].Nin
                    Nout[k,t*24:(t+1)*24] = self.assets[k].Nout
        
                    N[k,t*24:(t+1)*24] = np.asarray(self.assets[k].Nin) - np.asarray(self.assets[k].Nout)
                elif date_range[t].weekday() > 4:
                    Nin[k,t*24:(t+1)*24] = self.assets[k].Nin_weekend
                    Nout[k,t*24:(t+1)*24] = self.assets[k].Nout_weekend
        
                    N[k,t*24:(t+1)*24] = np.asarray(self.assets[k].Nin_weekend) - np.asarray(self.assets[k].Nout_weekend)
            
            self.assets[k].N = N[k,:] + self.assets[k].initial_number
            self.assets[k].Nin = Nin[k,:]      
            self.assets[k].Nout = Nout[k,:]
        
        
        
                
        # for i in range(0,self.n_assets):            
        #     # Check that the input timeseries are divisible by 24
        #     if not self.assets[i].Nin.size % 24 == 0:
        #         print('Nin/Nout data for fleet ' + self.Mult_aggEV.assets[i].name + ' is not exactly divisible by 24hrs, could lead to unnatural periodicities.')

        #     #increase the length of the in/out plugin series to be longer than the simulation
        #     repeat_num = timehorizon // self.assets[i].Nin.size
        #     self.assets[i].Nin = np.tile(self.assets[i].Nin,repeat_num+1)
        #     repeat_num = timehorizon // self.assets[i].Nout.size
        #     self.assets[i].Nout = np.tile(self.assets[i].Nout,repeat_num+1)
            
        # N = np.empty([self.n_assets,timehorizon]) #the normalised number of EVs connected at a given time (EV connections/disconnections are assumed to occur at teh start of the timestep)
        # for k in range(self.n_assets):
        #     for t in range(timehorizon):
        #         if t == 0:
        #             N[k,t] = self.assets[k].initial_number
        #         else:
        #             N[k,t] = N[k,t-1] + self.assets[k].Nin[t] - self.assets[k].Nout[t]
                               
        #     self.assets[k].N = N[k,:]
        

    