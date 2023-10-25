"""
Created: 07/05/2020 by C.CROZIER, updated by C Quarton

File description:
This file contains code that performs analysis using the generation and storage
models.

Pre-requisite modules: csv, numpy, matplotlib, cartopy
"""
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from cartopy import config
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry as sgeom
from shapely.ops import unary_union
from shapely.prepared import prep
from generation import (OffshoreWindModel, OffshoreWindModel10000,
                        OffshoreWindModel12000, OffshoreWindModel15000,
                        OffshoreWindModel17000, OffshoreWindModel20000,
                        OnshoreWindModel, OnshoreWindModel3600,
                        OnshoreWindModel2000, OnshoreWindModel3000,
                        OnshoreWindModel4000, OnshoreWindModel5000,
                        OnshoreWindModel6000, OnshoreWindModel7000,
                        SolarModel, TidalStreamTurbineModel)

class LoadFactorEstimator:
    
    def __init__(self, gen_type, data_loc=None):
        self.gen_type = gen_type
        self.load_factors = {}

        self.datapath = data_loc
        
        self.filepath = 'stored_model_runs/'+gen_type+'_load_factors.csv'

        recover = self.check_for_saved_run()
        if recover is False:
            self.calculate_load_factors()

    def check_for_saved_run(self):
        '''
        == description ==
        This function checks to see whether this simulation has been previously
        run, and if so sets power_out to the stored values.

        == parameters ==
        path: (str) location the csv file would stored if it exists

        == returns ==
        True if a previous run has been recovered
        False otherwise
        '''
        try:
            with open(self.filepath, 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                n = 1
                for row in reader:
                    self.load_factors[n] = [float(row[0]),float(row[1]),
                                            float(row[2])*100]
                    n += 1 
            return True
        except:
            return False
        
    def store_results(self):
        with open(self.filepath,'w') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Latitude','Longitude','Load Factor'])
            for n in self.load_factors:
                writer.writerow(self.load_factors[n][:2] +
                                [self.load_factors[n][2]/100])               

    def calculate_load_factors(self):
        if self.datapath is None:
            raise Exception('a data location is required')
        
        gen_model = {'osw':OffshoreWindModel,
                      'osw20.0':OffshoreWindModel20000,
                     # 'osw17.0':OffshoreWindModel17000,
                     # 'osw15.0':OffshoreWindModel15000,
                     # 'osw12.0':OffshoreWindModel12000,
                     # 'osw10.0':OffshoreWindModel10000,
                      'w7.0':OnshoreWindModel7000,
                      'w6.0':OnshoreWindModel6000,
                      'w5.0':OnshoreWindModel5000,
                      'w3.6':OnshoreWindModel3600,
                      'w4.0':OnshoreWindModel4000,
                      'w3.0':OnshoreWindModel3000,
                      'w2.0':OnshoreWindModel2000,
                      's':SolarModel}
        locs = {}
        with open(self.datapath+'site_locs.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                locs[int(row[0])] = [float(row[1]),float(row[2])]

        for site in locs:
            if site == 0:
                continue
            gm = gen_model[self.gen_type](sites=[site],data_path=self.datapath,
                                          save=False,year_min=1985,
                                          year_max=2019)
            
            try:
                lf = gm.get_load_factor()
            except:
                continue
            self.load_factors[site] = locs[site]+[lf]
        self.store_results()

    def estimate(self,lat,lon,max_dist=1,num_pts=3):
        pts = []
        for n in self.load_factors:
            loc = self.load_factors[n][:2]
            d = np.sqrt(np.power(lat-loc[0],2)+np.power(lon-loc[1],2))
            if d > max_dist:
                continue
            pts.append([d,self.load_factors[n][2]])

        pts = sorted(pts)
        f = 0
        n = 0
        if len(pts) < num_pts:
            for i in range(len(pts)):
                w = max_dist-pts[i][0]
                f += pts[i][1]*w
                n += w
        else:
            for i in range(num_pts):
                w = max_dist-pts[i][0]
                f += pts[i][1]*w
                n += w
        if n == 0:
            f = None
        else:
            f = f/n
        return f
        
class CorrelationCalculator:
    
    def __init__(self, gen_type, comparator, year_min, year_max, data_loc=None):
        self.gen_type = gen_type
        self.comparator = comparator    #time series to be compared to
        self.year_min = year_min          #will correspond to the yearmin of the comparator
        self.year_max = year_max          #will correspond to the yearmax of the comparator
        self.correlation_factors = {}

        self.datapath = data_loc
        
        self.calculate_correlation_factors()          

    def calculate_correlation_factors(self):
        if self.datapath is None:
            raise Exception('a data location is required')
        
        gen_model = {'osw':OffshoreWindModel,
                     # 'osw20.0':OffshoreWindModel20000,
                     # 'osw17.0':OffshoreWindModel17000,
                     # 'osw15.0':OffshoreWindModel15000,
                     # 'osw12.0':OffshoreWindModel12000,
                     # 'osw10.0':OffshoreWindModel10000,
                     # 'w7.0':OnshoreWindModel7000,
                     # 'w6.0':OnshoreWindModel6000,
                     # 'w5.0':OnshoreWindModel5000,
                     # 'w3.6':OnshoreWindModel3600,
                     # 'w4.0':OnshoreWindModel4000,
                     'w3.0':OnshoreWindModel3000,
                     # 'w2.0':OnshoreWindModel2000,
                     's':SolarModel,
                     'tidal':TidalStreamTurbineModel}
        locs = {}
        with open(self.datapath+'site_locs.csv','r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in reader:
                locs[int(row[0])] = [float(row[1]),float(row[2])]

        for site in locs:
            if site == 0:
                continue
            gm = gen_model[self.gen_type](sites=[site],data_path=self.datapath,
                                          save=False,year_min=self.year_min,
                                          year_max=self.year_max)
            
            try:
                p = gm.power_out      # might make sense to include this as a function in Generation.py rather than here
                correl = np.corrcoef(self.comparator, p) #Pearson coefficient
                print('Site '+str(site)+' has a correlation of '+str(correl[0][1]))
            except:
                continue
            self.correlation_factors[site] = locs[site]+[correl[0][1]]

    def estimate(self,lat,lon,max_dist=1,num_pts=3):
        pts = []
        for n in self.correlation_factors:
            loc = self.correlation_factors[n][:2]
            d = np.sqrt(np.power(lat-loc[0],2)+np.power(lon-loc[1],2))
            if d > max_dist:
                continue
            pts.append([d,self.correlation_factors[n][2]])

        pts = sorted(pts)
        f = 0
        n = 0
        if len(pts) < num_pts:
            for i in range(len(pts)):
                w = max_dist-pts[i][0]
                f += pts[i][1]*w
                n += w
        else:
            for i in range(num_pts):
                w = max_dist-pts[i][0]
                f += pts[i][1]*w
                n += w
        if n == 0:
            f = None
        else:
            f = f/n
        return f

# first here is the code for drawing maps
class LoadFactorMap:

    def __init__(self, model_to_map, lat_min, lat_max, lon_min,
                 lon_max, lat_num, lon_num, quality, is_land, label=""):
        self.model_to_map = model_to_map
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lon_num = lon_num
        self.lat_num = lat_num
        self.quality = quality
        self.is_land = is_land
        self.label=label


# This is an updated draw_map function using Cartopy rather that the deprecated Basemap       
    def draw_map(self, show=True, savepath='', cmap=None, vmax=None, vmin=None, turb=None):    # CQ edit: turb=None
        if cmap is None:
            # get standard spectral colormap
            spec = cm.get_cmap('Spectral', 1000)
            # reverse so that red is largest
            new = spec(np.linspace(0, 1, 1000))
            new = np.array(list(reversed(new)))
            # set zero to be white so that unknown areas will not be shaded
            new[:1,:] =  np.array([1,1,1,1])
            cmap = ListedColormap(new)                

        # initialise location data
        x = np.linspace(self.lon_min,self.lon_max,num=self.lon_num)
        y = np.linspace(self.lat_min,self.lat_max,num=self.lat_num)

        Z = np.zeros((len(x),len(y)))
        X = np.zeros((len(x),len(y)))
        Y = np.zeros((len(x),len(y)))
        minz=100
        maxz=0
        #m.drawcoastlines()
        
        LC = LandCheck() #initialise LandCheck function
        
        for i in range(len(x)):
            for j in range(len(y)):
                #xpt,ypt = m(x[i],y[j])
                xpt = x[i]
                ypt = y[j]
                X[i,j] = xpt
                Y[i,j] = ypt
                if LC.is_land_check(xpt,ypt) == self.is_land:
                    if self.is_land is True:
                        # Ireland
                        if ((xpt < 200000) and (ypt < 930000) and
                            (ypt > 340000)):
                            continue
                        # France
                        if xpt > 2.55 and ypt < 49.75:
                            continue
                    Z[i,j] = self.model_to_map.estimate(y[j],x[i])
                    if Z[i,j] > maxz:
                        maxz = Z[i,j]
                    if Z[i,j] < minz:
                        minz = Z[i,j]
                else:
                     Z[i,j] = None
        if vmin is None:
            vmin = minz*0.99
        if vmax is None:
            vmax = maxz
            
        if turb is not None:                               # CQ edit to include this if statement, scaling load factor to power generation
            Z = Z*turb*8760/100000
            vmin = vmin*turb*8760/100000
            vmax = vmax*turb*8760/100000

       # m.pcolor(X,Y,Z,vmin=vmin,vmax=vmax,cmap=cmap)
        ax = plt.axes(projection=ccrs.Mercator())
        plt.contourf(x, y, np.transpose(Z), 60, vmin=vmin, vmax=vmax,
                     transform=ccrs.PlateCarree(), cmap=cmap)
        
        
        ax.coastlines(resolution='50m')
       
        plt.colorbar(label="Load Factor (%)")
        plt.title(f"{self.label} load factor")
        if savepath != '':
            plt.savefig(savepath+'map.pdf', format='pdf',
                        dpi=300, bbox_inches='tight', pad_inches=0)
                     
            with open(savepath+'map.csv','w') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['title','row'])
                for i in range(len(x)):
                    writer.writerow(Z[i,:])      
            
        if show is True:
            plt.show()
        
class LandCheck: # just a function to identify whether a point is land or sea - returns TRUE/FALSE (was already defined in Basemap)
                 
    def __init__(self):
        land_shp_fname = shpreader.natural_earth(resolution='50m',
                                       category='physical', name='land')

        land_geom = unary_union(list(shpreader.Reader(land_shp_fname).geometries()))
        self.land = prep(land_geom)
        
    def is_land_check(self,xpt,ypt):
        return self.land.contains(sgeom.Point(xpt, ypt))

class OffshoreWindMap(LoadFactorMap):

    def __init__(self, lat_num=400, lon_num=300, quality='h',data_loc=None):
        lfe = LoadFactorEstimator('osw',data_loc=data_loc)
        
        super().__init__(lfe, 48.2, 61.2, -10.0, 4.0 ,lat_num, lon_num, quality,
                         is_land=False, label="Offshore Wind")


class OnshoreWindMap(LoadFactorMap):

    def __init__(self, lat_num=400, lon_num=300, quality='h',turbine_size=3.6,
                 data_loc=None):
        lfe = LoadFactorEstimator('w'+str(float(turbine_size)),
                                  data_loc=data_loc)
        
        super().__init__(lfe, 49.9, 59.0, -7.5, 2.0 ,lat_num, lon_num, quality,
                         is_land=True, label="Onshore Wind")


class SolarMap(LoadFactorMap):

    def __init__(self, lat_num=400, lon_num=300, quality='h',data_loc=None):
        lfe = LoadFactorEstimator('s',data_loc=data_loc)
        
        super().__init__(lfe, 49.9, 59.0, -7.5, 2.0 ,lat_num, lon_num, quality,
                         is_land=True, label="Solar")
        


class generationmap():
    def __init__(self)