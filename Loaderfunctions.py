import numpy as np

def listoflatlongstosite(latlonglist, trimmedsites):
    """takes as input a list of latitude, longitude, and a list of sites. Returns a list of index of the closest sites, and a bool: if the closest site is more than
    100km away, the bool is False
    Parameters
    -----
    latlonglist: list
        list of lat and longs, wth + meaning N/E and - meaning S/W
    trimmedsites: numpy array
        array of data sites. The header should be removed before putting into the function
        Each line is: siteindex , latitude, longitude

    Returns
    -----
    output: list
        made up of rows:[index, close]:
            index: int
                index of closest site
            close: bool
                True if distance <100Km, False if more
    """
    latlongarray=np.array(latlonglist)
    pointslatlongrad=np.radians(latlongarray)
    siteslatlongrad=np.radians(trimmedsites[:,1:]) #takes the lat and long columns
    # needs to be finished
    # transsiteslatlongrad=siteslatlongrad.T #transposes the array for matrix multiplication

    
    # distances=6371*np.arccos(np.matmul(np.sin(pointslatlongrad[:,0]),np.sin(transsiteslatlongrad[:,1]) +np.cos(pointslatlongrad[:,0]*np.cos(pointslatlongrad[:,0]* )))
    




def latlongtosite(latitude, longitude, listofsites):
    """takes as input a latitude, longitude, and a list of sites. Returns the index of the closest sites, and a bool: if the closest site is more than
    100km away, the bool is False
    Parameters
    -----
    latitude: float
        latitude of the coordinate, wth + meaning N and - meaning south
    longitude: float
        longitude of the coordinate, with + meaning E and - meaning W
    listofsites: list
        list of the data sites. Header line is: Site, Latitude, Longitude
        Each subsequent line is: siteindex (int), latitude(float), longitude(float)

    Returns
    -----
    lowestsiteindex: int
        index of closest site
    close: bool
        True if distance <100Km, False if more
    """

    sites=listofsites
    # sites=listofsites[1:] #removes header
    greatcircledistances=[greatcircledistance([latitude, longitude], [float(i[1]), float(i[2])]) for i in sites] #calculates distance between coordinate and each site
    greatcircledistances=np.array(greatcircledistances)
    lowestindex=np.argmin(greatcircledistances)
    lowestsiteindex=int(sites[lowestindex][0])
    lowestdistance=greatcircledistances[lowestindex]
    if lowestdistance<100:
        close=True
    else:
        close=False
    
    return lowestsiteindex, close
    

def greatcircledistance(pointa, pointb):
    """takes 2 points, and calculates the great circle distance between them
    
    Parameters
    -----
    pointa: list
        list in order [lat, long], in degrees

    pointb: list
        list in order [lat, long], in degrees
        
    Returns
    -----
    
    dist: float
        distance between points in km
    """

    longa=np.radians(pointa[0])
    lata=np.radians(pointa[1])
    longb=np.radians(pointb[0])
    latb=np.radians(pointb[1])

    centralangle=np.arccos(np.sin(lata)*np.sin(latb)+np.cos(lata)*np.cos(latb)*np.cos(abs(longa-longb)))
    dist=6371*centralangle
    
    return dist