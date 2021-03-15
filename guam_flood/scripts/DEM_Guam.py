# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:50:36 2020

@author: Manuel Zornoza
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
import pandas as pd
import os
import os.path as path  
from pyproj import Proj
import xarray as xr
from subprocess import call
import rasterio as rio 
from glob import glob
from mpl_toolkits.basemap import Basemap
import cmocean.cm as cmo
from matplotlib import gridspec
import cmocean
import cir
from matplotlib.colors import ListedColormap
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D

#%%

def colormap(elev, topat, topag): #Elevation, maximum topo, minimum bati
    colors2 = 'YlGnBu_r'
    colors1 = cmocean.cm.turbid
    bottom = plt.get_cmap(colors2, -topag)
    top = plt.get_cmap(colors1, topat)
    newcolors = np.vstack((bottom(np.linspace(0, 0.8, -topag)),top(np.linspace(0.1, 1, topat))))
    return(ListedColormap(newcolors))

pathfBat='/media/administrador/HD/GUAM/Bathymetry/'
bati=xr.open_dataset(os.path.join(pathfBat,'Bati_flooding.nc'))
dem=np.copy(np.flipud(bati.Band1))

# Area of interest

#projGuam = Proj(proj='utm', zone='55N', ellps='WGS84', datum='WGS84', units='m', south=False)
#x,y = projGuam(144.661953736655,13.457253736655,inverse=False)

lon_f=[144.66,144.688]
lat_f=[13.4573,13.472]

s=np.where((bati.lat>=lat_f[0]) & (bati.lat<=lat_f[1]))[0]
s1=np.where((bati.lon>=lon_f[0]) & (bati.lon<lon_f[1]))[0]

bati_f=bati.isel(lat=s)
bati_f=bati_f.isel(lon=s1)

nrows, ncols = dem.shape
for i in range (nrows):
    for j in range (ncols):
        if (dem[i][j] < 0):
            dem[i][j] = -9999
            
#for i in range (67,88):
#    for j in range(0,3):
#        dem[i][j] = 20
#        
#for i in range (88,101):
#    for j in range(0,8):
#        dem[i][j] = 20

#%% Plot
fig, ax = plt.subplots(2, 1, figsize=(10,10)) 
c1=colormap(bati_f.Band1, 20,-10)
im = ax[0].pcolor(bati_f.lon.values, bati_f.lat.values, bati_f.Band1.values, cmap = c1, vmax = 20, vmin = -10)
im2 = ax[1].pcolor(bati_f.lon.values, bati_f.lat.values, np.flipud(dem), cmap = c1, vmax = 20, vmin = -10)
axcb = fig.colorbar(im, ax=ax.ravel().tolist(), pad=0.04, aspect = 30, extend = 'both')

#%% Save

print("Writing modified DEM...")

pathfRes = path.join(os.getcwd(), 'guam_mod.asc')
pathfTopo = '/media/administrador/HD/GUAM/scripts/CoordX.asc'

fTopo = open(pathfTopo, 'r') 
fRes = open(pathfRes, 'w')              
for i in range(6):
    fRes.write("{0}".format(fTopo.readline()))

np.savetxt(fRes,dem,fmt='%1.3f')    
       
fRes.close()
print("Finished.")


