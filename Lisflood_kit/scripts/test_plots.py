#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 14:59:51 2021

@author: administrador
"""
from pyproj import Proj
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import copy
from matplotlib import cm
import os.path as op
import sys
import os
import cmocean as cmo
import matplotlib as mpl
import xarray as xr

# dev lisflood library 
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# paths
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.abspath(op.join(op.dirname(__file__), '..', 'data', 'demo'))

def plot_raster(filepath, mask, bmap, max_wd, min_wd):
  with rio.open(filepath) as src:
      w = copy.copy(cm.get_cmap('cool')) 
      w.set_under(color='none')
      w.set_over(color=w(0.9999999))
      
      wd = np.flipud(src.read(1))
      wdg = np.empty((mask.shape[0],mask.shape[1]))
      maskg = np.flipud(mask)
      
      for j in range(mask.shape[0]):
          for k in range(mask.shape[1]):
              if (maskg[j,k] == 1):
                  wdg[j][k] = wd[j][k]
              else:
                  wdg[j][k] = -1
      
      bmap.imshow(wdg,cmap=w,vmin=min_wd,vmax=max_wd,alpha=1)
      
path_ort = op.join(p_demo,'GUAM_2018_5m.png')  # Orthoimage

#%%--------------------------LOAD DEM PARAMETERS-------------------------------
path_mdem = op.join(p_demo, 'GUAM_lisflood_UTM_5m.dem.asc')
depth = np.loadtxt(path_mdem,skiprows=6)

nrows = depth.shape[0]                    # Rows 
ncols = depth.shape[1]                    # Cols 
with open(path_mdem) as d:
    for i in range(6):
        line = d.readline()
        if(i == 2):
            xll = float(line[10:-1])      # Xllcorner
        elif(i == 3):
            yll = float(line[10:-1])      # Yllcorner
        elif(i == 4):
            cellsize = float(line[9:-1])  # Cellsize
        elif(i == 5): 
            nondata = int(line[13:-1])    # Nodata_value
d.close()

xx =  np.linspace(xll,xll+cellsize*(ncols-1),ncols)
yy =  np.linspace(yll,yll+cellsize*(nrows-1),nrows)
xxyy = np.meshgrid(xx,yy)
xmesh = xxyy[0]                           # X Coordinates
ymesh = np.flipud(xxyy[1])                # Y Coordinates


dem = xr.open_dataset(op.join(p_demo, 'GUAM_utm_5.nc')).fillna(-9999)

depth0 = dem.elevation.values

pos = np.where(depth0 >= 0)
mask = np.copy(depth0)
mask[pos[0], pos[1]] = 1

del xx,yy,xxyy,line,path_mdem

cmap = cmo.cm.algae
cmap.set_under(color='none')
norm = mpl.colors.Normalize(vmin=-10)

fig, ax = plt.subplots(1,1,figsize=(14,7))
im1 = ax.pcolor(xmesh, ymesh, depth, shading='auto', cmap=cmap, vmin=-2, vmax=10)

cmap = cmo.cm.algae
cmap.set_under(color='none')
norm = mpl.colors.Normalize(vmin=-10)

fig, ax = plt.subplots(1,1,figsize=(14,7))
im1 = ax.pcolor(xmesh, ymesh, mask, shading='auto', cmap=cmap, vmin=0.1, vmax=5)

#%%--------------------------FLOODING MAPS-------------------------------------

utm_guam = Proj(proj='utm', zone='55N', ellps='WGS84', datum='WGS84', units='m', south=False)

fig,ax = plt.subplots(1,1,figsize=(13,7))

''' ------------------------------------------------------------------
--------------Target: change basemap for cartopy/geopandas---------------------
    ------------------------------------------------------------------
'''        
#------------------------------------------------------------------------------
#bmap = plt.axes(projection=ccrs.Mercator())
#gl = bmap.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                    #linewidth=0.8,color='gray', alpha=0.5, 
                    #linestyle='--')
#gl.xlabels_top = False; gl.ylabels_right = False
#gl.xlabel_style = {'color':'black'}; gl.ylabel_style = {'color':'black'}     
#bmap.set_extent((144.662,144.688,13.4573,13.47), crs=ccrs.PlateCarree())
#gl.xlocator = mticker.FixedLocator([144.663,144.683])
#gl.ylocator = mticker.FixedLocator([13.458,13.468])

#img = plt.imread(path_ort)
#plt.imshow(img, origin='upper', transform=ccrs.PlateCarree(), 
#          extent=[144.662,144.688,13.4573,13.47])
#------------------------------------------------------------------------------

lonmin, latmin = utm_guam(246872.22726091,1488874.5187709,inverse=True)
lonmax,latmax = utm_guam(246872.22726091+cellsize*(ncols-1),
                         1488874.5187709+cellsize*(nrows-1),inverse=True)
lonc = lonmin +((lonmax-lonmin)/2)
latc = latmin +((latmax-latmin)/2)

bmap = Basemap(projection = 'tmerc', llcrnrlat = latmin, llcrnrlon = lonmin,
              urcrnrlat = latmax, urcrnrlon = lonmax, lon_0 = lonc, lat_0 = latc)       
paral = np.arange(round(latmin+0.001,3),round(latmax,3),0.01)
merid = np.arange(round(lonmin+0.001,3),round(lonmax,3),0.02)
bmap.drawparallels(paral,labels=[1,0,0,0])
bmap.drawmeridians(merid,labels=[0,0,0,1])


mp = op.join(os.getcwd(),'FloodMaps')            # Maps path
#os.mkdir(mp)           #Comment this if directory already exists

path_res = r'/media/administrador/HD/wrapLISFLOOD_march2021/Lisflood_kit/data/projects/test_wrap/cases/0._20/results'
path_max = op.join(path_res,'guam-0031.wd')

bmap.imshow(np.flipud(plt.imread(path_ort)), origin='lower')
plot_raster(path_max, mask, bmap, 1, 0.01) 


plt.title('Emulated flooding event (RP = 20 years, SLR = +0.0 m)',fontsize=20,pad=30)
 
cbar = plt.colorbar(extend='max')
cbar.set_alpha(1)
cbar.draw_all()
cbar.ax.set_xlabel('Water depth [m]')

plt.savefig(mp + '/em20_00.png')
plt.show()

print('Done.')

''' ----------------------------------------------------------------
------------Target: set orthoimage under flood maps--------------------
    ----------------------------------------------------------------
'''

#path_anim = op.join(ip,'Animation')
##        os.mkdir(path_anim)   # If the directory already exists comment this
#
#fileswd = sorted(glob(path_res + '/*.wd'))     
#arr = []               
#for file in fileswd:
#  with rio.open(file) as src:
#    arr.append(src.read(1))
#
#base = datetime.datetime(1979, 1, 3, 8)
#dates = np.array([base + datetime.timedelta(minutes=(int(saveint)/60)*i) for i in range(len(arr))])   
#
##        clips = [ImageClip(m).set_duration(1) for m in arr]
##        concatclip = concatenate_videoclips(clips,method="compose")
##        concatclip.write_videofile('animation_em_{0}_{1}_wd.mp4',fps=30)
#
#fig,ax2 = plt.subplots(figsize=(5,5))
#im0 = ax2.imshow(arr[0], cmap = 'cool', vmin=0.05,vmax=1)
#plt.close('all')
#
#fig, ax = plt.subplots(figsize=(8, 6))
#im = ax.imshow(mask, cmap = 'GnBu_r', vmin=0,vmax=1)
#anim = FuncAnimation(fig, update_wd, frames=range(len(arr)),interval=1000)
#cbar = plt.colorbar(im0,extend='max')
#cbar.ax.set_ylabel('Water depth [m]')
#plt.show()
#
##        anim.save(path_anim+'/animation_em_{0}_{1}_wd.gif'.format(rp,slr), dpi=80, writer='pillow')
#anim.save(path_anim+'/animation_em_{0}_{1}_wd.mp4'.format(rp,slr), writer=mpl.animation.FFMpegWriter)
#plt.close()