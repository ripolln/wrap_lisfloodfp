# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:50:36 2020

@author: Manuel Zornoza
"""

''' RAINFLOOD SAMOA '''

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import numpy as np
import os
#import pandas as pd
import os.path as op
#from pyproj import Proj
import xarray as xr
import copy
from subprocess import call
import rasterio as rio 
from glob import glob
#from osgeo import gdal
#from mpl_toolkits.basemap import Basemap
import cmocean as cmo
#from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import matplotlib.ticker as mticker

## ------------------------------ FUNCTIONS -------------------------------- ##
# Define perimeters
def dem_perimeter_points(mask, xmesh, ymesh, depth, outer):
        '''
        Find perimeter coordinates depth file mask
        option:
        outer = True    obtains outer perimeter points
        outer = False   obtains inner perimeter points
        '''
        if not outer:
            value = 1
            sum_min = -9999 * 7 + 1
            sum_max = -9999 + 7
        if outer:
            value = -9999
            sum_min = -9999 * 7 + 1
            sum_max = -9999 + 7
        # loop over depth array to find perimeter coordinates
        nrows, ncols = mask.shape
        xpos, ypos = [], []
        
        for i in np.arange(1, nrows - 1):
            for j in np.arange(1, ncols - 1):
                cell = mask[i,j]
                if cell == value:
                    cellul = mask[i - 1, j - 1]
                    cellu  = mask[i - 1, j    ]
                    cellur = mask[i - 1, j + 1]
                    celll  = mask[i    , j - 1]
                    cellr  = mask[i    , j + 1]
                    celldl = mask[i + 1, j - 1]
                    celld  = mask[i + 1, j    ]
                    celldr = mask[i + 1, j + 1]
                    neighbour_sum = np.sum([cellul,cellu,cellur,celll,cellr,
                                            celldl,celld,celldr])
                    if (neighbour_sum >= sum_min) & (neighbour_sum <= sum_max):
                        xpos.append(i)
                        ypos.append(j)
        x_pts = xmesh[xpos, ypos]
        y_pts = ymesh[xpos, ypos]
        z_pts = depth[xpos, ypos]
        return x_pts, y_pts, z_pts, xpos, ypos
    
# Assign hydrograph to each perimeter point
def assign_group_id(x_pts, y_pts, x_pf, y_pf):
    'Takes xarray perimeter points and assigns closest profile transect'
    
    id_trans = np.array(np.zeros((len(x_pts))))
    
    # loop perimeter points (pts)
    for j in range(x_pts.size):
        iz = np.array((x_pts[j], y_pts[j]))
        
        dist = []
        # loop transect points (pf)
        for i in range(x_pf.size):      
            izt = np.array((x_pf[i], y_pf[i]))
            
            # calculate distance between pts(j), pf(i)
            dist.append(np.linalg.norm(izt - iz, 2))
        
        # assign closest transect id
        id_trans[j] = np.where(dist == np.min(dist))[0][0]
        
    return id_trans
    
# Plot 1 flood map  
def plot_raster(filepath, bmap, max_wd, min_wd):
  with rio.open(filepath) as src:
      w = cm.get_cmap('cool') 
      w.set_under(color='none')
      bmap.imshow(np.flipud(src.read(1)),cmap=w,vmin=min_wd,vmax=max_wd,alpha=1)

# Plot animation
def update_wd(i):
    oce = copy.copy(cm.get_cmap('cool'))
    oce.set_under(color='k',alpha=0.4)
    im_normed = arr[i]
    
    wdg = np.empty((397,397))
    mask = np.copy(depth)

    pos = np.where(depth >= 0)
    mask[pos[0], pos[1]] = 1

    for j in range(ncols):
        for k in range(nrows):
            if (mask[j,k] == 1):
                wdg[j][k] = im_normed [j][k]
        
    ax.imshow(wdg, cmap = oce, vmin=0.01, vmax=3)
    ax.set_title('WATER DEPTHS - ' + dates[i], fontsize=20)
    ax.set_axis_off()
    
def update_el(i):
    oce = cmo.cm.haline
    oce.set_under(color='k',alpha=0.4)
    im_normed = arr[i]
    
    wdg = np.empty((397,397))
    mask = np.copy(depth)

    pos = np.where(depth >= 0)
    mask[pos[0], pos[1]] = 1

    for j in range(ncols):
        for k in range(nrows):
            if (mask[j,k] == 1):
                wdg[j][k] = im_normed [j][k]
        
    ax.imshow(wdg, cmap = oce, vmin=0.01, vmax=3)
    ax.set_title('WATER ELEVATIONS - ' + dates[i], fontsize=20)
    ax.set_axis_off()
    
#%% ----------------------PATHS AND FILES--------------------------------------
print('Loading files and paths ...')

# LISFLOOD case
T = list(['10','50','100'])     # Return period 
p = 'GUAM'                      # Location

# Filenames
demfile = '{0}_flood.dem.asc'.format(p)
parfile = '{0}_{1}.par'.format(p,T[0])
bcifile = '{0}_bc_{1}.bci'.format(p,T[0])
bdyfile = '{0}_tvbc_{1}.bdy'.format(p,T[0])
startfile = '{0}_start_{1}.wd'.format(p,T[0])
resroot = '{0}_{1}'.format(p,T[0])
dirroot = 'Res_{0}_{1}'.format(p,T[0])

sim_time = '79200'          # Simulation time (in seconds)
initial_tstep = '100'       # Initial guess for optimum and max time step
massint = '1200'            # Mass balance save interval
saveint = '1200'            # Results save interval
fpfric = '0.06'             # Manning

# Paths
wp = os.getcwd()
os.chdir('..')
dp = op.join(os.getcwd(),'data')
os.chdir(wp)

path_dem = op.join(dp,'Bati_flooding.nc')         # Original DEM
path_dem2 = op.join(dp,'Bati_flooding_mod.asc')   # DEM modified (onlytopo,gaps)
path_mdem = op.join(dp, demfile)                  # DEM for lisflood

#path_lisf = op.join(wp,'lisflood_euflood')
path_ort = op.join(dp,'guam_2018.png')
path_res = op.join(wp,dirroot)
path_anim = op.join(path_res, 'Animation')  
path_max = op.join(path_res, '{0}_{1}.max'.format(p,T[0])) 

# Original & Modified DEM
dem = xr.open_dataset(path_dem)             # Raw
#depth = np.copy(np.flipud(dem.Band1))
depth = np.loadtxt(path_dem2,skiprows=6)   # Copy to modify
depth2 = np.loadtxt(path_mdem,skiprows=6)   # Lisflood DEM

# Raster dimensions
nrows = depth.shape[0]                 # Rows
ncols = depth.shape[1]                 # Cols
nodata = -9999                         # NaN value
lonmin = float(min(dem.lon))           # Xllcorner
latmin = float(min(dem.lat))           # Yllcorner
step = np.abs(dem.lon[1]-dem.lon[0])   # Cellsize
xx =  np.linspace(lonmin,lonmin+step*(ncols-1),ncols)
yy =  np.linspace(latmin,latmin+step*(nrows-1),nrows)
xxyy = np.meshgrid(xx,yy)
xcoor = xxyy[0]                        # X Coordinates
ycoor = np.flipud(xxyy[1])             # Y Coordinates

del xx,yy,xxyy
# Modify DEM
#bati = np.where(depth<0)
#depth[bati] = -9999

#for i in range(nrows):
#    for j in range(ncols):
#        if(i >= 87 and depth[i][j] == -9999):
#            depth[i][j] = 20
#            
#for i in range(nrows):
#    for j in range(ncols):
#        if(i >= 57 and j >= 204 and j <= 261 and depth[i][j] == -9999):
#            depth[i][j] = 20
#        if(j > 245 and depth[i][j] == 20):
#           depth[i][j] = -9999

print('Done.')

#%% -----------------------------PERIMETERS------------------------------------
print('Applying perimeter functions ...')

pos = np.where(depth >= 0)
mask = np.copy(depth)
mask[pos[0], pos[1]] = 1

x1, y1, z1, xpos1, ypos1 = dem_perimeter_points(mask=mask,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth,
                                                outer=False)

x_p1, y_p1, z_p1, xpos_p1, ypos_p1 = dem_perimeter_points(mask=mask,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth,
                                                outer=True)
depth_out1 = np.copy(depth)
mask_out1 = np.copy(mask)
depth_out1[xpos_p1,ypos_p1] = 1
pos = np.where(depth_out1 >= 0)
mask_out1[pos[0], pos[1]] = 1

x_p2, y_p2, z_p2, xpos_p2, ypos_p2 = dem_perimeter_points(mask=mask_out1,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth_out1,
                                                outer=True)
depth_in1 = np.copy(depth)
for i in range(len(xpos1)):
    depth_in1[xpos1[i]][ypos1[i]] = -9999

mask_in1 = np.copy(depth_in1)
pos = np.where(depth_in1 >= 0)
mask_in1[pos[0], pos[1]] = 1

x4, y4, z4, xpos4, ypos4 = dem_perimeter_points(mask=mask_in1,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth_in1,
                                                outer=False)

depth_in2 = np.copy(depth_in1)
for i in range(len(xpos4)):
    depth_in2[xpos4[i]][ypos4[i]] = -9999

mask_in2 = np.copy(depth_in2)
pos = np.where(depth_in2 >= 0)
mask_in2[pos[0], pos[1]] = 1

x_q0, y_q0, z_q0, xpos_q0, ypos_q0 = dem_perimeter_points(mask=mask_in2,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth_in2,
                                                outer=False)

lim_q = np.where(np.array(ypos_q0) < 245)

xq = x_q0[lim_q]
yq = y_q0[lim_q]
zq = z_q0[lim_q]

xposq = np.zeros((len(lim_q[0]),1))
yposq = np.zeros((len(lim_q[0]),1))
for i in range(len(lim_q[0])):
    xposq[i] = int(xpos_q0[lim_q[0][i]])
    yposq[i] = int(ypos_q0[lim_q[0][i]])

#for i in range(len(xpos_p1)):
#    depth[xpos_p1[i]][ypos_p1[i]]=-1.2
#for i in range(len(xpos_p2)):
#    depth[xpos_p2[i]][ypos_p2[i]]=-1.2

del depth_out1,mask_out1,depth_in1,depth_in2,mask_in1,mask_in2
del x4,y4,z4,xpos4,ypos4,x1,y1,z1,xpos1,ypos1,pos
del x_q0, y_q0, z_q0, xpos_q0, ypos_q0

print('Done.')

#%% -------------------BERM LOCATION AND HOMOGENISATION------------------------

berm = xr.open_dataset(op.join(dp,'Berm_Guam.nc'))   
xb0 = np.array(berm['Lon'])
yb0 = np.array(berm['Lat']) 
zb0 = np.array(berm['Elevation'])

xb = []; yb = []; zb = []
    
for k in range(len(xb0)):
    if(k==0) or (k==4):
        xpos = np.array(np.where(dem['lon'] <= xb0[k]))+1
        ypos = np.array(np.where(dem['lat'] <= yb0[k]))
    elif(k==6) or (k==7):
        xpos = np.array(np.where(dem['lon'] <= xb0[k]))
        ypos = np.array(np.where(dem['lat'] <= yb0[k]))+1
    elif(k==9):
        xpos = np.array(np.where(dem['lon'] <= xb0[k]))
        ypos = np.array(np.where(dem['lat'] <= yb0[k]))
    else:
        xpos = np.array(np.where(dem['lon'] <= xb0[k]))+1
        ypos = np.array(np.where(dem['lat'] <= yb0[k]))+1
        
    xb = np.append(xb,float(dem['lon'][xpos[0,-1]]))
    yb = np.append(yb,float(dem['lat'][ypos[0,-1]]))
    zb = np.append(zb,float(dem.Band1[ypos[0,-1]][xpos[0,-1]]))
    
#CAMBIAR FUNCION--AHORA TIENE QUE ASIGNAR LA ANCHURA INDICADA EN berm['Distance']
id_trans = assign_group_id(xq, yq, xb, yb)

for i in range(len(xq)):
    depth[int(xposq[i])][int(yposq[i])] = float(berm['Elevation'][int(id_trans[i])])
    
del xb0,yb0,zb0

#%% ---------------PLOT DEM (CHECK MODIFICATIONS, PERIMETERS, ETC.)------------
cmap = cmo.cm.algae
cmap.set_under(color='none')
norm = mpl.colors.Normalize(vmin=-10)

fig, ax = plt.subplots(1,1,figsize=(14,7))
im1 = ax.pcolor(xcoor, ycoor, depth, shading='auto', cmap=cmap, vmin=-2, vmax=10)

#ax.plot(xq, yq,'.',color='C0')        #Blue dots     # OK
ax.plot(x_p1, y_p1, '.', color='C4')  #Purple dots   # OK
ax.plot(x_p2, y_p2, '.', color='C3')  #Red dots      # OK

#ax.plot(xb, yb, '.', color='r')

cbar = plt.colorbar(im1,cmap=cmap)
cbar.set_alpha(1)
cbar.draw_all()
cbar.ax.set_xlabel('Elevation [m]')
plt.show()

#%% --------------------------UPDATE DEM---------------------------------------
#print('Writing modified DEM ...')

#fDem = open(path_mdem,'w') 
#           
#fDem.write('ncols\t\t{0}\n'.format(ncols))  
#fDem.write('nrows\t\t{0}\n'.format(nrows))
#fDem.write('xllcorner\t{0:.6f}\n'.format(lonmin)) 
#fDem.write('yllcorner\t{0:.6f}\n'.format(latmin)) 
#fDem.write('cellsize\t{0:.6f}\n'.format(float(step)))          
#fDem.write('NODATA_value\t{0}\n'.format(nodata))
#np.savetxt(fDem,depth,fmt='%1.3f')    
#       
#fDem.close()

#print('Done.')

#%% ---------------------------STARTFILE---------------------------------------
print('Creating startfile ...')

path_sta = op.join(wp,startfile)
fRes = open(path_sta,'w') 
fTopo = open(path_mdem, 'r')  
         
for i in range(6):
    fRes.write("{0}".format(fTopo.readline()))

ini = np.zeros((nrows,ncols))

np.savetxt(fRes,ini,fmt='%1.3f')    
   
fTopo.close()  
fRes.close() 

print('Done.')

#%% ----------------------------BCI FILE---------------------------------------
print('Creating bcifile ...')

path_bci = op.join(wp,'{0}_{1}.bci'.format(p,T[0]))
fbci = open(path_bci,'w') 

# x1: puntos de la berma homogeneizada donde entran los caudales
for i in range(len(xq)):         
    fbci.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format('P', xq[i], yq[i], 'QVAR', 
               int(id_trans[i]+1)))

for i in range(len(x_p1)):         
    fbci.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format('P', x_p1[i], y_p1[i], 'HFIX', 1))

for i in range(len(x_p2)):
    fbci.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format('P', x_p2[i], y_p2[i], 'HFIX', 1))  
    
fbci.close()

print('Done.')

#%% -----------------------------BDY FILE--------------------------------------

# ¡¡OJO!! Los volúmenes de Alba están en L/s·m, hay que multiplicarlos por el 
# ancho de celda (10.288996177128) y dividir entre 1000 para pasarlo a m3/s/celda

qs = xr.open_dataset(op.join(dp,'Guam_q_ReturnP.nc'))    
qs = qs.squeeze('Rp')

w = float(step*110634.367496)

print('Creating bdyfile ...') 

for k in list(T):
    path_bdy = op.join(wp,'{0}_{1}.bdy'.format(p,k))
    fbdy = open(path_bdy,'w')  
    fbdy.write('GUAM: 12 profiles along North shore. Return period:{0}\n'.format(k))
    for i in range(qs['profile'].shape[0]):
        
        fbdy.write('{0}\n'.format(i+1))                               # Profile
        fbdy.write('{0}\t{1}\n'.format(qs['time'].shape[0],'hours'))  # Hydro length
        for j in range(qs['time'].shape[0]):
            
            if(k=='10'):
                q = qs['q'][0][i][j]
                if(np.isnan(q)):
                    fbdy.write('{0}\t{1}\n'.format(int(qs['time'][j]),float(0)))
                else:
                    fbdy.write('{0}\t{1}\n'.format(int(qs['time'][j]),float(q*w/1000)))
            elif(k=='50'):
                q = qs['q'][1][i][j]
                if(np.isnan(q)):
                    fbdy.write('{0}\t{1}\n'.format(int(qs['time'][j]),float(0)))
                else:
                    fbdy.write('{0}\t{1}\n'.format(int(qs['time'][j]),float(q*w/1000)))
            else:
                q = qs['q'][2][i][j]
                if(np.isnan(q)):
                    fbdy.write('{0}\t{1}\n'.format(int(qs['time'][j]),float(0)))
                else:
                    fbdy.write('{0}\t{1}\n'.format(int(qs['time'][j]),float(q*w/1000)))
        fbdy.write('\n')                     
    fbdy.close()

print('Done.')

#%% --------------------------PARFILE------------------------------------------
print('Creating parfile ...')

path_par = op.join(wp,'{0}_{1}.par'.format(p,T[0]))
fpar = open(path_par,'w') 

fpar.write("{0}\n".format('# Flooding Guam'))
fpar.write("{0}\t\t{1}\n".format('DEMfile',demfile))
fpar.write("{0}\t\t{1}\n".format('resroot',resroot))
fpar.write("{0}\t\t{1}\n".format('dirroot',dirroot))
fpar.write("{0}\t\t{1}\n".format('sim_time',sim_time))
fpar.write("{0}\t\t{1}\n".format('initial_tstep',initial_tstep))
fpar.write("{0}\t\t{1}\n".format('massint',massint))
fpar.write("{0}\t\t{1}\n".format('saveint',saveint))
fpar.write("{0}\t\t\t{1}\n".format('fpfric',fpfric))
fpar.write("{0}\t\t{1}\n".format('bcifile',bcifile))
fpar.write("{0}\t\t{1}\n".format('bdyfile',bdyfile))
fpar.write("{0}\t\t{1}\n".format('startfile',startfile))

fpar.write("\n".format())
fpar.write("{0}\n".format('elevoff'))
fpar.write("{0}\n".format('acceleration'))
fpar.write("{0}\n".format('latlong'))
fpar.write("{0}\n".format('SGC_enable'))
fpar.write("{0}\n".format('comp_out'))

fpar.close()

print('Done.')

#%% ---------------------------LISFLOOD-FP-------------------------------------
print('Running LISFLOOD-FP ...')

path_lisf = op.join(wp,'lisflood_euflood')
call([path_lisf, '-v', '-log', path_par])

print('Done.')

#%% -----------------------------RESULTS---------------------------------------

print('Setting up the basemap ...')

fig,ax = plt.subplots(1,1,figsize=(20,20))
bmap = plt.axes(projection=ccrs.Mercator())

gl = bmap.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,linewidth=0.8, color='gray', alpha=0.5, linestyle='--')
gl.xlabels_top = False; gl.ylabels_right = False
gl.xlabel_style = {'color':'black'}; gl.ylabel_style = {'color':'black'}     
bmap.set_extent((144.662,144.688,13.4573,13.47), crs=ccrs.PlateCarree())
gl.xlocator = mticker.FixedLocator([144.663,144.683])
gl.ylocator = mticker.FixedLocator([13.458,13.468])

img = plt.imread(path_ort)
plt.imshow(img, origin='upper', transform=ccrs.PlateCarree(), 
          extent=[144.662,144.688,13.4573,13.47])

print('Done.')

print('Plotting maximum depths ...')

# Removing created perimeters and saving new .max
wd = np.loadtxt(path_max, skiprows=6)     # Depth

wd2 = np.empty((397,397))

for i in range(ncols):
    for j in range(nrows):
        if (mask[i,j] == 1):
            wd2[i][j] = wd [i][j]

path_max2 = op.join(path_res, '{0}_{1}_2.max') 
fRes = open(path_max2,'w') 
fHead = open(path_max, 'r')  
         
for i in range(6):
    fRes.write("{0}".format(fHead.readline()))

np.savetxt(fRes,wd2,fmt='%1.3f')    
    
fHead.close()  
fRes.close() 

# Plotting new water depths
plot_raster(path_max2, bmap, np.amax(wd2), 0.002) 

cbar = plt.colorbar()
cbar.set_alpha(1)
cbar.draw_all()
cbar.ax.set_xlabel('Water depth [m]')
plt.show()

print('Done.')

#%% ----------------------------ANIMATION--------------------------------------
print('Creating animation ...')

#os.mkdir(path_anim)   # If the directory already exists comment this

fileswd = sorted(glob(path_res + '/*.wd'))     
arr = []
for file in fileswd:
  with rio.open(file) as src:
    arr.append(src.read(1))
    
dates = [str(x) for x in range(66)]
fig, ax = plt.subplots(figsize=(6, 5))

anim = FuncAnimation(fig, update_wd, frames=range(len(arr)), interval=750)
im = ax.imshow(arr[0], cmap = 'cool', vmin=0.01,vmax=3)
plt.colorbar(im)
plt.show()

anim.save(path_anim+'/animation_{0}_wd.gif'.format(T), dpi=80, writer = 'pillow')
plt.close()

#--
   
filesel = sorted(glob(path_res + '/*.elev'))     
arr2 = []
for file in filesel:
  with rio.open(file) as src:
    arr2.append(src.read(1))

dates = [str(x) for x in range(109)]
fig2, ax2 = plt.subplots(figsize=(6, 5))
 
anim2 = FuncAnimation(fig2, update_el, frames=range(len(arr2)), interval=750)
im2 = ax2.imshow(arr2[0], cmap = cmo.cm.haline,vmin=0.01,vmax=3)
plt.colorbar(im2)
plt.show()

anim2.save(path_anim+'/animation_{0}_elev.gif'.format(T), dpi=80, writer = 'pillow')
plt.close()

print('Done.')

print('ALL DONE.')