# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 12:50:36 2020

@author: Manuel Zornoza
"""

''' FLOOD GUAM '''

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import os.path as op
from pyproj import Proj
import xarray as xr
import copy
from subprocess import Popen
import rasterio as rio 
from mpl_toolkits.basemap import Basemap
import cmocean as cmo
import time

#from matplotlib.animation import FuncAnimation
#from glob import glob
#import datetime 

#import cartopy.crs as ccrs
#import matplotlib.ticker as mticker

## ------------------------------ FUNCTIONS -------------------------------- ##
# Define perimeters
def perim_points(mask, xmesh, ymesh, depth, outer):
        '''
        Find perimeter coordinates from depth file 
        · outer = True    obtains outer perimeter points
        · outer = False   obtains inner perimeter points
        '''
        if not outer:
            value = 1            
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

# Get next outer perimeter
def next_perim(depth,mask,xcoor,ycoor,xpos,ypos):
    depth_out = np.copy(depth)
    mask_out = np.copy(mask)
    depth_out[xpos,ypos] = 1
    pos = np.where(depth_out >= 0)
    mask_out[pos[0], pos[1]] = 1

    x, y, z, xpos, ypos = perim_points(mask=mask_out,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth_out,
                                                outer=True)
    return x,y,z,xpos,ypos,depth_out,mask_out

# Get previous inner perimeter
def prev_perim(depth,xcoor,ycoor,xpos,ypos):
    depth_in = np.copy(depth)
    for i in range(len(xpos)):
        depth_in[xpos[i]][ypos[i]] = -9999
    
    mask_in = np.copy(depth_in)
    pos = np.where(depth_in >= 0)
    mask_in[pos[0], pos[1]] = 1
    
    x, y, z, xpos, ypos = perim_points(mask=mask_in,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth_in,
                                                outer=False)  
    return x,y,z,xpos,ypos,depth_in

# Cut perimeter
def cut_perim(x,y,z,xpos,ypos,lim,x_axis,minor):   
    ''' lim:    coordinate of cut
        x_axis: cutting direction [True=y axis, False=x axis]
        minor:  selected values referred to 'lim' [True=under, False=above] '''
    
    if x_axis:
        cpos = xpos
    if not x_axis:
        cpos = ypos
    if minor:
        cut = np.where(np.array(cpos) < lim)
    if not minor:
        cut = np.where(np.array(cpos) > lim)
        
    xc = x[cut[0]]; yc = y[cut[0]]; zc = z[cut[0]]
    xposc = np.zeros((len(cut[0]),1))
    yposc = np.zeros((len(cut[0]),1))
    for i in range(len(cut[0])):
        xposc[i] = int(xpos[cut[0][i]])
        yposc[i] = int(ypos[cut[0][i]])

    return xc,yc,zc,xposc,yposc

# Plot 1 flood map  
def plot_raster(filepath, mask, bmap, max_wd, min_wd):
  with rio.open(filepath) as src:
      w = copy.copy(cm.get_cmap('cool')) 
      w.set_under(color='none')
      w.set_over(color=w(0.9999999))
      
      wd = np.flipud(src.read(1))
      wdg = np.empty((mask.shape[0],mask.shape[1]))
      maskg = np.flipud(mask)
      
      for j in range(mask.shape[1]):
          for k in range(mask.shape[0]):
              if (maskg[k,j] == 1):
                  wdg[k][j] = wd[k][j]
              else:
                  wdg[k][j] = -1
      
      bmap.imshow(wdg,cmap=w,vmin=min_wd,vmax=max_wd,alpha=1)

## Plot animation
#def update_wd(i):
#    oce = copy.copy(cm.get_cmap('cool'))
#    oce.set_under(color='lightgreen')
#    im_normed = arr[i]
#    
#    wdg = np.empty((im_normed.shape[0],im_normed.shape[1]))
#    #    mask = np.copy(depth)
#    
#    #    pos = np.where(depth >= 0)
#    #    mask[pos[0], pos[1]] = 1
#    
#    for j in range(ncols):
#        for k in range(nrows):
#            if (mask[k,j] == 1):
#                wdg[k][j] = im_normed [k][j]
#    
#    ax.imshow(wdg, cmap = oce, vmin=0.02, vmax=1)
#    ax.set_title('Water depths Kwajalein - ' + str(dates[i]), fontsize=15,pad=15)
#    ax.set_axis_off()  
#    
#def update_el(i):
#    oce = copy.copy(cmo.cm.haline)
#    oce.set_under(color='k',alpha=0.4)
#    im_normed = arr[i]
#    
#    wdg = np.empty((mask.shape[0],mask.shape[1]))
#    mask = np.copy(depth)
#
#    pos = np.where(depth >= 0)
#    mask[pos[0], pos[1]] = 1
#
#    for j in range(ncols):
#        for k in range(nrows):
#            if (mask[k,j] == 1):
#                wdg[k][j] = im_normed [k][j]
#        
#    ax.imshow(wdg, cmap = oce, vmin=0.01, vmax=1)
#    ax.set_title('WATER ELEVATIONS - ' + dates[i], fontsize=20)
#    ax.set_axis_off()
    
# Projection
utm_guam = Proj(proj='utm', zone='55N', ellps='WGS84', datum='WGS84', units='m', south=False)
    
#%% ----------------------PATHS AND FILES--------------------------------------
start = time.time()

print('Loading files and paths ...')

p = 'GUAM'                       # Location

# -------------------------LISFLOOD cases--------------------------------------
'''
 This has to be changed depending on the inputs (hydrogrpahs), as well as the
   simulation time (sim_time) and the saving interval (saveint and massint)
'''
RP = list(['20','100'])          # Return period
L = list(['00','03','04','06'])  # Sea Level
#------------------------------------------------------------------------------

# DEM and parameters
demfile = '{0}_flood_UTM_5m.dem.asc'.format(p)
initial_tstep = '100'       # Initial guess for optimum and max time step
fpfric = '0.06'             # Manning
sim_time = '72000'          # Simulation time (in seconds) [72000s=20h]
massint = '1200'            # Mass balance save interval   [1200s=20min]
saveint = '1200'            # Results save interval        [1200s=20min]

# Paths
wp = os.getcwd()                                 # Working path
os.chdir('..')
dp = op.join(os.getcwd(),'data')                 # Data path
mp = op.join(os.getcwd(),'FloodMaps')            # Maps path
#os.mkdir(mp)           #Comment this if directory already exists
lp = '/home/administrador/Documentos/Lisflood/'  # Lisflood path
os.chdir(wp)

path_mdem = op.join(dp, demfile)                    # DEM for lisflood (UTM)
path_ort = op.join(dp,'{0}_2018_5m.png'.format(p))  # Orthoimage

# Raster
depth = np.loadtxt(path_mdem,skiprows=6)

nrows = depth.shape[0]                    # Rows 
ncols = depth.shape[1]                    # Cols 

with open(path_mdem) as d:
    for i in range(6):
        line = d.readline()
        if(i == 2):
            lonmin = float(line[10:-1])   # Xllcorner
        elif(i == 3):
            latmin = float(line[10:-1])   # Yllcorner
        elif(i == 4):
            step = float(line[9:-1])      # Cellsize
        elif(i == 5): 
            nodata = int(line[13:-1])     # Nodata_value
d.close()

xx =  np.linspace(lonmin,lonmin+step*(ncols-1),ncols)
yy =  np.linspace(latmin,latmin+step*(nrows-1),nrows)
xxyy = np.meshgrid(xx,yy)
xcoor = xxyy[0]                           # X Coordinates
ycoor = np.flipud(xxyy[1])                # Y Coordinates

del xx,yy,xxyy,line

#%% -------------------------MODIFY DEM----------------------------------------
# 'Delete' bathymetry
#bati = np.where(depth<0)
#depth[bati] = -9999

# Manual correction of certain zones
#for i in range(nrows):
#    for j in range(ncols):
#        if(i >= 165 and i <= 276 and j >= 323 and j <= 484 and depth[i][j] == -9999):
#            depth[i][j] = 0
#
#for i in range(nrows):
#    for j in range(ncols):
#        if(i >= 49 and j <= 138 and depth[i][j] == 20):
#            depth[i][j] = -9999
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

x1, y1, z1, xpos1, ypos1 = perim_points(mask=mask,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth,
                                                outer=False)

x_p1, y_p1, z_p1, xpos_p1, ypos_p1 = perim_points(mask=mask,xmesh=xcoor,
                                                ymesh=ycoor,depth=depth,
                                                outer=True)
# --- 3 outer perimeters ---
x_p2, y_p2, z_p2, xpos_p2, ypos_p2, depth_out2, mask_out2 = next_perim(depth,mask,xcoor,
                                                                       ycoor,xpos_p1,ypos_p1)
x_p3, y_p3, z_p3, xpos_p3, ypos_p3, depth_out3, mask_out3 = next_perim(depth_out2,mask_out2,
                                                                       xcoor,ycoor,xpos_p2,ypos_p2)
x_p4, y_p4, z_p4, xpos_p4, ypos_p4, depth_out4, mask_out4 = next_perim(depth_out3,mask_out3,
                                                                       xcoor,ycoor,xpos_p3,ypos_p3)

# Set new depth for outer perimeters
for i in range(len(xpos_p1)):
    depth[xpos_p1[i]][ypos_p1[i]]=-1
for i in range(len(xpos_p2)):
    depth[xpos_p2[i]][ypos_p2[i]]=-1
for i in range(len(xpos_p3)):
    depth[xpos_p3[i]][ypos_p3[i]]=-1
for i in range(len(xpos_p4)):
    depth[xpos_p4[i]][ypos_p4[i]]=-1
      
# --- 1 inner perimeter ---
x_q1, y_q1, z_q1, xpos_q1, ypos_q1, depth_in1 = prev_perim(depth,xcoor,ycoor,
                                                           xpos1,ypos1)
 
# Set limits for inner perimeters
xq,yq,zq,xposq1,yposq1 = cut_perim(x_q1,y_q1,z_q1,xpos_q1,ypos_q1,159,True,True)
xq,yq,zq,xposq2,yposq2 = cut_perim(xq,yq,zq,xposq1,yposq1,485,False,True)
xq,yq,zq,xposq,yposq = cut_perim(xq,yq,zq,xposq2,yposq2,13,False,False)
  
# Berm location 
berm = xr.open_dataset(op.join(dp,'Berm_Guam.nc'))   
xb,yb = utm_guam(np.array(berm['Lon']),np.array(berm['Lat']),inverse=False)
zb = np.array(berm['Elevation'])
    
''' ----------------------------------------------------------------
---------------Target: Establish each transect on the--------------------------
--------corresponding perimeter to each berm coordinate (blue points)----------
    ----------------------------------------------------------------
'''
# Berm homogenisation (assign transect (and change z) to each point of perimeter)      
id_trans = assign_group_id(xq, yq, xb, yb)

for i in range(len(xq)):
    depth[int(xposq[i])][int(yposq[i])] = float(berm['Elevation'][int(id_trans[i])])
'----------------------------------------------------------------------'

del depth_out2,mask_out2,depth_out3,mask_out3,depth_out4,mask_out4,depth_in1
del xpos_p2,ypos_p2,xpos_p3,ypos_p3,xpos_p4,ypos_p4
   
print('Done.')
    
#%% ---------------PLOT DEM (CHECK MODIFICATIONS, PERIMETERS, ETC.)------------
cmap = cmo.cm.algae
cmap.set_under(color='none')
norm = mpl.colors.Normalize(vmin=-10)

fig, ax = plt.subplots(1,1,figsize=(14,7))
im1 = ax.pcolor(xcoor, ycoor, depth, shading='auto', cmap=cmap, vmin=-2, vmax=10)

ax.plot(x_p1, y_p1, '.', color='C4')    # Purple dots   
ax.plot(x_p2, y_p2, '.', color='C8')    # Yellow dots   
ax.plot(x_p3, y_p3, '.', color='C3')    # Red dots      
ax.plot(x_p4, y_p4, '.', color='C1')    # Orange dots   

color = ['c','k','g','b','r','y']*2
for i in range(xq.size):
    plt.plot(xq[i], yq[i], '.', c=color[int(id_trans[i])])
for j in range(12):
    plt.text(xq[id_trans==j].mean()+40, yq[id_trans==j].mean()+20, '{0}'.format(j+1), c='black')

ax.plot(xb, yb, '.', color='b')

cbar = plt.colorbar(im1,cmap=cmap)
cbar.set_alpha(1)
cbar.draw_all()
cbar.ax.set_xlabel('Elevation [m]')
plt.show()

del x_p1,y_p1,x_p4,y_p4

#%% --------------------------UPDATE DEM---------------------------------------
print('Writing modified DEM ...')

demfile = '{0}_lisflood_UTM_5m.dem.asc'.format(p)
path_mdem = op.join(lp, demfile)
fDem = open(path_mdem,'w') 
           
fDem.write('ncols\t\t{0}\n'.format(ncols))  
fDem.write('nrows\t\t{0}\n'.format(nrows))
fDem.write('xllcorner\t{0:.6f}\n'.format(lonmin)) 
fDem.write('yllcorner\t{0:.6f}\n'.format(latmin)) 
fDem.write('cellsize\t{0:.6f}\n'.format(float(step)))          
fDem.write('NODATA_value\t{0}\n'.format(nodata))
np.savetxt(fDem,depth,fmt='%1.3f')    
       
fDem.close()

print('Done.')

#%% ---------------------------STARTFILE---------------------------------------
print('Creating startfile ...')

startfile = '{0}_start_UTM.start'.format(p)
path_sta = op.join(lp,startfile)
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

bcifile = '{0}_bc_UTM.bci'.format(p)
path_bci = op.join(lp, bcifile)
fbci = open(path_bci,'w') 

# xq, yq: Points on the homegenised berm
for i in range(len(xq)):         
    fbci.write("{0}\t{1:.8f}\t\t{2:.8f}\t{3}\t{4}\n".format('P', xq[i], yq[i], 'QVAR', 
               int(id_trans[i]+1)))

# x_pn, y_pn: Points over the n perimeter
for i in range(len(x_p2)):         
    fbci.write("{0}\t{1:.8f}\t\t{2:.8f}\t{3}\t{4}\n".format('P', x_p2[i], y_p2[i], 'HFIX', -0.5))

for i in range(len(x_p3)):
    fbci.write("{0}\t{1:.8f}\t\t{2:.8f}\t{3}\t{4}\n".format('P', x_p3[i], y_p3[i], 'HFIX', -0.5))  
    
fbci.close()

print('Done.')

#%% -----------------------------RUNNING CASES---------------------------------

'''  Loop over scenarios. 
       1 - Create bdy file
       2 - Create par file
       3 - Run LISFLOOD-FP
       4 - Plot flooding map
       5 - Create GIF with all maps (under development) '''

# HISTORICAL T = 20 YEARS
#qsh = xr.open_dataset(op.join(dp,'Historical_20.nc')) 
#
#qh = np.empty((7,12))
#for i in range(11): 
#    cont = 0
#    for j in range(7):
#        if(j == 0 or j == 2 or j == 3 or j == 5 or j == 6):
#            qh[j,i] = float(qsh['q'][cont][int(i)])
#            cont = cont+1
#        elif(j==1):
#            qh[j,i] = (float(qsh['q'][0][int(i)])+float(qsh['q'][1][int(i)]))/2
#        else:
#            qh[j,i] = (float(qsh['q'][2][int(i)])+float(qsh['q'][3][int(i)]))/2
#    
#path_bdy = op.join(wp,'{0}_tvbc_{1}_UTM.bdy'.format(p,'hist20')) 
#fbdy = open(path_bdy,'w') 
#fbdy.write('GUAM: 12 profiles along North shore. Return period:{0}\n'.format('20')) 
#for i in range(12): 
#    fbdy.write('{0}\n'.format(i+1))                               # Profile
#    fbdy.write('{0}\t{1}\n'.format('7','hours'))  # Hydro length
#    
#    for j in range(7):
#        fbdy.write('{0:.9f}\t{1}\n'.format(float(qh[j][i]/1000),int(j)))
#    fbdy.write('\n')
#                     
#fbdy.close()
    

# EMULATOR RESULTS (T = 20,100 YEARS; SLR = +0.0, +0.3, +0.4, +0.6 m)
qs = xr.open_dataset(op.join(dp,'aggregate_Guam_q.nc')) 

'''  WATCH OUT!! As we are working with UTM coordinates, water inflows must
      be given in m2/s (or m3/s·m), which will be after multiplied by the
                    cell size while running LISFLOOD-FP. 
               Used dataset gives water water inflows in L/s·m            '''

for rp in RP:
    for slr in L:
        print('Creating bdyfile RP {0}  SLR +{1}...'.format(rp,slr))
                   
        path_bdy = op.join(lp,'{0}_tvbc_{1}_UTM_{2}.bdy'.format(p,rp,slr))
        fbdy = open(path_bdy,'w')  
        fbdy.write('GUAM: 12 profiles along North shore. Return period:{0}. Sea Level Rise: {1}\n'.format(rp,slr)) 
 
        for i in range(qs['profile'].shape[0]):
            
            fbdy.write('{0}\n'.format(i+1))                                # Profile
            fbdy.write('{0}\t{1}\n'.format(qs['time'].shape[0],'hours'))   # Hydro length
            for j in range(qs['time'].shape[0]):           
                if(rp=='20'):  
                    if (slr=='00'):
                        q = qs['q'][0][0][i][j]     # SLR, RP, profile, time 
                        
                        if(np.isnan(q)):
                            fbdy.write('{0:.9f}\t{1}\n'.format(0,j))
                        else:
                            fbdy.write('{0:.9f}\t{1}\n'.format(float(q/1000),j))
                            
                    elif (slr=='03'):
                        q = qs['q'][1][0][i][j]     # SLR, RP, profile, time 
                        
                        if(np.isnan(q)):
                            fbdy.write('{0:.9f}\t{1}\n'.format(0,j))
                        else:
                            fbdy.write('{0:.9f}\t{1}\n'.format(float(q/1000),j))
                            
                    elif (slr=='04'):
                        q = qs['q'][2][0][i][j]     # SLR, RP, profile, time 
                        
                        if(np.isnan(q)):
                            fbdy.write('{0:.9f}\t{1}\n'.format(0,j))
                        else:
                            fbdy.write('{0:.9f}\t{1}\n'.format(float(q/1000),j))
                            
                    else:
                        q = qs['q'][3][0][i][j]     # SLR, RP, profile, time 
                        
                        if(np.isnan(q)):
                            fbdy.write('{0:.9f}\t{1}\n'.format(0,j))
                        else:
                            fbdy.write('{0:.9f}\t{1}\n'.format(float(q/1000),j))
                else:
    
                    if (slr=='00'):
                        q = qs['q'][0][1][i][j]     # SLR, RP, profile, time                         
                        if(np.isnan(q)):
                            fbdy.write('{0:.9f}\t{1}\n'.format(0,j))
                        else:
                            fbdy.write('{0:.9f}\t{1}\n'.format(float(q/1000),j))
                            
                    elif (slr=='03'):
                        q = qs['q'][1][1][i][j]     # SLR, RP, profile, time                        
                        if(np.isnan(q)):
                            fbdy.write('{0:.9f}\t{1}\n'.format(0,j))
                        else:
                            fbdy.write('{0:.9f}\t{1}\n'.format(float(q/1000),j))
                            
                    elif (slr=='04'):
                        q = qs['q'][2][1][i][j]     # SLR, RP, profile, time                        
                        if(np.isnan(q)):
                            fbdy.write('{0:.9f}\t{1}\n'.format(0,j))
                        else:
                            fbdy.write('{0:.9f}\t{1}\n'.format(float(q/1000),j))
                            
                    else:
                        q = qs['q'][3][1][i][j]     # SLR, RP, profile, time                         
                        if(np.isnan(q)):
                            fbdy.write('{0:.9f}\t{1}\n'.format(0,j))
                        else:
                            fbdy.write('{0:.9f}\t{1}\n'.format(float(q/1000),j))
            
            fbdy.write('\n')                     
            
        fbdy.close()
            
        print('Done.')
            
        print('Creating parfile RP {0}  SLR +{1}...'.format(rp,slr)) 
        resroot = '{0}_{1}_UTM_{2}'.format(p,rp,slr)
        dirroot = 'Res_{0}_{1}_UTM_{2}'.format(p,rp,slr)
        
        parfile = '{0}_{1}_UTM_{2}.par'.format(p,rp,slr)
        path_par = op.join(lp,parfile)
        fpar = open(path_par,'w') 
        
        fpar.write("{0}\n".format('# Flooding Guam - Emulated events'))
        fpar.write("{0}\t{1}\n".format('DEMfile',demfile))
        fpar.write("{0}\t{1}\n".format('resroot',resroot))
        fpar.write("{0}\t{1}\n".format('dirroot',dirroot))
        fpar.write("{0}\t{1}\n".format('sim_time',sim_time))
        fpar.write("{0}\t{1}\n".format('initial_tstep',initial_tstep))
        fpar.write("{0}\t{1}\n".format('massint',massint))
        fpar.write("{0}\t{1}\n".format('saveint',saveint))
        fpar.write("{0}\t\t{1}\n".format('fpfric',fpfric))
        fpar.write("{0}\t{1}\n".format('bcifile',bcifile))
        fpar.write("{0}\t{1}\n".format('bdyfile','{0}_tvbc_{1}_UTM_{2}.bdy'.format(p,rp,slr)))
        fpar.write("{0}\t{1}\n".format('startfile',startfile))                           
        fpar.write("\n".format())
        
        fpar.write("{0}\n".format('elevoff'))
        fpar.write("{0}\n".format('acceleration'))
        fpar.write("{0}\t{1}\n".format('theta',1))
        fpar.write("{0}\n".format('comp_out'))
        
        fpar.close()
        print('Done.')
                
        print('Running LISFLOOD-FP ...')
        
        ''' ----------------------------------------------------------------
        -----------Target: Solve libiomp5.so issue for any device,-------------
        ----------this works for mine but who knows with another one-----------
            ----------------------------------------------------------------
        '''
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['LD_LIBRARY_PATH'] = r'/home/anaconda3/lib/'
        '----------------------------------------------------------------------' 
        
        os.chdir(lp)
        c = Popen(["./lisflood.exe",'-v',parfile])
        c.wait()
        
        print('Done.')
             
        path_res = op.join(lp,dirroot)
        path_max = op.join(path_res,'{0}_{1}_UTM_{2}-0031.wd'.format(p,rp,slr))
    
        fig,ax = plt.subplots(1,1,figsize=(13,7))
        
        ''' ---------------------------------------------------------------
        ------------Target: Change basemap for cartopy/geopandas---------------
            ---------------------------------------------------------------
        '''      
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
        '----------------------------------------------------------------------'

        lonmin, latmin = utm_guam(246872.22726091,1488874.5187709,inverse=True)
        lonmax,latmax = utm_guam(246872.22726091+step*(ncols-1),
                                 1488874.5187709+step*(nrows-1),inverse=True)
        lonc = lonmin +((lonmax-lonmin)/2)
        latc = latmin +((latmax-latmin)/2)
        
        bmap = Basemap(projection = 'tmerc', llcrnrlat = latmin, llcrnrlon = lonmin,
                      urcrnrlat = latmax, urcrnrlon = lonmax, lon_0 = lonc, lat_0 = latc)       
        paral = np.arange(round(latmin+0.001,3),round(latmax,3),0.01)
        merid = np.arange(round(lonmin+0.001,3),round(lonmax,3),0.02)
        bmap.drawparallels(paral,labels=[1,0,0,0])
        bmap.drawmeridians(merid,labels=[0,0,0,1])
        
        print('Plotting maximum depths RP {0}  SLR +{1}m ...'.format(rp,slr))

        bmap.imshow(np.flipud(plt.imread(path_ort)), origin='lower')
        plot_raster(path_max, mask, bmap, 1, 0.01) 
        
        if (slr == '00'):
            plt.title('Emulated flooding event (RP = {0} years, SLR = +0.0 m)'.format(rp),
                      fontsize=20,pad=30)
        elif (slr == '03'):
            plt.title('Emulated flooding event (RP = {0} years, SLR = +0.3 m)'.format(rp),
                      fontsize=20,pad=30)
        elif (slr == '04'):
            plt.title('Emulated flooding event (RP = {0} years, SLR = +0.4 m)'.format(rp),
                      fontsize=20,pad=30)
        else: 
            plt.title('Emulated flooding event (RP = {0} years, SLR = +0.6 m)'.format(rp),
                      fontsize=20,pad=30)
       
        cbar = plt.colorbar(extend='max')
        cbar.set_alpha(1)
        cbar.draw_all()
        cbar.ax.set_xlabel('Water depth [m]')
        
        plt.savefig(mp + '/em{0}_{1}.png'.format(rp,slr))
        plt.show()
        
        print('Done.')

#        # UNDER DEVELOPMENT: CREATE ANIMATION (GIF/MP4) 
        ''' --------------------------------------------------------------
         -----------TARGET: Set orthoimage under flood maps--------------------
            --------------------------------------------------------------
        '''
#        print('Creating animation RP {0}  SLR +{1}m ...'.format(rp,slr))
        
#        path_anim = op.join(mp,'Animation')
##        os.mkdir(path_anim)   # If the directory already exists comment this
#        
#        fileswd = sorted(glob(path_res + '/*.wd'))     
#        arr = []               
#        for file in fileswd:
#          with rio.open(file) as src:
#            arr.append(src.read(1))
#        
#        base = datetime.datetime(1979, 1, 3, 8)
#        dates = np.array([base + datetime.timedelta(minutes=20*i) for i in range(len(arr))])   
#        
#        fig,ax2 = plt.subplots(figsize=(5,5))
#        im0 = ax2.imshow(arr[0], cmap = 'cool', vmin=0.05,vmax=1)
#        plt.close()
#        
#        fig, ax = plt.subplots(figsize=(8, 6))
#        im = ax.imshow(mask, cmap = 'GnBu_r', vmin=0,vmax=1)
#        anim = FuncAnimation(fig, update_wd, frames=range(len(arr)),interval=1000)
#        cbar = plt.colorbar(im0,extend='max')
#        cbar.ax.set_ylabel('Water depth [m]')
#        plt.show()
#        
#        anim.save(path_anim+'/animation_em_{0}_{1}_wd.gif'.format(rp,slr), dpi=80, writer = 'pillow')
#        plt.close()
        '----------------------------------------------------------------------'
        
        print('Done.')
        
        print('*** CASE: RP {0}  SLR +{1}m DONE. ***'.format(rp,slr))
        
print('ALL DONE. It took {0} minutes and {1:.2f} seconds.'.format(int((time.time()-start)/60),(time.time()-start)%60))

