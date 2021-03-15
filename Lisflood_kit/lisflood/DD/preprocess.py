#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''  Preprocessing of Digital Elevation Model (DEM) for use in LISFLOOD-FP
     INPUTS:    · Raw DEM (NetCDF)
                · Berm location [x,y,z] (NetCDF)
                
     OUTPUTS:   · Modified DEM (Ascii file) with elevations above zero (NaNs below),
                  homogenised berm along each transect (water inflows), and 
                  outer perimeters with constant elevation (water outflows) 
                · Inflow/outflow coordinates and transect [x,y,id]

'''

import os
import os.path as op
import sys
import numpy as np
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmocean as cmo

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

#%% PATHS
    
sys.path.insert(0, op.join(op.dirname(__file__), '..'))
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.abspath(op.join(op.dirname(__file__), '..','..', 'data', 'demo'))
#sys.exit()

#%% DATA

# DEM
dem = xr.open_dataset(op.join(p_demo, 'GUAM_utm_5.nc')).fillna(-9999)

depth = dem.elevation.values
nrows = depth.shape[0]
ncols = depth.shape[1]
xutm = dem.x.values
yutm = dem.y.values
xcoor, ycoor = np.meshgrid(xutm, yutm)
nodata = -9999
cellsize = xcoor[0][1]-xcoor[0][0]

# berm
berm = xr.open_dataset(op.join(p_demo,'GUAM_Berm_UTM.nc')) 
xb,yb = np.array(berm['X']),np.array(berm['Y'])
zb = np.array(berm['Elevation'])

#%% PERIMETERS

print('Applying perimeter functions ...')

pos = np.where(depth >= 0)
mask = np.copy(depth)
mask[pos[0], pos[1]] = 1

x1, y1, z1, xpos1, ypos1 = perim_points(mask=mask, xmesh=xcoor, ymesh=ycoor,
                                        depth=depth, outer=False)

x_p1, y_p1, z_p1, xpos_p1, ypos_p1 = perim_points(mask=mask, xmesh=xcoor,
                                                ymesh=ycoor, depth=depth,
                                                outer=True)
# --- 3 outer perimeters ---
x_p2,y_p2,z_p2,xpos_p2,ypos_p2,depth_out2,mask_out2 = next_perim(depth,mask,xcoor,ycoor,
                                                                 xpos_p1,ypos_p1)
x_p3,y_p3,z_p3,xpos_p3,ypos_p3,depth_out3,mask_out3 = next_perim(depth_out2,mask_out2,
                                                                 xcoor,ycoor,xpos_p2,ypos_p2)
x_p4,y_p4,z_p4,xpos_p4,ypos_p4,depth_out4,mask_out4 = next_perim(depth_out3,mask_out3,
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
xq, yq, zq, xposq, yposq, depth_in1 = prev_perim(depth,xcoor,ycoor,xpos1,ypos1)

# Set limits for inner perimeters [ONLY FOR GUAM!!]
xq,yq,zq,xposq1,yposq1 = cut_perim(xq,yq,zq,xposq,yposq,159,True,True)
xq,yq,zq,xposq2,yposq2 = cut_perim(xq,yq,zq,xposq1,yposq1,485,False,True)
xq,yq,zq,xposq,yposq = cut_perim(xq,yq,zq,xposq2,yposq2,13,False,False)
    
''' ----------------------------------------------------------------
---------------Target: Establish each transect on the--------------------------
--------corresponding perimeter to each berm coordinate (blue points)----------
    ----------------------------------------------------------------
'''
# Berm homogenisation (assign transect (and change z) to each point of perimeter)   
id_trans = assign_group_id(xq, yq, xb, yb)

for i in range(len(xq)):
    depth[int(xposq[i])][int(yposq[i])] = float(zb[int(id_trans[i])])
'''  -----------------------------------------------------------------
-------------------------------------------------------------------------------
     -----------------------------------------------------------------
'''

del depth_out2,mask_out2,depth_out3,mask_out3,depth_out4,mask_out4,depth_in1
del xpos_p2,ypos_p2,xpos_p3,ypos_p3,xpos_p4,ypos_p4

print('Done.')

#%% ---------------PLOT DEM (CHECK MODIFICATIONS, PERIMETERS, ETC.)------------
print('Plotting modified DEM ...')

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

cbar = plt.colorbar(im1)
cbar.set_alpha(1)
cbar.draw_all()
cbar.ax.set_xlabel('Elevation [m]')
plt.show()

del x_p1,y_p1,x_p4,y_p4

print('Done.')

#%% --------------------------UPDATE DEM---------------------------------------
print('Saving modified DEM (.ascii)...')

demfile = 'GUAM_lisflood_UTM_5m.dem.asc'
path_mdem = op.join(p_demo, demfile)
fDem = open(path_mdem,'w') 
           
fDem.write('ncols\t\t{0}\n'.format(ncols))  
fDem.write('nrows\t\t{0}\n'.format(nrows))
fDem.write('xllcorner\t{0:.6f}\n'.format(xutm[0])) 
fDem.write('yllcorner\t{0:.6f}\n'.format(yutm[-1])) 
fDem.write('cellsize\t{0:.6f}\n'.format(float(cellsize)))          
fDem.write('NODATA_value\t{0}\n'.format(nodata))
np.savetxt(fDem,depth,fmt='%1.3f')    
       
fDem.close()

print('Done.')

#%% ---------------------SAVE INFLOW/OUTFLOW COORDS----------------------------
print('Saving inflow/outflow coordinates (.nc)...')

xh_pts = np.append(x_p2,x_p3)
yh_pts = np.append(y_p2,y_p3)
t_pts = np.ones(int(x_p2.shape[0])+int(x_p3.shape[0]))*(1)


q_pts = xr.Dataset({'x':(['pts'],xq),
                     'y':(['pts'],yq)},
                     coords={'profile':(['pts'],id_trans)})

h_pts = xr.Dataset({'x':(['pts'],xh_pts),
                     'y':(['pts'],yh_pts)},
                     coords={'profile':(['pts'],t_pts)})

q_pts.to_netcdf(op.join(p_demo, 'q_pts.nc'))
h_pts.to_netcdf(op.join(p_demo, 'h_pts.nc'))

print('Done.')