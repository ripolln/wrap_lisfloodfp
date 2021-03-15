#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 11 19:21:35 2020

@author: Sara Ortega 
"""

# pip
import os
import os.path as op
import sys
import numpy as np
import xarray as xr

# dev lisflood library 
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# LISFLOOD MODULE (lisflood.DD.lisflood)
from lisflood.DD.wrap import LisfloodProject, LisfloodWrap
#from lisflood.DD.plots import plot_raster, update_wd

# paths
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))
p_demo = op.abspath(op.join(op.dirname(__file__), '..', 'data', 'demo'))

#sys.exit()

import time
start = time.time()


#%%---------------------------LOAD DATA----------------------------------------
# Raster modified DEM
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

del xx,yy,xxyy,line,path_mdem

# Inflow/outflow points (obtained with preprocess.py)
q_pts = xr.open_dataset(op.join(p_demo, 'q_pts.nc'))
h_pts = xr.open_dataset(op.join(p_demo, 'h_pts.nc'))

'''  WATCH OUT!! As we are working with UTM coordinates, water inflows must
      be given in m2/s (or m3/s·m), which will be after multiplied by the
                    cell size while running LISFLOOD-FP. 
              Used dataset gives water water inflows in L/s·m            '''
              
# Hydrographs (one per profile, RP and SLR scenario)
qs = xr.open_dataset(op.join(p_demo,'GUAM_q.nc')) 

#sys.exit()

#%% ------------------------LISFLOOD PROJECT-----------------------------------

p_proj = op.join(p_data, 'projects')        # lisflood projects main directory
n_proj = 'test_wrap'                        # project name

lp = LisfloodProject(p_proj, n_proj)        # lisflood wrap

# DEM parameters
lp.cartesian = True
lp.depth = depth
lp.xmesh = xmesh
lp.ymesh = ymesh
lp.nrows = nrows
lp.ncols = ncols
lp.xll_corner = xll          # xleft
lp.yll_corner = yll         # ybottom
lp.cellsize = cellsize
lp.nondata = nondata

# LISFLOOD parameters
lp.DEMfile = '{0}.dem.asc'.format(n_proj) # DEM filename
lp.resroot = 'guam'             # results files name
lp.dirroot = 'results'          # results files path
lp.saveint = 1200               # interval saved files (s)
lp.massint = 1200               # interval .mass file (s)
lp.sim_time = 72000             # simulation length (s)
lp.initial_tstep = 100          # Initial guess for optimum and max time step
lp.fpfric = 0.06                # Manning's (floodplain) if spatially uniform

# Boundary conditions
# Location and type (BCI)
lp.bci = True
if lp.bci:
    lp.bcifile = '{0}_{1}.bci'.format(lp.resroot, n_proj)
    lp.bc_ide = 'P'             # boundary identifier ('N','E','S','W' or 'P')

# Time varying conditions (BDY)
lp.bdy = True
if lp.bdy:
    lp.bc_freq = 'hours'
    lp.bdyfile = '{0}_{1}.bdy'.format(lp.resroot, n_proj) 

# Model solver
lp.solver = 'acceleration'      

# Water balance parameters
#lp.rainfall = '{0}_{1}.rain'.format(lp.resroot, n_proj)   # (mm/h) uniform
#lp.routing = 'routing'        # RECOMMENDED FOR RAINFALL WHEN STEEP SLOPES
#lp.depththresh
#lp.routingspeed
#lp.infiltration = 0.0000001   # (m/s) uniform

# Other parameters
#lp.theta = 1              # adds numerical diffusion to the inertial model if below 1
#lp.latlong = 'latlong'    # only for Subgrid and 2D

# Output options
lp.overpass = None        # time in seconds at which save flood image for validation
lp.depthoff = None        # suppress depth files (*.wd) 
lp.elevoff = 'elevoff'    # suppress water surface files (*.elev)
#lp.mint_hk = None        # allows calculation of maxH... at the mass interval instead of every tstep

# Other parameters case-specific
lp.xq = q_pts.x
lp.yq = q_pts.y
lp.prof = q_pts.profile
lp.twl = -0.5

lp.xh = h_pts.x
lp.yh = h_pts.y

#sys.exit()

#%% -------------------------LISFLOOD WRAP-------------------------------------

lw = LisfloodWrap(lp) 

# Build cases
lw.build_cases(q_pts,h_pts,qs)

# Run LISFLOOD-FP
lw.run_cases()

print('ALL DONE. It took {0} minutes and {1:.2f} seconds.'.format(int((time.time()-start)/60),(time.time()-start)%60))
#sys.exit()

# Extract results
#if not op.isdir(lw.proj.p_cases): os.makedirs(lw.proj.p_cases)
#os.chdir(lw.proj.p_cases)

#xds_mass = lw.extract_output('.mass')       # .mass
#xds_res = lw.extract_output('2d')           # .inittm .maxtm .totaltm .max .mxe
#xds_bc_Q, xds_bc_H = lw.extract_bci()       # input BC

'''-------------------Target: extract_bdy, extract_start...?----------------'''








