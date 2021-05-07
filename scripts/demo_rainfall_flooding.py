#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip
import os
import os.path as op
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# LISFLOOD module 
from wraplisfloodfp.wrap import LisfloodProject, LisfloodWrap 
from wraplisfloodfp.wrap import LisfloodDEM, LisfloodCase

from wraplisfloodfp.plots import plot_output_2d


# --------------------------------------
# data folder
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))

# projects path
p_proj = op.join(p_data, 'projects')

# demo data
p_demo = op.join(p_data, 'sites', 'Guam')


# --------------------------------------
# input files 

# DEM file
p_dem = op.join(p_demo, 'GUAM_utm_5.nc')

# load site DEM file
dem = xr.open_dataset(p_dem)



# --------------------------------------
# Lisflood DEM
lf_dem = LisfloodDEM()

# Set DEM
lf_dem.set_dem(dem)

# plot preprocessed dem
#lf_dem.plot_raw()
#plt.show()


# --------------------------------------
# Lisflood CASES

# input rain (demo)
time = np.arange(20)
rain = np.ones(20)*5
xds_rain = xr.Dataset(
    {'rain': (('time',), rain)},
    coords = {'time': time},
    attrs = {'units': 'mm/h'},
)

# prepare Lisflood case input 
lc = LisfloodCase()
lc.id = '000_test_rainfall'

# set time varying inflow boundaries
lc.rain_activate = True
lc.rain_value = xds_rain['rain']
lc.rain_freq = 'hours' 


# --------------------------------------
# Lisflood PROJECT 

p_proj = op.join(p_data, 'projects')  # lisflood projects main directory
n_proj = 'demo_rainfall_flooding'     # project name

lp = LisfloodProject(p_proj, n_proj)  # lisflood wrap

# DEM parameters
lp.cartesian = True
lp.depth = lf_dem.depth_preprocessed  # preprocessed DEM for coastal flooding
lp.xmesh = lf_dem.xmesh
lp.ymesh = lf_dem.ymesh
lp.nrows = lf_dem.nrows
lp.ncols = lf_dem.ncols
lp.xll_corner = lf_dem.xll            # xleft
lp.yll_corner = lf_dem.yll            # ybottom
lp.cellsize = lf_dem.cellsize
lp.nondata = lf_dem.nodataval

# LISFLOOD parameters
lp.saveint = 1200           # interval saved files (s)
lp.massint = 1200           # interval .mass file (s)
lp.sim_time = 72000         # simulation length (s)
lp.initial_tstep = 100      # Initial guess for optimum and max time step
lp.fpfric = 0.06            # Manning's (floodplain) if spatially uniform

# Water balance parameters
lp.routing = True           # recommended for rainfall when steep sloopes
lp.depththresh = 0.005      # depth at which a cell is considered wet (m)
lp.routesfthresh = 0.1      # Water surface slope above which routing occurs
#lp.infiltration = 0.0000001   # (m/s) uniform

# Output options
lp.overpass = None          # time in seconds at which save flood image for validation
lp.depthoff = True          # suppress depth files (*.wd) 
lp.elevoff = True           # suppress water surface files (*.elev)
#lp.mint_hk = None          # allows calculation of maxH... at the mass interval instead of every tstep

# Model solver
lp.solver = 'acceleration'

# Other parameters
#lp.theta = 1               # adds numerical diffusion to the inertial model if below 1
#lp.latlong = 'latlong'     # only for Subgrid and 2D

# Optional: can modify default case filenames
#lp.PARfile = 'params.par'      # model parameteres file
#lp.DEMfile = 'DEM.asc'         # digital elevation model filename
#lp.resroot = 'out'             # root for naming the output files
#lp.dirroot = 'outputs'         # name for results files folder
#lp.rainfile = 'rainfall.rain'  # rainfall conditions filename (.rain)


# --------------------------------------
# Lisflood WRAP 
lw = LisfloodWrap(lp)

# Build case: rainfall flooding
lw.io.build_case(lc)

# Run LISFLOOD-FP
lw.run_cases()


# --------------------------------------
# Lisflood OUTPUT

# .mass
xds_mass = lw.extract_output('.mass')
print(xds_mass)

# .inittm .maxtm .totaltm .max .mxe
xds_res = lw.extract_output('2d')
print(xds_res)


# Plot 2D output
sim_time = int(xds_res.attrs['simtime(h)'])
cmap = 'jet'
name = 'Guam rainfall flooding'
p_export = 'temp_export_fig_rain'
var_plot = 'max'

plot_output_2d(name, xds_res, var_plot, p_export, sim_time, cmap)

