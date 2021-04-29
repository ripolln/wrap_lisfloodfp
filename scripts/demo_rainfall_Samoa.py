#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip
import os
import os.path as op
import sys
import numpy as np
import xarray as xr
from pyproj import Proj
import matplotlib.pyplot as plt

# dev
sys.path.insert(0, op.join(op.dirname(__file__), '..'))

# LISFLOOD module 
from lisfloodpy.wrap import LisfloodProject, LisfloodWrap 
from lisfloodpy.wrap import LisfloodDEM, LisfloodCase

from lisfloodpy.plots import plot_output_2d


# --------------------------------------
# data folder
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))

# projects path
p_proj = op.join(p_data, 'projects')

# demo data
p_demo = op.join(p_data, 'sites', 'Samoa')


# --------------------------------------
# input files 

# DEM file (the 3 Samoa islands)
p_dem = op.join(p_demo, 'samoa_dem.nc')

# load site DEM file
dem = xr.open_dataset(p_dem)
lats = dem.lat.values
lons = dem.lon.values


# Cut the middle island: Apia
lon_cut = [-172.09, -171.42]
lat_cut = [-14.05, -13.78]

ix_lat = np.where((lats > lat_cut[0]) & (lats < lat_cut[1]))[0]
ix_lon = np.where((lons > lon_cut[0]) & (lons < lon_cut[1]))[0]

dem_apia = dem.isel(lat=ix_lat, lon=ix_lon)


# change DEM projection to UTM
utm_proj = Proj(
    proj='utm',
    zone='2L',
    ellps='WGS84',
    datum='WGS84',
    units='m',
    south=True,
)


lon_mg, lat_mg = np.meshgrid(dem_apia.lon.values, dem_apia.lat.values)
x_utm, y_utm = utm_proj(lon_mg, lat_mg) 

dem_apia_utm = xr.Dataset(
    {
        'elevation':(('y','x',), dem_apia.elevation.values)
    },
    coords = {
        'x': x_utm[0,:],
        'y': y_utm[:,0],
    }
)


# --------------------------------------
# Lisflood DEM
lf_dem = LisfloodDEM()

# Set DEM
lf_dem.set_dem(dem_apia_utm)

## Preprocess outer perimeters
#lf_dem.preprocess_outer_perimeters()

## Preprocess inner perimeters
#lf_dem.preprocess_inner_perimeters()

## get outflow coordinates
#h_water_level = lf_dem.get_outflow_coordinates()

# plot dem
#lf_dem.plot_raw(vmin=-2, vmax=600)
#plt.show()


# --------------------------------------
# Lisflood CASES (rainfall with and without coastal "outflow")

# input rain (demo)
time = np.arange(20)
rain = np.ones(20)*10
xds_rain = xr.Dataset(
    {'rain': (('time',), rain)},
    coords = {'time': time},
    attrs = {'units': 'mm/h'},
)


# ---
# prepare Lisflood case: without coastal outflow 
lc0 = LisfloodCase()
lc0.id = '000_rainfall_10mmh'

# set rain 
lc0.rain_activate = True
lc0.rain_value = xds_rain['rain']
lc0.rain_freq = 'hours'


## ---
## prepare Lisflood case: with coastal outflow 
#lc1 = LisfloodCase()
#lc1.id = '001_rainfall_10mmh_coastal_outflow'

## set time varying inflow boundaries
#lc1.rain_activate = True
#lc1.rain_value = xds_rain['rain']
#lc1.rain_freq = 'hours'

## set constant water level boundaries (coastal "outflow")
#lc1.hfix_activate = True
#lc1.hfix_xcoord = h_water_level.x.values
#lc1.hfix_ycoord = h_water_level.y.values
#lc1.hfix_value = -0.5


# ---
# store Lisflood cases inside a list
l_cases = [lc0]  #, lc1]


# --------------------------------------
# Lisflood PROJECT 

p_proj = op.join(p_data, 'projects')  # lisflood projects main directory
n_proj = 'demo_rainfall_Samoa'        # project name

lp = LisfloodProject(p_proj, n_proj)  # lisflood wrap

# DEM parameters
lp.cartesian = True
#lp.depth = lf_dem.depth_preprocessed  # preprocessed DEM for coastal flooding
lp.depth = lf_dem.depth  # raw DEM for coastal flooding
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

# Build cases: rainfall flooding with and without coastal "outflow" 
lw.build_cases(l_cases)

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
name = 'Samoa 10mm/h rainfall'
p_export = 'export_fig_samoa_rf10_depthraw'
var_plot = 'max'

plot_output_2d(name, xds_res, var_plot, p_export, sim_time, cmap)

