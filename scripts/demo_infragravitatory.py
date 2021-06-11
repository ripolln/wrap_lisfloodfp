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
from wlisflood.wrap import LisfloodProject, LisfloodWrap
from wlisflood.wrap import LisfloodDEM, LisfloodCase

from wlisflood.plots import plot_output_2d


# --------------------------------------
# data folder
p_data = op.abspath(op.join(op.dirname(__file__), '..', 'data'))

# projects path
p_proj = op.join(p_data, 'projects')

# demo data
p_demo = op.join(p_data, 'sites', 'Guam')

# DEM file
p_dem = op.join(p_demo, 'GUAM_utm_5.nc')


# --------------------------------------
# input synthetic infragravitatory wave

# time 
tt = 3600  # seconds
time = np.arange(tt)

# wave
a = 3                               # amplitude
l = 180                             # wave longitude
eta = a * np.sin(2*np.pi*time/l)    # water elevation

# generate dataset (using only one input profile for the entire coast)
dset = xr.Dataset(
    {
        'eta': (['profile', 'time'], np.reshape(eta, (1,-1)))
    },
    coords = {
        'profile': [1],
        'time': time,
    },
)


# --------------------------------------
# load site DEM file
dem = xr.open_dataset(p_dem)

# Lisflood DEM preprocesser
lf_dem = LisfloodDEM()

# Set DEM
lf_dem.set_dem(dem)

# get coastal points coordinates for water inflow
coastal_points = lf_dem.get_coastal_points()


# --------------------------------------
# Lisflood CASE: infragravitatory

# prepare Lisflood case input 
lc = LisfloodCase()
lc.id = '0000_test_infra'

# set time varying inflow boundaries
lc.hvar_activate = True
lc.hvar_xcoord = coastal_points.x.values
lc.hvar_ycoord = coastal_points.y.values
lc.hvar_profile = coastal_points.profile.values + 1
lc.hvar_value = dset['eta']
lc.hvar_freq = 'seconds'


# --------------------------------------
# Lisflood PROJECT 

p_proj = op.join(p_data, 'projects')  # lisflood projects main directory
n_proj = 'demo_infra'                 # project name

lp = LisfloodProject(p_proj, n_proj)  # lisflood wrap

# DEM parameters
lp.cartesian = True
lp.depth = lf_dem.depth               # raw DEM for coastal flooding
lp.xmesh = lf_dem.xmesh
lp.ymesh = lf_dem.ymesh
lp.nrows = lf_dem.nrows
lp.ncols = lf_dem.ncols
lp.xll_corner = lf_dem.xll            # xleft
lp.yll_corner = lf_dem.yll            # ybottom
lp.cellsize = lf_dem.cellsize
lp.nondata = lf_dem.nodataval

# LISFLOOD parameters
lp.saveint = 10            # interval saved files (s)
lp.massint = 10            # interval .mass file (s)
lp.sim_time = 3600         # simulation length (s)
lp.initial_tstep = 10      # Initial guess for optimum and max time step
lp.fpfric = 0.06           # Manning's (floodplain) if spatially uniform

# Output options
lp.overpass = None          # time in seconds at which save flood image for validation
lp.depthoff = False         # suppress depth files (*.wd) 
lp.elevoff = True           # suppress water surface files (*.elev)
#lp.mint_hk = None          # allows calculation of maxH... at the mass interval instead of every tstep

# Model solver
lp.solver = 'acceleration'

# Other parameters
#lp.theta = 1               # adds numerical diffusion to the inertial model if below 1


# --------------------------------------
# Lisflood WRAP 
lw = LisfloodWrap(lp)

# Build case: coastal flooding from infragravitatory
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

# read elevation along simulation
xds_flood = lw.extract_output_wd()
print(xds_flood)


# Plot 2D output
sim_time = int(xds_res.attrs['simtime(h)'])
cmap = 'jet'
name = 'Guam Infragravitatory flooding'
var_plot = 'max'
p_export = lw.proj.p_export

plot_output_2d(name, xds_res, var_plot, p_export, sim_time, cmap)


