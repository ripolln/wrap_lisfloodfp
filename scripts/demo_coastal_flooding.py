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

# BERM file (for coastal flooding)
p_berm = op.join(p_demo, 'GUAM_Berm_UTM.nc')

# Hydrographs (for coastal flooding)
p_hydro = op.join(p_demo, 'GUAM_q.nc')


# --------------------------------------
# load site DEM file
dem = xr.open_dataset(p_dem)

# load site BERM file
berm = xr.open_dataset(p_berm)

# load site hydrographs (one per profile, RP, and SLR scenario)
qs = xr.open_dataset(p_hydro)
qs['q'] = qs['q'] / 1000  # set q units: L/s*m -> m3/s*m (UTM coordinates)


# --------------------------------------
# Lisflood DEM preprocesser for coastal flooding
lf_dem = LisfloodDEM()

# Set DEM
lf_dem.set_dem(dem)

# Preprocess outer perimeters
lf_dem.preprocess_outer_perimeters()

# Preprocess inner perimeters
lf_dem.preprocess_inner_perimeters()

# Cut last inner perimeter
lf_dem.cut_perimeter(159, 'x', 'under')
lf_dem.cut_perimeter(485, 'y', 'under')
lf_dem.cut_perimeter(13, 'y', 'above')

# berm homogenization
xb = np.array(berm['X'])
yb = np.array(berm['Y'])
zb = np.array(berm['Elevation'])

lf_dem.berm_homogenization(xb, yb, zb)

# store preprocessed dem (ascii)
p_dem_preprocessed = op.join(p_demo, 'GUAM_lisflood_UTM_5m.dem.nc')
lf_dem.save_dem_preprocessed(p_dem_preprocessed)

# get and store inflow/outflow coordinates
q_inflow = lf_dem.get_inflow_coordinates()
h_water_level = lf_dem.get_outflow_coordinates()


# plot preprocessed dem
#lf_dem.plot_preprocessed()
#plt.show()


# --------------------------------------
# Lisflood CASES

# build one case for each inflow event (sea level and return period selection)
l_cases = []
c = 1
for sl in qs.sl.values:
    for rp in qs.rp.values:

        # select sea level and return period and build case 
        qs_sel = qs.sel(sl=sl, rp=rp)

        # prepare Lisflood case input 
        lc = LisfloodCase()
        lc.id = '{0:03d}_sl{1}_rp{2}'.format(c, sl, rp)

        # set time varying inflow boundaries
        lc.qvar_activate = True
        lc.qvar_xcoord = q_inflow.x.values
        lc.qvar_ycoord = q_inflow.y.values
        lc.qvar_profile = q_inflow.profile.values + 1
        lc.qvar_value = qs_sel['q']
        lc.qvar_freq = 'hours' 

        # set constant water level boundaries
        lc.hfix_activate = True
        lc.hfix_xcoord = h_water_level.x.values
        lc.hfix_ycoord = h_water_level.y.values
        lc.hfix_value = -0.5 

        # append case to list
        l_cases.append(lc)

        c+=1


# --------------------------------------
# Lisflood PROJECT 

p_proj = op.join(p_data, 'projects')  # lisflood projects main directory
n_proj = 'demo_coastal_flooding'      # project name

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

# Output options
lp.overpass = None          # time in seconds at which save flood image for validation
lp.depthoff = False         # suppress depth files (*.wd) 
lp.elevoff = True           # suppress water surface files (*.elev)
#lp.mint_hk = None          # allows calculation of maxH... at the mass interval instead of every tstep

# Model solver
lp.solver = 'acceleration'

# Other parameters
#lp.theta = 1               # adds numerical diffusion to the inertial model if below 1
#lp.latlong = 'latlong'     # only for Subgrid and 2D

# Optional: can modify default case filenames
#lp.PARfile = 'params.par'    # model parameteres file
#lp.DEMfile = 'DEM.asc'       # digital elevation model filename
#lp.resroot = 'out'           # root for naming the output files
#lp.dirroot = 'outputs'       # name for results files folder
#lp.bcifile = 'boundary.bci'  # boundary condition filename (.bci)
#lp.bdyfile = 'boundary.bdy'  # varying boundary conditions filename (.bdy)


# --------------------------------------
# Lisflood WRAP 
lw = LisfloodWrap(lp)

# Build cases: coastal flooding from hydrographs 
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
name = 'Guam coastal flooding'
p_export = 'temp_export_fig_coastal'
var_plot = 'max'

plot_output_2d(name, xds_res, var_plot, p_export, sim_time, cmap)

