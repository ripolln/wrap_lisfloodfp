#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import subprocess as sp
from subprocess import call

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import copy
import rasterio as rio 
from mpl_toolkits.basemap import Basemap
import cmocean as cm
from pyproj import Proj

from .io import LisfloodIO


class LisfloodProject(object):
    'LISFLOOD-FP flood model project parameters.'
    'Used inside LisfloodWrap and LisfloodIO'

    def __init__(self, p_proj, n_proj): # p=path, n=name
        '''
        This class stores LISFLOOD-FP project information 
        '''

        self.p_main = op.join(p_proj, n_proj)    # project path
        self.name = n_proj                       # project name
        
        # sub folders
        self.p_cases = op.join(self.p_main, 'cases')

        # topobathymetry depth value (2D numpy.array)
        self.cartesian = None       # units convention
        self.depth = None           # topo-bathy data
        self.xmesh = None
        self.ymesh = None

        # .dem file parameters
        self.nrows = None           # number of rows
        self.ncols = None           # number of columns 
        self.xll_corner = None        # x low left corner
        self.yll_corner = None        # y low left corner
        self.cellsize = None        # cell size
        self.nondata = None         # value for non-data

        #----------------------------------------------------------------------       
        # source points
        self.xq = None
        self.yq = None
        
        # elevation points
        self.xh = None
        self.yh = None
        
        # total water level
        self.twl = None
        
        #number of profiles
        self.prof = None
        #----------------------------------------------------------------------

        # .par file parameters
        self.DEMfile = None         # digital elevation model filename
        self.resroot = None         # root for naming the results files
        self.dirroot = None         # path for results files
        self.saveint = None         # interval for saved files (in seconds)
        self.massint = None         # interval for .mass file (in seconds)
        self.sim_time = None        # simulation length (in seconds)
        self.initial_tstep = None   # model time step (in seconds)
        self.fpfric = None          # Manning's value (floodplain) if spatially uniform
        self.manningfile = None     # filename for grid Manning values
        
        # .bci file parameters
        self.bci = False            # activation of .bci
        self.bcifile = None         # boundary condition filename (.bci)
        self.bc_ide = None          # boundary identifier (boundaries, or point source)
        self.q_bc_names = None      
        self.lev_bc_names = None
        
        # .bdy file parameters (optional)
        self.bdy = False            # activation of .bdy
        self.bdyfile = None         # varying boundary conditions filename (.bdy)
        self.bc_freq = None         # time units
        
        #----------------------------------------------------------------------
        # .start file
        self.start = False          # activation of .start
        self.startfile = None       # start conditions filename (.start)
        #----------------------------------------------------------------------
        
        self.solver = None          # model solver
        
        # water parameters
        self.rainfall = None        # (mm/h) uniform
        self.routing = None         # RECOMMENDED FOR RAINFALL WHEN STEEP SLOPES
        self.depththresh = None
        self.routingspeed = None
        self.infiltration = None    # (m/s) uniform

        # optional additions
        self.theta = None           # reset numerical diffusion for 'acceleration'
        self.latlong = None         # only for Subgrid and 2D
        
        # additional output settings
        self.overpass = None        # time for flood image available (s) ??????
        self.depthoff = None        # suppress depth files (*.wd) 
        self.elevoff = None         # suppress water surface files (*.elev)
        self.mint_hk = None         # allows calculation of maxH... at the mass interval instead of every tstep
        
        # # output points
        # self.x_out = []
        # self.y_out = []
        
        # # save plot vortex
        # self.plot_vortex = None 


class LisfloodWrap(object):
    'LISFLOOD-FP flood model wrap for multi-case handling'

    def __init__(self, lisflood_proj):  #lisflood_proj ->  created with LisfloodProject
        '''
        initializes wrap

        lisflood_proj - LisfloodProject() instance, contains project parameters
        '''

        self.proj = lisflood_proj           # lisflood project parameters
        self.io = LisfloodIO(self.proj)     # lisflood input/output 

        # resources
        p_res = op.join(op.dirname(op.realpath(__file__)), 'resources')

        # lisflood executable
        self.bin = op.abspath(op.join(p_res, 'lisflood.exe'))

    def build_cases(self, q_pts, h_pts, qs):
        '''
        generates all files needed for lisflood-case execution
        
        event_list - list of inflow events (xarray.dataset) 
        '''

        # make main project directory
        self.io.make_project()

        # one case for each inflow event
             # build case 
        for i in range(qs.sl.shape[0]):
            for j in range(qs.rp.shape[0]):
                case_id = '{0}_{1}'.format(np.array2string(qs.sl[i]),
                                           np.array2string(qs.rp[j]))
                self.io.build_case(case_id, qs, qs.sl[i], qs.rp[j])

    def get_run_folders(self):
        'return sorted list of project cases folders'

        ldir = sorted(os.listdir(self.proj.p_cases))
        fp_ldir = [op.join(self.proj.p_cases, c) for c in ldir]

        return [p for p in fp_ldir if op.isdir(p)]

    def run_cases(self):
        'run all cases inside project "cases" folder'

        # get sorted execution folders
        run_dirs = self.get_run_folders()
        
        for p_run in run_dirs:
            
            # run case
            self.run(p_run)

            # log
            p = op.basename(p_run)
            print('LISFLOOD-FP CASE: {0} SOLVED'.format(p))
            
    def run(self, p_run):
        'Execution commands for launching LISFLOOD-FP'

        os.chdir(p_run)

        print('\n Running LISFLOOD-FP case...\n')

        ''' The next couple of lines solve the libiomp5.so problem, setting its
            path to /home/anaconda3/lib/ and verifying if it is duplicated.
            It works for this device (Linux), but may not work for others.
           
           TARGET: solve libiomp5.so problem with a more <<universal>> solution
        '''

        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        os.environ['LD_LIBRARY_PATH'] = r'/home/anaconda3/lib/'

        # execute file case
        
        par_file = op.join(p_run, '{0}.par'.format(self.proj.resroot))

        call([self.bin, '-v', par_file])
        
        
    def extract_output(self, output_run):
        '''
        extract output from all cases generated by "build_cases"

        return xarray.Dataset (uses new dim "case" to join output)
        '''

        # get sorted execution folders
        run_dirs = self.get_run_folders()

        # exctract output case by case and concat in list
        l_out = []
        for p_run in run_dirs:

            # read output file
            xds_case_out = self.io.output_case(p_run, output_run)
            l_out.append(xds_case_out)

        # concatenate xarray datasets (new dim: case)
        xds_out = xr.concat(l_out, dim='case')

        return(xds_out)


    def extract_bci(self):
        '''
        extract input from all cases generated by "build_cases"

        return xarray.Dataset (uses new dim "case" to join output)
        '''

        # get sorted execution folders
        run_dirs = self.get_run_folders()

        # exctract input case by case and concat in list
        l_out_Q, l_out_H = [], []
        for p_run in run_dirs:

            # read input file
            xds_case_Q, xds_case_H = self.io.output_bci(p_run)
            l_out_Q.append(xds_case_Q)
            l_out_H.append(xds_case_H)

        # concatenate xarray datasets (new dim: case)
        xds_out_Q = xr.concat(l_out_Q, dim='case')
        xds_out_H = xr.concat(l_out_H, dim='case')
        
        return(xds_out_Q, xds_out_H)