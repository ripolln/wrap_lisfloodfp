#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as op

import numpy as np
import pandas as pd
import xarray as xr


class LisfloodIO(object):
    'LISFLOOD model input/output'

    def __init__(self, lisflood_proj):

        # needs LisfloodProject 
        self.proj = lisflood_proj

    def make_project(self):
        'makes lisflood project folder'

        if not op.isdir(self.proj.p_main): os.makedirs(self.proj.p_main)

    def build_case(self, case_id, qs, sl, rp):
        '''
        Builds lisflood case input files for a given inflow event
        
        case_id     - LISFLOOD case index (int)
        event_dict  - input event dictionary with variables:
                    'q_in':          overtoppings (xr.dataset)
                    'lev':           levels (xr.dataset)
                    'perimeter_q':   source points (P1, P2, P3...)
                    'perimeter_lev': level points (P0, P1, P2...)
                    'zbeach':        mean option
        '''

        # LISFLOOD-FP case path
        p_case = op.join(self.proj.p_cases, case_id)

        # make execution dir
        if not op.isdir(p_case): os.makedirs(p_case)
        
        # initialize
        self.proj.bcifile = None
        self.proj.bdyfile = None
        
        # print(' Writing input files...')

        # make DEM *.dem.asc file
        self.make_dem(p_case)
        ' DEM is already a .ascii --> TARGET: copy-paste file in p_case??'
        
        # make boundary conditions *.bci file
        if self.proj.bci:   
            self.make_bci(p_case)
        
        # make time-varying BCs *.bdy file (optional)
        if self.proj.bdy:   
            self.make_bdy(p_case, qs, sl, rp)
        
        # make rainfall *.rain file
        
        # make sensors *.stage file

        # make input *.par file
        self.make_par(p_case)

        print(' Case {0} built.'.format(case_id))                    

    def make_bci(self, p_file):
        '''
        Generates boundary condition type file (.bci)       
        resroot:    project name
        name :      output filename
        bc_ide :    boundary identifier ('N','E','S','W', or 'P' point source)
        q_pts :     inflow coordinates
        h_pts :     outflow (elevations) points  
        '''
        os.chdir(p_file)
        
        # file name
        self.proj.bcifile = '{0}_{1}.bci'.format(self.proj.resroot, self.proj.name)
                    
        # open and write .bci file
        f = open(self.proj.bcifile, 'wb')
        f = self.write_line_var(f)
        f = self.write_line_fix(f)
        ''' TARGET: write HVAR conditions too'''
        
        f.close()
            
    
    def write_line_var(self, f):
        'Write line when boundary condition is variable (mode: QVAR)'
        
        for i in range(self.proj.xq.size):
            line = '{0}\t{1}\t{2}\t{3}\t{4} \n'.format(self.proj.bc_ide,
                    float(self.proj.xq[i]), float(self.proj.yq[i]),'QVAR', 
                    int(self.proj.prof[i])+1)
            f.write(line.encode())
            
        return f

    def write_line_fix(self, f):
        'Write line when boundary condition is variable (mode: HFIX)'
        
        for i in range(self.proj.xh.size):
            line = '{0}\t{1}\t{2}\t{3}\t{4} \n'.format(self.proj.bc_ide,
                    float(self.proj.xh[i]), float(self.proj.yh[i]),'HFIX', 
                    float(self.proj.twl))
            f.write(line.encode())
            
        return f


    def make_bdy(self, p_file, qs, sl, rp):
        '''
        Generates time varying boundary condition file (.bdy)
        
        Line 1 :    (bc_type) comment line, ignored by LISFLOOD-FP
        Line 2 :    (bc_ide)  boundary identifier (from .bci)
        Line 3 :    number or time points and time units ('hours','seconds')
        Line 4+ :   (bc_value, bc_time) value i   time i
        
        '''
        os.chdir(p_file)
        
        # file name
        self.proj.bdyfile = '{0}_{1}.bdy'.format(self.proj.resroot, self.proj.name)
                    
        # open and write .bdy file
        f = open(self.proj.bdyfile, 'wb')
        
        # Line 1
        f.write(('{0} flooding\n'.format(self.proj.resroot)).encode())
        
        ''' LOOP AROUND PROFILES, RP AND SLR '''
        # write time-series for all profiles
        for i in range(qs.profile.size):
            f = self.write_bdy_identifier(f,qs,sl,rp,i)
                    
        f.close()
        

    def write_bdy_identifier(self, f, qs, sl, rp, i):
        'Write line when boundary condition is variable (mode: QVAR, HVAR)'
        
        # Line 2
        f.write(('{0}\n'.format(int(qs.profile[i]))).encode())
        
        # Line 3
        f.write(('{0}\t{1}\n'.format(int(qs.time.size),self.proj.bc_freq)).encode())
        
        # Line 4 to N
        for t in range(qs.time.size):
            q = float(qs.q[int(np.where(qs.sl==sl)[0])][int(np.where(qs.rp==rp)[0])][i][t])/1000
            if (np.isnan(q)):
                f.write(('{0:.9f}\t{1}\n'.format(0,int(qs.time[t]))).encode())
            else:
                f.write(('{0:.9f}\t{1}\n'.format(q,int(qs.time[t]))).encode()) 
            
        f.write(('\n').encode())

        return f
    
    
    def make_par(self, p_file):
        '''
        Generates configuration file (.par)
        
        DEMfile :       digital elevation model filename
        resroot :       root for naming the results files
        dirroot :       path for results files
        saveint :       interval for saved files (in seconds)
        massint :       interval for .mass file (in seconds)
        sim_time :      simulation length (in seconds)
        initial_tstep : model time step (in seconds)
        bcifile :       boundary condition filename (.bci)
        bdyfile :       varying boundary conditions filename (.bdy)
        fpfric :        Manning's value for floodplain if spatially uniform
        manningfile :   filename for grid Manning values
        
        diffusive / adapt / acceleration / Roe:     model solvers
        
        (other optional commands...)
        '''
        os.chdir(p_file)
    
        # open and write .par file
        f = open('{0}.par'.format(self.proj.resroot), 'wb')
        f.write(('# Flooding {0}\n'.format(self.proj.resroot)).encode())
        
        # input data
        f.write(('{0}\t{1}\n'.format('DEMfile',self.proj.DEMfile)).encode())
        f.write(('{0}\t{1}\n'.format('resroot',self.proj.resroot)).encode())
        f.write(('{0}\t{1}\n'.format('dirroot',self.proj.dirroot)).encode())
        f.write(('{0}\t{1}\n'.format('sim_time',self.proj.sim_time)).encode())
        f.write(('{0}\t{1}\n'.format('initial_tstep',self.proj.initial_tstep)).encode())
        f.write(('{0}\t{1}\n'.format('saveint',self.proj.saveint)).encode())
        f.write(('{0}\t{1}\n'.format('massint',self.proj.massint)).encode())
        
        f.write(('{0}\t\t{1}\n'.format('fpfric',self.proj.fpfric)).encode())

        # boundary conditions
        if self.proj.bcifile:
            f.write(('{0}\t{1}\n'.format('bcifile',self.proj.bcifile)).encode())
        if self.proj.bdyfile:
            f.write(('{0}\t{1}\n'.format('bdyfile',self.proj.bdyfile)).encode())
        
        # model solver
        f.write(('{0}\n'.format(self.proj.solver)).encode())
    
        # water inputs/outputs
        if self.proj.rainfall:
            f.write(('{0}\t{1}\n'.format('rainfall',self.proj.rainfall)).encode())

        # optional additions
        if self.proj.theta:
            f.write(('{0}\t{1}\n'.format('theta',self.proj.theta)).encode())
            
        if self.proj.latlong:
            f.write(('{0}\n'.format(self.proj.latlong)).encode())
            f.write(('SGC_enable\n').encode())
        
        # additional output settings
        if self.proj.elevoff:
            f.write(('{0}\n'.format(self.proj.elevoff)).encode())
        
        if self.proj.depthoff:
            f.write(('{0}\n'.format(self.proj.depthoff)).encode())
        
        if self.proj.overpass:
            f.write(('{0}\t{1}\n'.format('overpass',self.proj.overpass)).encode())
        
        f.close()
        
    def make_dem(self, p_file):
        '''
        Generates topobathy_grid asci file .dem.asc (LISFLOOD-FP compatible)
        '''
        
        os.chdir(p_file)
    
        # open and write ascii file
        f = open(self.proj.DEMfile, 'wb')
        f.write(('ncols\t{0}\n'.format(self.proj.ncols)).encode())
        f.write(('nrows\t{0}\n'.format(self.proj.nrows)).encode())
        f.write(('xllcorner\t{0}\n'.format(self.proj.xll_corner)).encode())
        f.write(('yllcorner\t{0}\n'.format(self.proj.yll_corner)).encode())
        f.write(('cellsize\t{0}\n'.format(self.proj.cellsize)).encode())
        f.write(('NODATA_value\t{0}\n'.format(self.proj.nondata)).encode())
        np.savetxt(f, self.proj.depth, fmt='%.1f')
        f.close()
    

    #--------------------------------------------------------------------------
    #----------HEREAFTER: AUGUST 2020 WRAP, MUST UPDATE/COMPLETE---------------
    #--------------------------------------------------------------------------
    
    def output_case(self, p_case, output_run):
        'read output files and returns xarray.Dataset'
    
        if output_run == '.mass':
            
            p_file = op.join(p_case, 'results', '{0}{1}'.format(self.proj.resroot, 
                                                                output_run))
            # read ascii      
            with open(p_file) as fInput:
                header = fInput.readline().split()
                massdata = np.loadtxt(p_file, skiprows=1)
                
            xds_out = xr.Dataset(
                {
                    header[1]: (('time',), massdata[:,1], {'units':'s'}),       # Tstep
                    header[2]: (('time',), massdata[:,2], {'units':'s'}),       # MinTstep
                    header[3]: (('time',), massdata[:,3], {'units':'[]'}),      # NumTsteps
                    header[4]: (('time',), massdata[:,4], {'units':'m2'}),      # Area
                    header[5]: (('time',), massdata[:,5], {'units':'m3'}),      # Vol
                    header[6]: (('time',), massdata[:,6], {'units':'m3/s'}),    # Qin
                    header[7]: (('time',), massdata[:,7], {'units':'m'}),       # Hds
                    header[8]: (('time',), massdata[:,8], {'units':'m3/s'}),    # Qout
                    header[9]: (('time',), massdata[:,8], {'units':'m3/s'}),    # Qerror
                    header[10]: (('time',), massdata[:,8], {'units':'m3'}),     # Verror
                    header[11]: (('time',), massdata[:,8], {'units':'10^3m3'}), # Rain-(Inf+Evap)
                },
                coords = {'time': massdata[:,0]}
            )
            
            # add info
            xds_out.attrs['time'] = 'seconds'
        
        elif output_run == '2d':
            
            var_ls, units_ls, data_ls = [], [], []
            
            for i,v in enumerate(['.inittm', '.maxtm', '.totaltm', '.max', '.mxe']):
                
                p_file = op.join(p_case, 'results', '{0}{1}'.format(self.proj.resroot, 
                                                                    v))
                var = v.replace('.', '')
                if var in ['inittm', 'maxtm', 'totaltm']:    var_units = 'hour'
                if var in ['max', 'mxe']:                    var_units = 'm'

                var_ls.append(var)
                units_ls.append(var_units)
                data_ls.append(np.loadtxt(p_file, skiprows=6))
                
            xds_out = xr.Dataset(
                {
                    var_ls[0]: (('Y','X',), data_ls[0], {'units':units_ls[0]}), # .inittm
                    var_ls[1]: (('Y','X',), data_ls[1], {'units':units_ls[1]}), # .maxtm
                    var_ls[2]: (('Y','X',), data_ls[2], {'units':units_ls[2]}), # .totaltm
                    var_ls[3]: (('Y','X',), data_ls[3], {'units':units_ls[3]}), # .max
                    var_ls[4]: (('Y','X',), data_ls[4], {'units':units_ls[4]}), # .mxe
                },
            )
            
            # set X and Y values
            X = np.arange(self.proj.x_corner, 
                          self.proj.x_corner+self.proj.cellsize*self.proj.ncols,
                          self.proj.cellsize)
            Y = np.arange(self.proj.y_corner, 
                          self.proj.y_corner+self.proj.cellsize*self.proj.nrows,
                          self.proj.cellsize)
            xds_out = xds_out.assign_coords(X=X)
            xds_out = xds_out.assign_coords(Y=Y)
            
            # add info
            xds_out.attrs['nondata'] = self.proj.nondata
            xds_out.attrs['simtime(h)'] = self.proj.sim_time / 3600

        # add info about coordinates 
        if self.proj.cartesian:
            xds_out.attrs['xlabel'] = 'X_utm (m)'
            xds_out.attrs['ylabel'] = 'Y_utm (m)'

        return xds_out
        

    def output_bci(self, p_case):
        'read xarray.Dataset xds_X.nc input case file'
    
        xds_Q = xr.open_dataset(op.join(p_case, 'xds_Q.nc'))
        xds_H = xr.open_dataset(op.join(p_case, 'xds_H.nc'))
        
        # add info
        xds_Q['val'].attrs['units'] = 'Overtopping volume (m3/s·m)' 
        xds_H['val'].attrs['units'] = 'TWL (m)'
        
        return xds_Q, xds_H
    
        
        
