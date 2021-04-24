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
        'Initialize IO with a LisfloodProject'

        self.proj = lisflood_proj

        # inner parameters
        self.write_bci = False
        self.write_bdy = False
        self.write_rain = False

    def make_project(self):
        'Makes lisflood project folder'

        if not op.isdir(self.proj.p_main): os.makedirs(self.proj.p_main)

    def build_case(self, lisflood_case):
        '''
        Builds lisflood case input files for a LisfloodCase 

        lc     - LisfloodCase instance.
        '''

        # LISFLOOD-FP case path
        p_case = op.join(self.proj.p_cases, lisflood_case.id)
        if not op.isdir(p_case): os.makedirs(p_case)

        # make DEM *.dem.asc file
        p_file = op.join(p_case, self.proj.DEMfile)
        self.make_dem(p_file)

        # make boundary conditions *.bci file
        p_file = op.join(p_case, self.proj.bcifile)
        self.make_bci(p_file, lisflood_case)

        # make time-varying BCs *.bdy file
        p_file = op.join(p_case, self.proj.bdyfile)
        self.make_bdy(p_file, lisflood_case)

        # make rainfall *.rain file
        p_file = op.join(p_case, self.proj.rainfile)
        self.make_rain(p_file, lisflood_case)

        # TODO make sensors *.stage file

        # TODO make start *.start file

        # make input *.par file
        p_file = op.join(p_case, self.proj.PARfile)
        self.make_par(p_file, lisflood_case)

        print('Case {0} built.'.format(lisflood_case.id))

    def make_bci(self, p_file, lc):
        '''
        Generates .bci boundary file

        p_file - path to write .bci file
        lc     - LisfloodCase instance.
        '''

        # restricted to 'P' bc_ide 
        bc_ide = 'P'

        # check if have to write file or not
        if not any([lc.qfix_activate, lc.qvar_activate, lc.hfix_activate, lc.hvar_activate]):
            return

        self.write_bci = True

        # open and write .bci file
        with open(p_file, 'w') as f:

            # QFIX
            if lc.qfix_activate:
                for x, y in zip(lc.qfix_xcoord, lc.qfix_ycoord):
                    f.write('{0}\t{1:f}\t{2:f}\t{3}\t{4:f}\n'.format(
                        bc_ide, x, y,'QFIX', lc.qfix_value))
            # QVAR
            if lc.qvar_activate:
                for x, y, p in zip(lc.qvar_xcoord, lc.qvar_ycoord, lc.qvar_profile):
                    f.write('{0}\t{1:f}\t{2:f}\t{3}\tqp_{4}\n'.format(
                        bc_ide, x, y,'QVAR', int(p)))
            # HFIX
            if lc.hfix_activate:
                for x, y in zip(lc.hfix_xcoord, lc.hfix_ycoord):
                    f.write('{0}\t{1:f}\t{2:f}\t{3}\t{4:f}\n'.format(
                        bc_ide, x, y,'HFIX', lc.hfix_value))
            # HVAR
            if lc.hvar_activate:
                for x, y, p in zip(lc.hvar_xcoord, lc.hvar_ycoord, lc.hvar_profile):
                    f.write('{0}\t{1:f}\t{2:f}\t{3}\thp_{4:f}\n'.format(
                        bc_ide, x, y,'HVAR', int(p)))

    def make_bdy(self, p_file, lc):
        '''
        Generates .bdy boundary file

        p_file - path to write .bdy file
        lc     - LisfloodCase instance.
        '''

        # check if have to write file or not
        if not any([lc.qvar_activate, lc.hvar_activate]):
            return

        # update project
        self.write_bdy = True

        # open and write .bdy file
        with open(p_file, 'w') as f:

            # header
            f.write('{0} time varying profiles\n'.format(lc.id))

            # QVAR
            if lc.qvar_activate:
                for p in lc.qvar_value.profile:
                    v = lc.qvar_value.sel(profile=p)

                    # profile ID
                    f.write('qp_{0}\n'.format(int(p)))

                    # profile time (len, units)
                    f.write('{0}\t{1}\n'.format(int(v.time.size), lc.qvar_freq))

                    for t in v.time:
                        q = v.sel(time=t).values
                        if np.isnan(q): q = 0
                        f.write('{0:.9f}\t{1}\n'.format(q, int(t)))
                    f.write('\n')

            # HVAR
            if lc.hvar_activate:
                for p in lc.hvar_value.profile:
                    v = lc.hvar_value.sel(profile=p)

                    # profile ID
                    f.write('hp_{0}\n'.format(int(p)))

                    # profile time (len, units)
                    f.write('{0}\t{1}\n'.format(int(v.time.size), lc.hvar_freq))

                    for t in v.time:
                        h = v.sel(time=t).values
                        if np.isnan(h): h = 0
                        f.write('{0:.9f}\t{1}\n'.format(h, int(t)))
                    f.write('\n')

    def make_rain(self, p_file, lc):
        '''
        Generates time varying rainfall file (.rain)

        p_file - path to write .rain file
        lc     - LisfloodCase instance.
        '''

        # check if have to write file or not
        if not lc.rain_activate:
            return

        # update project
        self.write_rain = True

        # rain dataArray
        r = lc.rain_value

        # open and write .rain file
        with open(p_file, 'w') as f:

            # header
            f.write('{0} time varying rainfall\n'.format(lc.id))

            # rainfall time (len, units)
            f.write('{0}\t{1}\n'.format(int(r.time.size), lc.rain_freq))

            for t in r.time:
                v = r.sel(time=t).values
                if np.isnan(v): v = 0
                f.write('{0:.9f}\t{1}\n'.format(v, int(t)))
            f.write('\n')

    def make_par(self, p_file, lc):
        '''
        Generates configuration file (.par)

        p_file - path to write .bci file
        lc     - LisfloodCase instance.

        parameteres

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

        # open and write .par file
        with open(p_file, 'w') as f:
            f.write('# Flooding {0}\n'.format(lc.id))

            # input data
            f.write('{0}\t{1}\n'.format('DEMfile',self.proj.DEMfile))
            f.write('{0}\t{1}\n'.format('resroot',self.proj.resroot))
            f.write('{0}\t{1}\n'.format('dirroot',self.proj.dirroot))
            f.write('{0}\t{1}\n'.format('sim_time',self.proj.sim_time))
            f.write('{0}\t{1}\n'.format('initial_tstep',self.proj.initial_tstep))
            f.write('{0}\t{1}\n'.format('saveint',self.proj.saveint))
            f.write('{0}\t{1}\n'.format('massint',self.proj.massint))
            f.write('{0}\t\t{1}\n'.format('fpfric',self.proj.fpfric))

            # boundary conditions
            if self.write_bci:
                f.write('{0}\t{1}\n'.format('bcifile',self.proj.bcifile))
            if self.write_bdy:
                f.write('{0}\t{1}\n'.format('bdyfile',self.proj.bdyfile))

            # model solver
            f.write('{0}\n'.format(self.proj.solver))

            # rain
            if self.write_rain: 
                f.write('rainfall\t{0}\n'.format(self.proj.rainfile))

            # routing
            if self.proj.routing:
                f.write('routing\n')
                if self.proj.depththresh:
                    f.write('depththresh\t{0}\n'.format(self.proj.depththresh))
                if self.proj.routesfthresh:
                    f.write('routesfthresh\t{0}\n'.format(self.proj.routesfthresh))
                if self.proj.routingspeed:
                    f.write('routingspeed\t{0}\n'.format(self.proj.routingspeed))

            # TODO infiltration

            # optional additions
            if self.proj.theta:
                f.write('theta\t{0}\n'.format(self.proj.theta))

            if self.proj.latlong:
                f.write('{0}\n'.format(self.proj.latlong))
                f.write('SGC_enable\n')

            # additional output settings
            if self.proj.elevoff:
                f.write('elevoff\n')

            if self.proj.depthoff:
                f.write('depthoff\n')

            if self.proj.overpass:
                f.write('overpass\t{0}\n'.format(self.proj.overpass))

    def make_dem(self, p_file):
        '''
        Generates topobathy_grid asci file .dem.asc (LISFLOOD-FP compatible)
        '''

        # open and write ascii file
        with open(p_file, 'wb') as f:

            f.write(('ncols\t{0}\n'.format(self.proj.ncols)).encode())
            f.write(('nrows\t{0}\n'.format(self.proj.nrows)).encode())
            f.write(('xllcorner\t{0}\n'.format(self.proj.xll_corner)).encode())
            f.write(('yllcorner\t{0}\n'.format(self.proj.yll_corner)).encode())
            f.write(('cellsize\t{0}\n'.format(self.proj.cellsize)).encode())
            f.write(('NODATA_value\t{0}\n'.format(self.proj.nondata)).encode())

            np.savetxt(f, self.proj.depth, fmt='%.1f')

    def output_case_mass(self, p_case):
        'Read output .mass files from and returns xarray.Dataset'

        nf = '{0}.mass'.format(self.proj.resroot)
        p_file = op.join(p_case, self.proj.dirroot, nf)

        # read ascii      
        with open(p_file) as f:
            header = f.readline().split()
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
            coords = {'time': massdata[:,0]},
        )

        # add info
        xds_out.attrs['time'] = 'seconds'

        # add info about coordinates 
        if self.proj.cartesian:
            xds_out.attrs['xlabel'] = 'X_utm (m)'
            xds_out.attrs['ylabel'] = 'Y_utm (m)'

        return xds_out

    def output_case_2D(self, p_case):
        '''
        Read output from 2D files and returns xarray.Dataset

        2D files:
            - .inittm: time of initial inundation
            - .maxtm: time of maximum depth
            - .totaltm: total time of inundation
            - .max: maximum water depth
            - .mxe: maximum water surface elevation
        '''
        var_ls, units_ls, data_ls = [], [], []

        for i, v in enumerate(['.inittm', '.maxtm', '.totaltm', '.max', '.mxe']):
            nf = '{0}{1}'.format(self.proj.resroot, v)
            p_file = op.join(p_case, self.proj.dirroot, nf)

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
        X = np.arange(
            self.proj.xll_corner,
            self.proj.xll_corner + self.proj.cellsize*self.proj.ncols,
            self.proj.cellsize
        )
        Y = np.arange(
            self.proj.yll_corner,
            self.proj.yll_corner + self.proj.cellsize*self.proj.nrows,
            self.proj.cellsize
        )
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

