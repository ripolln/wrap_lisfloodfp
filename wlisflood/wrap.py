#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import os.path as op
import sys
import subprocess as sp

import numpy as np
import xarray as xr

from .io import LisfloodIO
from .plots import plot_dem_preprocessed, plot_dem_raw


class LisfloodDEM(object):
    '''
    Lisflood custom DEM.

    Handles perimeters and berm preprocessing methodology for lisflood coastal
    flooding cases
    '''

    def __init__(self):
        'Initialize listflood DEM preprocessor'

        # config
        self.nodataval = -9999

        # DEM properties
        self.depth = None
        self.nrows = None
        self.ncols = None
        self.xutm = None
        self.yutm = None
        self.xcoor = None
        self.ycoor = None
        self.cellsize = None

        # DEM properties for LisfloodProject
        self.xll = None
        self.yll = None
        self.xmesh = None
        self.ymesh = None

        # DEM mask
        self.mask = None

        # preprocessed DEM
        self.depth_preprocessed = None

        # list of outer and inner perimeters
        self.list_outer_perimeters = []
        self.list_inner_perimeters = []

        # BERM properties
        self.berm = None
        self.xb = None
        self.yb = None
        self.zb = None

        # transect id for each inside perimeter point
        self.id_trans = None

    def set_dem(self, dem):
        '''
        Sets DEM properties from DEM.nc file

        variables: elevation (y,x)
        coordinates: x, y
        '''

        # change nans to nodatavalue 
        dem = dem.fillna(self.nodataval)

        # get DEM properties
        depth = dem.elevation.values[:]

        nrows = depth.shape[0]
        ncols = depth.shape[1]

        xutm = dem.x.values[:]
        yutm = dem.y.values[:]

        xcoor, ycoor = np.meshgrid(xutm, yutm)

        cellsize = xcoor[0][1]-xcoor[0][0]

        # generate DEM mask
        pos = np.where(depth >= 0)
        mask = np.copy(depth)
        mask[pos[0], pos[1]] = 1

        # store DEM properties
        self.depth = depth
        self.nrows = nrows
        self.ncols = ncols
        self.xutm = xutm
        self.yutm = yutm
        self.xcoor = xcoor
        self.ycoor = ycoor
        self.cellsize = cellsize
        self.mask = mask

        # prepare DEM information for LisfloodProject
        xll = xutm[0]
        yll = yutm[-1]
        xx =  np.linspace(xll, xll+cellsize*(ncols-1), ncols)
        yy =  np.linspace(yll, yll+cellsize*(nrows-1), nrows)
        xxyy = np.meshgrid(xx,yy)
        xmesh = xxyy[0]                           # X Coordinates
        ymesh = np.flipud(xxyy[1])                # Y Coordinates

        self.xll = xll
        self.yll = yll
        self.xmesh = xmesh
        self.ymesh = ymesh

        # prepare copy of depth for preprocessing
        self.depth_preprocessed = np.copy(self.depth)

    def get_coastal_points(self):
        '''
        Finds coastal points
        '''

        # get dem data
        xcoor = self.xcoor
        ycoor = self.ycoor
        mask = self.mask

        # find coastal points
        xpos, ypos = self.perimeter_position(mask, outer = False)
        id_trans = np.zeros(len(xpos))

        # return xarray.Dataset
        return xr.Dataset(
            {
                'x': (['pts'], xcoor[xpos, ypos]),
                'y' :(['pts'], ycoor[xpos, ypos]),
                'profile' :(['pts'], id_trans),
            },
        )

    def preprocess_inner_perimeters(self):
        '''
        Preprocess DEM: inner perimeters

        - Find DEM inner perimeter (p1)
        - Get 1 previous inner perimeter (p2)
        '''

        # get dem data
        xcoor = self.xcoor
        ycoor = self.ycoor
        depth = self.depth
        mask = self.mask

        # inner perimeter (p1)
        xpos_p1, ypos_p1 = self.perimeter_position(mask, outer = False)
        self.append_inner_perimeter(xpos_p1, ypos_p1, xcoor, ycoor, depth)

        # previous inner perimeter (p2)
        xpos_p2, ypos_p2, depth_p2, mask_p2 = self.prev_inner_perimeter(
            self.depth, self.mask, xpos_p1, ypos_p1,
        )
        self.append_inner_perimeter(xpos_p2, ypos_p2, xcoor, ycoor, depth_p2)

    def preprocess_outer_perimeters(self):
        '''
        Preprocess DEM: outer perimeters

        - Find DEM outer perimeter (p1)
        - Get 3 next outer perimeters (p2, p3, p4)
        - update depth_preprocessed
        '''

        # get dem data
        xcoor = self.xcoor
        ycoor = self.ycoor
        depth = self.depth
        mask = self.mask

        # get depth to update
        depth_prep = self.depth_preprocessed

        # outter perimeter (p1)
        xpos_p1, ypos_p1 = self.perimeter_position(mask, outer = True)
        self.append_outer_perimeter(xpos_p1, ypos_p1, xcoor, ycoor, depth)

        # next outer perimeter (p2)
        xpos_p2, ypos_p2, depth_p2, mask_p2 = self.next_outer_perimeter(
            depth, mask, xpos_p1, ypos_p1,
        )
        self.append_outer_perimeter(xpos_p2, ypos_p2, xcoor, ycoor, depth_p2)

        # next outer perimeter (p3)
        xpos_p3, ypos_p3, depth_p3, mask_p3 = self.next_outer_perimeter(
            depth_p2, mask_p2, xpos_p2, ypos_p2,
        )
        self.append_outer_perimeter(xpos_p3, ypos_p3, xcoor, ycoor, depth_p3)

        # next outer perimeter (p4)
        xpos_p4, ypos_p4, depth_p4, mask_p4 = self.next_outer_perimeter(
            depth_p3, mask_p3, xpos_p3, ypos_p3,
        )
        self.append_outer_perimeter(xpos_p4, ypos_p4, xcoor, ycoor, depth_p4)

        # update depth_preprocessed
        for dp in self.list_outer_perimeters:
            for xp, yp in zip(dp['xpos'], dp['ypos']):
                depth_prep[xp][yp] = -1

        self.depth_preprocessed = depth_prep

    def perimeter_position(self, mask, outer):
        '''
        Find DEM perimeter position

        mask                 - depth file mask
        outer (True / False) - obtains outer / inner perimeter points

        returns position of the perimeter: xpos, ypos
        '''

        # inner / outer detector switch
        if outer:
            value = -9999
        else:
            value = 1

        sum_min = -9999 * 7 + 1
        sum_max = -9999 + 7

        # loop over depth array to find perimeter coordinates
        nrows, ncols = mask.shape
        xpos, ypos = [], []

        for i in np.arange(1, nrows - 1):
            for j in np.arange(1, ncols - 1):
                cell = mask[i,j]
                if cell == value:

                    # cell neighbours
                    cellul = mask[i - 1, j - 1]
                    cellu  = mask[i - 1, j    ]
                    cellur = mask[i - 1, j + 1]
                    celll  = mask[i    , j - 1]
                    cellr  = mask[i    , j + 1]
                    celldl = mask[i + 1, j - 1]
                    celld  = mask[i + 1, j    ]
                    celldr = mask[i + 1, j + 1]

                    # sum neighbours and check if is perimeter
                    neighbour_sum = np.sum(
                        [cellul,cellu,cellur,celll,cellr,celldl,celld,celldr]
                    )
                    if (neighbour_sum >= sum_min) & (neighbour_sum <= sum_max):
                        xpos.append(i)
                        ypos.append(j)

        return xpos, ypos

    def next_outer_perimeter(self, depth_prev, mask_prev, xpos_prev, ypos_prev):
        'Get next outer perimeter'

        depth_next = np.copy(depth_prev)
        depth_next[xpos_prev, ypos_prev] = 1
        pos = np.where(depth_next >= 0)

        mask_next = np.copy(mask_prev)
        mask_next[pos[0], pos[1]] = 1

        xpos_next, ypos_next = self.perimeter_position(mask_next, outer = True)

        return xpos_next, ypos_next, depth_next, mask_next

    def prev_inner_perimeter(self, depth_next, mask_next, xpos_next, ypos_next):
        'Get previous inner perimeter'

        depth_prev = np.copy(depth_next)
        for xp, yp in zip(xpos_next, ypos_next): depth_prev[xp,yp] = -9999
        pos = np.where(depth_prev >= 0)

        mask_prev = np.copy(depth_prev)
        mask_prev[pos[0], pos[1]] = 1

        xpos_prev, ypos_prev = self.perimeter_position(mask_prev, outer = False)

        return xpos_prev, ypos_prev, depth_prev, mask_prev

    def append_outer_perimeter(self, xpos, ypos, xcoor, ycoor, depth):
        'Append perimeter parameters to outer perimeters list'

        self.list_outer_perimeters.append(
            {
                'xpos': xpos,
                'ypos': ypos,
                'xp': xcoor[xpos, ypos],
                'yp': ycoor[xpos, ypos],
                'zp': depth[xpos, ypos],
            }
        )

    def append_inner_perimeter(self, xpos, ypos, xcoor, ycoor, depth):
        'Append perimeter parameters to inner perimeters list'

        self.list_inner_perimeters.append(
            {
                'xpos': xpos,
                'ypos': ypos,
                'xp': xcoor[xpos, ypos],
                'yp': ycoor[xpos, ypos],
                'zp': depth[xpos, ypos],
            }
        )

    def cut_perimeter(self, limit, axis='x', mode='under'):
        '''
        Cut last inner perimeter.

        limit - coordinate limit to cut
        axis  - 'x' or 'y', cutting axis selection
        mode  - 'under' or 'above', refered to limit
        '''

        # get last inner perimeter
        ip = self.list_inner_perimeters[-1]

        # switch axis input
        if axis == 'x':
            cpos = ip['xpos']
        elif axis == 'y':
            cpos = ip['ypos']

        # switch mode input
        if mode == 'under':
            cut = np.where(np.array(cpos) < limit)[0]
        elif mode == 'above':
            cut = np.where(np.array(cpos) > limit)[0]

        # cut perimeter points
        xc = ip['xp'][cut]
        yc = ip['yp'][cut]
        zc = ip['zp'][cut]

        # cut perimeter positions
        xposc = np.array(ip['xpos'])[cut]
        yposc = np.array(ip['ypos'])[cut]

        #  update last inner perimeter
        self.list_inner_perimeters[-1] = {
            'xpos': xposc,
            'ypos': yposc,
            'xp': xc,
            'yp': yc,
            'zp': zc,
        }

    def berm_homogenization(self, xb, yb, zb):
        '''
        Establish each transect on the corresponding perimeter to each berm
        coordinate

        xb, yb, zb - berm profiles X, Y, and Elevation

        - assign group id
        - update depth_preprocessed
        '''

        # get last inner perimeter
        ip = self.list_inner_perimeters[-1]
        xps = ip['xp']
        yps = ip['yp']
        xpos = ip['xpos']
        ypos = ip['ypos']

        # get preprocessed depth
        depth_prep = self.depth_preprocessed

        # assign closest profile transect to each point of the inner perimeter 
        id_trans = []

        # loop perimeter points
        for xp, yp in zip(xps, yps):

            dist = []
            # loop profiles points
            for xpb, ypb in zip(xb, yb):

                dist.append(np.linalg.norm(
                    np.array((xpb, ypb))-np.array((xp, yp)), 2)
                )

            # assign closest transect id
            id_trans.append(np.where(dist == np.min(dist))[0][0])

        # update id_trans and berm x,y
        self.id_trans = id_trans
        self.xb = xb
        self.yb = yb

        # update depth_preprocessed
        for xp, yp, it in zip(xpos, ypos, id_trans):
            depth_prep[xp, yp] = zb[it]

        self.depth_preprocessed = depth_prep

    def save_dem_preprocessed(self, p_save):
        'Stores preprocessed DEM to ascii file'

        with open(p_save, 'w') as fw:

            # write header
            fw.write('ncols\t\t{0}\n'.format(self.ncols))
            fw.write('nrows\t\t{0}\n'.format(self.nrows))
            fw.write('xllcorner\t{0:.6f}\n'.format(self.xutm[0]))
            fw.write('yllcorner\t{0:.6f}\n'.format(self.yutm[-1]))
            fw.write('cellsize\t{0:.6f}\n'.format(float(self.cellsize)))
            fw.write('NODATA_value\t{0}\n'.format(self.nodataval))

            # write depth
            np.savetxt(fw, self.depth_preprocessed, fmt='%1.3f')

    def get_inflow_coordinates(self):
        'Return inflow coordinates and profiles'

        # get last inner perimeter
        ip = self.list_inner_perimeters[-1]

        # get id transect
        id_trans = self.id_trans

        # return xarray.Dataset
        return xr.Dataset(
            {
                'x': (['pts'], ip['xp']),
                'y' :(['pts'], ip['yp']),
                'profile' :(['pts'], id_trans),
            },
        )

    def get_outflow_coordinates(self):
        'Return outflow coordinates'

        # get second and third outer perimeter
        ip2 = self.list_outer_perimeters[1]
        ip3 = self.list_outer_perimeters[2]

        xh_pts = np.append(ip2['xp'], ip3['xp'])
        yh_pts = np.append(ip2['yp'], ip3['yp'])

        # return xarray.Dataset
        return xr.Dataset(
            {
                'x': (['pts'], xh_pts),
                'y' :(['pts'], yh_pts),
            },
        )

    def plot_preprocessed(self):
        'Plot preprocessed DEM, perimeters, transects'

        # get x,y coordinates and preprocessed depth
        xcoor = self.xcoor
        ycoor = self.ycoor
        depth = self.depth_preprocessed

        # get outer perimeters x,y points
        x_p1 = self.list_outer_perimeters[0]['xp']
        y_p1 = self.list_outer_perimeters[0]['yp']
        x_p2 = self.list_outer_perimeters[1]['xp']
        y_p2 = self.list_outer_perimeters[1]['yp']
        x_p3 = self.list_outer_perimeters[2]['xp']
        y_p3 = self.list_outer_perimeters[2]['yp']
        x_p4 = self.list_outer_perimeters[3]['xp']
        y_p4 = self.list_outer_perimeters[3]['yp']

        # get most inner perimeter x,y points and id_trans
        xq = self.list_inner_perimeters[-1]['xp']
        yq = self.list_inner_perimeters[-1]['yp']
        id_trans = np.array(self.id_trans)
        xb = self.xb
        yb = self.yb

        fig = plot_dem_preprocessed(
            xcoor, ycoor, depth,
            x_p1, y_p1, x_p2, y_p2, x_p3, y_p3, x_p4, y_p4,
            xq, yq, id_trans, xb, yb,
        )

        return fig

    def plot_raw(self, vmin=-2, vmax=10):
        'Plot raw DEM'

        # get x,y coordinates and preprocessed depth
        xcoor = self.xcoor
        ycoor = self.ycoor
        depth = self.depth_preprocessed

        fig = plot_dem_raw(xcoor, ycoor, depth, vmin, vmax)

        return fig


class LisfloodCase(object):
    'LISFLOOD-FP case input definition'

    def __init__(self):
        '''
        This class stores LISFLOOD-FP case input information
        '''

        # case ID
        self.id = None

        # constant inflow
        self.qfix_activate  = False
        self.qfix_xcoord    = None   # x coordinates (int list / 1D np array)
        self.qfix_ycoord    = None   # y coordinates (int list / 1D np array)
        self.qfix_value     = None   # inflow value (float)

        # time varying inflow
        self.qvar_activate  = False
        self.qvar_profile   = None   # inflow profiles (int list / 1D np array)
        self.qvar_xcoord    = None   # x coordinates (int list / 1D np array)
        self.qvar_ycoord    = None   # y coordinates (int list / 1D np array)
        self.qvar_freq      = None   # inflow time frequency (HOURS, ...)
        self.qvar_value     = None   # inflow values (xarray.DataArray. coords: profile, time)
                                     # inflow units: m3/s*m (UTM coordinates)

        # constant water level
        self.hfix_activate  = False
        self.hfix_value     = None   # water level value (float)
        self.hfix_xcoord    = None   # x coordinates (int list / 1D np array)
        self.hfix_ycoord    = None   # y coordinates (int list / 1D np array)

        # time varying water level
        self.hvar_activate  = False
        self.hvar_profile   = None   # level profiles (int list / 1D np array)
        self.hvar_xcoord    = None   # x coordinates (int list / 1D np array)
        self.hvar_ycoord    = None   # y coordinates (int list / 1D np array)
        self.hvar_freq      = None   # level time frequency (HOURS, ...)
        self.hvar_value     = None   # level values (xarray.DataArray. coords: profile, time)
                                     # level units: m 

        # rain
        self.rain_activate  = False
        self.rain_freq      = None   # rain time frequency (HOURS, ...)
        self.rain_value     = None   # rain values (xarray.DataArray. coords: time)
                                     # rain units: mm/h 


class LisfloodProject(object):
    '''
    LISFLOOD-FP flood model project parameters.
    '''

    def __init__(self, p_proj, n_proj):
        '''
        This class stores LISFLOOD-FP project information
        '''

        self.p_main = op.join(p_proj, n_proj)    # project path
        self.name = n_proj                       # project name

        # sub folders
        self.p_cases = op.join(self.p_main, 'cases')    # cases folder
        self.p_export = op.join(self.p_main, 'export')  # export figures folder

        # topobathymetry depth value (2D numpy.array)
        self.cartesian = None       # units convention
        self.depth = None           # topo-bathy data
        self.xmesh = None
        self.ymesh = None

        # .dem file parameters
        self.nrows = None           # number of rows
        self.ncols = None           # number of columns 
        self.xll_corner = None      # x low left corner
        self.yll_corner = None      # y low left corner
        self.cellsize = None        # cell size
        self.nondata = None         # value for non-data

        # .par file parameters
        self.PARfile = 'params.par' # parameters file
        self.DEMfile = 'DEM.asc'    # digital elevation model filename
        self.resroot = 'out'        # root for naming the output files
        self.dirroot = 'outputs'    # name for results files folder
        self.saveint = None         # interval for saved files (in seconds)
        self.massint = None         # interval for .mass file (in seconds)
        self.sim_time = None        # simulation length (in seconds)
        self.initial_tstep = None   # model time step (in seconds)
        self.fpfric = None          # Manning's value (floodplain) if spatially uniform
        self.manningfile = None     # filename for grid Manning values

        # .bci file parameters
        self.bcifile = 'boundary.bci'  # boundary condition filename (.bci)

        # .bdy file parameters
        self.bdyfile = 'boundary.bdy'  # varying boundary conditions filename (.bdy)

        # .rain file parameteres
        self.rainfile = 'rainfall.rain'  # rainfall conditions filename (.rain)

        # .start file
        self.start = False           # activation of .start
        self.startfile = 'st.start'  # start conditions filename (.start)

        self.solver = None          # model solver

        # water parameters
        self.routing = False        # RECOMMENDED FOR RAINFALL WHEN STEEP SLOPES
        self.depththresh = None     # depth at which a cell is considered wet (m)
        self.routesfthresh = None   # Water surface slope above which routing occurs
        self.routingspeed = None
        self.infiltration = None    # (m/s) uniform

        # optional additions
        self.theta = None           # reset numerical diffusion for 'acceleration'
        self.latlong = None         # only for Subgrid and 2D

        # additional output settings
        self.overpass = None        # time for flood image available (s) ??????
        self.depthoff = False       # suppress depth files (*.wd) 
        self.elevoff = False        # suppress water surface files (*.elev)
        self.mint_hk = None         # allows calculation of maxH... at the mass interval instead of every tstep


class LisfloodWrap(object):
    'LISFLOOD-FP flood model wrap for multi-case handling'

    def __init__(self, lisflood_proj):
        '''
        lisflood_proj - LisfloodProject() instance, contains project parameters
        '''

        self.proj = lisflood_proj           # lisflood project parameters
        self.io = LisfloodIO(self.proj)     # lisflood input/output 

        # lisflood executable
        self.p_bin = op.join(op.dirname(op.realpath(__file__)), 'resources', 'bin')

    def build_cases(self, list_cases):
        '''
        Generates all files needed for a list of Lisflood cases

        list_cases     - list of LisfloodCase instances.
        '''

        # make main project directory
        self.io.make_project()

        # build each case
        for lc in list_cases:
            self.io.build_case(lc)

    def get_run_folders(self):
        'Return sorted list of project cases folders'

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
            print('\nCase {0} solved.\n'.format(op.basename(p_run)))

    def run(self, p_run):
        'Execution commands for launching LISFLOOD-FP'

        # TODO: linux computers can have a problem finding libiomp4.so library 
        # use next lines to fix it
        #os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        #os.environ['LD_LIBRARY_PATH'] = r'/home/anaconda3/lib/' <-- library path

        # aux. func. for launching bash command
        def bash_cmd(str_cmd, out_file=None, err_file=None):
            'Launch bash command using subprocess library'

            _stdout = None
            _stderr = None

            if out_file:
                _stdout = open(out_file, 'w')
            if err_file:
                _stderr = open(err_file, 'w')

            s = sp.Popen(str_cmd, shell=True, stdout=_stdout, stderr=_stderr)
            s.wait()

            if out_file:
                _stdout.flush()
                _stdout.close()
            if err_file:
                _stderr.flush()
                _stderr.close()

        # parameters file
        par_file = op.join(p_run, self.proj.PARfile) 

        # lisflood args
        args = '-v -log'

        # get OS lisflood binary
        if sys.platform.startswith('win'):
            # Windows
            # TODO obtain  windows binary
            print('LISFLOOD-FP binary not available for current OS')
            return

        elif sys.platform.startswith('linux'):
            fbin = op.abspath(op.join(self.p_bin, 'lisflood_euflood_lnx'))

        elif sys.platform == 'darwin':
            fbin = op.abspath(op.join(self.p_bin, 'lisflood_euflood_osx'))

        else:
            print('LISFLOOD-FP binary not available for current OS')
            return

        print('\n Running LISFLOOD-FP case...\n')
        cmd = 'cd {0} && {1} {2} {3}'.format(p_run, fbin, args, par_file)
        bash_cmd(cmd)

    def extract_output(self, output_run='2d'):
        '''
        extract output from all cases generated by "build_cases"

        - output_run = '.mass' / '2d'

        return xarray.Dataset (uses new dim "case" to join output)
        '''

        # choose output extraction function
        if output_run == '.mass':
            fo = self.io.output_case_mass
        elif output_run == '2d':
            fo = self.io.output_case_2D
        else:
            print('not output extraction for {0} file'.format(output_run))
            return None

        # get sorted execution folders
        run_dirs = self.get_run_folders()

        # exctract output case by case and concat in list
        l_out = []
        for p_run in run_dirs:

            # read output file
            xds_case_out = fo(p_run)
            l_out.append(xds_case_out)

        # concatenate xarray datasets (new dim: case)
        xds_out = xr.concat(l_out, dim='case')

        return(xds_out)

    def extract_output_wd(self, case_ix = 0):
        '''
        extract output water level from wd files for a choosen case

        case_ix - case index

        return xarray.Dataset
        '''

        # get selected case folder
        run_dirs = self.get_run_folders()
        p_case = run_dirs[case_ix]

        # read wd files to xarray.Dataset
        wds = self.io.output_case_wd(p_case)

        return wds

