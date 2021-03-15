#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import os.path as op
import pandas as pd
import numpy as np
from math import sqrt
# from glob import glob

import rasterio as rio
import cmocean.cm as cmo
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.animation import FuncAnimation
from natsort import natsorted, ns
from moviepy.editor import *
import glob
import copy

# Plot 1 flood map  
def plot_raster(filepath, mask, bmap, max_wd, min_wd):
  with rio.open(filepath) as src:
      w = copy.copy(cmo.get_cmap('cool')) 
      w.set_under(color='none')
      w.set_over(color=w(0.9999999))
      
      wd = np.flipud(src.read(1))
      wdg = np.empty((mask.shape[0],mask.shape[1]))
      maskg = np.flipud(mask)
      
      for j in range(mask.shape[1]):
          for k in range(mask.shape[0]):
              if (maskg[k,j] == 1):
                  wdg[k][j] = wd[k][j]
              else:
                  wdg[k][j] = -1
      
      bmap.imshow(wdg,cmap=w,vmin=min_wd,vmax=max_wd,alpha=1)
      
# Create water depths animation
def update_wd(i):
    oce = copy.copy(cm.get_cmap('cool'))
    oce.set_under(color='lightgreen')
    im_normed = arr[i]
    
    wdg = np.empty((im_normed.shape[0],im_normed.shape[1]))
    #    mask = np.copy(depth)
    
    #    pos = np.where(depth >= 0)
    #    mask[pos[0], pos[1]] = 1
    
    for j in range(im_normed.shape[1]):
        for k in range(im_normed.shape[0]):
            if (mask[k,j] == 1):
                wdg[k][j] = im_normed [k][j]
    
    ax.imshow(wdg, cmap = oce, vmin=0.02, vmax=1)
    ax.set_title('Water depths Kwajalein - ' + str(dates[i]), fontsize=15,pad=15)
    ax.set_axis_off() 



def plot_output_2d(name, xds_out, var_name, p_export, sim_time, cmap=cmo.ice_r):
    '''
    Plots LISFLOOD execution output for every case

    name       - project name
    xds_out    - lisflood output (xarray.Dataset)
    var_name   - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export   - path for exporting figures
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    for case_ix in xds_out.case.values[:]:

        # select case
        xds_out_case = xds_out.sel(case=case_ix)

        # output case subfolder
        case_id = '{0:04d}'.format(case_ix)
        p_export_case = op.join(p_export, case_id)

        # plot variable times
        plot_var(
            name, xds_out_case, var_name, p_export_case, case_id, sim_time, cmap
            )

def plot_var(name, xds_out_case, var_name, p_export_case, case_id, sim_time, cmap):
    '''
    Plots LISFLOOD execution output for selected var and case

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export     - path for exporting figures
    '''

    if not op.isdir(p_export_case): os.makedirs(p_export_case)

    xds_oct = xds_out_case

    # get mesh data from output dataset
    X = xds_oct.X.values[:]
    Y = xds_oct.Y.values[:]

    # get variable and units
    var = np.flipud(xds_oct[var_name].values[:])
    var[var==0] = np.nan
    var[var==-9999] = np.nan
    var[var>=100] = np.nan
    var_units = xds_oct[var_name].attrs['units']
    
    if var_name in ['inittm', 'maxtm', 'totaltm']:
        var_min = 0
        var_max = sim_time
    if var_name in ['max', 'mxe']:
        var_max = np.nanmax(var)
        var_min = np.nanmin(var)

    # new figure
    fig, ax0 = plt.subplots(nrows=1, figsize=(12, 12))
    var_title = '{0} - {1} - case{2}'.format(var_name, name, case_id)  # title

    # pcolormesh
    im = plt.pcolormesh(X, Y, var, cmap=cmap, vmin= var_min, vmax= var_max)

    # customize pcolormesh
    plt.title(var_title, fontsize = 14, fontweight='bold')
    plt.xlabel(xds_oct.attrs['xlabel'], fontsize = 12)
    plt.ylabel(xds_oct.attrs['ylabel'], fontsize = 12)

    plt.axis('scaled')
    plt.xlim(X[0], X[-1])
    plt.ylim(Y[0], Y[-1])

    # add custom colorbar
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    plt.ylabel('{0} ({1})'.format(var_name, var_units), fontsize = 12)

    # export fig
    p_ex = op.join(p_export_case, 'outmap_{0}_{1}.png'.format(var_name, case_id))
    fig.savefig(p_ex)
    
    # close fig 
    plt.close()
 
def plot_output_volumes(name, xds_out, var_name, p_export):
    '''
    Plots LISFLOOD execution volume of water time-series

    name       - project name
    xds_out    - lisflood output (xarray.Dataset)
    var_name   - 'mass'
    p_export   - path for exporting figures
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    for case_ix in xds_out.case.values[:]:

        # select case
        xds_out_case = xds_out.sel(case=case_ix)

        # output case subfolder
        case_id = '{0:04d}'.format(case_ix)
        p_export_case = op.join(p_export, case_id)

        # plot variable times
        plot_volumes(
            name, xds_out_case, var_name, p_export_case, case_id,
            )

def plot_volumes(name, xds_out_case, var_name, p_export_case, case_id):
    '''
    Plots LISFLOOD execution output for selected var and case

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'mass'
    p_export     - path for exporting figures
    '''

    if not op.isdir(p_export_case): os.makedirs(p_export_case)

    xds_oct = xds_out_case

    # get variable and units
    var_qin = xds_oct['Qin'].values[:]
    var_qout = xds_oct['Qout'].values[:]
    var_units = xds_oct['Qin'].attrs['units']
    time = xds_oct['time'].values[:]
    var_vol = xds_oct['Vol'].values[:]

    # new figure
    fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(16, 5), sharex=True)

    # plot overtoppings
    ax0.plot(time, var_qin, '.-r', linewidth=1, markersize=1, label='Qin')
    ax0.plot(time, var_qout, '.-b', linewidth=1, markersize=1, label='Qout')

    # customize plot
    var_title = '{0} - {1} - case{2}'.format(var_name, name, case_id)  # title
    ax0.set_title(var_title, fontsize = 14, fontweight='bold')
    ax0.set_ylabel('Overtopping volumes ({0})'.format(var_units))
    ax0.legend()

    # plt.axis('scaled')
    ax0.set_xlim(time[0], time[-1])
    ax0.set_ylim(0, np.nanmax((var_qin, var_qout))+10)

    # plot overtoppings
    ax1.plot(time, var_vol, '.-k', linewidth=1, markersize=1, label='Vol')

    ax1.set_ylabel('Volume of water (m3)')
    ax1.set_xlabel('Time (s)')
    ax1.legend()

    # export fig
    p_ex = op.join(p_export_case, 'outmap_{0}_{1}.png'.format(var_name, case_id))
    fig.savefig(p_ex)
    
    # close fig 
    plt.close()

def plot_volumes_matrix(name, xds_out_ls, var_name, p_export, 
                        n_rows=False, n_cols=False, show=False):
    '''
    Plots LISFLOOD execution output for selected var, for every case

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'mass'
    p_export     - path for exporting figures
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    # bound limits
    var_max_all, var_min_all, n_clusters = [], [], []
    for i in range(len(xds_out_ls)):
        var_max_all.append(np.nanmax(xds_out_ls[i]['Qout'].values))
        var_min_all.append(np.nanmin(xds_out_ls[i]['Qout'].values))
        n_clusters.append(xds_out_ls[i].case.values.size)
    var_max_all = np.max(var_max_all) + 20
    var_min_all = np.min(var_min_all)

    # get number of rows and cols for gridplot 
    if not n_rows:
        n_clusters = np.sum(n_clusters)
        n_rows, n_cols = GetBestRowsCols(n_clusters)

    # plot figure
    fig = plt.figure(figsize=(20*1.5, 15*1.5))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0, hspace=0)
    gr, gc = 0, 0

    for j in range(len(xds_out_ls)):
        xds_out = xds_out_ls[j]
        
        for case_n, (case_ix) in enumerate(xds_out.case.values[:]):
            
            # select case
            xds_out_case = xds_out.sel(case=case_ix)
            var_units = xds_out_case[var_name].attrs['units']
    
            # plot variable times
            ax = plt.subplot(gs[gr, gc])
            ax = volumes_matrix(
                ax, case_ix, xds_out_case, var_name, 
                var_max=var_max_all, var_min=var_min_all, 
            )
    
            # counter
            gc += 1
            if gc >= n_cols:
                gc = 0
                gr += 1

    # show and return figure
    if show: plt.show()
    
    p_ex = op.join(p_export, 'volumes_{0}_matrix.png'.format(var_name))
    fig.savefig(p_ex)
    
    plt.close()

def volumes_matrix(ax, case_ix, xds_out_case, var_name, 
                   cmap='jet', var_max=None, var_min=None):
    '''
    Plots LISFLOOD execution output 

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export     - path for exporting figures
    cmap         - matplotlib colormap
    '''

    # ax1.set_ylabel('Volume of water (m3)')
    # ax1.set_xlabel('Time (s)')

    # get variable and units
    var_qin = xds_out_case['Qin'].values[:]
    var_qout = xds_out_case['Qout'].values[:]
    time = xds_out_case['time'].values[:]
    
    # plot 
    ax.plot(time, var_qin, '.-r', linewidth=1, markersize=1, label='Qin')
    ax.plot(time, var_qout, '.-b', linewidth=1, markersize=1, label='Qout')
    ax.legend()

    # number
    ax.text(500, var_max-100, case_ix, color='k', 
            fontweight='bold', fontsize=25)

    # plt.spines.set_color('k')
    ax.axhline(y=0, color='k')
    ax.axvline(x=time[0], color='k')
    ax.axhline(y=var_max, color='k')
    ax.axvline(x=time[-1], color='k')

    plt.xlim(time[0], time[-1])
    plt.ylim(0, var_max)
    plt.axis('off')
    

def GetDivisors(x):
    l_div = []
    i = 1
    while i<x:
        if x%i == 0:
            l_div.append(i)
        i = i + 1
    return l_div

def GetBestRowsCols(n):
    'try to square number n, used at gridspec plots'

    sqrt_n = sqrt(n)
    if sqrt_n.is_integer():
        n_r = int(sqrt_n)
        n_c = int(sqrt_n)
    else:
        l_div = GetDivisors(n)
        n_c = l_div[int(len(l_div)/2)]
        n_r = int(n/n_c)

    return n_r, n_c

def plot_input_profiles(name, xds_bc, var_name, p_export):
    '''
    Plots LISFLOOD execution output for every case

    name       - project name
    xds_bc     - lisflood input (xarray.Dataset)
    var_name   - 'val'
    p_export   - path for exporting figures
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    for case_ix in xds_bc.case.values[:]:

        # select case
        xds_bc_case = xds_bc.sel(case=case_ix)

        # output case subfolder
        case_id = '{0:04d}'.format(case_ix)
        p_export_case = op.join(p_export, case_id)

        # plot variable times
        plot_profiles_matrix(
            name, xds_bc_case, var_name, p_export_case, case_id
            )

def plot_profiles_matrix(name, xds_bc_case, var_name, p_export, case_id, 
                         n_rows=False, n_cols=False, show=False):
    '''
    Plots LISFLOOD execution input for selected var, for every case

    name         - project name
    xds_out_case - lisflood input (xarray.Dataset)
    var_name     - 'val'
    p_export     - path for exporting figures
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    profiles = np.unique(xds_bc_case.pts.values)
    var_max_all = xds_bc_case[var_name].values.max(axis=0).max()
    var_min_all = xds_bc_case[var_name].values.min(axis=0).min()
    n_clusters = profiles.size

    # get number of rows and cols for gridplot 
    if not n_rows:
        n_rows, n_cols = GetBestRowsCols(n_clusters)

    # plot figure
    fig = plt.figure(figsize=(20*1.5, 20*1.5))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0, hspace=0)
    gr, gc = 0, 0

    xds_out = xds_bc_case
    for pf_n, (pf_ix) in enumerate(profiles):
        
        # select case
        xds_out_pf = xds_out.sel(pts=pf_ix)

        # plot variable times
        ax = plt.subplot(gs[gr, gc])
        ax = input_matrix(
            ax, pf_ix, xds_out_pf, var_name, 
            var_max=var_max_all, var_min=var_min_all, 
        )
        
        if gc != 0:     ax.set_yticklabels([])

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    time = xds_out_pf.time.values[:]
    dates = pd.date_range(start=time[0], periods=time.size, freq='1H'
                          ).strftime('%Y/%m/%d-%H:%M')
    
    fig.suptitle('Max Overtopping: {0}m3/s/m,  Time: {1} - {2}'.format(
        var_max_all, dates[0], dates[-1]), fontsize=30, fontweight='bold')
    
    # show and return figure
    if show: plt.show()
    
    p_ex = op.join(p_export, 'input_profiles_{0}_matrix.png'.format(case_id))
    fig.savefig(p_ex)
    
    plt.close()

def input_matrix(ax, pf_ix, xds_out_pf, var_name, var_max=None, var_min=None):
    'Single matrix plot'

    # get variable and units
    var = xds_out_pf[var_name].values[0,:]
    time = xds_out_pf.time.values[:]
    
    # plot
    ax.plot(time, var, '.-r', linewidth=1, markersize=1)

    # plt.axis('scaled')
    ax.set_xlim(time[0], time[-1])
    ax.set_ylim(0, var_max)

    # number
    ax.text(time[0], var_max*9/10, pf_ix, color='k', fontweight='bold', fontsize=20)
    ax.set_xticklabels([])

    return ax

def plot_output_matrix(name, xds_out_ls, var_name, p_export, cmap='jet',
                        n_rows=False, n_cols=False, show=False):
    '''
    Plots LISFLOOD execution output for selected var, for every case

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export     - path for exporting figures
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    # bound limits
    var_max_all, var_min_all, n_clusters = [], [], []
    for i in range(len(xds_out_ls)):
        var_max_all.append(xds_out_ls[i][var_name].max(axis=0).max())
        var_min_all.append(xds_out_ls[i][var_name].min(axis=0).min())
        n_clusters.append(xds_out_ls[i].case.values.size)
    var_max_all = np.max(var_max_all)
    var_min_all = np.min(var_min_all)

    # get number of rows and cols for gridplot 
    if not n_rows:
        n_clusters = np.sum(n_clusters)
        n_rows, n_cols = GetBestRowsCols(n_clusters)

    # plot figure
    fig = plt.figure(figsize=(20*1.5, 15*1.5))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0, hspace=0)
    gr, gc = 0, 0

    for j in range(len(xds_out_ls)):
        xds_out = xds_out_ls[j]
        
        for case_n, (case_ix) in enumerate(xds_out.case.values[:]):
            
            # select case
            xds_out_case = xds_out.sel(case=case_ix)
            var_units = xds_out_case[var_name].attrs['units']
    
            # plot variable times
            ax = plt.subplot(gs[gr, gc])
            pc = output_matrix(
                ax, case_ix, xds_out_case, var_name, cmap,
                var_max=var_max_all, var_min=var_min_all, 
            )
    
            # get lower positions
            if gr==n_rows-1 and gc==0:
                pax_l = ax.get_position()
            elif gr==n_rows-1 and gc==n_cols-1:
                pax_r = ax.get_position()
    
            # counter
            gc += 1
            if gc >= n_cols:
                gc = 0
                gr += 1

    # colorbar
    cbar_ax = fig.add_axes([pax_l.x0, pax_l.y0-0.05, pax_r.x1 - pax_l.x0, 0.02])
    cb = fig.colorbar(pc, cax=cbar_ax, orientation='horizontal')
    cb.set_label(label='{0} ({1})'.format(var_name, var_units), size=20, weight='bold')
    cb.ax.tick_params(labelsize=15)

    # show and return figure
    if show: plt.show()
    
    p_ex = op.join(p_export, 'output_{0}_matrix.png'.format(var_name))
    fig.savefig(p_ex)
    
    plt.close()

def output_matrix(ax, case_ix, xds_out_case, var_name, cmap,
                  var_max=None, var_min=None):
    '''
    Plots LISFLOOD execution output 

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export     - path for exporting figures
    cmap         - matplotlib colormap
    '''

    # get mesh data from output dataset
    X = xds_out_case.X.values[:]
    Y = xds_out_case.Y.values[:]

    # get variable and units
    var = np.flipud(xds_out_case[var_name].values[:])
    
    # newcmap
    # if var_name=='W': 	 newcmap = custom_cmap(100, 'plasma_r', 0.05, 0.9, 'viridis', 0.2, 1)
    # if var_name=='Hsig': newcmap = custom_cmap(100, 'YlOrRd', 0.09, 0.9, 'YlGnBu_r', 0, 0.88)

    # pcolormesh
    pc = ax.pcolormesh(X, Y, var, cmap=cmap, vmin= var_min, vmax= var_max)
    
    # number
    ax.text(X[5], Y[5], case_ix, color='fuchsia', fontweight='bold', fontsize=20)

    # plt.spines.set_color('k')
    ax.axhline(y=Y[0], color='k')
    ax.axvline(x=X[0], color='k')
    ax.axhline(y=Y[-1], color='k')
    ax.axvline(x=X[-1], color='k')

    plt.axis('scaled')
    plt.xlim(X[0], X[-1])
    plt.ylim(Y[0], Y[-1])
    plt.axis('off')
    
    return pc

def perimeter_matrix(ax, case_ix, xds_out_case, var_name, 
                    cmap='jet', var_max=None, var_min=None):
    '''
    Plots LISFLOOD execution output 

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export     - path for exporting figures
    cmap         - matplotlib colormap
    '''

    # # maximum value of output variable over time --->  axis = (case, Y, X)
    # xds_out_max = xds_out_case.max()  # there is no time axis!!!

    # # get mesh data from output dataset
    # X = xds_out_case.X.values[:]
    # Y = xds_out_case.Y.values[:]

    # get variable and units
    var = np.flipud(xds_out_case[var_name].values[:])
    
    # pcolormesh
    pc = ax.pcolormesh(var, cmap=cmap, vmin= var_min, vmax= var_max)
    
    # number
    ax.text(5, 5, case_ix, color='fuchsia', fontweight='bold', fontsize=20)

    # plt.spines.set_color('k')
    ax.axhline(y=0, color='k')
    ax.axvline(x=0, color='k')
    ax.axhline(y=var.shape[1], color='k')
    ax.axvline(x=var.shape[0], color='k')

    plt.axis('scaled')
    # plt.xlim(X[0], X[-1])
    # plt.ylim(Y[0], Y[-1])
    plt.axis('off')
    
    return pc

def plot_bci_perimeter(p_cases, name, depth, xds_Q, xds_H, p_export, 
                       cmap=cmo.ice_r):
    '''
    Plots LISFLOOD execution output for every case

    name       - project name
    xds_out    - lisflood output (xarray.Dataset)
    var_name   - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export   - path for exporting figures
    '''
    
    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    # get folder cases
    ldir = sorted(os.listdir(p_cases))
    fp_ldir = [op.join(p_cases, c) for c in ldir]
    
    for case_ix, p_case in enumerate(fp_ldir):

        # select case files
        xds_q = xds_Q.sel(case=case_ix)
        xds_h = xds_H.sel(case=case_ix)

        # output case subfolder
        case_id = '{0:04d}'.format(case_ix)
        p_export_case = op.join(p_export, case_id)

        # plot variable times
        plot_perimeter(
            name, xds_q, xds_h, depth, p_export_case, case_id, cmap
            )
        
def plot_perimeter(name, xds_q, xds_h, depth, p_export_case, case_id, cmap):
    '''
    Plots LISFLOOD execution output for selected var and case

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export     - path for exporting figures
    '''
    
    if not op.isdir(p_export_case): os.makedirs(p_export_case)
    
    # set masks
    mask_q, mask_h = depth.copy(), depth.copy()
    
    mask_q[:,:] = np.nan
    mask_q[xds_q['x_pos'].values, xds_q['y_pos']] = xds_q['val'].max(axis=1)
    
    mask_h[:,:] = np.nan
    mask_h[xds_h['x_pos'].values, xds_h['y_pos']] = xds_h['val'].max(axis=1)
    
    # var info
    mask = [mask_q, mask_h]
    var_name = ['Overtopping', 'Sea level']
    var_units = ['m3/s/m', 'm']
    cmap = [cmo.thermal, cmo.balance]
    
    # new figure
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(20, 12))

    for i,ax in enumerate([ax0,ax1]):
        # pcolormesh
        im = ax.pcolormesh(np.flipud(mask[i]), cmap=cmap[i])#, vmin= var_min, vmax= var_max)
    
        # customize pcolormesh
        var_title = '{0} - {1} - case{2}'.format(var_name[i], name, case_id)  # title
        ax.set_title(var_title, fontsize = 14, fontweight='bold')
        ax.axis('off')
    
        ax.axis('scaled')
       
        # add custom colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im, cax=cax)
        plt.ylabel('{0} ({1})'.format(var_name[i], var_units[i]), fontsize = 12)
    
    # export fig
    p_ex = op.join(p_export_case, 'perimeter_{0}.png'.format(case_id))
    fig.savefig(p_ex)
    
    # close fig 
    plt.close()
 

def plot_bc_matrix(name, xds_q_ls, xds_h_ls, depth, var_name, p_export, 
                   n_rows=False, n_cols=False, show=False):
    '''
    Plots LISFLOOD execution output for selected var, for every case

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export     - path for exporting figures
    '''

    # make export dir
    if not op.isdir(p_export): os.makedirs(p_export)

    var_max_q, var_min_q, n_clusters = [], [], []
    for i in range(len(xds_q_ls)):
        var_max_q.append(xds_q_ls[i][var_name].max(axis=0).max())
        var_min_q.append(xds_q_ls[i][var_name].min(axis=0).min())
        n_clusters.append(xds_q_ls[i].case.values.size)
    var_max_q = np.max(var_max_q)
    var_min_q = np.min(var_min_q)
    var_max_h = 1.2
    var_min_h = -1.2

    # get number of rows and cols for gridplot 
    if not n_rows:
        n_clusters = np.sum(n_clusters)
        n_rows, n_cols = GetBestRowsCols(n_clusters)

    # plot figure
    fig = plt.figure(figsize=(20*1.5, 15*1.5))

    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0, hspace=0)
    gr, gc = 0, 0

    for j in range(len(xds_q_ls)):
        xds_q = xds_q_ls[j]
        xds_h = xds_h_ls[j]
        
        for case_n, (case_ix) in enumerate(xds_q.case.values[:]):
            
            # select case
            xds_q_case = xds_q.sel(case=case_ix)
            var_units_q = xds_q_case[var_name].attrs['units']
            xds_h_case = xds_h.sel(case=case_ix)
            var_units_h = xds_h_case[var_name].attrs['units']
    
            # plot variable times
            ax = plt.subplot(gs[gr, gc])
            
            pc_H = bc_matrix(
                ax, case_ix, xds_h_case, depth, var_name, 
                var_max=var_max_h, var_min=var_min_h, cmap='coolwarm'
            )
            pc_Q = bc_matrix(
                ax, case_ix, xds_q_case, depth, var_name, 
                var_max=var_max_q, var_min=var_min_q, cmap='viridis_r'
            )
            
            norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
            color = norm(xds_h_case['val'].max(axis=1).max())
            cmap = matplotlib.cm.get_cmap('coolwarm')
            rgba = cmap(color)
            ax.set_facecolor(rgba)
            
            # BC modes
            ax.text(xds_h_case['x'].min(), xds_h_case['y'].max() - 100, 
                    '{0} (P{1})'.format(str(xds_h_case.mode.values), 
                                        int(xds_h_case.perimeter.values)), 
                    color='k', fontsize=22)

            ax.text(xds_q_case['x'].min(), xds_q_case['y'].max() - 300, 
                    '{0} (P{1})'.format(str(xds_q_case.mode.values), 
                                        int(xds_q_case.perimeter.values)), 
                    color='k', fontsize=22)

            # get lower positions
            if gr==n_rows-1 and gc==0:
                pax_l = ax.get_position()
            elif gr==n_rows-1 and gc==n_cols-1:
                pax_r = ax.get_position()
    
            # counter
            gc += 1
            if gc >= n_cols:
                gc = 0
                gr += 1
    
    cbar_ax = fig.add_axes([pax_l.x0, pax_l.y0-0.05, pax_r.x1 - pax_l.x0, 0.02])
    cb = fig.colorbar(pc_Q, cax=cbar_ax, orientation='horizontal')
    cb.set_label(label='{0}'.format(var_units_q), size=20, weight='bold')
    cb.ax.tick_params(labelsize=15)

    cbar_ax = fig.add_axes([pax_l.x0, pax_l.y1*2.45, pax_r.x1 - pax_l.x0, 0.02])
    cb = fig.colorbar(pc_H, cax=cbar_ax, orientation='horizontal')
    cb.set_label(label='{0}'.format(var_units_h), size=20, weight='bold')
    cb.ax.tick_params(labelsize=15)

    # show and return figure
    if show: plt.show()
    
    p_ex = op.join(p_export, 'bci_{0}_matrix.png'.format(var_name))
    fig.savefig(p_ex)
    
    plt.close()

def bc_matrix(ax, case_ix, xds_case, depth, var_name, 
                    cmap='jet', var_max=None, var_min=None):
    '''
    Plots LISFLOOD execution output 

    name         - project name
    xds_out_case - lisflood output (xarray.Dataset)
    var_name     - 'max', 'mxe', 'inittm', 'maxtm', 'totaltm'
    p_export     - path for exporting figures
    cmap         - matplotlib colormap
    '''

    # scatter
    pc = ax.scatter(xds_case['x'], xds_case['y'], c=xds_case['val'].max(axis=1), 
                    cmap=cmap, vmin= var_min, vmax= var_max)

    # number
    ax.text(xds_case['x'].min(), xds_case['y'].min(), case_ix, 
            color='fuchsia', fontweight='bold', fontsize=25)
    
    ax.axis('scaled')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    
    return pc

def plot_zbeach(name, xds_case, p_export, cmap, show=False):
    
    fig, ax = plt.subplots(nrows=1, figsize=(8, 8))

    im = ax.scatter(xds_case['x'], xds_case['y'], c=xds_case['zbeach'], 
                    cmap=cmap, vmin= 0, vmax= 5)
    
    # add profiles number
    profiles = np.unique(xds_case.pts.values)
    for i in profiles:
        xds_pf = xds_case.sel(pts=i)
        x_pf = xds_pf['x'].mean()
        y_pf = xds_pf['y'].mean()
        ax.text(x_pf+30, y_pf, i+1, c='b')
    
    # properties
    ax.set_facecolor('gainsboro')
    ax.axis('scaled')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.set_yticks([])
    ax.set_xticks([])
    
    # title
    ax.set_title('PERIMETER P{0}: minZ={1}m, maxZ={2}m'.format(
        xds_case.perimeter,
        xds_case['zbeach'].values.min().round(2), 
        xds_case['zbeach'].values.max().round(2)), fontweight='bold')

    # add custom colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    plt.ylabel('Zbeach (m)', fontsize = 12)

    # show and return figure
    if show: plt.show()
    
    p_ex = op.join(p_export, 'zbeach_{0}.png'.format(name))
    fig.savefig(p_ex)
    
    plt.close()
