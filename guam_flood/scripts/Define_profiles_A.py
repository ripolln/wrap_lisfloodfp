#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  7 10:35:56 2020

@author: administrador
"""

import xarray as xr
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import cmocean
import cir
import os
from matplotlib import cm
from matplotlib.colors import ListedColormap
from scipy import interpolate
from mpl_toolkits.mplot3d import Axes3D


#%%

path = r'/media/administrador/HD/GUAM/Bathymetry/'
bati=xr.open_dataset(os.path.join(path,'Bati_flooding.nc'))

def colormap(elev, topat, topag): #Elevation, maximum topo, minimum bati
    colors2 = 'YlGnBu_r'
    colors1 = cmocean.cm.turbid
    bottom = plt.get_cmap(colors2, -topag)
    top = plt.get_cmap(colors1, topat)
    newcolors = np.vstack((bottom(np.linspace(0, 0.8, -topag)),top(np.linspace(0.1, 1, topat))))
    return(ListedColormap(newcolors))

def calc_distance(x,y):
    lon1=np.deg2rad(x[0]); lon2=np.deg2rad(x[1])
    lat1=np.deg2rad(y[0]); lat2=np.deg2rad(y[1])
    R = 6371 
    x = (lon2 - lon1) * np.cos( 0.5*(lat2+lat1) )
    y = lat2 - lat1
    delta_d = R * np.sqrt( x*x + y*y )*1000
    return delta_d 

#%% Cut area of interest

lon_f=[144.66,144.688]
lat_f=[13.4573,13.472]

s=np.where((bati.lat>=lat_f[0]) & (bati.lat<=lat_f[1]))[0]
s1=np.where((bati.lon>=lon_f[0]) & (bati.lon<lon_f[1]))[0]

bati_f=bati.isel(lat=s)
bati_f=bati_f.isel(lon=s1)


#%% PROFILES

#Variables needed:
number_points=500 #Number of points to define between start and end position


#Interpolate surface

number_profiles=int(input("Please enter the number of profiles you want to define:\n"))
surface= interpolate.interp2d(bati_f.lon.values, bati_f.lat.values, bati_f.Band1.values, kind='linear')

fig=plt.figure(figsize=[17,14])
gs1=gridspec.GridSpec(1,1)
ax=fig.add_subplot(gs1[0])
c1=colormap(bati_f.Band1, 10,-40)
im = ax.pcolormesh(bati_f.lon.values, bati_f.lat.values, bati_f.Band1.values, cmap = c1, vmax = 10, vmin = -40)

s=np.where(bati_f.Band1.values<=0)
ax.plot(bati_f.lon.values[s[1]], bati_f.lat.values[s[0]], '.',color='white',alpha=0.5,markersize=1.5)

ax.set_aspect('equal')
ax.set_xticks(np.arange(lon_f[0], lon_f[1], step=0.01))
ax.ticklabel_format(useOffset=False)
ax.set_xlabel('Lon (º)',fontsize=15)
ax.set_ylabel('Lat (º)',fontsize=15)

gs1.tight_layout(fig, rect=[0.02, 0.4, 0.90,  0.95])
    
gs2=gridspec.GridSpec(1,1)
ax1=fig.add_subplot(gs2[0])
plt.colorbar(im,cax=ax1,extend='both')
ax1.set_ylabel('Elevation (m)')
gs2.tight_layout(fig, rect=[0.82, 0.45, 0.89, 0.85])

#%

gs3=gridspec.GridSpec(1,1)
ax3=fig.add_subplot(gs3[0])
gs3.tight_layout(fig, rect=[0.1, 0.02, 0.88, 0.37])
colors=['royalblue','crimson','gold','darkmagenta','darkgreen','darkorange','mediumpurple','coral','pink','lightgreen','darkgreen','darkorange']


# points=plt.ginput(n=number_profiles*2)

X=np.full([number_points,number_profiles],np.nan)
Y=np.full([number_points,number_profiles],np.nan)
Z=np.full([number_points,number_profiles],np.nan)
DIST_PROF=np.full([number_points,number_profiles],np.nan)
DIST_REP=np.full([number_points,number_profiles],np.nan)

for a in range(number_profiles):
        
    points=[]
    while len(points) < 2:
        print('Define profile with two points')
        points = np.asarray(plt.ginput(2, timeout=20))
        if len(points) < 2:
            print('Too few points, starting over')
    
    # points=plt.ginput(n=2,timeout=5)
    
    ax.plot([points[0][0],points[1][0]],[points[0][1],points[1][1]],'-',color=colors[a])
    ax.plot([points[0][0]],[points[0][1]],'s',color=colors[a],markersize=4)
    
    f = interpolate.interp1d([points[0][0],points[1][0]], [points[0][1],points[1][1]])
    dif=abs(points[0][0]-points[1][0])
    if points[0][0] > points[1][0]:
        xnew = np.linspace(points[1][0],points[0][0], number_points)
    else:
        xnew = np.linspace(points[0][0],points[1][0], number_points)
    xnew = np.linspace(points[0][0],points[1][0], number_points)
    ynew = f(xnew)
    # xx,yy=np.meshgrid(xnew, ynew)
    # ax.plot(xnew,ynew,'k.')
    ind=np.argsort(ynew)
    ynew=ynew[ind]
    xnew=xnew[ind]    
    pos=3
    lon1=np.deg2rad(xnew[pos]); lon2=np.deg2rad(xnew[pos+1])
    lat1=np.deg2rad(ynew[pos]); lat2=np.deg2rad(ynew[pos+1])
    R = 6371 
    x = (lon2 - lon1) * np.cos( 0.5*(lat2+lat1) )
    y = lat2 - lat1
    delta_d = R * np.sqrt( x*x + y*y )*1000
 
        
    xx,yy=np.meshgrid(xnew, ynew)
    pf=surface(xnew,ynew)
    if xnew[-1]>xnew[0]:
        im = ax.pcolormesh(xx, yy, pf, cmap = c1, vmax = 10, vmin = -40,zorder=1)
        Z[:,a]=np.diagonal(pf)
    else:
        im = ax.pcolormesh(xx,yy, np.fliplr(pf), cmap = c1, vmax = 10, vmin = -40,zorder=1)
        Z[:,a]=np.diagonal(np.fliplr(pf))
       
    X[:,a]=xnew
    Y[:,a]=ynew  
    ax.scatter(X[:,a],Y[:,a],8,Z[:,a],cmap=c1, vmax = 10, vmin = -40,zorder=2)
    
    dist_rep=np.arange(len(xnew))*delta_d
    dist_save=dist_rep[ind]
    DIST_PROF[:,a]=dist_save    
    ax3.plot(dist_rep,Z[:,a],color=colors[a],label='Profile: ' +str(a))

# Plot intersection with 0
for a in range(number_profiles):
    d=Z[:,a]
    s=np.where(d<0)[0][0]
    ax.plot(X[:,a],Y[:,a],'.',color=colors[a],markersize=3,zorder=3)
    ax.plot(X[s,a],Y[s,a],'s',color=colors[a],markersize=10,zorder=3)
    ax3.plot(DIST_PROF[s,a],Z[s,a],'x',color=colors[a],zorder=3)
    
plt.show()

ax3.plot([0,1500],[0,0],':',color='plum',alpha=0.5)
ax3.set_xlabel(r'(m)')
ax3.set_ylabel(r'Elevation (m)')
ax3.legend()
ax3.set_xlim([0,np.nanmax(DIST_PROF)])

# Define distance for each profile
a=0
value = input("Please enter yes if you want to obtain the distance associated to each profile:\n")

if value=='yes':
    distance=np.full([number_profiles],0)
    
    for a in range(number_profiles):
        points=[]
        while len(points) < 8:
            print('Define coast associated to each profile (8 points)')
            points = np.asarray(plt.ginput(8, timeout=20))
            if len(points) < 8:
                print('Too few points, starting over')
        
        ax.plot(points[:,0],points[:,1],'-',color=colors[a],linewidth=3)
        
        for m in range(len(points)-1):
            distance[a]=distance[a]+calc_distance([points[m,0],points[m+1,0]],[points[m,1],points[m+1,1]])
    

if value=='yes':
    profiles = xr.Dataset({'Lon': (['number_points','profile'],X),
                          'Lat': (['number_points','profile'],Y),
                          'Elevation': (['number_points','profile'],Z),
                          'Distance_profile': (['number_points','profile'],DIST_PROF),
                          'Rep_coast_distance': (['profile'],distance),
                         
                          }, 
                        coords={'number_points': range(number_points), 
                                'profile': range(number_profiles)})
else:
    profiles = xr.Dataset({'Lon': (['number_points','profile'],X),
                          'Lat': (['number_points','profile'],Y),
                          'Elevation': (['number_points','profile'],Z),
                          'Distance_profile': (['number_points','profile'],DIST_PROF),
                                                   
                          }, 
                        coords={'number_points': range(number_points), 
                                'profile': range(number_profiles)})
    

# plt.savefig(os.path.join(path,'profiles.png'), dpi=600)
# profiles.to_netcdf(path=os.path.join(path, 'Profiles_Guam.nc'))  


fig=plt.figure(figsize=[17,14])
ax = fig.add_subplot(111, projection='3d') 
xx,yy=np.meshgrid(bati_f.lon.values, bati_f.lat.values)
ax.plot_surface(xx,yy, bati_f.Band1.values,cmap = c1, vmax = 10, vmin = -40)
for a in range(number_profiles):
    d=Z[:,a]
    s=np.where(d<0)[0]
    ax.plot(X[s,a],Y[s,a],Z[s,a],'.',color=colors[a],markersize=7)
    ax.plot(X[:,a],Y[:,a],Z[:,a],'.',color=colors[a],markersize=1)
ax.axis('equal')
ax.view_init(38, 128)
