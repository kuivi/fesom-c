#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 09:22:18 2021

@author: I. Kuznetsov (kuivi)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from matplotlib.dates import DateFormatter, date2num
import glob
import os
from scipy.spatial import cKDTree
import pyproj
import matplotlib.pyplot as plt
from scipy import interpolate
from netCDF4 import Dataset
from netCDF4 import num2date
import cftime
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
import shutil
from   matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
from scipy import signal

exname='medi_v9_1_'
pathmesh='/work/projects/medi/mesh/mesh_v9_1/'


fnod = pathmesh+'nod2d.out'
with open(fnod, 'r') as f:  # We need to re-open the file
    nnod2d = int(f.readline().strip().split()[0]) # number of nods
    nodx  = np.zeros(nnod2d)
    nody  = np.zeros(nnod2d)
    nindx = np.zeros(nnod2d)
    i = 0
    for line in f:
        columns = line.strip().split()
        nodx[i] = columns[1]
        nody[i] = columns[2]    
        nindx[i] = columns[3]    
        i=i+1
felem = pathmesh+'elem2d.out'
with open(felem, 'r') as f:  # We need to re-open the file
    nelem2d = int(f.readline().strip().split()[0]) # number of nods
    elem2d  = np.zeros((nelem2d,4),dtype=int)
    i = 0
    for line in f:
        columns = line.strip().split()
        elem2d[i,:] = columns[:]
        i=i+1
        
depth=nodx.copy()
depth[:]=np.nan

############################# Gebco ##########################################

f='/work/projects/medi/data/GEBCO_04_Nov_2021_866e4b608088/gebco_medi.nc'
f='/work/data/bathymetry/GRIDONE_2D.nc'
print(f)
ncf = Dataset(f,'r')
lon = ncf.variables['lon'][:].data
lat = ncf.variables['lat'][:].data
d = ncf.variables['elevation'][:,:].data
ncf.close()
d[d>0.0]=0.0
d=d*(-1)
fdepth = RegularGridInterpolator((lon, lat), d.T,bounds_error=False,fill_value=None)
depth=fdepth((nodx,nody))
ind=np.where(np.isnan(depth))
print("NAN",len(ind[0]))
depth[depth<1]=1.0


f='/work/projects/medi/data/GEBCO_04_Nov_2021_866e4b608088/gebco_medi.nc'
print(f)
ncf = Dataset(f,'r')
lon2 = ncf.variables['lon'][:].data
lat2 = ncf.variables['lat'][:].data
d2 = ncf.variables['elevation'][:,:].data
ncf.close()
d2[d2>0.0]=0.0
d2=d2*(-1)
d3=d2.copy()
d3[d3<10.0]=10.0

from scipy.signal import savgol_filter
from scipy.signal import convolve2d
def smooth(y, box_pts):
    box = np.ones((box_pts,box_pts)) / box_pts**2
    y_smooth = convolve2d(y, box, mode="same")
    return y_smooth

d2_sm=savgol_filter(d3, 51, polyorder=1,axis=0)
d2_sm=savgol_filter(d2_sm, 51, polyorder=1,axis=1)

d2_sm = smooth(d2,11)
d2_sm = smooth(d2_sm,11)
d2_sm = smooth(d2_sm,11)



ddd=d2_sm.copy()
ddd[ddd>500]=500

plt.contourf(lon2,lat2,ddd)
plt.colorbar()
plt.show(block=False)

d2_sm[d2_sm<10]=10

fdepth2 = RegularGridInterpolator((lon2, lat2), d2_sm.T,bounds_error=False,fill_value=None)
depth2=fdepth2((nodx,nody))
ind=np.where(np.isnan(depth2))
print("NAN",len(ind[0]))


from scipy.spatial import cKDTree
tree = cKDTree(list(zip(nodx, nody)))
d_st, inds_st = tree.query(list(zip(nodx, nody)), k=20, workers=-1)
depth2_sm=depth2.copy()
for i in range(len(nodx)):
    depth2_sm[i] = np.mean(depth2[inds_st[i,:]])
    
f = open(pathmesh+exname+'depth_smoothed_v3.out','w')       
for i in range(len(nodx)):
    f.write("%15.10f\n" % (depth2_sm[i]))
f.close()



f='/work/projects/medi/data/lagoonVe_2002_GBe_wgs.nc'
print(f)
ncf = Dataset(f,'r')
lonv = ncf.variables['lon'][:].data
latv = ncf.variables['lat'][:].data
dv = ncf.variables['Band1'][:]
dv[dv.mask]=np.nan
dv=dv.data
dv[dv>0.0]=0.0
dv=dv*(-1)

fdepthdv = RegularGridInterpolator((lonv, latv), dv.T,bounds_error=False,fill_value=np.nan)
depthdv=fdepthdv((nodx,nody))



ind=np.where(~np.isnan(depthdv))[0]

depth2[ind]=depthdv[ind]
depth[ind]=depthdv[ind]

plt.scatter(nodx,nody,1,depth2-depth)


f = open(pathmesh+exname+'depth_smoothed_v2.out','w')       
for i in range(len(nodx)):
    f.write("%15.10f\n" % (depth2[i]))
f.close()
shutil.copyfile(pathmesh+exname+'depth.out',pathmesh+'depth.out')

#-----------------------------------------------------------------------------

exname='medi_v8_1_gebco_1min_'
pathmesh='/work/projects/medi/mesh/mesh_v8_1/'
f = open(pathmesh+exname+'depth.out','w')       
for i in range(len(nodx)):
    f.write("%15.10f\n" % (depth[i]))
f.close()

exname='medi_v8_1_'
pathmesh='/work/projects/medi/mesh/mesh_v8_1/'
f = open(pathmesh+exname+'depth.out','w')       
for i in range(len(nodx)):
    f.write("%15.10f\n" % (depth2[i]))
f.close()

shutil.copyfile(pathmesh+exname+'depth.out',pathmesh+'depth.out')

#plt.contourf(lon,lat,d,levels=[0,2,5])
#plt.colorbar()
plt.plot(nodx,nody,'.k',alpha=0.1)

ind=np.where((nodx<13) & (nodx>9) &(nody<36)&(nody>32.5))

plt.figure()
ax=plt.gca()
dd=depth2.copy()-depth.copy()
dd[dd>100]=100
dd[dd<-30]=-30
plt.scatter(nodx[ind],nody[ind],1,dd[ind])
plt.colorbar()



ax.set_xlim=[9,13]
ax.set_ylim=[32.5,36]

f='/work/projects/medi/data/GEBCO_04_Nov_2021_866e4b608088/gebco_medi.nc'
print(f)
ncf = Dataset(f,'r')
lon2 = ncf.variables['lon'][:].data
lat2 = ncf.variables['lat'][:].data
d2 = ncf.variables['elevation'][:,:].data
ncf.close()
d2[d2>0.0]=0.0
d2=d2*(-1)
fdepth2 = RegularGridInterpolator((lon2, lat2), d2.T,bounds_error=False,fill_value=None)
depth2=fdepth2((nodx,nody))
ind=np.where(np.isnan(depth2))
print("NAN",len(ind[0]))
depth2[depth<2]=2.0













