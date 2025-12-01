#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Created on Fri Aug 20 12:08:49 2021

@author: I. Kuznetsov (kuivi)

Regrid output from fesom_c nc files to a regular mesh in nc file.

todo: 
* interpoltion in Z direction
* interpolate multiple variables, make variables list by reading from infile
* autodetect 2d vars
* save inds and distances for future use
"""


# In[2]:


import os
import numpy as np
import gsw
from netCDF4 import Dataset
import joblib
import copy
import pyproj


# In[3]:


import time as tm


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
sys.path.append(os.path.abspath("/home/ollie/ikuznets/python/pyfesom2/pyfesom2/"))
from load_mesh_c_data import *
from load_mesh_c_data import read_fesomc_sect,contourf_sect,fesom_c_mesh#load_c_station,


# In[5]:


from regriding import *


# In[6]:


start = tm.process_time()


# In[7]:


#define paths
infile = '/work/ollie/ikuznets/RUN/fang_eddy/impl_small/nc_output/ts.nc'
outfile = '/work/ollie/ikuznets/RUN/fang_eddy/impl_small/nc_output/salinity_reg.nc'
path2output = '/work/ollie/ikuznets/RUN/fang_eddy/impl_small/nc_output/'
path2png    = '/work/ollie/ikuznets/RUN/fang_eddy/plots/png/impl_small/'
if not os.path.exists(path2png): os.makedirs(path2png)


# In[8]:


#define variables to be saved
var='salinity'
#define resolution x,y
resolution = [2500,2500]
#define regular mesh boundary [xmin,xmax,ymin,ymax]
box = [110, 125,85.3, 86.9]
box = [88.6, 151.4,84.7, 87.2]


# In[9]:


#read mesh from file
mesh = fesom_c_mesh(infile)


# In[10]:


print("time steps in infile: ",np.size(mesh.mtime))


# In[11]:


#define "z" from sigma layers with constant depth from 0 node
#z for sigma
z=(1-mesh.sigma_lev[:])*mesh.topo[0]
#z for sigmam1
z=(1-(mesh.sigma_lev[:-1]+mesh.sigma_lev[1:])/2)*mesh.topo[0]


# In[12]:


#open infile
ncf = Dataset(infile)


# In[13]:


#read time fomr infile
intime = ncf.variables['time'][:]


# In[14]:


#define regular grid

left, right, down, up = box
wgs84_geod = pyproj.Geod(ellps='WGS84') # get geoid

#mesh size from resolution 
res = [round(wgs84_geod.inv(left,(up+down)*0.5,right,(up+down)*0.5)[2]/resolution[0])
       ,round(wgs84_geod.inv((left+right)*0.5,down,(left+right)*0.5,up)[2]/resolution[1])]


influence=10000

lonNumber, latNumber = res
lonreg = np.linspace(left, right, lonNumber)
latreg = np.linspace(down, up, latNumber)
lonreg2, latreg2 = np.meshgrid(lonreg, latreg)
distances_path = None
inds_path = None
radius_of_influence=influence
basepath = None


# In[15]:


print("points number along X: ",lonNumber)
print("points number along Y: ",latNumber)


# In[16]:


#convert 2d lon,lat of regular grid to a stereographic coordinates 
pste = pyproj.Proj(proj="stere", errcheck="True",ellps='WGS84', lat_0=latreg2.mean(), lon_0=lonreg2.mean())
(slon,slat)=pste(lonreg2,latreg2)
#slon=slon/1000
#slat=slat/1000


# In[17]:


#prepare output file
filename=outfile
if os.path.exists(filename):
    os.remove(filename)
ifile = Dataset(filename,'w')

did_depth =  ifile.createDimension('depth', z.size )
did_lon   =  ifile.createDimension('lon',lonreg.size);
did_lat   =  ifile.createDimension('lat',latreg.size);
did_time  =  ifile.createDimension('time',np.size(mesh.mtime));

vid_lat  = ifile.createVariable('lat','f8',('lat',));
vid_lon  = ifile.createVariable('lon','f8',('lon',));
vid_time = ifile.createVariable('time','f8',('time',));
vid_depth  = ifile.createVariable('depth','f8',('depth',));
vid_data = ifile.createVariable(var,'f8',('time','depth','lat','lon',));

vid_lon.axis  =  'X'
vid_lon.units =  'degrees_east'
vid_lon.standard_name =  'longitude'
vid_lon.long_name = 'longitude'
vid_lon[:] = lonreg[:]
vid_lat.axis  =  'Y'
vid_lat.units =  'degrees_north'
vid_lat.standard_name =  'latitude'
vid_lat.long_name = 'latitude'
vid_lat[:] = latreg[:]
#use time from infile
vid_time.units =  ncf.variables['time'].units
vid_time.standard_name =  'time'
vid_time.long_name = 'time'
vid_time.calendar = 'noleap'
vid_time[:] = intime[:]
vid_depth.axis  =  'Z'
vid_depth.units =  'm'
vid_depth.standard_name =  'depth'
vid_depth.long_name = 'depth'
vid_depth[:] = z[:]
vid_data.units =  'psu'
vid_data.standard_name =  'salinity'
vid_data.long_name = 'sea water salinity'


# In[18]:


#make inds and distances
(xs, ys) = pste(mesh.x2,mesh.y2)
(xt, yt) = pste(lonreg2.flatten(), latreg2.flatten())
tree = cKDTree(list(zip(xs, ys)))
distances, inds = tree.query(list(zip(xt, yt)), k=1, n_jobs=-1)


# In[19]:


datai3 = np.ndarray((z.size,latNumber,lonNumber))


# In[20]:


for itime,time in enumerate(intime):
    if (itime%(intime.size%25)==0): print(round(itime/np.size(mesh.mtime)*100))
    #read data
    datain = ncf.variables[var][itime,:,:].data
    #loop over depth
    for iz,depth in enumerate(z): 

        datai = datain[inds,iz]
        datai[distances >= radius_of_influence] = np.nan
        datai = datai.reshape(lonreg2.shape)
        datai3[iz,:,:] = datai[:,:]

        #start = tm.process_time()
    vid_data[itime,:,:,:]=datai3


# In[21]:


ncf.close()
ifile.close()


# In[22]:


print("Done: ",tm.process_time() - start)

