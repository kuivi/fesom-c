#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Created on Feb 8 2023

@author: I. Kuznetsov (kuivi)

Get tidal amplitudes and phases from fesom-C output
Regrid result on regular mesh

"""


# In[2]:


import os
import numpy as np
from netCDF4 import Dataset
import pyproj
from scipy import fft
from scipy.spatial import cKDTree


# In[3]:


import time as tm


# In[4]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import sys
sys.path.append(os.path.abspath("/albedo/home/ikuznets/python/pyfesom2/pyfesom2/"))
from load_mesh_c_data import *
from load_mesh_c_data import read_fesomc_sect,contourf_sect,fesom_c_mesh#load_c_station,


# In[5]:


start = tm.process_time()


# In[6]:


#define paths
titl = ''
path2='/albedo/work/projects/p_fesomc_mosaic/RUN/med/medi_v8_1_cd_2d_manning/nc_output/'+titl
#path2='/albedo/work/projects/p_fesomc_mosaic/RUN/med/medi_v5_1/nc_output/'+titl


infilete = path2+'eta_128.nc'
meshfile = path2+'eta_128.nc'
outfile = path2+'amppha.nc'
outfile2 = path2+'amppha_reg.nc'
path2output = path2
path2png    = path2
if not os.path.exists(path2png): os.makedirs(path2png)


# In[7]:


mesh = fesom_c_mesh(infilete)


# In[8]:


ncfte = Dataset(infilete)


# In[9]:


#get timestep
t1=ncfte.variables['time'][0]
t2=ncfte.variables['time'][1]
dt=(t2-t1)*86400
print("dt=",dt)
# set dt manuly
dt=23.2875*30/2
print("dt=",dt)


# In[10]:


Nper = 1 # number of M2 periods (1,... 29.5)
# set up range of data for input (model output)
tind1 = ncfte.variables['time'].shape[0]-2
print("total time step:",tind1)

#get start index for input data
tind0 = tind1 - int((Nper*(3600*12.42/(dt))))
print("start index for input: ",tind0)
tindw=np.arange(tind0,tind1,1)
tot_time = tindw.size
print("total input indexes: ",tot_time," indexes: ", tindw)
if tind0<0: print("========= ERROR  ============= tind0<0")


# In[11]:


#read data 
eta=ncfte.variables['eta'][tindw,:].data.squeeze()


# In[12]:


#get wet mask
wet_mask=np.unique(np.where(eta[:,:]==0)[1])


# In[13]:


# number of points:
Np=eta.shape[0]
# perform DFFT for real eta
Y = fft.rfft(eta,axis=0,workers=-1)


# In[14]:


#tidal waves and freq.
wname=['M2 ', 'S2 ', 'N2 ', 'K2 ','K1 ', 'O1 ','P1 ', 'Q1 ', 'Mf ', 'Mm ', 'Ssa', 'M4 ']
wper=[12.4200, 12.00, 12.6584, 11.9673, 23.9344,25.8194, 24.0659, 26.8684, 327.85, 661.31, 4383.05,6.2103]


# In[15]:


#get freq. for input data (time step dt in second) convert to hours
f = fft.fftfreq(Np,d=dt/3600)[:Np//2] 
wind=[]
#find position (index) for each wave
for i in range(len(wname)):
    #skip freq. =0 , add 1 to position due to skiping
    wind.append(1+(np.abs(1/f[1:] - wper[i])).argmin())
    print(wname[i],wind[-1])


# In[16]:


Y=Y[wind,:]
#get aplitudes
a = np.abs(Y) *2/Np
#get phases
p = 180-np.angle(Y,deg=True)


# In[17]:


a[:,wet_mask]=np.nan
p[:,wet_mask]=np.nan


# In[18]:


#prepare output file
filename=outfile
if os.path.exists(filename):
    os.remove(filename)
ifile = Dataset(filename,'w')

#did_depth =  ifile.createDimension('depth', nz)#z.size )
did_node   =  ifile.createDimension('node',Y.shape[1]);
did_wave  =  ifile.createDimension('wave',Y.shape[0]);

vid_lat  = ifile.createVariable('lat','f8',('node',));
vid_lon  = ifile.createVariable('lon','f8',('node',));

vid_wave  = ifile.createVariable('period','f8',('wave',));
vid_amp = ifile.createVariable('amp','f8',('wave','node',));
vid_ph = ifile.createVariable('ph','f8',('wave','node',));
vid_r = ifile.createVariable('real','f8',('wave','node',));
vid_i = ifile.createVariable('imag','f8',('wave','node',));

vid_lat[:] = mesh.y2[:]
vid_lon[:] = mesh.x2[:]
vid_wave[:] = wper[:]
vid_amp[:] = a[:]
vid_ph[:] = p[:]
vid_r[:] = Y.real[:]
vid_i[:] = Y.imag[:]
ifile.close()


# In[19]:


print("output file:")
print(outfile)


# In[20]:


#define variables to be saved
var='v'
#define resolution x,y
resolution = [1000,1000]
#define regular mesh boundary [xmin,xmax,ymin,ymax]
box = [mesh.x2.min(), mesh.x2.max(), mesh.y2.min(), mesh.y2.max()]


# In[21]:


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


# In[22]:


print("points number along X: ",lonNumber)
print("points number along Y: ",latNumber)


# In[23]:


#convert 2d lon,lat of regular grid to a stereographic coordinates 
pste = pyproj.Proj(proj="stere", errcheck="True",ellps='WGS84', lat_0=latreg2.mean(), lon_0=lonreg2.mean())
(slon,slat)=pste(lonreg2,latreg2)
#slon=slon/1000
#slat=slat/1000


# In[24]:


#prepare output file
filename=outfile2
if os.path.exists(filename):
    os.remove(filename)
ifile = Dataset(filename,'w')

#did_depth =  ifile.createDimension('depth', nz)#z.size )
did_lon   =  ifile.createDimension('lon',lonreg.size);
did_lat   =  ifile.createDimension('lat',latreg.size);
did_wave  =  ifile.createDimension('wave',Y.shape[0]);

vid_lat  = ifile.createVariable('lat','f8',('lat',));
vid_lon  = ifile.createVariable('lon','f8',('lon',));
vid_wave = ifile.createVariable('wave','f8',('wave',));
vid_period  = ifile.createVariable('period','f8',('wave',));


vid_amp = ifile.createVariable('amp','f8',('wave','lat','lon',));
vid_ph = ifile.createVariable('pha','f8',('wave','lat','lon',));
vid_r = ifile.createVariable('real','f8',('wave','lat','lon',));
vid_i = ifile.createVariable('imag','f8',('wave','lat','lon',));

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

vid_wave[:] = wper[:]
vid_period[:] = wper[:]

vid_amp.units =  'm'
vid_amp.standard_name =  'amplitude'
vid_ph.units =  'deg'
vid_ph.standard_name =  'phase'


# In[25]:


#convert  lon,lat to a stereographic coordinates 
pste = pyproj.Proj(proj="stere", errcheck="True",ellps='WGS84', lat_0=latreg2.mean(), lon_0=lonreg2.mean())
(xsn, ysn) = pste(mesh.x2,mesh.y2)
(xt, yt) = pste(lonreg2.flatten(), latreg2.flatten())
#build tree
treen = cKDTree(list(zip(xsn, ysn)))
#make inds and distances
distancesn, indsn = treen.query(list(zip(xt, yt)), k=1, workers=-1)


# In[26]:


for iw in range(Y.shape[0]):
    print(iw)
    #read data
    datain = a[iw,:].squeeze()
    datai = datain[indsn]
    datai[distancesn >= radius_of_influence] = np.nan
    datai = datai.reshape(lonreg2.shape)
    vid_amp[iw,:,:]=datai

    datain = p[iw,:].squeeze()
    datai = datain[indsn]
    datai[distancesn >= radius_of_influence] = np.nan
    datai = datai.reshape(lonreg2.shape)
    vid_ph[iw,:,:]=datai    
    
    datain = Y[iw,:].real.squeeze()
    datai = datain[indsn]
    datai[distancesn >= radius_of_influence] = np.nan
    datai = datai.reshape(lonreg2.shape)
    vid_r[iw,:,:]=datai        
    
    datain = Y[iw,:].imag.squeeze()
    datai = datain[indsn]
    datai[distancesn >= radius_of_influence] = np.nan
    datai = datai.reshape(lonreg2.shape)
    vid_i[iw,:,:]=datai            


# In[27]:


ifile.close()


# In[ ]:





# In[ ]:




