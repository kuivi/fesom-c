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
import shapefile

import pyTMD.time
import pyTMD.model
import pyTMD.spatial
import pyTMD.utilities
from pyTMD.calc_delta_time import calc_delta_time
from pyTMD.infer_minor_corrections import infer_minor_corrections
from pyTMD.predict_tide import predict_tide
from pyTMD.predict_tide_drift import predict_tide_drift
from pyTMD.read_tide_model import extract_tidal_constants
from pyTMD.read_netcdf_model import extract_netcdf_constants
from pyTMD.read_GOT_model import extract_GOT_constants
from pyTMD.read_FES_model import extract_FES_constants
from shapely.geometry import Point, Polygon 

import pyTMD.model
import pyTMD
from pyTMD.read_tide_model import extract_tidal_constants
from pyTMD.read_netcdf_model import extract_netcdf_constants
from pyTMD.read_netcdf_model import read_netcdf_grid
from pyTMD.read_GOT_model import extract_GOT_constants
from pyTMD.read_FES_model import extract_FES_constants

from pyTMD.load_constituent import load_constituent
from pyTMD.load_nodal_corrections import load_nodal_corrections

fnod = '/work/projects/indiga/mesh/out/nod2d.out'
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
ob=np.where(nindx==2)        
nodx[ob].max()
nody[ob].min()

felem = '/work/projects/indiga/mesh/out/elem2d.out'
with open(felem, 'r') as f:  # We need to re-open the file
    nelem2d = int(f.readline().strip().split()[0]) # number of nods
    elem2d  = np.zeros((nelem2d,4),dtype=int)
    i = 0
    for line in f:
        columns = line.strip().split()
        elem2d[i,:] = columns[:]
        i=i+1

model = pyTMD.model( '/work/tides/tide_models/',format='OTIS',compressed=False).elevation('AOTIM-5')

amp,ph,D,c = pyTMD.extract_tidal_constants(nodx[ob], nody[ob], model.grid_file,
                model.model_file, model.projection, TYPE=model.type,
                    METHOD='spline', GRID=model.format)

amp= amp.filled(np.nan)
ph= ph.filled(np.nan)
amp[1,:]=amp[2,:]
ph[1,:]=ph[2,:]

exname='indiga_v2_'
pathmesh='/work/projects/indiga/mesh/out/'
f = open(pathmesh+exname+'m2.out','w')       
f.write("%7i\n" % (len(ob[0])))
for i in range(len(ob[0])):
    f.write("%7i " % (ob[0][i]+1))
    for j in range(8):
        f.write("%4.8f %4.8f " % (amp[i,j],ph[i,j]))
    f.write("\n")        
f.close()
shutil.copyfile(pathmesh+exname+'m2.out',pathmesh+'m2.out')




path2tmodels='/work/tides/tide_models/'
tmodel='TPXO9-atlas-v4'
compressed=False
atlas = 'netcdf'#'netcdf'
model = pyTMD.model(path2tmodels, format=atlas, compressed=compressed).elevation(tmodel)

#define resolution x,y
resolution = [1000,1000]
#define regular mesh boundary [xmin,xmax,ymin,ymax]
box = [47, 50.,67, 69]

left, right, down, up = box
wgs84_geod = pyproj.Geod(ellps='WGS84') # get geoid

#mesh size from resolution 
res = [round(wgs84_geod.inv(left,(up+down)*0.5,right,(up+down)*0.5)[2]/resolution[0])
       ,round(wgs84_geod.inv((left+right)*0.5,down,(left+right)*0.5,up)[2]/resolution[1])]
lonNumber, latNumber = res
lonreg = np.linspace(left, right, lonNumber)
latreg = np.linspace(down, up, latNumber)
lonreg2, latreg2 = np.meshgrid(lonreg, latreg)

lons=lonreg2.flatten()
lats=latreg2.flatten()
amp,ph,D,c = pyTMD.extract_tidal_constants(lons, lats, model.grid_file,
                model.model_file, model.projection, TYPE=model.type,
                    METHOD='spline', GRID=model.format)
ampr=np.reshape(amp,(lonreg2.shape[0],lonreg2.shape[1],8))
phr=np.reshape(ph,(lonreg2.shape[0],lonreg2.shape[1],8))
Dr=np.reshape(D,(lonreg2.shape[0],lonreg2.shape[1]))



#save to nc
f='/work/projects/indiga/mesh/aotim5.nc'
os.remove(f)
ncf = Dataset(f,'w')
xdim=ncf.createDimension('x',size=len(lonreg))
ydim=ncf.createDimension('y',size=len(latreg))
xvar=ncf.createVariable('x', datatype='f',dimensions=('x'))
xvar.setncattr_string('units','Degrees_east')
xvar[:]=lonreg[:]
yvar=ncf.createVariable('y', datatype='f',dimensions=('y'))
yvar.setncattr_string('units','Degrees_north')
yvar[:]=latreg[:]
dvar=ncf.createVariable('depth', datatype='f',dimensions=('y','x'))
dvar.setncattr_string('units','m')
dvar[:,:]=Dr[:,:]
dam=[]
dph=[]
for i in range(8):
    dam.append(ncf.createVariable(c[i],datatype='f',dimensions=('y','x')))
    dam[i].setncattr_string('units','m')
    dam[i][:,:]=ampr[:,:,i]
    dph.append(ncf.createVariable(c[i]+'_p',datatype='f',dimensions=('y','x')))
    dph[i].setncattr_string('units','grad')
    dph[i][:,:]=phr[:,:,i]
ncf.close()

amp,ph,D,c = pyTMD.extract_tidal_constants(nodx[ob], nody[ob], model.grid_file,
                model.model_file, model.projection, TYPE=model.type,
                    METHOD='spline', GRID=model.format)

amp= amp.filled(np.nan)
ph= ph.filled(np.nan)
amp[1,:]=amp[2,:]
ph[1,:]=ph[2,:]


cc=['m2', 's2', 'n2', 'k2', 'k1', 'o1', 'p1', 'q1', 'mf', 'mm', 'm4']


f = open(pathmesh+exname+'m2.out','w')       
f.write("%7i\n" % (len(ob[0])))
for i in range(len(ob[0])):
    f.write("%7i " % (ob[0][i]+1))
    for j in [5,6,4,7,3,1,2,0,12,13,8]:
        f.write("%4.8f %4.8f " % (amp[i,j],ph[i,j]))
    f.write("\n")        
f.close()
shutil.copyfile(pathmesh+exname+'m2.out',pathmesh+'m2.out')



exname='indiga_v1_'
pathmesh='/work/projects/indiga/mesh/out/'
f = open(pathmesh+exname+'m2.out','w')       
f.write("%7i\n" % (len(ob[0])))
for i in range(len(ob[0])):
    f.write("%7i " % (ob[0][i]+1))
    for j in range(8):
        f.write("%4.8f %4.8f " % (amp[i,j],ph[i,j]))
    f.write("\n")        
f.close()
shutil.copyfile(pathmesh+exname+'m2.out',pathmesh+'m2.out')



plt.plot(amp[:,0])

for j in range(amp.shape[1]):
    amp[:,j]= amp[:,j].filled(np.nan)
    ph[:,j]= ph[:,j].filled(np.nan)
    ind=np.where(np.isnan(amp[:,j]))[0]
    while (len(ind)>0):
        for i in range(len(ind)):
            m1=ind[i]-1
            p1=ind[i]+1
            if ind[i]==0: 
                m1=0
            if ind[i]==len(amp):
                p1=ind[i]
            #print(amp[m1:p1+1,j])    
            amp[ind[i],j]=np.nanmean(amp[m1:p1+1,j])
            ph[ind[i],j]=np.nanmean(ph[m1:p1+1,j])
        ind=np.where(np.isnan(amp[:,j]))[0]            
        




ix=np.where((lon<56.57) & (lon>53.1))[0]
iy=np.where((lat<69.6) & (lat>67.8))[0]
lon=lon[ix[0]:ix[-1]]
lat=lat[iy[0]:iy[-1]]
bathymetry=bathymetry[iy[0]:iy[-1],ix[0]:ix[-1]]
plt.contourf(lon,lat,bathymetry)
plt.plot(nodx,nody,'.')
plt.colorbar()
#lon=np.arange(-10.0,37,0.1)
#lat=np.arange(29.0,48,0.1)
#long,latg = np.meshgrid(lon,lat)
#lons = long.flatten()
#lats = latg.flatten()
lons=nodx[ob]
lats=nody[ob]
amp,ph,D,c = extract_netcdf_constants(lons, lats, model.grid_file,
        model.model_file, TYPE=model.type, METHOD='linear', 
        SCALE=model.scale, GZIP=model.compressed)
m2= amp[:,5].filled(np.nan)
m2p=ph[:,5].filled(np.nan)
Df=D.filled(np.nan)
m2s=m2.copy()
m2ps=m2p.copy()
Ds=Df.copy()
ind=np.where(np.isnan(m2))[0]
for i in range(len(ind)):
    m1=ind[i]-1
    p1=ind[i]+1
    if ind[i]==0: 
        m1=0
    if ind[i]==len(m2):
        p1=ind[i]
    m2s[ind[i]]=np.nanmean(m2[m1:p1+1])
    m2ps[ind[i]]=np.nanmean(m2p[m1:p1+1])
    Ds[ind[i]]=np.nanmean(Df[m1:p1+1])


for i in range(len(m2)):
    m1=i-1
    p1=i+1
    if i==0:
        m1=0
    if i==len(m2)-1:
        p1=len(m2)-1
    m2[i]=np.mean(m2s[m1:p1+1])
    m2p[i]=np.mean(m2ps[m1:p1+1])
    Df[i]=np.mean(Ds[m1:p1+1])

m2d=m2.copy()
for i in range(len(m2)):
    m2d[i] = m2[i]*Df[i]/depth[ob[0]][i]



exname='medi_v4_'
pathmesh='/work/projects/medi/mesh/mesh2/'
f = open(pathmesh+exname+'m2.out','w')       
f.write("%7i\n" % (len(ob[0])))
for i in range(len(ob[0])):
    f.write("%7i " % (ob[0][i]+1))
    for j in range(1):
        f.write("%4.8f %4.8f " % (m2[i],m2p[i]))
    f.write("\n")        
f.close()
shutil.copyfile(pathmesh+exname+'m2.out',pathmesh+'m2.out')

    
x=np.arange(len(m2))

plt.plot(x,m2s,'-')
plt.plot(x,m2,'-')
plt.plot(x,m2d,'-')
plt.plot(x,m2/m2s,'-')

plt.plot(x,Ds)
plt.plot(x,Df)
plt.plot(x,Df/Ds)

plt.plot(x,depth[ob[0]])

plt.plot(lats,m2s,'.')
plt.plot(lats,m2,'.')
plt.plot(lats,m2_spline,'.')

plt.scatter(lons,lats,10,m2)

plt.contourf(long,latg,m2*100,levels=np.arange(0, 30,2.5))
plt.colorbar()
c = model.constituents
pu,pf,G = load_nodal_corrections(0 + 48622.0, c,
        DELTAT=0, CORRECTIONS=model.format)



# function to extrapolate array with nans (fill nans with surounding min values)
def rmnan(d,iter=10):
    o=d.copy()
    ind=np.where(np.isnan(d))
    print(np.shape(ind)[1])    
    for i in range(iter):
        o[:-1,:]=np.fmin(d[:-1,:],d[1:,:])
        ind=np.where(np.isnan(d))
        d[ind]=o[ind]
        o[:,:-1]=np.fmin(d[:,:-1],d[:,:-1])
        ind=np.where(np.isnan(d))
        d[ind]=o[ind]
        o[1:,:]=np.fmin(d[1:,:],d[:-1,:])
        ind=np.where(np.isnan(d))
        d[ind]=o[ind]
        o[:,1:]=np.fmin(d[:,1:],d[:,:-1])
        ind=np.where(np.isnan(d))
        d[ind]=o[ind]
    ind=np.where(np.isnan(d))
    print(np.shape(ind)[1])    
    return d


f='/work/tides/tpxo9/tp9/h_tpxo9.v1.nc'
ncf = Dataset(f,'r')
lon = ncf.variables['lon_z'][:,:].data
lat = ncf.variables['lat_z'][:,:].data
ind=np.where((lon>360-10) & (lon<360-5) & (lat>32) & (lat<39))
ix0=ind[0].min()
ix1=ind[0].max()
iy0=ind[1].min()
iy1=ind[1].max()
lon = ncf.variables['lon_z'][ix0:ix1,iy0:iy1].data
lat = ncf.variables['lat_z'][ix0:ix1,iy0:iy1].data
ha = ncf.variables['ha'][:,ix0:ix1,iy0:iy1].data
ha[ha==0]=np.nan
ha = rmnan(ha)
hp = ncf.variables['hp'][:,ix0:ix1,iy0:iy1].data
hp[hp==0]=np.nan
hp = rmnan(hp)
con = ncf.variables['con'][:].data
names=[]
for j in range(np.shape(con)[0]):
    a=[str(con[j][i])[2] for i in [0,1,2,3]]
    names.append((a[0]+a[1]+a[2]+a[3]).strip())
names.append('dummy')
tide=np.zeros((2,16,len(ob[0])))
lona=lon[:,0]
lata=lat[0,:]
j=0
for j in range(15):
    fa = RegularGridInterpolator((lona-360, lata), ha[j,:,:],bounds_error=False,fill_value=None)
    tide[0,j,:]=fa((nodx[ob],nody[ob]))
    fp = RegularGridInterpolator((lona-360, lata), hp[j,:,:],bounds_error=False,fill_value=None)
    tide[1,j,:]=fp((nodx[ob],nody[ob]))

#fesom order
fnames=['M2 ', 'S2 ', 'N2 ', 'K2 ','K1 ', 'O1 ','P1 ', 'Q1 ', 'Mf ', 'Mm ', 'Ssa', 'M4 ']
#tpxo order
indtpxo=[0,1,2,3,4,5,6,7,9,8,15,10]
exname='medi_v1_'
pathmesh='/work/projects/medi/mesh/mesh2/'
f = open(pathmesh+exname+'m2.out','w')       
f.write("%7i\n" % (len(ob[0])))
for i in range(len(ob[0])):
    f.write("%7i " % (ob[0][i]+1))
    for j in range(12):
        jcon=indtpxo[j]
        f.write("%4.8f %4.8f " % (tide[0,jcon,i],tide[1,jcon,i]))
    f.write("\n")        
f.close()
shutil.copyfile(pathmesh+exname+'m2.out',pathmesh+'m2.out')



############################# plots    
for i in range(15):
    plt.plot(nody[ob],tide[0,i,:],label=names[i])
plt.legend()

plt.contourf(lon-360,lat,hp[0,:,:],levels=31)
plt.colorbar()
plt.plot(nodx[ob],nody[ob],'.r')
plt.figure()
#plt.contourf(lon-360,lat,ha[0,:,:],levels=101,vmax=1.5,vmin=0.5)
plt.scatter(lon-360,lat,10,hp[0,:,:],vmax=1.5,vmin=0.5)
plt.scatter(lon-360,lat,10,ha[0,:,:],vmax=1.5,vmin=0.5)
plt.colorbar()
plt.plot(nodx[ob],nody[ob],'.r')


plt.colorbar()

j=179576
plt.plot(nodx[elem2d[j]-1],nody[elem2d[j]-1],'-')

ind=np.where((nodx[elem2d[:,0]]>24.8) & (nodx[elem2d[:,0]]<25.5) &
             (nody[elem2d[:,0]]>37.1) & (nody[elem2d[:,0]]<37.3))

elem_coordx = np.zeros((nelem2d))
elem_coordy = np.zeros((nelem2d))
elem_coordx =np.array([nodx[elem2d[j]-1] for j in range(nelem2d)])
elem_coordy =np.array([nody[elem2d[j]-1] for j in range(nelem2d)])
ind=np.where((elem_coordx>24.8) & (elem_coordx<25.5) &
             (elem_coordy>37.1) & (elem_coordy<37.3))

for j in ind[0]:
    plt.plot(nodx[elem2d[j]-1],nody[elem2d[j]-1])

j=179576
plt.plot(nodx[elem2d[j]-1],nody[elem2d[j]-1],'*y')

#define resolution x,y
resolution = [1000,1000]
#define regular mesh boundary [xmin,xmax,ymin,ymax]
box = [-10, 37.,30, 46]

left, right, down, up = box
wgs84_geod = pyproj.Geod(ellps='WGS84') # get geoid

#mesh size from resolution 
res = [round(wgs84_geod.inv(left,(up+down)*0.5,right,(up+down)*0.5)[2]/resolution[0])
       ,round(wgs84_geod.inv((left+right)*0.5,down,(left+right)*0.5,up)[2]/resolution[1])]
lonNumber, latNumber = res
lonreg = np.linspace(left, right, lonNumber)
latreg = np.linspace(down, up, latNumber)
lonreg2, latreg2 = np.meshgrid(lonreg, latreg)


lons=lonreg2.flatten()
lats=latreg2.flatten()
amp,ph,D,c = extract_netcdf_constants(lons, lats, model.grid_file,
        model.model_file, TYPE=model.type, METHOD='linear', 
        SCALE=model.scale, GZIP=model.compressed)
ampr=np.reshape(amp,(lonreg2.shape[0],lonreg2.shape[1],14))
phr=np.reshape(ph,(lonreg2.shape[0],lonreg2.shape[1],14))
Dr=np.reshape(D,(lonreg2.shape[0],lonreg2.shape[1]))

fname='/work/projects/medi/data/tides_2_5km_tpxo.npy'
fname='/work/projects/medi/data/tides_1_0km_tpxo.npy'

np.save(fname,(ampr,phr,Dr,c),allow_pickle=True)

pfield=plt.scatter(nodx,nody,1,amp[:,5],cmap='plasma')

####
amp,ph,D,c = extract_netcdf_constants(-8.1575562796, 36.9442493435, model.grid_file,
        model.model_file, TYPE=model.type, METHOD='linear', 
        SCALE=model.scale, GZIP=model.compressed)

amp


