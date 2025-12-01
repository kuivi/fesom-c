#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
example how to write river data so fesom-c can read it
It is done only for 1 (one) river
One need to modify script to add more rivers.

The output file, for th FESOM-C model, is a list of dates (when we have runoff),
list of nodes where we have runoff (for more rivers one need to add runoff from
 different rivers to nodes, so one node can have more than 1 river data), and runoff list


@author: I. Kuznetsov (kuivi)

"""



import numpy as np
import matplotlib.pyplot as plt
import pyproj
import jdcal

ver='v9_1'

path2mesh='/work/projects/medi/mesh/mesh_v9_1/'
#read mesh
fnod = path2mesh+'nod2d.out'
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
felem = path2mesh+'elem2d.out'
with open(felem, 'r') as f:  # We need to re-open the file
    nelem2d = int(f.readline().strip().split()[0]) # number of nods
    elem2d  = np.zeros((nelem2d,4),dtype=int)
    i = 0
    for line in f:
        columns = line.strip().split()
        elem2d[i,:] = columns[:]
        i=i+1

#plt.plot(nodx,nody,'.')
#river coordinates
riv0x=13.21
riv0y=45.71

#plt.plot([riv0x,riv1x],[riv0y,riv1y],'*r')

#radius of river influence
riv_radius = 5000.

#ini geod
g = pyproj.Geod(ellps='WGS84') # get geoid

#distance from each nod to river position
dist = g.inv(riv0x*np.ones(nnod2d),riv0y*np.ones(nnod2d),nodx,nody)[2] 

#create mask to one river based on distance
riv_mask = dist/riv_radius
riv_mask[dist>riv_radius]=1.0
riv_mask=1.0-riv_mask
#identify indexes of nodes indlunced by rivers
idx = np.where(riv_mask>0)[0]
nrivnod = np.size(idx) #tot number of nodes for this river
totw = riv_mask[idx].sum() # total weight of nodes

#plt.plot(nodx[idx],nody[idx],'.')
#plt.scatter(nodx[idx],nody[idx],c=riv_mask[idx])
#plt.colorbar()

#read river data
#here is just dummy example, 
#one need to read data and store it in "roff" (runoff) and "tiv_time" (julian days)
ndays=365
daysofyear=np.array([i+1 for i in range(ndays)])  #days
roff = np.sin(np.linspace(0,3.14,ndays))*500 #runoff Q m**3/s , example of runoff
riv_time=[]    
for n in range(ndays):
    jd=jdcal.gcal2jd(2009,1,daysofyear[n]) #convert river date to julian days
    riv_time.append(jd[0]+jd[1]) #add julian days to riv_time
   
# devide runoff by tot weights    
roff=roff/totw

#plt.plot(riv_time,roff)

#write runoff,dates, and node indexes to ascii file, so fesom-c can read

nriv_time=len(riv_time)

f = open(path2mesh+'rivers_daily.dat','w')       
f.write("%i\n" % (nriv_time))
f.write("%i\n" % (nrivnod))
for i in range(nriv_time):
    f.write("%17.7f" % (riv_time[i]))
f.write("\n")    
for i in range(nrivnod):
    f.write("%i " % (idx[i]+1))
f.write("\n")    
for i in range(nriv_time):
    for j in range(nrivnod):
        f.write("%17.10f " % (roff[i]*riv_mask[idx[j]]))
    f.write("\n")    

f.close()
