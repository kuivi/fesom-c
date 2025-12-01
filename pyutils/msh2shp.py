#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to convert FESOM-C mesh (nod2d.out format) to shapefile format for visualization.

Created on Wed Jan 26 23:42:29 2022

@author: I. Kuznetsov (kuivi)
"""
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
import numpy as np

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

depth=nodx*0.0
pathmesh='/work/projects/indiga/mesh/out/indiga_v2_depth.out'
f = open(pathmesh,'r')       
for i in range(len(nodx)):
    depth[i]=f.readline()
f.close()

delem=[]
for i in range(nelem2d):
    coord=[]
    d=depth[elem2d[i][:]-1].mean()
    delem.append(d)


poly=[]
il=[]
for i in range(nelem2d):
    coord=[]
    for j in range(4):
        coord.append((nodx[elem2d[i][j]-1],nody[elem2d[i][j]-1]))
    poly.append(Polygon(coord))
    il.append(i+1)
        
gdf = gpd.GeoDataFrame(geometry=poly, crs="EPSG:4326")
gdf['depth']=delem
gdf.to_file("/work/projects/indiga/mesh/indiga_v2d.shp",driver='ESRI Shapefile')
