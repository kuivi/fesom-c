#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:35:34 2022

@author: I. Kuznetsov (kuivi)
"""

import pyTMD.model
from pyTMD.read_tide_model import extract_tidal_constants
from pyTMD.read_netcdf_model import extract_netcdf_constants
from pyTMD.read_GOT_model import extract_GOT_constants
from pyTMD.read_FES_model import extract_FES_constants

from pyTMD.load_constituent import load_constituent
from pyTMD.load_nodal_corrections import load_nodal_corrections

import datetime
import shutil

#t: days relative to 1992-01-01T00:00:00

t=datetime.datetime(2010,1,2,0,0,0)-datetime.datetime(1992,1,1)
t=t.days+t.seconds/86400

mjd = t+48622.0

DELTAT=0.0

CORRECTIONS='GOT'
CORRECTIONS='OTIS'

c=['m2','s2','n2', 'k2','k1', 'o1', 'p1', 'q1', 'mf', 'mm', 'ssa', 'm4']
c=['m2','s2','n2', 'k2','k1', 'o1', 'p1', 'q1', 'mf', 'mm', 'ssa', 'm4']
pu,pf,G =load_nodal_corrections(mjd,c, DELTAT=DELTAT,CORRECTIONS=CORRECTIONS)
pug = pu/np.pi*180.
phas=G[0]+pug[0]
amp=pf[0]
  

import pytide
wt = pytide.WaveTable(['M2','S2','N2', 'K2','K1', 'O1', 'P1', 'Q1', 'Mf', 'Mm', 'Ssa', 'M4'])
tt=[datetime.datetime(2000,1,1,0,0,0)]
f, vu = wt.compute_nodal_modulations(tt)

amp = [i[0] for i in f]
phas = [i[0]/np.pi*180. for i in vu]



exname='medi_v5_1_'
pathmesh='/work/projects/medi/mesh/mesh_v5_1/'


f = open(pathmesh+'2010_01_02_ampl_factor.out','w')       
for i in range(len(c)):
    f.write("%5.10f %5.10f \n" % (amp[i],phas[i]))
f.write("%s \n" % ('------------ stop reading ---------'))    
f.write("%s \n" % ('for: '+str(datetime.datetime(2010,1,2,0,0,0))))    
f.write("%s \n" % ('by pyTMD, load_nodal_corrections, CORRECTIONS=OTIS'))    
f.close()
shutil.copyfile(pathmesh+'2010_01_02_ampl_factor.out',pathmesh+'ampl_factor.out')






#GOT
pug = pu/np.pi*180.
phas=G[0]+pug[0]
ampl=1.0
eta_got=pf[0][0]*ampl*np.cos(0+phas[0])


c='m2'
amp,ph,omega,alpha,species = load_constituent(c)
th = (omega*t*86400.0 + ph + pu[0,0])/np.pi*180.
eta_otis=pf[0][0]*ampl*np.cos(0+th)

th=(G[0,0]*np.pi/180.0 + pu[0,0])%(np.pi*2) 
th=(omega*t*86400.0 + ph + pu[0,0])%(np.pi*2) 

np.cos(0+th)

exname='medi_v5_1_'
pathmesh='/work/projects/medi/mesh/mesh_v5_1/'

