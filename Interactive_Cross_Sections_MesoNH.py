#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:35:05 2022

@author: philippotn
"""
import time
import numpy as np
from numpy.lib.stride_tricks import as_strided
from matplotlib import pyplot as plt 
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import matplotlib.colors as mcolors

from matplotlib.lines import Line2D
from matplotlib.backend_tools import ToolBase , ToolToggleBase
plt.rcParams['toolbar'] = 'toolmanager'
#from matplotlib.backend_bases import MousseButton
import xarray as xr
from numba import njit,prange
import os
from compute_RH_lcl import RH_water_ice, dewpoint
from Module_LES_mountains import RG_lin_interp, AL_lin_interp, AGL_lin_interp, AGL_lin_anomaly, horizontal_convergence, boundary_layer_diags, surface_layer_diags

floatype = np.float16

# - Data LES
# simu = 'AMMA'  ; Zmax = 16000
# simu = 'AMA0W' ; Zmax = 16000
# simu = 'BOMEX' ; Zmax = 16000

# simu = 'ABL0V' ; Zmax = 16000
# simu = 'AMOPL' ; Zmax = 16000
# simu = 'A0W0V' ; Zmax = 16000
# simu = 'ATRI2' ; Zmax = 16000
# simu = 'ACON2' ; Zmax = 16000
# simu = 'AMO20' ; Zmax = 16000
# simu = 'CTRI2' ; Zmax = 5000
# simu = 'CFLAT' ; Zmax = 5000
# simu = 'CMO10' ; Zmax = 5000
# simu = 'DMOPL' ; Zmax = 5000
# simu = 'DMO10' ; Zmax = 4500

# simu = 'E600M' ; Zmax = 10000
simu = 'E620M' ; Zmax = 15000

# simu = 'AROME' ; Zmax = 15000

# userPath = '/cnrm/tropics/user/philippotn'
userPath = '/home/philippotn'

if userPath == '/home/philippotn':
    if simu[0] in ['A','C']: dataPath = userPath+'/Documents/SIMU_LES/'
    elif simu[0] in ['D']: dataPath = userPath+'/Documents/NO_SAVE/'
    elif simu[0] in ['E']: dataPath = userPath+'/Documents/Canicule_Août_2023/'+simu+'/'
        
    
    # 
    savePath = userPath+'/Images/LES_'+simu+'/figures_maps/'
else:
    dataPath = userPath+'/LES_'+simu+'/NO_SAVE/'
    savePath = userPath+'/LES_'+simu+'/figures_maps/'

put_in_RAM = True
orog = True


lFiles = []
# LES du stage
if simu == 'AMMA':
    lFiles = [dataPath + 'AMMH3.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(307,325,2)]
elif simu == 'AMA0W':
    lFiles = [dataPath + 'AMA0W.1.GD551.OUT.{:03d}.nc'.format(i) for i in range(1,7)] + [dataPath + 'AMA0W.1.R6H1M.OUT.{:03d}.nc'.format(i) for i in range(1,361)]
elif simu == 'LBA':
    dataPath = '/cnrm/tropics/user/couvreux/POUR_NATHAN/LBA/SIMU_LES/'
    lFiles = [dataPath + 'LBA__.1.RFLES.OUT.{:03d}.nc'.format(i) for i in range(1,7)]
    tFile = dataPath + 'new_LBA__.1.RFLES.000.nc'
elif simu == 'BOMEX':
    dataPath = '/cnrm/tropics/user/couvreux/POUR_AUDE/SIMU_MNH/bigLES/'
    hours = np.arange(100,102,1)
    lFiles = [dataPath + 'L25{:02d}.1.KUAB2.OUT.{:03d}.nc'.format(i//6+1,i%6+1) for i in hours-1]
# LES relief idéalisé
elif simu == 'ABL0V':
    lFiles = [dataPath + 'ABL0V.1.SEG01.OUT.{:03d}.nc'.format(i) for i in range(50,201,50)]
elif simu == 'AMOPL':
    seg = 'R200m' ; lFiles = [dataPath + 'AMOPL.1.200m1.OUT.{:03d}.nc'.format(i) for i in [60, 240]]+[dataPath + 'AMOPL.1.200m2.OUT.{:03d}.nc'.format(i) for i in [180,360,540,720]]
elif simu== 'A0W0V':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,60)]
elif simu== 'ATRI2':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,60)]
elif simu== 'ACON2':
    seg = 'S200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,962,60)]
elif simu== 'AMO20':
    seg = 'M200m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in [600]]#range(1,961,60)]
elif simu== 'CTRI2':
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in [40,140,240,340]]#range(1,961,60)]
elif simu== 'CFLAT':
    seg = 'S100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(1,872,60)]
elif simu== 'CMO10':
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:03d}.nc'.format(i) for i in range(2,3,1)]
elif simu=='DMOPL':
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(6,109,6)]
elif simu=='DMO10':
    seg = 'M100m' ; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(6,109,6)]
    
# LES canicule alpes
elif simu=='E600M':
    # seg = '19AUG'; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(6,7,1)]
    seg = '19AU2'; lFiles = [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(31,38,6)]
elif simu=='E620M':
    seg = '19AU1'; lFiles += [dataPath + simu+'.2.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(1,38,12)]
    seg = '19AU2'; lFiles += [dataPath + simu+'.2.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(6,37,12)]
    
    # seg = '19AU1'; lFiles += [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(1,8,1)]
    # seg = '19AU2'; lFiles += [dataPath + simu+'.1.'+seg+'.OUT.{:04d}.nc'.format(i) for i in range(1,7,1)]
    
if not os.path.exists(savePath): os.makedirs(savePath) ; print('Directory created !')

f0 = xr.open_dataset(lFiles[0])
var_names = [i for i in f0.data_vars]

#%%

nx1 = 1 ; nx2 = len(f0.ni)-1
ny1 = 1 ; ny2 = len(f0.nj)-1
x = np.array(f0.ni)[nx1:nx2]/1000
y = np.array(f0.nj)[ny1:ny2]/1000

dx=x[1]-x[0]
x_ = np.append(x,x[-1]+dx) -dx/2
y_ = np.append(y,y[-1]+dx) -dx/2


nz1 = 1
Zm = f0.level.data[nz1:]
ZTOP = f0.level_w.data[-1]
nz2 = nz1+np.argmin( np.abs(Zm - Zmax))

Z = f0.level.data[nz1:nz2]

DZm = f0.level_w.data[nz1+1:nz2+1] - f0.level_w.data[nz1:nz2]
dzmax = DZm[-1]
# z = np.array(f0.level)[nz1:nz2]/1000
# z_ = (np.array(f0.level)[nz1:nz2+1]+np.array(f0.level)[nz1-1:nz2])/2000
# z__ = np.array(f0.level)[nz1-1:nz2+1]/1000

AL__ = np.arange(-dzmax/2,Zmax+dzmax/2,dzmax) # AL = Altitude Levels
AL = AL__[1:-1]
z = AL/1000
z_ = (AL__[:-1]+AL__[1:])/2000
z__ = AL__/1000

nt,nz,nx,ny = len(lFiles),len(z),len(x),len(y)

if orog:
    if simu[0] in ['A','C','D']:
        ZS = xr.open_dataset(dataPath+simu+'_init_M'+str(round(dx*1000))+'m_pgd.nc')['ZS'][ny1:ny2,nx1:nx2].data
    else:
        ZS = xr.open_dataset(dataPath+'PGD_'+simu+'_'+str(round(dx*1000))+'m.nc')['ZS'][ny1:ny2,nx1:nx2].data
        # ZS = xr.open_dataset(dataPath+'PGD_200m.neste1.nc')['ZS'][ny1:ny2,nx1:nx2].data
        # ZS = xr.open_dataset(dataPath+'PGD_600m.neste1.nc')['ZS'][ny1:ny2,nx1:nx2].data
    # z = np.arange(dx/2,f0.level[nz2]/1000,dx)
else:
    ZS = np.zeros((ny,nx))
    
#%% preparation of wind speed interpolation for streamlines # doesn't work with orography
new_z = np.linspace(dx,z[-1]-z[-1]%dx,int(z[-1]/dx))
ios = np.zeros(len(new_z),dtype=int)
c0s = np.zeros(len(new_z))
c1s = np.zeros(len(new_z))
for iz in range(len(new_z)):
    ios[iz] = np.argmin(np.abs(new_z[iz]-z))
    io=ios[iz]
    if new_z[iz]<z[io]:
        io-=1
    dz = z[io+1]-z[io]
    c0s[iz] = (z[io+1]-new_z[iz])/dz
    c1s[iz] = (new_z[iz]-z[io])/dz
    
@njit()
def interp_vertical(var,ios,c0s,c1s):
    nz,nx = np.shape(var)
    new_var = np.zeros((len(ios),nx),dtype=var.dtype)
    for iz in range(len(ios)):
        io=ios[iz]
        new_var[iz] = var[io]*c0s[iz] + var[io+1]*c1s[iz]
    return new_var

#%%
    
if not(put_in_RAM):
    times = []
    list_f = []
    for it,fname in enumerate(lFiles):
        f = xr.open_dataset(fname)
        list_f.append(f)
        times.append(' {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))) # year = str(f.time.data[0])[:10]+
        
else:
    def npf(v,dtype=np.float32):
        if v in var_names:
            return np.array(f[v][0,nz1:nz2,ny1:ny2,nx1:nx2],dtype=dtype)
        else:
            return np.zeros((nz2-nz1,ny,nx),dtype=dtype)
    def npf2D(v,dtype=np.float32):
        if v in var_names:
            return np.array(f[v][0,ny1:ny2,nx1:nx2],dtype=dtype)
        else:
            return np.zeros((ny,nx),dtype=dtype)
    
    AGL = [2,10,100,500,1000,2000] # above ground levels
    nAGL = len(AGL)
    izAGL = [ ]
    data3D = np.zeros((11,nt,nz+len(AGL),ny,nx),dtype=floatype)
    data2D = np.zeros((13,nt,ny,nx),dtype=np.float32)
    data1D = np.zeros((3,nt,nz),dtype=np.float32)
    dataZm = np.zeros((3,nt,nz2-nz1,ny,nx),dtype=np.float32)
    
    times = []
    for it,fname in enumerate(lFiles):
        time0 = time.time()
        f = xr.open_dataset(fname)
        times.append(' {:02d}h{:02d}'.format(int(f.time.dt.hour),int(f.time.dt.minute))) # year = str(f.time.data[0])[:10]+
        print(times[-1],end=' ')
        
        U = npf('UT') ; U = (U+np.roll(U,1,axis=2))/2
        V = npf('VT') ; V = (V+np.roll(V,1,axis=1))/2
        W = npf('WT') ; W = (W+np.roll(W,-1,axis=0))/2
        TKE = npf('TKET')
        P = npf('PABST')
        TH = npf('THT')
        RV = npf('RVT')
        RC = npf('RCT')+npf('RIT')
        RP = npf('RRT')+npf('RST')+npf('RGT')
        TR = npf('SVT001')
        
        T = TH * (P/100000)**(2/7)
        THV = TH * (1.+ 1.61*RV)/(1.+RV+RC+RP)
        rhl,rhi = RH_water_ice(RV,T,P)
        RH = np.maximum(rhl,rhi)
        RHO = P/287.06/ (THV * (P/100000.)**(2/7))  # Rd = 287.06     # J/kg/K 
        
        
        dataZm[0,it] = TH
        dataZm[1,it] = dewpoint(T,RH) / (P/100000)**(2/7)
        dataZm[2,it] = W
        
        for iv,VAR in enumerate([U,V,W,TKE,P,THV,RH,RV,RC,RP,TR]):
            if iv in [4,5]: # Special case for pressure and THV anomalies
                VARz = AL_lin_interp(VAR,AL,Zm,ZS)
                data1D[iv-4,it] = np.nanmean(VARz,axis=(1,2))
                data3D[iv,it,nAGL:] = VARz - data1D[iv-4,it].reshape(nz,1,1)
                for iz in range(nAGL):
                    data3D[iv,it,iz] = AGL_lin_anomaly( AGL_lin_interp(VAR,AGL[iz],Zm,ZS) ,AGL[iz],AL,data1D[iv-4,it],ZS)
            else:
                data3D[iv,it,len(AGL):] = AL_lin_interp(VAR,AL,Zm,ZS)
                for iz in range(nAGL):
                    data3D[iv,it,iz] = AGL_lin_interp(VAR,AGL[iz],Zm,ZS)
                
            if iv in [7,8,9]: # Change kg/kg into g/kg for mixing ratios RV,RC,RP
                data3D[iv,it] *= 1000 
                
        # data1D[2,it] = dewpoint( data1D[1,it] * (data1D[0,it]/100000)**(2/7) , np.nanmean(AL_lin_interp(RH,AL,Zm,ZS),axis=(1,2)) ) / (data1D[0,it]/100000)**(2/7)
        data1D[2,it] = dewpoint( data1D[1,it] * (data1D[0,it]/100000)**(2/7) , np.nanmean(np.array(data3D[6,it,len(AGL):],dtype=np.float32)) ) / (data1D[0,it]/100000)**(2/7)
        
        iv=0
        data2D[iv,it] = ZS
        iv+=1; data2D[iv,it] = T[0]-273.15 # near surface temperature °C
        iv+=1; data2D[iv,it] = np.sqrt(U[0]**2+V[0]**2+W[0]**2) # near surface wind m/s
        iv+=1; data2D[iv,it] = np.sqrt(U[0]**2+V[0]**2+W[0]**2) + 4*np.sqrt(TKE[0]) #     
        iv+=1; data2D[iv,it] = np.sqrt( npf2D('DUMMY2D3')**2 + npf2D('DUMMY2D4')**2 ) # momentum flux
        iv+=1; data2D[iv,it] = np.sum((RC+RP)*RHO*DZm.reshape(len(DZm),1,1),axis=0) * (ZTOP-ZS)/ZTOP  # total condensed water kg/m²
        iv+=1; data2D[iv,it] = npf2D('ACPRRSTEP') * 3600 # precipitation rate mm/h
        iv+=1; data2D[iv,it] = npf2D('DUMMY2D1') * 1004.71 * RHO[0]  # sensible heat flux K*m/s  to   W/m²   mutliplying by rho and     Cp = 1004.71 J/K/kg
        iv+=1; data2D[iv,it] = npf2D('DUMMY2D2') * 2.5008e6 * RHO[0] # latent heat flux from kg/kg*m/s  to  W/m²  multipling by rho and Lv = 2.5008e6 J/kg
        
        BLH,CBM,THVBL,WBL,UBLH,VBLH = boundary_layer_diags(Zm,ZS,RC,THV,RHO,W,U,V,deltaTH=0.3)
        iv+=1; data2D[iv,it] = BLH
        iv+=1; data2D[iv,it] = CBM
        iv+=1; data2D[iv,it] = THVBL
        iv+=1; data2D[iv,it] = WBL
        
        # SLH,THVSL,MFSL = surface_layer_diags(Zm,ZS,THV,RHO,W)
        
        print(time.ctime()[-13:-5] ,str(round(time.time()-time0,1))+' s')
        
#%%
k = 2*np.pi*np.fft.rfftfreq(nx,dx)
m = 2*np.pi*np.fft.fftfreq(ny,dx)
kk,mm = np.meshgrid(k,m)
k2 = kk**2+mm**2
del(kk,mm)
def gaussian_filter(periodic_2Darray,lc):
    periodic_2Darray[np.isnan(periodic_2Darray)] = 0
    kc = 2*np.pi/lc
    return np.fft.irfft2(np.exp(-k2/kc**2/2)*np.fft.rfft2(periodic_2Darray) )

@njit()
def TDMAsolver(ac, bc, cc, dc):
    n = len(dc)
    for it in range(1, n):
        mc = ac[it]/bc[it-1]
        bc[it] = bc[it] - mc*cc[it-1] 
        dc[it] = dc[it] - mc*dc[it-1]
    bc[-1] = dc[-1]/bc[-1]
    for il in range(n-2, -1, -1):
        bc[il] = (dc[il]-cc[il]*bc[il+1])/bc[il]
    return bc

k = 2*np.pi*np.fft.rfftfreq(nx, d=dx)
nk = len(k)
kk , _ = np.meshgrid(k,np.zeros(nz))


b_diag = ( kk**2 + (( 2/(z__[2:]-z__[1:-1]) + 2/(z__[1:-1]-z__[:-2])  )/(z__[2:]-z__[:-2])).reshape((nz,1)) )*np.ones((nz,nk),dtype=complex)
a_side = ( -2/(z__[1:-1]-z__[:-2])/(z__[2:]-z__[:-2]) ).reshape((nz,1)) *np.ones((nz,nk),dtype=complex)
c_side = ( -2/(z__[2:]-z__[1:-1])/(z__[2:]-z__[:-2])  ).reshape((nz,1)) *np.ones((nz,nk),dtype=complex)
    
def gaussian_sp_fft_tridiag_filter(T,lc):
    T[np.isnan(T)] = 0
    
    kc2 = (2*np.pi/lc)**2
    # Tfilter = np.copy(T)
    Tf = np.fft.rfft(T)
    # Tf0 = Tf[0]# ; Tfn = Tf[-1]
    S = 2*kc2*Tf[:,:]
    # S[0,:] -= a_side[0]*Tf0
    # S[-1,:] -= a_side[-1]*Tfn
    Tf = TDMAsolver(a_side, 2*kc2 + b_diag , c_side , S)
    # T[1:-1,:] = np.fft.irfft(Tf)
    # return T
    return np.fft.irfft(Tf)

def get_data2D(var,it,axis,i):
    if floatype == np.float16:
        np.float32(data3D[var,it,i+1,:,:])
            
def get_data(var,it,axis,i):
    if put_in_RAM:
        if floatype == np.float16:
            if axis==0:
                return np.float32(data3D[var,it,i+nAGL,:,:])
            elif axis==1:
                return np.float32(data3D[var,it,nAGL:,i,:])
            elif axis==2:
                return np.float32(data3D[var,it,nAGL:,:,i])
        else:
            if axis==0:
                return data3D[var,it,i,:,:]
            elif axis==1:
                return data3D[var,it,:,i,:]
            elif axis==2:
                return data3D[var,it,:,:,i]
        
    else:
        if var==0:
            if axis==0:
                return list_f[it]['WT'][0,nz1+i,ny1:ny2,nx1:nx2].data
            elif axis==1:
                return list_f[it]['WT'][0,nz1:nz2,ny1+i,nx1:nx2].data
            elif axis==2:
                return list_f[it]['WT'][0,nz1:nz2,ny1:ny2,nx1+i].data
        if var==1:
            if axis==0:
                data = list_f[it]['THT'][0,nz1+i,ny1:ny2,nx1:nx2].data * ( 1+0.61*list_f[it]['RVT'][0,nz1+i,ny1:ny2,nx1:nx2].data )
                return data-np.mean(data,axis=(0,1),keepdims=True) # to be replaced by 000.nc mean data
                # return data-np.median(data,axis=(0,1),keepdims=True) # to be replaced by 000.nc mean data
            elif axis==1:
                data = list_f[it]['THT'][0,nz1:nz2,ny1+i,nx1:nx2].data * ( 1+0.61*list_f[it]['RVT'][0,nz1:nz2,ny1+i,nx1:nx2].data )
                return data-np.mean(data,axis=1,keepdims=True) # to be replaced by 000.nc mean data
            elif axis==2:
                data = list_f[it]['THT'][0,nz1:nz2,ny1:ny2,nx1+i].data * ( 1+0.61*list_f[it]['RVT'][0,nz1:nz2,ny1:ny2,nx1+i].data )
                return data-np.mean(data,axis=1,keepdims=True) # to be replaced by 000.nc mean data
            
        if var==2:
            if axis==0:
                data = list_f[it]['RVT'][0,nz1+i,ny1:ny2,nx1:nx2].data
            elif axis==1:
                data = list_f[it]['RVT'][0,nz1:nz2,ny1+i,nx1:nx2].data
            elif axis==2:
                data = list_f[it]['RVT'][0,nz1:nz2,ny1:ny2,nx1+i].data
            return data*1000
        if var==3:
            if axis==0:
                data = list_f[it]['RCT'][0,nz1+i,ny1:ny2,nx1:nx2].data + list_f[it]['RIT'][0,nz1+i,ny1:ny2,nx1:nx2].data
            elif axis==1:
                data = list_f[it]['RCT'][0,nz1:nz2,ny1+i,nx1:nx2].data + list_f[it]['RIT'][0,nz1:nz2,ny1+i,nx1:nx2].data
            elif axis==2:
                data = list_f[it]['RCT'][0,nz1:nz2,ny1:ny2,nx1+i].data + list_f[it]['RIT'][0,nz1:nz2,ny1:ny2,nx1+i].data
            return data*1000
        if var==4:
            if axis==0:
                data = list_f[it]['RRT'][0,nz1+i,ny1:ny2,nx1:nx2].data + list_f[it]['RST'][0,nz1+i,ny1:ny2,nx1:nx2].data + list_f[it]['RGT'][0,nz1+i,ny1:ny2,nx1:nx2].data
            elif axis==1:
                data = list_f[it]['RRT'][0,nz1:nz2,ny1+i,nx1:nx2].data + list_f[it]['RST'][0,nz1:nz2,ny1+i,nx1:nx2].data + list_f[it]['RGT'][0,nz1:nz2,ny1+i,nx1:nx2].data
            elif axis==2:
                data = list_f[it]['RRT'][0,nz1:nz2,ny1:ny2,nx1+i].data + list_f[it]['RST'][0,nz1:nz2,ny1:ny2,nx1+i].data + list_f[it]['RGT'][0,nz1:nz2,ny1:ny2,nx1+i].data
            return data*1000
        
        if var==5:
            if axis==0:
                data = list_f[it]['PABST'][0,nz1+i,ny1:ny2,nx1:nx2].data
                return data-np.mean(data,axis=(0,1),keepdims=True) # to be replaced by 000.nc mean data
            elif axis==1:
                data = list_f[it]['PABST'][0,nz1:nz2,ny1+i,nx1:nx2].data
                return data-np.mean(data,axis=1,keepdims=True) # to be replaced by 000.nc mean data
            elif axis==2:
                data = list_f[it]['PABST'][0,nz1:nz2,ny1:ny2,nx1+i].data
                return data-np.mean(data,axis=1,keepdims=True) # to be replaced by 000.nc mean data
            
        if var==6:
            if axis==0:
                return list_f[it]['UT'][0,nz1+i,ny1:ny2,nx1:nx2].data
            elif axis==1:
                return list_f[it]['UT'][0,nz1:nz2,ny1+i,nx1:nx2].data
            elif axis==2:
                return list_f[it]['UT'][0,nz1:nz2,ny1:ny2,nx1+i].data
        if var==7:
            if axis==0:
                return list_f[it]['VT'][0,nz1+i,ny1:ny2,nx1:nx2].data
            elif axis==1:
                return list_f[it]['VT'][0,nz1:nz2,ny1+i,nx1:nx2].data
            elif axis==2:
                return list_f[it]['VT'][0,nz1:nz2,ny1:ny2,nx1+i].data
            


def regrid_mean(a,n):
    return as_strided(a, shape= tuple(map(lambda x: int(x / n), a.shape)) + (n, n), strides=tuple(map(lambda x: x*n, a.strides)) + a.strides).mean(axis=(2,3))

def get_precip_plot(precip,X,Y,step,threshold=10**-2):
    flat_precip = np.ndarray.flatten(regrid_mean(precip,step))
    select_precip = flat_precip > threshold
    flat_precip = flat_precip[select_precip]
    flat_X = np.ndarray.flatten(regrid_mean(X,step))[select_precip]
    flat_Y = np.ndarray.flatten(regrid_mean(Y,step))[select_precip]
    return flat_X, flat_Y, flat_precip

# Zxx,Zyy = np.meshgrid(x,y)
X,Y = np.meshgrid(x,y)
Yxx,Yzz = np.meshgrid(x,z)
Xyy,Xzz = np.meshgrid(y,z)

def get_hexagonal_flat_coords(x,y,spacing=3,mirror=0):
    ddot = spacing*(x[1]-x[0])
    xf = np.arange(x[0],x[-1]-ddot/2,ddot)
    yf = np.arange(y[0],y[-1],ddot*np.sqrt(3)/2)
    flat_X,flat_Y = np.meshgrid(xf,yf)
    flat_X[mirror::2,:] += ddot/2
    return flat_X.ravel(),flat_Y.ravel()

def get_all_hexagonal_flat_coords(x,y,z,spacings=[5,3,3],mirror=0):
    flat_Zxx,flat_Zyy = get_hexagonal_flat_coords(x,y,spacing=spacings[0],mirror=mirror)
    flat_Yxx,flat_Yzz = get_hexagonal_flat_coords(x,z,spacing=spacings[1],mirror=mirror)
    flat_Xyy,flat_Xzz = get_hexagonal_flat_coords(y,z,spacing=spacings[2],mirror=mirror)
    return flat_Zxx,flat_Zyy,flat_Yxx,flat_Yzz,flat_Xyy,flat_Xzz
flat_Zxx,flat_Zyy,flat_Yxx,flat_Yzz,flat_Xyy,flat_Xzz = get_all_hexagonal_flat_coords(x,y,z,spacings=[nx/70,nx/100,nx/100])
qv_Zxx,qv_Zyy,qv_Yxx,qv_Yzz,qv_Xyy,qv_Xzz = get_all_hexagonal_flat_coords(x,y,z,spacings=[nx/70,nx/100,nx/100],mirror=1)
@njit(parallel=True)
def get_hexagonal_flat_precip(x,y,flat_X,flat_Y,precip):
    n_flat = len(flat_X)
    flat_precip = np.zeros(n_flat)
    for i in prange(n_flat):
        ix = np.argmin(np.abs(x-flat_X[i]))
        iy = np.argmin(np.abs(y-flat_Y[i]))
        flat_precip[i] = precip[iy,ix]
    return flat_precip

def zm2alt(zm,zs):
    return zs + (ZTOP-zs) * zm/ZTOP

#%%
class Player(FuncAnimation):
    def __init__(self,  pv, pv2D, frames=None, init_func=None, fargs=None,save_count=None,animated=False,blit=False,
                 frame_on=True,ticks=True, vert_ratio = 2.3 ,width=15, player_height=0.3, **kwargs): #,width=19.2,
        
        self.fs = 10
        self.clouds_levels,self.clouds_color,self.clouds_lw,self.clouds_alpha = [1e-3],'k',1.,0.7
        self.precip_color,self.precip_alpha = 'k',1.
        self.line_color,self.linestyles = 'grey',['--','-.',':']
        self.quiver_parameters = ('k','mid',10,'xy','xy',0.1,2,3)
        
        self.it = 0
        self.iz = nz//10
        self.iy = ny//2
        self.ix = nx//2
        self.pv = pv
        self.pv2D = pv2D
        self.var= 2
        self.var2D=-1
        
        self.zlimit = [ z_[0] , z_[-1] ]
        self.xlimit = [ x_[0] , x_[-1] ]
        self.ylimit = [ y_[0] , y_[-1] ]
        self.ticks = ticks
        
        self.dhori = ( z_[-1] - z_[0] )*vert_ratio/2
        self.xlim = [ x_[self.ix]-self.dhori , x_[self.ix]+self.dhori ]
        self.ylim = [ y_[self.iy]-self.dhori , y_[self.iy]+self.dhori ]
        
        self.min=0
        self.max=nt-1
        self.runs = ( self.max>2 )
        self.forwards = True
        self.interpolation = False
        self.clouds_contour = True
        self.precip_dots = True
        self.precip_step = 5
        self.streamlines = False
        self.quiver = False
        self.profile = True  # temperature, humidity and wind profile ( ax_T and ax_W )
        self.vcrosss = not(self.profile) # vertical cross section ( ax_X and ax_Y )
        
        vert_fact = 0.8
        hori_ratio = nx/ny
        hori_height = width / (hori_ratio + vert_fact*vert_ratio/2)
        height = player_height + hori_height
        
        self.fig = plt.figure(figsize=(width,height))
        
        mid = hori_height*hori_ratio/width
        top = hori_height / height
        
        playerax_pos = [0.01 , top + 0.1*(1-top)/10 , 0.78 , 0.9*(1-top)]
        self.playerax   = self.fig.add_axes( playerax_pos,  xticks=[],yticks=[],frame_on=frame_on)
        self.setup_playerax(self.playerax , self.pv,self.pv2D)
        
        self.filterax = self.fig.add_axes([0.88,playerax_pos[1] ,0.1, playerax_pos[3] ],  xticks=[],yticks=[],frame_on=frame_on)
        self.lamda_filter = 0.
        self.filter_slider = matplotlib.widgets.Slider(self.filterax, 'Filter',0., 6, valinit=0.,orientation ='horizontal' ,color='darkorange')
        self.filter_text = self.filterax.text(0.5,0.5,'0 km', transform=self.filterax.transAxes,ha='center',va='center')
        self.filter_slider.on_changed(self.set_lamda_filter)
        
        self.ax_CB = self.fig.add_axes([mid+0.1*(1-mid), top - 0.08  , 0.8*(1-mid), 0.04 ],  xticks=[],yticks=[],frame_on=frame_on)
        
        if self.ticks:
            pad_down = 0.04
            pad_left = pad_down*height/width
            self.ax_Z_pos = [pad_left, pad_down, mid-pad_left, top-pad_down]
            self.ax_Y_pos = [mid+pad_left, vert_fact/2*top+pad_down, (1-mid)-pad_left, vert_fact/2*top-pad_down]
            self.ax_X_pos = [mid+pad_left, pad_down, (1-mid)-pad_left, vert_fact/2*top-pad_down]
            self.ax_T_pos = [mid+pad_left, pad_down, (1-mid)/2-pad_left, vert_fact*top-pad_down]
            self.ax_W_pos = [mid+(1-mid)/2+pad_left, pad_down, (1-mid)/2-pad_left, vert_fact*top-pad_down]
            self.ax_Z = self.fig.add_axes(self.ax_Z_pos,frame_on=frame_on)
        else:
            self.ax_Z_pos = [0, 0, mid, top]
            self.ax_Y_pos = [mid, vert_fact/2*top, 1-mid, vert_fact/2*top]
            self.ax_X_pos = [mid, 0, 1-mid, vert_fact/2*top]
            self.ax_T_pos = [mid, 0, (1-mid)/2, vert_fact*top]
            self.ax_W_pos = [mid+(1-mid)/2, 0, (1-mid)/2, vert_fact*top]
            self.ax_Z = self.fig.add_axes(self.ax_Z_pos,  xticks=[],yticks=[],frame_on=frame_on)
            
        self.ax_Z.set_xlim(self.xlimit)
        self.ax_Z.set_ylim(self.ylimit)
        self.ax_Z.set_xlabel('X km')
        self.ax_Z.set_xlabel('Y km')
        self.ax_Z.set_aspect('equal')
        self.ax_Z.set_facecolor((0., 0., 0.))
        self.legend_elements = [ [Line2D([0], [0], color=self.line_color,linestyle=self.linestyles[i], lw=1, label=lb+' plane')] for i,lb in enumerate(["X-Y","X-Z","Y-Z"]) ]
        self.legend_Z = self.ax_Z.legend(handles=self.legend_elements[0],fancybox=True,fontsize=self.fs,loc='upper left', title=str(round(1e3*z[self.iz]))+' m',title_fontsize=2*self.fs)#,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
        
        self.image_Z = self.ax_Z.imshow( get_data(self.var,self.it,0,self.iz) ,norm=self.get_norm(), cmap=self.pv[self.var][3], extent=(x_[0],x_[-1],y_[0],y_[-1]), origin= 'lower', animated=animated,interpolation='none',zorder=1)
        
        self.clouds = [self.ax_Z.contour(x,y,get_data(8,self.it,0,self.iz), levels=self.clouds_levels,colors=self.clouds_color,linewidths=self.clouds_lw,alpha=self.clouds_alpha)]
        flat_precip = get_hexagonal_flat_precip(x,y,flat_Zxx,flat_Zyy,get_data(9,self.it,0,self.iz))
        select_precip = flat_precip > 10**-2
        self.precip_Z = self.ax_Z.scatter(flat_Zxx[select_precip], flat_Zyy[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
        self.precip_X = None
        self.stream_Z = None
        self.stream_X = None
        
        qvc,qvp,qvs,qvsu,qvu,qvw,qvhw,qvhl = self.quiver_parameters
        flat_U = get_hexagonal_flat_precip(x,y,qv_Zxx,qv_Zyy,get_data(0,self.it,0,self.iz))
        flat_V = get_hexagonal_flat_precip(x,y,qv_Zxx,qv_Zyy,get_data(1,self.it,0,self.iz))
        self.quiver_Z = self.ax_Z.quiver(qv_Zxx,qv_Zyy,flat_V,flat_U, color=qvc,pivot=qvp,scale=qvs,scale_units=qvsu,units=qvu,width=qvw,headwidth=qvhw,headlength=qvhl,visible=self.quiver,zorder=2)
        
        # self.add_topography()
        
        self.CB = plt.colorbar(self.image_Z,cax=self.ax_CB,orientation='horizontal',format='%.1f')
        self.CB_fact = 1.5
        self.CB.ax.tick_params(labelsize=self.fs)
        self.CB.set_label(self.pv[self.var][0]+'  ( '+self.pv[self.var][4]+' )',fontsize=1.5*self.fs)
        
        # self.z_text = self.ax_Z.text(*1/10,zmax,"Ra = %.2e"%les_Ra[0],fontsize=18)
        if self.vcrosss: self.setup_vcrosssax()
        if self.profile: self.setup_profileax()
        self.data_update()
        
        self.fig.canvas.mpl_connect('button_press_event',self.on_press)
        
        FuncAnimation.__init__(self,self.fig, self.update, frames=self.play(), 
                                           init_func=init_func, fargs=fargs,blit=blit,
                                           save_count=save_count, **kwargs )
    def setup_vcrosssax(self,animated=False):
        if self.ticks:
            self.ax_Y = self.fig.add_axes(self.ax_Y_pos)
            self.ax_X = self.fig.add_axes(self.ax_X_pos)
        else:
            self.ax_Y = self.fig.add_axes(self.ax_Y_pos,xticks=[],yticks=[])
            self.ax_X = self.fig.add_axes(self.ax_X_pos,xticks=[],yticks=[])
              
        self.ax_Y.set_xlim(self.xlim)
        self.ax_Y.set_ylim(self.zlimit)
        self.ax_X.set_xlim(self.ylim)
        self.ax_X.set_ylim(self.zlimit)
        self.ax_Y.set_aspect('equal')
        self.ax_X.set_aspect('equal')
        self.ax_Y.set_facecolor((0., 0., 0.))
        self.ax_X.set_facecolor((0., 0., 0.))
        self.line_ZX = self.ax_Z.axvline(x[self.ix],color=self.line_color,ls=self.linestyles[2])
        self.line_ZY = self.ax_Z.axhline(y[self.iy],color=self.line_color,ls=self.linestyles[1])
        self.line_YX = self.ax_Y.axvline(x[self.ix],color=self.line_color,ls=self.linestyles[2])
        self.line_YZ = self.ax_Y.axhline(z[self.iz],color=self.line_color,ls=self.linestyles[0])
        self.line_XY = self.ax_X.axvline(y[self.iy],color=self.line_color,ls=self.linestyles[1])
        self.line_XZ = self.ax_X.axhline(z[self.iz],color=self.line_color,ls=self.linestyles[0])
        self.ax_Y.legend(handles=self.legend_elements[1],fancybox=True,fontsize=self.fs,loc='upper left')#,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
        self.ax_X.legend(handles=self.legend_elements[2],fancybox=True,fontsize=self.fs,loc='upper left')#,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
        self.image_Y = self.ax_Y.pcolormesh(x_,z_, get_data(self.var,self.it,1,self.iy) ,norm=self.get_norm(), cmap=self.pv[self.var][3], animated=animated,shading='flat',zorder=1)
        self.image_X = self.ax_X.pcolormesh(y_,z_, get_data(self.var,self.it,2,self.ix) ,norm=self.get_norm(), cmap=self.pv[self.var][3], animated=animated,shading='flat',zorder=1)
        
        qvc,qvp,qvs,qvsu,qvu,qvw,qvhw,qvhl = self.quiver_parameters
        flat_U = get_hexagonal_flat_precip(x,z,qv_Yxx,qv_Yzz,get_data(2,self.it,1,self.iy))
        flat_V = get_hexagonal_flat_precip(x,z,qv_Yxx,qv_Yzz,get_data(1,self.it,1,self.iy))
        self.quiver_Y = self.ax_Y.quiver(qv_Yxx,qv_Yzz,flat_V,flat_U, color=qvc,pivot=qvp,scale=qvs,scale_units=qvsu,units=qvu,width=qvw,headwidth=qvhw,headlength=qvhl,visible=self.quiver,zorder=2)
        flat_U = get_hexagonal_flat_precip(y,z,qv_Xyy,qv_Xzz,get_data(2,self.it,2,self.ix))
        flat_V = get_hexagonal_flat_precip(y,z,qv_Xyy,qv_Xzz,get_data(0,self.it,2,self.ix))
        self.quiver_X = self.ax_X.quiver(qv_Xyy,qv_Xzz,flat_V,flat_U, color=qvc,pivot=qvp,scale=qvs,scale_units=qvsu,units=qvu,width=qvw,headwidth=qvhw,headlength=qvhl,visible=self.quiver,zorder=2)
                
    def setup_profileax(self):
        if self.ticks:
            self.ax_T = self.fig.add_axes(self.ax_T_pos)
            self.ax_W = self.fig.add_axes(self.ax_W_pos)
        else:
            self.ax_T = self.fig.add_axes(self.ax_T_pos,xticks=[],yticks=[])
            self.ax_W = self.fig.add_axes(self.ax_W_pos,xticks=[],yticks=[])
        self.ax_T.set_xlim([300,340])#[270,350])
        self.ax_T.set_ylim(self.zlimit)
        self.ax_W.set_xlim([-5,5])
        self.ax_W.set_ylim(self.zlimit)
        
        
        self.line_TZ = self.ax_T.axhline(z[self.iz],color=self.line_color,ls=self.linestyles[0])
        self.line_WZ = self.ax_W.axhline(z[self.iz],color=self.line_color,ls=self.linestyles[0])
        # self.ax_T.legend(handles=self.legend_elements[3],fancybox=True,fontsize=self.fs,loc='upper left')#,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
        # self.ax_W.legend(handles=self.legend_elements[3],fancybox=True,fontsize=self.fs,loc='upper left')#,bbox_to_anchor=(1, 1), loc=1, borderaxespad=0)
        
        
        self.horimean_TH, = self.ax_T.plot(data1D[1,self.it,:],z,'--r',linewidth=1,alpha=0.5)
        self.horimean_THD, = self.ax_T.plot(data1D[2,self.it,:],z,'--b',linewidth=1,alpha=0.5)
        
        self.profile_Z = zm2alt(Z,ZS[self.iy,self.ix])/1000
        self.profile_marker = self.ax_Z.scatter(x[self.ix],y[self.iy],marker='x',color='k')
        self.profile_TH, = self.ax_T.plot(dataZm[0,self.it,:,self.iy,self.ix],self.profile_Z,'-r')
        self.profile_THD, = self.ax_T.plot(dataZm[1,self.it,:,self.iy,self.ix],self.profile_Z,'-b')
        self.profile_W, = self.ax_W.plot(dataZm[2,self.it,:,self.iy,self.ix],self.profile_Z,'-k')
        
        
    def get_norm(self,norm2D=False):
        if not(norm2D):
            if self.pv[self.var][5]=='lin':
                norm=matplotlib.colors.Normalize(vmin=self.pv[self.var][1], vmax=self.pv[self.var][2])
            elif self.pv[self.var][5]=='log':
                norm=matplotlib.colors.LogNorm(vmin=self.pv[self.var][1], vmax=self.pv[self.var][2],clip=True)
            elif self.pv[self.var][5]=='asinh':
                norm=matplotlib.colors.AsinhNorm(linear_width=self.pv[self.var][6],vmin=self.pv[self.var][1], vmax=self.pv[self.var][2])
        else:
            if self.pv2D[self.var2D][5]=='lin':
                norm=matplotlib.colors.Normalize(vmin=self.pv2D[self.var2D][1], vmax=self.pv2D[self.var2D][2])
            elif self.pv2D[self.var2D][5]=='log':
                norm=matplotlib.colors.LogNorm(vmin=self.pv2D[self.var2D][1], vmax=self.pv2D[self.var2D][2],clip=True)
            elif self.pv2D[self.var2D][5]=='asinh':
                norm=matplotlib.colors.AsinhNorm(linear_width=self.pv2D[self.var2D][6],vmin=self.pv2D[self.var2D][1], vmax=self.pv2D[self.var2D][2])
        return norm
    
    def interpolation_update(self):
        if self.interpolation==True:
            self.image_Z.set_interpolation('bicubic')
            if self.vcrosss:
                self.image_Y.remove() ; self.image_Y = self.ax_Y.pcolormesh(x,z,get_data(self.var,self.it,1,self.iy) ,norm=self.get_norm(), cmap=self.pv[self.var][3],shading='gouraud',zorder=1)
                self.image_X.remove() ; self.image_X = self.ax_X.pcolormesh(y,z,get_data(self.var,self.it,2,self.ix) ,norm=self.get_norm(), cmap=self.pv[self.var][3],shading='gouraud',zorder=1)
        else:
            self.image_Z.set_interpolation('none')
            if self.vcrosss:
                self.image_Y.remove() ; self.image_Y = self.ax_Y.pcolormesh(x_,z_,get_data(self.var,self.it,1,self.iy) ,norm=self.get_norm(), cmap=self.pv[self.var][3],shading='flat',zorder=1)
                self.image_X.remove() ; self.image_X = self.ax_X.pcolormesh(y_,z_,get_data(self.var,self.it,2,self.ix) ,norm=self.get_norm(), cmap=self.pv[self.var][3],shading='flat',zorder=1)
        self.fig.canvas.draw_idle()
    
    def add_topography(self):
        if orog:
            self.isoZS = self.ax_Z.contour(x,y,ZS, levels=[0,1000,2000,3000,4000],linewidths=1.,alpha=0.3,colors=['k'] , zorder=2 )
            plt.clabel(self.isoZS, self.isoZS.levels, inline=True, inline_spacing=0, fmt='%d', fontsize=20/3)
            self.fig.canvas.draw_idle()
        else:
            self.isoZS = None
        
    def remove_topography(self):
        if self.isoZS is not None:
            for tp in self.isoZS.collections:
                tp.remove()
            self.isoZS = None
            self.fig.canvas.draw_idle()
            
    def remove_clouds(self):
        if self.clouds is not None:
            for i in range(len(self.clouds)):
                for tp in self.clouds[i].collections:
                    tp.remove()
            self.clouds = None
        
    def remove_precip(self):
        if self.precip_Z is not None:
            self.precip_Z.remove()
            self.precip_Z = None
        if self.precip_X is not None:
            self.precip_Y.remove()
            self.precip_X.remove()
            self.precip_X = None
            
    def remove_streamlines(self):
        if self.stream_Z is not None: 
            self.stream_Z.lines.remove()
            for ax in [self.ax_Z]:
                for art in ax.get_children():
                    if isinstance(art,matplotlib.patches.FancyArrowPatch):
                        art.remove()
            self.stream_Z = None
        if self.stream_X is not None: 
            self.stream_Y.lines.remove()
            self.stream_X.lines.remove()
            for ax in [self.ax_Y,self.ax_X]:
                for art in ax.get_children():
                    if isinstance(art,matplotlib.patches.FancyArrowPatch):
                        art.remove()
            self.stream_X = None
            
    def data_update(self):
        if self.lamda_filter>0.:
            if self.var2D >= 0:
                self.image_Z.set_data( gaussian_filter(data2D[self.var2D,self.it],self.lamda_filter) )
            else:
                self.image_Z.set_data( gaussian_filter(get_data(self.var,self.it,0,self.iz),self.lamda_filter) )
            if self.vcrosss:
                self.image_Y.set_array( np.ravel(gaussian_sp_fft_tridiag_filter(get_data(self.var,self.it,1,self.iy),self.lamda_filter),order='C') )
                self.image_X.set_array( np.ravel(gaussian_sp_fft_tridiag_filter(get_data(self.var,self.it,2,self.ix),self.lamda_filter),order='C') )
        else:
            if self.var2D >= 0:
                self.image_Z.set_data( data2D[self.var2D,self.it] )
            else:
                self.image_Z.set_data( get_data(self.var,self.it,0,self.iz) )
                
            if self.vcrosss:
                self.image_Y.set_array( np.ravel(get_data(self.var,self.it,1,self.iy),order='C') )
                self.image_X.set_array( np.ravel(get_data(self.var,self.it,2,self.ix),order='C') )
        
        if self.profile:
            self.profile_Z = zm2alt(Z,ZS[self.iy,self.ix]) /1000
            self.horimean_TH.set_data([data1D[1,self.it,:],z])
            self.horimean_THD.set_data([data1D[2,self.it,:],z])
            self.profile_TH.set_data([dataZm[0,self.it,:,self.iy,self.ix],self.profile_Z])
            self.profile_THD.set_data([dataZm[1,self.it,:,self.iy,self.ix],self.profile_Z])
            self.profile_W.set_data([dataZm[2,self.it,:,self.iy,self.ix],self.profile_Z])
            
        if self.clouds_contour:
            self.remove_clouds()
            if self.vcrosss:
                self.clouds = [self.ax_Z.contour(x,y,get_data(8,self.it,0,self.iz), levels=self.clouds_levels,colors=self.clouds_color,linewidths=self.clouds_lw,alpha=self.clouds_alpha),
                              self.ax_Y.contour(x,z,get_data(8,self.it,1,self.iy), levels=self.clouds_levels,colors=self.clouds_color,linewidths=self.clouds_lw,alpha=self.clouds_alpha),
                              self.ax_X.contour(y,z,get_data(8,self.it,2,self.ix), levels=self.clouds_levels,colors=self.clouds_color,linewidths=self.clouds_lw,alpha=self.clouds_alpha)]
            else:
                self.clouds = [self.ax_Z.contour(x,y,get_data(8,self.it,0,self.iz), levels=self.clouds_levels,colors=self.clouds_color,linewidths=self.clouds_lw,alpha=self.clouds_alpha)]
                
        if self.precip_dots:
            self.remove_precip()
            flat_precip = get_hexagonal_flat_precip(x,y,flat_Zxx,flat_Zyy,get_data(9,self.it,0,self.iz))
            select_precip = flat_precip > 10**-2
            self.precip_Z = self.ax_Z.scatter(flat_Zxx[select_precip], flat_Zyy[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
            if self.vcrosss:
                flat_precip = get_hexagonal_flat_precip(x,z,flat_Yxx,flat_Yzz,get_data(9,self.it,1,self.iy))
                select_precip = flat_precip > 10**-2
                self.precip_Y = self.ax_Y.scatter(flat_Yxx[select_precip], flat_Yzz[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
                flat_precip = get_hexagonal_flat_precip(y,z,flat_Xyy,flat_Xzz,get_data(9,self.it,2,self.ix))
                select_precip = flat_precip > 10**-2
                self.precip_X = self.ax_X.scatter(flat_Xyy[select_precip], flat_Xzz[select_precip], s= flat_precip[select_precip]*2, c=self.precip_color,alpha=self.precip_alpha,zorder=100)
            
        if self.streamlines:
            self.remove_streamlines()
            U = get_data(0,self.it,0,self.iz) ; V = get_data(1,self.it,0,self.iz)
            self.stream_Z = self.ax_Z.streamplot(x, y,V,U,density=1., color='k', linewidth=np.sqrt(U**2+V**2)/10) #
            if self.vcrosss:
                U = interp_vertical(get_data(0,self.it,1,self.iy),ios,c0s,c1s) ; V = interp_vertical(get_data(2,self.it,1,self.iy),ios,c0s,c1s)
                self.stream_Y = self.ax_Y.streamplot(x, new_z, U,V ,density=2., color='k', linewidth=np.sqrt(U**2+V**2)/10)
                U = interp_vertical(get_data(1,self.it,2,self.ix),ios,c0s,c1s) ; V = interp_vertical(get_data(2,self.it,2,self.ix),ios,c0s,c1s)
                self.stream_X = self.ax_X.streamplot(y, new_z, U,V,density=2., color='k', linewidth=np.sqrt(U**2+V**2)/10)
            
        if self.quiver:
            flat_U = get_hexagonal_flat_precip(x,y,qv_Zxx,qv_Zyy,get_data(0,self.it,0,self.iz))
            flat_V = get_hexagonal_flat_precip(x,y,qv_Zxx,qv_Zyy,get_data(1,self.it,0,self.iz))
            self.quiver_Z.set_UVC(flat_U,flat_V)
            if self.vcrosss:
                flat_U = get_hexagonal_flat_precip(x,z,qv_Yxx,qv_Yzz,get_data(0,self.it,1,self.iy))
                flat_V = get_hexagonal_flat_precip(x,z,qv_Yxx,qv_Yzz,get_data(2,self.it,1,self.iy))
                self.quiver_Y.set_UVC(flat_U,flat_V)
                flat_U = get_hexagonal_flat_precip(y,z,qv_Xyy,qv_Xzz,get_data(1,self.it,2,self.ix))
                flat_V = get_hexagonal_flat_precip(y,z,qv_Xyy,qv_Xzz,get_data(2,self.it,2,self.ix))
                self.quiver_X.set_UVC(flat_U,flat_V)
        self.fig.canvas.draw_idle()

    def cmap_update(self):
        if self.var2D>=0:
            self.image_Z.set_norm(self.get_norm(True))
            self.image_Z.set_cmap(self.pv2D[self.var2D][3])
            self.CB.set_label(self.pv2D[self.var2D][0]+'  '+self.pv2D[self.var2D][4])
        else:
            self.image_Z.set_norm(self.get_norm())
            self.image_Z.set_cmap(self.pv[self.var][3])
            self.CB.set_label(self.pv[self.var][0]+'  '+self.pv[self.var][4])
        if self.vcrosss:
            self.image_Y.set_norm(self.get_norm())
            self.image_X.set_norm(self.get_norm())
            self.image_Y.set_cmap(self.pv[self.var][3])
            self.image_X.set_cmap(self.pv[self.var][3])
        
    def play(self):
        while self.runs:
            if self.forwards:
                self.it+=1
            else:
                self.it-=1
            if self.it > self.min and self.it < self.max:
                yield self.it
            else:
                self.stop()
                yield self.it

    def start(self):
        self.runs=True
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()
    def backward(self, event=None):
        self.forwards = False
        self.start()
    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()
    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.it > self.min and self.it < self.max:
            self.it = self.it+self.forwards-(not self.forwards)
        elif self.it == self.min and self.forwards:
            self.it+=1
        elif self.it == self.max and not self.forwards:
            self.it-=1
        self.data_update()
        self.slider.set_val(self.it)
        self.fig.canvas.draw_idle()

    def var0(self,event=None):
        self.var = 0 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var1(self,event=None):
        self.var = 1 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var2(self,event=None):
        self.var = 2 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var3(self,event=None):
        self.var = 3 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var4(self,event=None):
        self.var = 4 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var5(self,event=None):
        self.var = 5 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var6(self,event=None):
        self.var = 6 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var7(self,event=None):
        self.var = 7 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var8(self,event=None):
        self.var = 8 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var9(self,event=None):
        self.var = 9 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var10(self,event=None):
        self.var = 10 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    def var11(self,event=None):
        self.var = 11 ; self.var2D = -1 ; self.cmap_update() ; self.data_update()
    
    def var2D0(self,event=None):
        self.var2D = 0 ; self.cmap_update() ; self.data_update()
    def var2D1(self,event=None):
        self.var2D = 1 ; self.cmap_update() ; self.data_update()
    def var2D2(self,event=None):
        self.var2D = 2 ; self.cmap_update() ; self.data_update()
    def var2D3(self,event=None):
        self.var2D = 3 ; self.cmap_update() ; self.data_update()
    def var2D4(self,event=None):
        self.var2D = 4 ; self.cmap_update() ; self.data_update()
    def var2D5(self,event=None):
        self.var2D = 5 ; self.cmap_update() ; self.data_update()
    def var2D6(self,event=None):
        self.var2D = 6 ; self.cmap_update() ; self.data_update()
    def var2D7(self,event=None):
        self.var2D = 7 ; self.cmap_update() ; self.data_update()
    def var2D8(self,event=None):
        self.var2D = 8 ; self.cmap_update() ; self.data_update()
    def var2D9(self,event=None):
        self.var2D = 9 ; self.cmap_update() ; self.data_update()
    def var2D10(self,event=None):
        self.var2D = 10 ; self.cmap_update() ; self.data_update()
    def var2D11(self,event=None):
        self.var2D = 11 ; self.cmap_update() ; self.data_update()
    def var2D12(self,event=None):
        self.var2D = 12 ; self.cmap_update() ; self.data_update()
        
    def setup_playerax(self, playerax, pv, pv2D):
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        self.sliderax = divider.append_axes("right", size="500%", pad=0.07)
        self.button_oneback = matplotlib.widgets.Button(playerax, label='$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, label='$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, label='$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, label='$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, label='$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider = matplotlib.widgets.Slider(self.sliderax, '', self.min, self.max, valinit=self.it, color='deepskyblue')
        self.time_text = self.sliderax.text(0.5,0.5, times[self.it], transform=self.sliderax.transAxes,ha='center',va='center')
        self.slider.on_changed(self.set_pos)

        for v,pvar in enumerate(pv):
            ax = divider.append_axes("right", size="80%", pad=0.05)
            if v==0:
                self.button0 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button0.on_clicked(self.var0)
            if v==1:
                self.button1 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button1.on_clicked(self.var1)
            if v==2:
                self.button2 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2.on_clicked(self.var2)
            if v==3:
                self.button3 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button3.on_clicked(self.var3)
            if v==4:
                self.button4 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button4.on_clicked(self.var4)
            if v==5:
                self.button5 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button5.on_clicked(self.var5)
            if v==6:
                self.button6 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button6.on_clicked(self.var6)
            if v==7:
                self.button7 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button7.on_clicked(self.var7)
            if v==8:
                self.button8 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button8.on_clicked(self.var8)
            if v==9:
                self.button9 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button9.on_clicked(self.var9)
            if v==10:
                self.button10 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button10.on_clicked(self.var10)
            if v==11:
                self.button11 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button11.on_clicked(self.var11)
                
        for v,pvar in enumerate(pv2D):
            ax = divider.append_axes("right", size="80%", pad=0.05)
            if v==0:
                self.button2D0 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D0.on_clicked(self.var2D0)
            if v==1:
                self.button2D1 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D1.on_clicked(self.var2D1)
            if v==2:
                self.button2D2 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D2.on_clicked(self.var2D2)
            if v==3:
                self.button2D3 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D3.on_clicked(self.var2D3)
            if v==4:
                self.button2D4 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D4.on_clicked(self.var2D4)
            if v==5:
                self.button2D5 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D5.on_clicked(self.var2D5)
            if v==6:
                self.button2D6 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D6.on_clicked(self.var2D6)
            if v==7:
                self.button2D7 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D7.on_clicked(self.var2D7)
            if v==8:
                self.button2D8 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D8.on_clicked(self.var2D8)
            if v==9:
                self.button2D9 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D9.on_clicked(self.var2D9)
            if v==10:
                self.button2D10 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D10.on_clicked(self.var2D10)
            if v==11:
                self.button2D11 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D11.on_clicked(self.var2D11)
            if v==12:
                self.button2D12 = matplotlib.widgets.Button(ax, label=pvar[0])
                self.button2D12.on_clicked(self.var2D12)
                
    def set_pos(self,i):
        self.it = int(i)#int(self.slider.val)
        self.data_update()
        self.time_text.set_text(times[self.it])
    def update(self,i):
        self.slider.set_val(i)
    
    def set_lamda_filter(self,l):
        self.lamda_filter = l**3
        if self.lamda_filter<1:
            self.filter_text.set_text('{:.2f} km'.format(self.lamda_filter))
        elif self.lamda_filter<10:
            self.filter_text.set_text('{:.1f} km'.format(self.lamda_filter))
        else:
            self.filter_text.set_text('{:.0f} km'.format(self.lamda_filter))
        self.data_update()
    
    def iz_update(self):
        if self.vcrosss:
            self.line_XZ.set_ydata([z[self.iz]]) 
            self.line_YZ.set_ydata([z[self.iz]])
        if self.profile:
            self.line_TZ.set_ydata([z[self.iz]]) 
            self.line_WZ.set_ydata([z[self.iz]])
        self.legend_Z.set_title(str(round(1e3*z[self.iz]))+' m')
        self.data_update()
        self.fig.canvas.draw_idle()
    def iy_update(self):
        self.ylim = [ y_[self.iy]-self.dhori , y_[self.iy]+self.dhori ]
        if self.vcrosss:
            self.line_ZY.set_ydata([y[self.iy]])
            self.ax_X.set_xlim(self.ylim)
            self.line_XY.set_xdata([y[self.iy]])
        if self.profile:
            self.profile_marker.set_offsets([x[self.ix],y[self.iy]])
        self.data_update()
        self.fig.canvas.draw_idle()
    def ix_update(self):
        self.xlim = [ x_[self.ix]-self.dhori , x_[self.ix]+self.dhori ]
        if self.vcrosss:
            self.line_ZX.set_xdata([x[self.ix]])
            self.ax_Y.set_xlim(self.xlim)
            self.line_YX.set_xdata([x[self.ix]])
        if self.profile:
           self.profile_marker.set_offsets([x[self.ix],y[self.iy]])
        self.data_update()
        self.fig.canvas.draw_idle()
    def on_press(self,event):
        if event.inaxes == self.ax_Z:
            if event.button == 3:
                self.iy = np.argmin(np.abs(y - event.ydata)) ; self.iy_update()
            if event.button == 1:
                if self.profile: self.iy = np.argmin(np.abs(y - event.ydata))
                self.ix = np.argmin(np.abs(x - event.xdata)) ; self.ix_update()
                
        if self.vcrosss:
            if event.inaxes == self.ax_Y:
                if event.button == 3:
                    self.iz = np.argmin(np.abs(z - event.ydata)) ; self.iz_update()
                if event.button == 1:
                    self.ix = np.argmin(np.abs(x - event.xdata)) ; self.ix_update()
            if event.inaxes == self.ax_X:
                if event.button == 3:
                    self.iz = np.argmin(np.abs(z - event.ydata)) ; self.iz_update()
                if event.button == 1:
                    self.iy = np.argmin(np.abs(y - event.xdata)) ; self.iy_update()
        if self.profile:
            if event.inaxes == self.ax_T or event.inaxes == self.ax_W:
                if event.button == 3:
                    self.iz = np.argmin(np.abs(z - event.ydata)) ; self.iz_update()
        if event.inaxes == self.ax_CB:
            fact = 0.
            if event.button == 3: fact = self.CB_fact
            if event.button == 1: fact = 1./self.CB_fact
            if fact > 0:
                if self.var2D >= 0:
                    self.pv2D[self.var2D][1] *= fact
                    self.pv2D[self.var2D][2] *= fact
                else:
                    self.pv[self.var][1] *= fact
                    self.pv[self.var][2] *= fact
                self.cmap_update() ; self.fig.canvas.draw_idle()

# Wcm = 'seismic'
# Wcm = mcolors.LinearSegmentedColormap.from_list('seismic2',['k','b','c','w','y','r','k'])
# Wcm = mcolors.LinearSegmentedColormap.from_list('seismic2',[(0.2,0.2,0.2),(0,0,1),(0,1,1),(1,1,1),(1,1,0),(1,0,0),(0.2,0.2,0.2)])
Wcm = mcolors.LinearSegmentedColormap.from_list('seismic2',[(0,(0.2,0.2,0.2)),(0.25,(0,0,1)),(0.4,(0,1,1)),(0.5,(1,1,1)),(0.6,(1,1,0)),(0.75,(1,0,0)),(1,(0.2,0.2,0.2))])
Rcm = 'YlGnBu'

Wmax = 10 # m/s  
Wnorm = 'lin' # 'asinh'
Wasinh = 5
pv3D = [['u',-Wmax,Wmax,Wcm,'m/s',Wnorm,Wasinh],['v',-Wmax,Wmax,Wcm,'m/s',Wnorm,Wasinh],['w',-Wmax/2,Wmax/2,Wcm,'m/s',Wnorm,Wasinh]]
pv3D += [['$TKE$',1e-1,1,'Reds','J/kg','lin']]
# pv3D += [['$TKE$',1e-4,10,'Reds','J/kg','log']]
pv3D += [['$p^{\prime}$',-100,100,'seismic','Pa','lin',5],['$θ_v^{\prime}$',-2,2, 'RdBu_r','K','lin',0.15]]
pv3D += [['$RH$',0,100,'rainbow_r','%','lin']]
pv3D += [['$r_v$',0,20,Rcm,'g/kg','lin'], ['$r_c$',0,1,Rcm,'g/kg','lin'],['$r_p$',0,1,Rcm,'g/kg','lin']]       
pv3D += [['$TR1$',1e-4,10,'Greens','','log']]

ZScm = matplotlib.colors.LinearSegmentedColormap.from_list("", ["green","yellow","saddlebrown","grey","w"])
        
pv2D = [['$z_s$',0,np.max(ZS),ZScm,'m','lin']]
pv2D += [['$T_s$',0,40,'rainbow','°C','lin']]
pv2D += [['$W_s$',0,Wmax,'rainbow','m/s','lin'],['$G_s$',0,Wmax,'rainbow','m/s','lin']]
pv2D += [['$w·u_h^{\prime}$',0,0.2,'rainbow','','lin']]
pv2D += [['TCW',1e-4,10,'Blues','kg/m²','log'],['$PR_s$',0,100,'Blues','mm/h','lin']]
pv2D += [['H',-1000,1000,'seismic','W/m²','lin'], ['LE',-1000,1000,'seismic','W/m²','lin']]      
pv2D += [['$z_{top}^{BL}$',0,5000,'rainbow','m','lin'],['CB',0,1,'Blues','','lin'],['$θ_v^{BL}$',300,320,'rainbow','K','lin'],['$w^{BL}$',-5,5,Wcm,'m/s','lin']]


ani = Player(pv3D,pv2D,interval=10,ticks=True,save_count=nt)

class AGL_tool(ToolBase):
    # default_keymap = 'control'
    description = 'Show variables at Above Ground Levels (AGL) instead of altitude levels'
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        self.izAGL = 0
        super().__init__(*args, **kwargs)
    def trigger(self,*args):
        self.izAGL = (self.izAGL+1)%nAGL
        self.ani.iz = self.izAGL-nAGL
        self.ani.legend_Z.set_title(str(round(AGL[self.izAGL]))+' m AGL')
        self.ani.data_update()
ani.fig.canvas.manager.toolmanager.add_tool('AGL', AGL_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('AGL', 'animation')

class Topography_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off topographic isolines'
    default_toggled = True
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.topography = True
        self.ani.add_topography()
    def disable(self, *args):
        self.ani.topography= False
        self.ani.remove_topography()
ani.fig.canvas.manager.toolmanager.add_tool('Topography', Topography_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Topography', 'animation')

class Interpolation_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off interpolation ( bicubic for X-Y & gouraud for X-Z and Y-Z )'
    default_toggled = False
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.interpolation= True
        self.ani.interpolation_update()
    def disable(self, *args):
        self.ani.interpolation= False
        self.ani.interpolation_update()
ani.fig.canvas.manager.toolmanager.add_tool('Interpolation', Interpolation_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Interpolation', 'animation')

class Clouds_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off clouds contour ( 0.001 g/kg )'
    default_toggled = True
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.clouds_contour = True
        self.ani.data_update()
    def disable(self, *args):
        self.ani.clouds_contour = False
        self.ani.remove_clouds()
        self.ani.fig.canvas.draw_idle()
ani.fig.canvas.manager.toolmanager.add_tool('Clouds', Clouds_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Clouds', 'animation')

class Precip_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off precipitation dots ( > 0.1 g/kg )'
    default_toggled = True
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.precip_dots = True
        self.ani.data_update()
    def disable(self, *args):
        self.ani.precip_dots = False
        self.ani.remove_precip()
        self.ani.fig.canvas.draw_idle()
ani.fig.canvas.manager.toolmanager.add_tool('Precipitations', Precip_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Precipitations', 'animation')

class Streamlines_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off streamlines ( m/s )'
    default_toggled = False
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.streamlines = True
        self.ani.data_update()
    def disable(self, *args):
        self.ani.streamlines = False
        self.ani.remove_streamlines()
        self.ani.fig.canvas.draw_idle()
ani.fig.canvas.manager.toolmanager.add_tool('Streamlines', Streamlines_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Streamlines', 'animation')

class Quiver_tool(ToolToggleBase):
    # default_keymap = 'control'
    description = 'Turn on/off Quiver ( m/s )'
    default_toggled = False
    def __init__(self,*args, ani, **kwargs):
        self.ani = ani
        super().__init__(*args, **kwargs)
    def enable(self,*args):
        self.ani.quiver = True
        self.ani.quiver_Z.set_visible(True)
        self.ani.quiver_Y.set_visible(True)
        self.ani.quiver_X.set_visible(True)
        self.ani.data_update()
    def disable(self, *args):
        self.ani.quiver = False
        self.ani.quiver_Z.set_visible(False)
        self.ani.quiver_Y.set_visible(False)
        self.ani.quiver_X.set_visible(False)
        self.ani.fig.canvas.draw_idle()
ani.fig.canvas.manager.toolmanager.add_tool('Quiver', Quiver_tool,ani=ani)
ani.fig.canvas.manager.toolbar.add_tool('Quiver', 'animation')
    