#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 22:44:17 2024

@author: philippotn
"""
from numba import njit,prange
import numpy as np
floatype = np.float32
intype = np.int32


@njit(parallel=True)
def RG_lin_interp(VAR,Z,Zm): # Regular Grid linear interpolation
    # VAR (3D) =  VARiable defined on Zm altitude levels
    # Z (1D) =  altitude levels on which new_VAR in interpolated
    # Zm (1D) = model altitude levels
    nzVAR,nx,ny = np.shape(VAR)
    nz = np.shape(Z)
    new_VAR = np.zeros((nz,nx,ny),dtype=VAR.dtype)
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                l = 0
                while Zm[l+1]<Z[k]:
                    l+=1
                if l<nzVAR-1:
                    dZ = Zm[l+1]-Zm[l]
                    new_VAR[k,i,j] = ( VAR[l,i,j]*(Zm[l+1]-Z[k]) + VAR[l+1,i,j]*(Z[k]-Zm[l]) )/dZ
    return new_VAR

@njit(parallel=True)
def AL_lin_interp(VAR,AL,Zm,ZS): # Altitude Levels linear interpolation
    # VAR (3D) =  VARiable defined on Zm levels with Gal-Chen and Somerville terrain-following coordinates
    # AL (1D) =  altitude levels on which VAL in interpolated
    # Zm (1D) = terrain-following model levels
    # ZS (2D) = surface altitude
    nzVAR,ny,nx = np.shape(VAR)
    nz, = np.shape(AL)
    ZTOP = (Zm[-1]+Zm[-2])/2
    VAL = np.full((nz,ny,nx), np.nan,dtype=VAR.dtype)
    for j in prange(ny):
        for i in range(nx):
            zs = ZS[j,i]
            l = 0
            for k in range(nz):
                if AL[k]>zs:
                    zm = ZTOP * (AL[k]-zs) / (ZTOP-zs)
                    while Zm[l+1]<zm:
                        l+=1
                    if l<nzVAR-1:
                        dZ = Zm[l+1]-Zm[l]
                        VAL[k,j,i] = ( VAR[l,j,i]*(Zm[l+1]-zm) + VAR[l+1,j,i]*(zm-Zm[l]) )/dZ
    return VAL
    
@njit(parallel=True)
def AGL_lin_interp(VAR,AGL,Zm,ZS): # Above Ground Level linear interpolation
    # VAR (3D) =  VARiable defined on Zm levels with Gal-Chen and Somerville terrain-following coordinates
    # AGL =  above ground level on which VAGL in interpolated
    # Zm (1D) = terrain-following model levels
    # ZS (2D) = surface altitude
    nzVAR,ny,nx = np.shape(VAR)
    ZTOP = (Zm[-1]+Zm[-2])/2
    VAGL = np.zeros((ny,nx), dtype=VAR.dtype)
    for j in prange(ny):
        for i in range(nx):
            zm = ZTOP * AGL / (ZTOP-ZS[j,i])
            l = 0
            while Zm[l+1]<zm:
                l+=1
            if l<nzVAR-1:
                dZ = Zm[l+1]-Zm[l]
                VAGL[j,i] = ( VAR[l,j,i]*(Zm[l+1]-zm) + VAR[l+1,j,i]*(zm-Zm[l]) )/dZ
    return VAGL

@njit(parallel=True)
def AGL_lin_anomaly(VAGL,AGL,z,mean,ZS): 
    # VAGL (2D) is obtained with AGL_lin_interp
    # mean (1D) is nanmean(axis=(1,2)) of VAL obtained with AL_lin_interp
    VAGLA = np.copy(VAGL)
    ny,nx = np.shape(VAGLA)
    for j in prange(ny):
        for i in range(nx):
            zVAGL = ZS[j,i] + AGL
            k = 0
            while zVAGL>z[k+1] or np.isnan(mean[k]):
                k+=1
            dz = z[k+1]-z[k]
            VAGLA[j,i] -= ( mean[k]*(z[k+1]-zVAGL) + mean[k+1]*(zVAGL-z[k]) )/dz
    return VAGLA

@njit(parallel=True)
def MO_flux(MO,U,V,RHO,RV,RT,TH):
    nz,ny,nx = np.shape(U)
    flux =  np.zeros((4,nz),dtype=U.dtype)
    for j in prange(ny):
        for i in range(nx):
            i0 = i-1 if i>=1 else nx-1
            j0 = j-1 if j>=1 else ny-1
            for j_,i_,wind in [ (j,i0,U[:,j,i]) , (j0,i,V[:,j,i]) ]:
                if MO[j_,i_] != MO[j,i]:
                    sign = 1. if MO[j,i] else -1.
                    for k in range(nz):
                        massflux = sign*wind[k]*(RHO[k,j_,i_]+RHO[k,j,i])/2
                        flux[0,k] += massflux
                        flux[1,k] += massflux*(RV[k,j_,i_]+RV[k,j,i])/2 # vapor flux
                        flux[2,k] += massflux*(RT[k,j_,i_]+RT[k,j,i])/2 # water flux
                        flux[3,k] += massflux*(TH[k,j_,i_]+TH[k,j,i])/2 # heat flux
    return flux

@njit(parallel=True)
def boundary_layer_diags(Zm,ZS,rc,thv,rho,W,U,V,deltaTH=0.3):
    nz,ny,nx = np.shape(rc)
    BLH = np.full((ny,nx),np.nan,dtype=floatype)
    THVBL = np.zeros((ny,nx),dtype=floatype) 
    WBL = np.zeros((ny,nx),dtype=floatype)
    UBLH = np.zeros((ny,nx),dtype=floatype)
    VBLH = np.zeros((ny,nx),dtype=floatype)
    cloud_base_mask = np.zeros((ny,nx),dtype=intype)
    ZTOP = (Zm[-1]+Zm[-2])/2
    for j in prange(ny):
        for i in range(nx):
            mean_thv = thv[0,j,i]
            mean_w = W[0,j,i]
            mass = Zm[0]*rho[0,j,i]
            for h in range(0,nz-1):
                if rc[h,j,i]>1e-6:
                    BLH[j,i] = ZS[j,i] + (ZTOP-ZS[j,i]) * Zm[h]/ZTOP
                    cloud_base_mask[j,i] = 1
                    break
                if thv[h,j,i] > mean_thv + deltaTH:
                    BLH[j,i] = ZS[j,i] + (ZTOP-ZS[j,i]) * Zm[h]/ZTOP
                    break
                else:
                    layer_mass = (rho[h+1,j,i]+rho[h,j,i])/2 * (Zm[h+1]-Zm[h])
                    mean_thv = ( mean_thv*mass + (thv[h+1,j,i]+thv[h,j,i])/2 * layer_mass ) / ( mass+layer_mass)
                    mean_w = ( mean_w*mass + W[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                    mass += layer_mass
            THVBL[j,i] = mean_thv
            WBL[j,i] = mean_w
            UBLH[j,i] = U[h,j,i]
            VBLH[j,i] = V[h,j,i]
    return BLH,cloud_base_mask,THVBL,WBL,UBLH,VBLH

@njit(parallel=True)
def surface_layer_diags(Zm,ZS,thv,rho,W):
    nz,ny,nx = np.shape(thv)
    SLH = np.full((ny,nx),np.nan,dtype=floatype)
    THVSL = np.zeros((ny,nx),dtype=floatype)
    WSL = np.zeros((ny,nx),dtype=floatype)
    MFSL = np.zeros((ny,nx),dtype=floatype)
    ZTOP = (Zm[-1]+Zm[-2])/2
    for j in prange(ny):
        for i in range(nx):
            mean_thv = thv[0,j,i]
            mean_w = W[0,j,i]
            mass = Zm[0]*rho[0,j,i]
            for h in range(0,nz-1):
                if thv[h+1,j,i] > thv[h,j,i]:
                    SLH[j,i] = ZS[j,i] + (ZTOP-ZS[j,i]) * Zm[h]/ZTOP
                    break
                else:
                    layer_mass = (rho[h+1,j,i]+rho[h,j,i])/2 * (Zm[h+1]-Zm[h])
                    mean_thv = ( mean_thv*mass + (thv[h+1,j,i]+thv[h,j,i])/2 * layer_mass ) / ( mass+layer_mass)
                    mean_w = ( mean_w*mass + W[h+1,j,i] * layer_mass ) / ( mass+layer_mass)
                    mass += layer_mass
            THVSL[j,i] = mean_thv
            WSL[j,i] = mean_w
            MFSL[j,i] = mean_w*mass/SLH[j,i]
    return SLH,THVSL,WSL

@njit()
def horizontal_convergence(U,V,dx):
    CONV = np.zeros_like(U)
    CONV[:,:-1,:] -= ( V[:,1:,:] - V[:,:-1,:] )/dx
    CONV[:,-1,:] -= ( V[:,0,:] - V[:,-1,:] )/dx
    CONV[:,:,:-1] -= ( U[:,:,1:] - U[:,:,:-1] )/dx
    CONV[:,:,-1] -= ( U[:,:,0] - U[:,:,-1] )/dx
    return CONV