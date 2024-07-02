#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D
import skgstat
from reader import Reader
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from datetime import datetime
from datetime import date as date_creator
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from numba import njit
import pandas as pd
import os
from numba import jit,njit
from tqdm import tqdm
import math
import matplotlib.ticker as mticker
import xarray as xr
from tqdm import tqdm
import string

get_ipython().run_line_magic('matplotlib', 'inline')

savefig_path = 'T:/C2H/STAGES/Wiki_glide/Figures/'

@njit
def find_closest_neighbour_inside_tab(lat_l, lon_l, lat, lon):
    h,k = np.shape(lat_l)
    dist = np.ones((h,k))
    for i in range(h):
        for j in range(k):
            dist[i,j] = (lat_l[i,j] - lat)**2 + (lon_l[i,j] - lon)**2
    return np.argmin(dist)//h,np.argmin(dist)%k

@njit
def find_closest_neighbour_inside_list(lat_l, lon_l, lat, lon):
    dist = np.arange(len(lat_l))
    for i in range(len(lat_l)):
        dist[i] = (lat_l[i] - lat)**2 + (lon_l[i] - lon)**2
    return np.argmin(dist)

def select_inside_xarray(xarray,img_extent,time_step,iteration,type):
    
    if type == 'simu' :
        selected_events = xarray.where(
            (xarray['lat'] >= img_extent[2]) & (xarray['lat'] <= img_extent[3]) &
            (xarray['lon'] >= img_extent[0]) & (xarray['lon'] <= img_extent[1]) &
            (xarray['iteration'] >= iteration),
            drop=True
        )
        
    elif type == 'traces' :
        selected_events = xarray.where(
            (xarray['latitude_inf'] >= img_extent[2]) & (xarray['latitude_inf'] <= img_extent[3]) &
            (xarray['latitude_sup'] >= img_extent[2]) & (xarray['latitude_sup'] <= img_extent[3]) &
            (xarray['longitude_sup'] >= img_extent[0]) & (xarray['longitude_sup'] <= img_extent[1]) &
            (xarray['longitude_inf'] >= img_extent[0]) & (xarray['longitude_inf'] <= img_extent[1]) &
            (xarray['time_stemp'] == time_step),
            drop=True
        )       

    elif type == 'comp' :
        selected_events = xarray.where(
            (xarray['iteration'] == iteration),
            drop=True
        )

    elif type == 'comp_topo' :
        selected_events = xarray.where(
            (xarray['iteration_i'] == iteration),
            drop=True
        ) 

    elif type == 'comp_T':
        selected_events = xarray.where(
            (xarray['iteration_T'] == iteration),
            drop=True
        ) 

    elif type == 'speed' :
        selected_events = xarray.where(
            (xarray['latitude'] >= img_extent[2]) & (xarray['latitude'] <= img_extent[3]) &
            (xarray['longitude'] >= img_extent[0]) & (xarray['longitude'] <= img_extent[1]) &
            (xarray['hours'] == time_step),
            drop=True)
            
    return selected_events

from scipy.interpolate import RegularGridInterpolator

def potential_temp(temp_i,press_i,alt_i,type):
    gamma = 2/7
    rho = 1
    g = 9.81
    p0 = 100000
    
    if type == 'net':
        if np.isnan(temp_i) == False and np.isnan(alt_i) == False :
            press_net = p0-rho*g*alt_i
            t_pot=(temp_i+273.15)*(press_net/p0)**gamma - 273.15
        else :
            t_pot = np.nan
    
    elif type == 'mf':
        
        if np.isnan(temp_i) == False :
            if np.isnan(press_i) == False :
                t_pot = temp_i*(press_i/p0)**gamma - 273.15
            elif np.isnan(alt_i) == False :
                press_net = p0-rho*g*alt_i
                t_pot=(temp_i)*(press_net/p0)**gamma - 273.15
            else :
                t_pot = np.nan
        else :
            t_pot = np.nan 
    return t_pot
        

def select_inside_pdframe_2(df,iteration,extent,type):
    if type == 'net':
        selected_pd = df[(df['lat'] >=extent[2]) & (df['lat'] <= extent[3]) &
              (df['lon'] >= extent[0]) & (df['lon'] <= extent[1]) &
            (df['rawtime'] == 10*60+6*60*60+iteration*10*60)]

    elif type == 'mf':
        selected_pd = df[(df['lat'] >=extent[2]) & (df['lat'] <= extent[3]) &
              (df['lon'] >= extent[0]) & (df['lon'] <= extent[1]) &
            (df['rawtime'] == 10*60+6*60*60+iteration*10*60)]

    return selected_pd

def select_inside_pdframe(df,iteration,extent,type):
    if type == 'net':
        selected_pd = df[(df['lat'] >=extent[2]) & (df['lat'] <= extent[3]) &
              (df['lon'] >= extent[0]) & (df['lon'] <= extent[1]) &
            (df['rawtime'] == iteration*3600)]

    elif type == 'mf':
        selected_pd = df[(df['lat'] >=extent[2]) & (df['lat'] <= extent[3]) &
              (df['lon'] >= extent[0]) & (df['lon'] <= extent[1]) &
            (df['rawtime'] == iteration*3600)]

    return selected_pd

def simple_idw(x, y, z, xi, yi):
    dist = distance_matrix(x,y, xi,yi)

    # In IDW, weights are 1 / distance
    weights = 1.0 / dist

    # Make weights sum to one
    weights /= weights.sum(axis=0)

    # Multiply the weights for each interpolated point by all observed Z-values
    zi = np.dot(weights.T, z)
    return zi


def distance_matrix(x0, y0, x1, y1):
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    # Make a distance matrix between pairwise observations
    # Note: from <http://stackoverflow.com/questions/1871536>
    # (Yay for ufuncs!)
    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])

    return np.hypot(d0, d1)

@njit
def find_indice_min_dist(liste_lat,liste_lon,latitude,longitude):
    n = len(liste_lat)
    res = np.zeros((1,n))[0]
    for i in range(n):
            res[i] = np.sqrt((liste_lat[i]-latitude)**2 + (liste_lon[i]-longitude)**2)
    return np.argmin(res)

@njit
def compute_press(alt_i):
    rho = 1
    g = 9.81
    p0 = 100000
    press_net = p0-rho*g*alt_i
    return press_net

def compute_dist(lat1,lon1,lat2,lon2,rad=True):
    if not(rad):
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)
    # approximate radius of earth in m
    R = 6373_000.0
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * ca

