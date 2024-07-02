#!/usr/bin/env python
# coding: utf-8

# In[ ]:


@njit()
def arange_netcdf(simu_array,it,n,m):

    ZS = np.ravel(simu_array[0,it])
    TS = np.ravel(simu_array[1,it])
    BLH = np.ravel(simu_array[9,it])
    W_max = np.ravel(simu_array[-1,it])
    iteration = np.ravel(np.ones((n,m))*it)

    return ZS,TS,BLH,W_max,iteration


def to_netcdf_simu(coords_netcdf, simu_array, jour, heure_debut, save_name): 

    vars,its,n,m = np.shape(simu_array)

    latitudes = np.ravel(coords_netcdf['latitude'].values[1:-1,1:-1])
    longitudes = np.ravel(coords_netcdf['longitude'].values[1:-1,1:-1])
    
    latitudes_i = latitudes.copy()
    longitudes_i = longitudes.copy()
    
    var = 0
    ZS_i = np.ravel(simu_array[var,0])
    var = 1
    TS_i= np.ravel(simu_array[var,0])
    var = 9
    BLH_i = np.ravel(simu_array[var,0])
    var = -1
    W_max_i = np.ravel(simu_array[var,0])
    heure_i = np.ravel(np.ones((n,m))*heure_debut)

    #variables = ['ZS','TS','UVWS','UVWTKES','MOMFLUX','TCONDW','PRECR','SENSHF','LATHF','BLH','CBM','THVBL','WBL']

    for it in range(heure_debut+1,heure_debut+its): 

        ZS,TS,BLH,W_max,iteration = arange_netcdf(simu_array,it,n,m)

        ZS_i = np.concatenate((ZS_i, ZS), axis=None)
        TS_i = np.concatenate((TS_i, TS), axis=None)
        BLH_i = np.concatenate((BLH_i, BLH), axis=None)
        W_max_i = np.concatenate((W_max_i, W_max), axis=None)
        latitudes_i = np.concatenate((latitudes_i, latitudes), axis=None)
        longitudes_i = np.concatenate((longitudes_i, longitudes), axis=None)
        heure_i = np.concatenate((heure_i, np.ravel(np.ones((n,m))*it)), axis=None)
        print(len(ZS_i),len(longitudes_i))
        
        print(it, ' ok')
        
    num_events = len(longitudes_i)
    num_data_points = 8
    Simu_netcdf = xr.Dataset(
        {
            'ZS': (['event'], ZS_i.astype(np.float32)),
            'W_max': (['event'], W_max_i.astype(np.float32)),
            'TS': (['event'], TS_i.astype(np.float32)),
            'BLH': (['event'], BLH_i.astype(np.float32)),
            'lat': (['event'], latitudes_i.astype(np.float32)),
            'lon': (['event'], longitudes_i.astype(np.float32)),
            'heure': (['event'], heure_i.astype(int))
        },
        coords={
            'event': np.arange(num_events)
        },
    )
    
    Simu_netcdf.to_netcdf('T:/C2H/STAGES/LEO_BARROIS/Netcdfffs/'+save_name + '.nc')

### Glide 

@njit()
def BHL_arange(lon_inf,lon_sup,lat_inf,lat_sup,PBLH_array):

    lon_sup = np.ravel(lon_sup)
    lon_inf = np.ravel(lon_inf)
    lat_sup = np.ravel(lat_sup)
    lat_inf = np.ravel(lat_inf)
    PBLH = np.ravel(PBLH_array[:-1,:-1])

    lon_sup = lon_sup[PBLH != np.nan]
    lon_inf = lon_inf[PBLH != np.nan]
    lat_sup = lat_sup[PBLH != np.nan]
    lat_inf = lat_inf[PBLH != np.nan]
    PBLH = PBLH[PBLH != np.nan]
    
    return lat_inf,lat_sup,lon_inf,lon_sup,PBLH

@njit()
def W_arange(lon,lat,W_array):

    lon_w = np.ravel(lon)
    lat_w = np.ravel(lat)
    max_speed = np.ravel(W_array)

    values = max_speed[max_speed > -1000]
    percenth = np.nanpercentile(values,98)
    percentl = np.nanpercentile(values,1)

    lon_w = lon_w[(max_speed > percentl) & (max_speed < percenth)]
    lat_w = lat_w[(max_speed > percentl) & (max_speed < percenth)]
    max_speed = max_speed[(max_speed > percentl) & (max_speed < percenth)]
    
    return lon_w,lat_w,max_speed
    
def to_netcdf_glide(day,save_name,type):

    img_extent = (4.7942, 8.1545, 43.3545, 46.6707)

    time_stemps = np.array([[i,(i+1)] for i in range(10,19)])
    its_gliders = len(time_stemps)

    if type == 'BLH' :

        nlon,nlat = 50,50
        lons = np.linspace(img_extent[0],img_extent[1],nlon)
        lats = np.linspace(img_extent[2],img_extent[3],nlat)
    
        lon_sup,lat_sup = np.meshgrid(lons[1:],lats[1:])
        lon_inf,lat_inf = np.meshgrid(lons[:-1],lats[:-1])
    
        latitude_inf_i = np.zeros((1,1))[0]
        latitude_sup_i = np.zeros((1,1))[0]
        lonitude_inf_i = np.zeros((1,1))[0]
        longitude_sup_i = np.zeros((1,1))[0]
        PBLH_i = np.zeros((1,1))[0]
        time_stemp_of_flight_i = np.zeros((1,1))[0]
    
        for it in tqdm(range(its_gliders)):
            
            time_stemp = time_stemps[it]
            PBLH_array = np.load('T:/C2H/STAGES/LEO_BARROIS/ndarray/maximum_height_map/IGC_'+str(day) +'-08-2023/both/1h/both_'+str(time_stemp[0])+'_'+str(time_stemp[1]) +'_large.npy')
    
            latitude_inf,latitude_sup,lonitude_inf,longitude_sup,PBLH = BHL_arange(lon_inf,lon_sup,lat_inf,lat_sup,PBLH_array)
    
            latitude_inf_i = np.concatenate((latitude_inf_i,latitude_inf), axis = 0)
            latitude_sup_i = np.concatenate((latitude_sup_i,latitude_sup), axis = 0)
            lonitude_inf_i = np.concatenate((lonitude_inf_i,lonitude_inf), axis = 0)
            longitude_sup_i = np.concatenate((longitude_sup_i,longitude_sup), axis = 0)
            
            PBLH_i = np.concatenate((PBLH_i,PBLH), axis = 0)
            time_stemp_of_flight_i = np.concatenate((time_stemp_of_flight_i,np.ones((1,len(PBLH)))[0]*time_stemp[0]), axis = 0)
                        
        num_events = len(latitude_inf_i)-1
        num_data_points = 7
        PBLH_glide = xr.Dataset(
            {
                'latitude_inf': (['event'], latitude_inf_i[1:].astype(np.float32)),
                'latitude_sup': (['event'], latitude_sup_i[1:].astype(np.float32)),
                'longitude_inf': (['event'], lonitude_inf_i[1:].astype(np.float32)),
                'longitude_sup': (['event'], longitude_sup_i[1:].astype(np.float32)),
                'PBLH': (['event'], PBLH_i[1:].astype(np.float32)),
                'heure': (['event'], time_stemp_of_flight_i[1:].astype(np.float32))
            },
            coords={
                'event': np.arange(num_events)
            },
        )
        
        PBLH_glide.to_netcdf('T:/C2H/STAGES/LEO_BARROIS/Netcdf_new/'+save_name + '.nc')

    if type == 'W_max' :

        nlon,nlat = 500,500
        lons = np.linspace(img_extent[0],img_extent[1],nlon)
        lats = np.linspace(img_extent[2],img_extent[3],nlat)

        llon,llat = np.meshgrid(lons,lats)

        lon_i = np.zeros((1,1))[0]
        lat_i = np.zeros((1,1))[0]
        max_speed_i = np.zeros((1,1))[0]
        time_stemp_of_flight_i = np.zeros((1,1))[0]

        for it in tqdm(range(its_gliders)):

            time_stemp = time_stemps[it]
            W_array = np.load('T:/C2H/STAGES/LEO_BARROIS/ndarray/maximum_speed_map/'+'IGC_'+str(day) +'-08-2023/both_1h/'+str(time_stemp[0])+'_'+str(time_stemp[1]) +'.npy')
            
            lon,lat,max_speed = W_arange(llon,llat,W_array)
            lon_i = np.concatenate((lon_i,lon), axis = 0)
            lat_i = np.concatenate((lat_i,lat), axis = 0)
            max_speed_i = np.concatenate((max_speed_i,max_speed), axis = 0)
            time_stemp_of_flight_i = np.concatenate((time_stemp_of_flight_i,np.ones((1,len(max_speed)))[0]*time_stemp[0]), axis = 0)

        num_events = len(time_stemp_of_flight_i)-1
        num_data_points = 4
        W_glide = xr.Dataset(
            {
                'lon': (['event'], lon_i[1:].astype(np.float32)),
                'lat': (['event'], lat_i[1:].astype(np.float32)),
                'max_speed': (['event'], max_speed_i[1:].astype(np.float32)),
                'heure': (['event'], time_stemp_of_flight_i[1:].astype(int))
            },
            coords={
                'event': np.arange(num_events)
            },
        )
        
        W_glide.to_netcdf('T:/C2H/STAGES/LEO_BARROIS/Netcdfffs/'+save_name + '.nc')
    
def save_all() :
    days = np.arange(19,25,1)
    types = ['BLH','W_max']
    
    for day in days :
        for type in types :
            save_name = str(day)+'_'+type
            to_netcdf_glide(day,save_name,type)
            print(day,type,' ok')

