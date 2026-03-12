import xarray
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import datetime as datetime
import argparse
import json
import os
import time


def main():
    t0 = time.time()
    directory = '/Volumes/ToshibaDrive/gridded_originals/'
    gridmet_dir = directory + 'gridmet' # temperature and specific humidity
    era5_dir = directory + 'era5land_daily_Rn' # net radiation
    sfe_save_dir = directory + 'SFE'
    ground_heat_flux = 10 # as a fraction of Rn

    print(f"\nAssuming ground heat flux of {ground_heat_flux*100}%\n")

    start_year = 1979
    stop_year = 2025
    
    for y in range(start_year, stop_year+1):
        print('\n', y)
        bowen_ratio = calculate_bowen_ratio(year=y, gridmet_dir=gridmet_dir)
        et = calculate_et(year=y, bowen_ds=bowen_ratio, era5_dir=era5_dir, ground_heat_flux = ground_heat_flux)
        et = et.where(et["ET"] >= 0) # ensure NaN when ET is negative; due to negative Rn.
        et.to_netcdf(sfe_save_dir + '/SFE_' + str(y) + '.nc')  
    
    print(f"Total time: {time.time() - t0} seconds")


def calculate_bowen_ratio(year, gridmet_dir):
    RV = 461.5 #J kg^-1 K^-1
    CP = 1005 #J kg^-1 K^-1
    LAMBDA = 2500800 #J kg^-1 

    # Open gridMET
    qa = xarray.open_mfdataset(paths = gridmet_dir + "/specific_humidity/sph*" +  str(year)  + ".nc") 
    qa = qa['specific_humidity'] #kg/kg 

    ta_max_data = xarray.open_mfdataset(paths = gridmet_dir + "/temperature_max/tmmx*" + str(year) + ".nc") 
    ta_max = ta_max_data['air_temperature']

    # Calculate sensible and latent heat
    sensible_heat = RV*CP*(ta_max*ta_max)
    latent_heat = (LAMBDA*LAMBDA)*qa
    bowen_ratio = sensible_heat/latent_heat  

    return bowen_ratio

   
def calculate_et(year, bowen_ds, era5_dir, ground_heat_flux):
    WATER_VAPORIZATION = 2260000 #J /kg

    # Open ERA5-Land Net radiation
    era5_ds = xarray.open_mfdataset(era5_dir + '/era5land_Rn_' + str(year) + '*', engine="netcdf4") 
    name_dict = {"longitude":"lon", 
             "latitude":"lat",
             "time_local":"day"}
    era5_ds = era5_ds.rename(name_dict)
    
    # Save fig of Rn before interpolation
    #era5_ds['Rn_daily_Wm2'].sel(day=pd.to_datetime(f'{year}-06-01'), method='nearest').plot()
    #plt.savefig('figs/intermediate_steps/rn_years/rn_raw_' + str(year) + '.png')
    #plt.close()
   
    # Ensure grid of era5 matches bowen ratio using linear method
    era5_interp = era5_ds.interp_like(bowen_ds)
    
    # Save fig of Rn after interpolation
    #era5_interp['Rn_daily_Wm2'].sel(day=pd.to_datetime(f'{year}-06-01'), method='nearest').plot()
    #plt.savefig('figs/intermediate_steps/rn_years/rn_interpolated_' + str(year) + '.png')
    #plt.close()

    era5_interp['Rn_daily_Wm2_min_ground'] = era5_interp["Rn_daily_Wm2"] * (1-ground_heat_flux)
    
    latent_heat_flux = era5_interp["Rn_daily_Wm2_min_ground"] / (bowen_ds + 1) 
    latent_heat_flux_ds = latent_heat_flux.to_dataset(name = "ET")

    # Convert to ET (mm/day)
    et = latent_heat_flux_ds["ET"] / WATER_VAPORIZATION
    et_ds = et.to_dataset(name = "ET")
    et_ds["ET"] = et_ds["ET"] * (86400)

    print(et_ds)

    era5_ds.close()
    bowen_ds.close()
    
    if 'expver' in et_ds.dims:
        print("Found 'expver' dimension. Removing it...")
        et_ds = et_ds.drop_dims('expver')

    return et_ds


if __name__ == '__main__':
    main()