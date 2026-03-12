import glob
import zipfile
import os
import shutil
import numpy as np
import xarray as xr
from timezonefinder import TimezoneFinder
from pytz import timezone
import pandas as pd
import sys
from datetime import timedelta, datetime
from pytz import timezone
import time
import dask

import era5land_utils


def main():
    t0 = time.time()
    
    ### If you need to unzip files and move them to a new folder:
    #input_folder = 'era5'
    #output_folder = 'era5_unzipped'
    #era5land_utils.extract_and_save_zip_files(input_folder, output_folder)

    ### For converting from accumulated, hourly LE in J/m2 to daily mm/day ET
    directory = '/Volumes/ToshibaDrive/gridded_originals/'
    input_folder = directory + 'era5land_hourly_le'
    output_folder = directory + 'era5land_daily_ET'
        
    start_year = 1979
    stop_year = 2023
    for year in np.arange(start_year, stop_year+1):
        print(f"Converting ERA5-Land LE data to daily for {year} and saving in {output_folder}")
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            
            # Open dataset
            files = glob.glob(os.path.join(input_folder, '*' + str(year) + '*.nc'))
            ds = xr.open_mfdataset(files)
            
            # Clean up extra dimensions
            ds = ds.drop('expver')
            ds = ds.drop('number')

            # Fix longitudes and crop to CONUS IF NECESSARY
            if (ds.longitude.values.min()==0) & (ds.longitude.values.max()==359.9):
                max_lat = 50
                min_lat = 24
                min_lon = 360-125
                max_lon  = 360-66
                ds = ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon,max_lon))
                ds['longitude'] = (ds['longitude'] - 360) # So that longitude is in correct format
            
            elif (ds.longitude.values.min()!=-125) or (ds.longitude.values.max()!=-66):
                print(f"Longitude range is {ds.longitude.values.min()} to {ds.longitude.values.max()}. May want to clip.")

            # Convert LE from J/m2 to W/m2
            ds['slhf'] = ds['slhf'] * -1 # So that 'going upward' is positive instead of negative
            ds['slhf_Wm2'] = ds['slhf'] / 3600 # Originally in J/m2, so convert to W/m2 by dividing by seconds in an hour
        
            # De-accumulate LE to get hourly values
            ds_hourly = era5land_utils.convert_accumulated_to_hourly(ds=ds, var_name='slhf_Wm2', new_var_name='slhf_Wm2_hourly', time_name='valid_time')

            # Convert from UTC to local time
            ds_local = era5land_utils.convert_utc_to_local(ds=ds_hourly, time_name='valid_time')

            # Resample to daily
            ds_daily = ds_local.resample(time_local='1D').mean()

            # Convert from W/m2 to mm/day
            # Dividing latent heat of vaporization by seconds in a day gives units of W/kg
            # W/m2 divided by W/kg gives kg/m2/day which is mm/day
            latent_heat_of_vaporization = 2.45e6 # J/kg
            seconds_in_a_day = 86400 # seconds
            ds_daily['ET'] = ds_daily['slhf_Wm2_hourly'] / (latent_heat_of_vaporization / seconds_in_a_day)

            # Clean up and drop extra variables
            ds_daily = ds_daily[['ET', 'latitude', 'longitude', 'time_local']]

            # Save to new folder
            ds_daily.to_netcdf(output_folder + '/era5land_ET_' + str(year) + '.nc')

    t1 = time.time()
    print(f"Elapsed time: {t1-t0} seconds")







if __name__ == '__main__':
    main()