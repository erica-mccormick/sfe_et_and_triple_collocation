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
    input_folder = directory + 'era5land_hourly_rn' # I didn't keep these files around; can be re-downloaded from ERA5 website
    output_folder = directory + 'era5land_daily_Rn'
    
    start_year = 1979
    stop_year = 2025
    
    for year in np.arange(start_year, stop_year+1):
            print(f"Converting ERA5-Land LE data to daily for {year} and saving in {output_folder}")
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                
                # Open dataset
                files = glob.glob(os.path.join(input_folder, '*' + str(year) + '*.nc'))
                ds = xr.open_mfdataset(files)
                
                # If there's the expver dim, select one and drop the rest
                if 'expver' in ds.dims:
                    ds = ds.sel(expver = 1, drop=True)
                    print("Found 'expver' dimension. Removing it...")


                # Calculate W/m2 from J/m2 by dividing by seconds in an hour
                ds['ssr_acc_wm2'] = ds['ssr'] / 3600
                ds['str_acc_wm2'] = ds['str'] / 3600
                ds['Rn_acc'] = ds['ssr'] + ds['str']
                ds['Rn_acc_Wm2'] = ds['Rn_acc'] / 3600
                
                
                # De-accumulate LE to get hourly values
                ds_hourly = era5land_utils.convert_accumulated_to_hourly(ds=ds, var_name='Rn_acc_Wm2', new_var_name='Rn_hourly_Wm2', time_name='time')

                # Convert from UTC to local time
                ds_local = era5land_utils.convert_utc_to_local(ds=ds_hourly, time_name='time')

                # Resample to daily
                ds_daily = ds_local.resample(time_local='1D').mean()

                # Clean up and drop extra variables
                ds_daily['Rn_daily_Wm2'] = ds_daily['Rn_hourly_Wm2']
                ds_daily = ds_daily[['Rn_daily_Wm2', 'latitude', 'longitude', 'time_local']]

                # Save to new folder
                ds_daily.to_netcdf(output_folder + '/era5land_Rn_' + str(year) + '.nc')

    t1 = time.time()
    print(f"Elapsed time: {t1-t0} seconds")




if __name__ == '__main__':
    main()