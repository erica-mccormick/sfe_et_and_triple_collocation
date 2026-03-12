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

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
print("Warning from era5land_utils.py: Deprecation warnings have been silenced. Timezonefinder has warnings.")


# This function extracts the "data.nc" file from each zip file in the input folder
def extract_and_save_zip_files(input_folder, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Use glob to find all zip files in the input folder
    zip_files = glob.glob(os.path.join(input_folder, '*.zip'))
    
    for zip_file in zip_files:
        # Extract the "data.nc" file from each zip file
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            data_nc = zip_ref.extract('data.nc', path=output_folder)
        
        # Rename the extracted file to match the zip file's name
        base_name = os.path.basename(zip_file)
        new_nc_file = os.path.join(output_folder, base_name.replace('.netcdf.zip', '.nc'))
        shutil.move(data_nc, new_nc_file)
        
        

def convert_accumulated_to_hourly(ds, var_name, new_var_name, time_name):
    """
    ds is the dataset
    var_name is the variable of interest, for example, 'slhf_Wm2'
    new_var_name is what you want it saved as once it is de-accumulated, for example 'slhf_Wm2_hourly'
    time_name is the name of the time dimension. For Rn, that is 'time' but for LE it is 'valid_time'
    """
    ds['diff'] = ds[var_name].diff(time_name)
    ds['hour_utc'] = ds[time_name].dt.hour
    first_hour_mask = ds['hour_utc'] == 1
    ds[new_var_name] = ds['diff'].where(~first_hour_mask, ds[var_name])
    return ds

   
        
def convert_utc_to_local(ds, time_name):        
    t = ds[time_name].values
    lat = ds.latitude.values
    lon = ds.longitude.values
    
    total_lats = len(lat)

    # Create TimezoneFinder object
    tf = TimezoneFinder()

    # Initialize the local_array with the correct shape
    local_array = np.zeros((len(t), len(lat), len(lon)), dtype='datetime64[ns]')

    # Precompute time offsets for each lat/lon point
    tz_offsets = np.zeros((len(lat), len(lon)), dtype=int)
    
    for i in range(len(lat)):
        if i%100 == 0:
            print("Latitude progress is " + str(round((i/total_lats)*100)) + "%")
        for j in range(len(lon)):
            tz_name = tf.timezone_at(lat=lat[i], lng=lon[j])
            if tz_name:
                tz = timezone(tz_name)
                utc_offset = int(tz.utcoffset(pd.Timestamp(t[0])).total_seconds() / 3600)
                tz_offsets[i, j] = utc_offset

    # Vectorized time conversion
    for k in range(len(t)):
        local_array[k, :, :] = np.datetime64(pd.Timestamp(t[k])) + np.timedelta64(1, 'h') * tz_offsets
        #print(f"t[k] = {t[k]}, tz_offsets={tz_offsets[30,30]}, new array: {local_array[k,30,30]}")

    # Convert the local_array to an xarray DataArray
    local_array_da = xr.DataArray(local_array, coords=[local_array[:,0,0], lat, lon], dims=['time_local', 'latitude', 'longitude'])

    # Create a new dataset with the updated time dimension
    new_ds = xr.Dataset()
    for var_name, da in ds.data_vars.items():
        if time_name in da.dims:
            new_dims = tuple('time_local' if dim == time_name else dim for dim in da.dims)
            new_ds[var_name] = (new_dims, da.values, da.attrs)
        else:
            new_ds[var_name] = da

    # Assign new coordinates
    new_ds = new_ds.assign_coords({dim: ds.coords[dim] for dim in ds.coords if dim != time_name})
    new_ds = new_ds.assign_coords(time_local=local_array_da['time_local'])

    return new_ds