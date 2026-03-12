import numpy as np
import xarray
import cftime 
import matplotlib.pyplot as plt
import argparse
import time
import dask
from dask.distributed import Client
import pandas as pd
import os

# suppress userwarnings
print("Note: Warning messages are suppressed.")
import warnings
warnings.filterwarnings("ignore")


def drop_first_day(ds):
    """Remove the first time step because it causes duplicate days"""
    return ds.isel(time_local=slice(1, None))  

def import_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-start_month', type=int, default = 1)
    parser.add_argument('-end_month', type=int, default = 12)
    parser.add_argument('-fig_file_path', type=str, default = 'figs')
    parser.add_argument('-out_netcdf_file_path', type=str, default = 'output')
    parser.add_argument('-ds0_name', type=str, default = 'False')
    parser.add_argument('-ds1_name', type=str, default = 'False')
    parser.add_argument('-ds2_name', type=str, default = 'False')
    parser.add_argument('-use_rolling_average_anomaly', default='False')
    return parser.parse_args()


def select_datasets(ds_name):
    print(f"Preparing dataset {ds_name}...")
    fluxcom_path = '/Volumes/ToshibaDrive/gridded_originals/CRUNCEP_v8/daily/'
    era5_path = '/Volumes/ToshibaDrive/gridded_originals/era5land_daily_ET/era5land_et_halfdegree/'
    gleam_path = '/Volumes/ToshibaDrive/gridded_originals/gleam_v4/gleam_e_halfdegree/'
    sfe_path = '/Volumes/ToshibaDrive/gridded_originals/SFE_2024_run/SFE_ET_halfdegree/'
    
    if ds_name == 'fluxcom':
        return prep_fluxcom(fluxcom_path)
    elif ds_name == 'gleam':
        return prep_gleam(gleam_path)
    elif ds_name == 'sfe':
        return prep_sfe(sfe_path)
    elif ds_name == 'era5':
        return prep_era5(era5_path)
    else:
        raise ValueError(f"Dataset {ds_name} not found. Please choose from 'fluxcom', 'gleam', 'sfe', or 'era5'.")


def main():

    t0 = time.time()
    args = import_args()
        
    # Out paths
    fig_file_path = args.fig_file_path + '/'
    netcdf_file_path = args.out_netcdf_file_path
    
    # Check if fig file path and netcdf file path exist and if not, create them
    if not os.path.isdir(fig_file_path):
        os.makedirs(fig_file_path)
        print(f'Created folder {fig_file_path} to save figure outputs', flush=True)

    if not os.path.isdir(netcdf_file_path):
        os.makedirs(netcdf_file_path)
        print(f'Created folder {netcdf_file_path} to save netcdfs', flush=True)

    ### Open datasets
    list_of_datasets = [args.ds0_name, args.ds1_name, args.ds2_name]
    ds0 = select_datasets(args.ds0_name)
    ds1 = select_datasets(args.ds1_name)
    ds2 = select_datasets(args.ds2_name)
    
    print(f"\nDS0: {ds0}\n\nDS1: {ds1}\n\nDS2: {ds2}")
    
    ### Apply rolling average anomaly if requested
    if args.use_rolling_average_anomaly.lower() == 'true':
        print("Applying rolling average anomaly to datasets.")
        ds0 = rolling_average_anomaly(ds0, args.ds0_name)
        ds1 = rolling_average_anomaly(ds1, args.ds1_name)
        ds2 = rolling_average_anomaly(ds2, args.ds2_name)
    
    ### Merge datasets together
    print("Requested start and end months: ", args.start_month, args.end_month)
    et = xarray.merge([ds0, ds1, ds2], join='inner', compat = 'equals' )
    
    ds0.close()
    ds1.close()
    ds2.close()
    
    # Add month information to allow season selection
    et['month'] = et['time.month']
    et = et.where(et['month'].isin([np.arange(args.start_month, args.end_month+1)]), drop = True)
    print(f"\n\nMerged ET: {et}")
    t1 = time.time()
    print(f"Loaded data and selected months. Took {round(t1-t0)} seconds.")
        
    # Make maps of days with NaN then drop those days for all datasets
    if 'sfe' in list_of_datasets:
        et = count_nan_days_and_drop(et, fig_file_path, args)
    

    # Do TC
    print("Performing TC analysis.")
    triple_collocation(et, fig_file_path, netcdf_file_path, args)
    t2 = time.time()
    print(f"Done. Elapsed time {round((t2-t0)/60)} minutes.")
    

def plot(data, name, file_path, vmin = 0, vmax = None):
    if vmax: 
        data.plot(vmin = vmin, vmax = vmax, cmap = 'viridis')
    else:
        data.plot(cmap = 'viridis')
    plt.title(name)
    plt.xlim(-125, -66.5)
    plt.ylim(24.5, 49.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(file_path + name)
    plt.close()


def count_nan_days_and_drop(et, fig_file_path, args):
    # Make maps of how many NaNs there are in each dataset
    #print("Plotting NaN maps", flush=True)
    #ds0_nans = et[args.ds0_name].isnull().sum(dim='time')
    #ds1_nans = et[args.ds1_name].isnull().sum(dim='time')
    #ds2_nans = et[args.ds2_name].isnull().sum(dim='time')
    #plot(ds0_nans, f"nans_from_raw_{args.ds0_name}", fig_file_path)
    #plot(ds1_nans, f"nans_from_raw_{args.ds1_name}", fig_file_path)
    #plot(ds2_nans, f"nans_from_raw_{args.ds2_name}", fig_file_path)
    
    # For days when any dataset is NaN, drop data for that day for all datasets
    # This SFE mask is already applied when SFE is created, so this step is somewhat redundant.
    sfe_mask = et['sfe'] < 0
    et_masked = et.where(~sfe_mask)
    return et_masked

def prep_fluxcom(fluxcom_path):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        #Load in fluxcom, which is X1
        #The no data value is -999 and there is no scaling factor
        fluxcom = xarray.open_mfdataset(fluxcom_path + "*", decode_times=False, engine = "netcdf4")
        fluxcom = xarray.decode_cf(fluxcom, mask_and_scale = True, decode_times = True, use_cftime = True, drop_variables = ["lat_bnds", "lon_bnds", "time_bnds"])
        # Rename the dimensions and variables to match the other datasets
        flux_name_dict = {"LE": "fluxcom"}
        fluxcom = fluxcom.rename(flux_name_dict)
        # convert LE from MJ/dm^2 to J/dm^2
        fluxcom["fluxcom"] = fluxcom["fluxcom"] * (1000000)
        # convert LE to ET by dividing by water vaporization 
        fluxcom["fluxcom"] = fluxcom["fluxcom"] / 2500800 #J/kg
        fluxcom = fluxcom['fluxcom'].to_dataset() # drop all other variables
    return fluxcom


def prep_sfe(sfe_path):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        sfe = xarray.open_mfdataset(sfe_path + "*.tiff")
        # Rename the dimensions and variables to match the other datasets
        sfe_name_dict = {"y":"lat",
                        "x":"lon",
                        "ET":"sfe",
                        "day": "time"}
        sfe = sfe.rename(sfe_name_dict)
        # Confirm all of SFE ET negative values are nan
        sfe["sfe"] = sfe["sfe"].where(sfe["sfe"] >= 0, np.nan)
    return sfe
    
    
def prep_gleam(gleam_path):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        gleam = xarray.open_mfdataset(gleam_path + "*.tiff", decode_times=True)
        gleam = xarray.decode_cf(gleam, mask_and_scale = True, decode_times = True, use_cftime = True)
        gleam = gleam.where(gleam != -999, np.nan)
        #fix GLEAM time
        time_val = gleam["time"].values
        datetime_val = np.array([np.datetime64(t) for t in time_val])
        gleam["time"] = datetime_val
        # Rename the dimensions and variables to match the other datasets
        gleam_name_dict = {"y":"lat",
                            "x":"lon",
                            "E":"gleam"}
        gleam = gleam.rename(gleam_name_dict)
        # Get rid of extra variable unit attributes for gleam to avoid long print-outs
        for var in gleam.data_vars:
            if 'units' in gleam[var].attrs:
                del gleam[var].attrs['units']
    return gleam
            

def prep_era5(era5_path):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        era5 = xarray.open_mfdataset(era5_path + "*.tiff", preprocess=drop_first_day)
        # Rename the dimensions and variables to match the other datasets
        era5_name_dict = {"y":"lat",
                        "x":"lon",
                        "ET":"era5",
                        "time_local": "time"}
        era5 = era5.rename(era5_name_dict)
    return era5


def rolling_average_anomaly(ds, ds_name, window_size=30):
    ds['rolling_mean'] = ds[ds_name].rolling(time=window_size, center=True).mean()
    ds[ds_name] = ds[ds_name] - ds['rolling_mean']
    return ds.drop_vars('rolling_mean')



def save_tc_result(array, final_dims, final_coords, netcdf_file_path, name):
    ds = xarray.Dataset(
        {f"ds{i}": xarray.DataArray(array[i], dims=final_dims, coords=final_coords)
            for i in range(3)})
    ds.to_netcdf(netcdf_file_path + f'/{name}.nc')


def triple_collocation(et, fig_file_path, netcdf_file_path, args):
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        # Extract data as numpy arrays for faster processing
        datasets = [et[args.ds0_name].values, et[args.ds1_name].values, et[args.ds2_name].values]

        # When we transform things back into xarrays we just want a 2D version of the original xarray dims and coords
        final_dims = et.drop_vars('time')[list(et.data_vars.keys())[0]].dims[1:]
        final_coords = et.drop_vars('time')[list(et.data_vars.keys())[0]].coords
        
        # Calculate variances
        print("Calculating variances and means", flush=True)
        variances = np.array([np.nanvar(ds, axis=0) for ds in datasets])
        save_tc_result(variances, final_dims, final_coords, netcdf_file_path, 'var')

        # Calculate means
        means = np.array([np.nanmean(ds, axis=0) for ds in datasets])
        save_tc_result(means, final_dims, final_coords, netcdf_file_path, 'mean')

        # Calculate standard deviations
        std_devs = np.array([np.nanstd(ds, axis=0) for ds in datasets])
        save_tc_result(std_devs, final_dims, final_coords, netcdf_file_path, 'std')

        print("Calculating covariances", flush=True)
        # Calculate covariances
        list_of_pairs = [(0, 1), (0, 2), (1, 2)]
        covariances = []

        for pair in list_of_pairs:
            ds1 = datasets[pair[0]]
            ds2 = datasets[pair[1]]

            # Create a valid mask: only keep time points where both datasets have valid values
            valid_mask = np.isfinite(ds1) & np.isfinite(ds2)  # True where both are not NaN

            # Initialize covariance array
            cov_map = np.full((ds1.shape[1], ds1.shape[2]), np.nan)  # Start with NaNs

            # Calculate covariance for each pixel
            for m in range(ds1.shape[1]):  # Loop over lat
                for n in range(ds1.shape[2]):  # Loop over lon
                    valid_time_points = valid_mask[:, m, n]

                    if valid_time_points.sum() > 1:  # Need at least 2 valid time points
                        cov_map[m, n] = np.cov(ds1[valid_time_points, m, n], 
                                            ds2[valid_time_points, m, n], 
                                            bias=True)[0,1]
            
            covariances.append(cov_map)
        print("Saving covariances")
        save_tc_result(covariances, final_dims, final_coords, netcdf_file_path, 'cov')


        # Calculate multiplicative biases
        print("Calculating multiplicative biases", flush=True)
        B2 = covariances[2] / covariances[1]
        B3 = covariances[2] / covariances[0]

        # Calculate additive biases
        print("Calculating additive biases", flush=True)
        A2 = means[1] - B2 * means[0]
        A3 = means[2] - B3 * means[0]
        
        # Save alpha and beta
        print("Saving alpha and beta", flush=True)
        alpha_ds = xarray.Dataset(
            { "A2": xarray.DataArray(A2, dims=final_dims, coords=final_coords),
              "A3": xarray.DataArray(A3, dims=final_dims, coords=final_coords),})
        
        print(alpha_ds, flush=True)
        
        beta_ds = xarray.Dataset(
            {"B2": xarray.DataArray(B2, dims=final_dims, coords=final_coords),
                "B3": xarray.DataArray(B3, dims=final_dims, coords=final_coords),})
    
        alpha_ds.to_netcdf(netcdf_file_path + '/alpha.nc')
        beta_ds.to_netcdf(netcdf_file_path + '/beta.nc')

        # Calculate RMSE
        print("Calculating random errors", flush=True)
        errors = np.empty(3, dtype=object)
        errors[0] = variances[0] - (covariances[0] * covariances[1]) / covariances[2]
        errors[1] = variances[1] - (covariances[0] * covariances[2]) / covariances[1]
        errors[2] = variances[2] - (covariances[1] * covariances[2]) / covariances[0]
        
        rmses = [np.sqrt(errors[i]) for i in range(3)]
        rmse_ds = xarray.Dataset(
            {f"ds{i}": xarray.DataArray(rmses[i], dims=final_dims, coords=final_coords)
                for i in range(3)})
        rmse_ds.to_netcdf(netcdf_file_path + '/rmse.nc')
        
        print(rmse_ds, flush=True)
      

        # Calculate correlation coefficients 
        print("Calculating correlation coefficients", flush=True)
        correlations = np.empty(3, dtype=object)

        #list_of_pairs = [(0, 1), (0, 2), (1, 2)], Q12 = 0, Q13 = 1, Q23 = 2
        correlations[0] = np.sqrt((covariances[0] * covariances[1]) / (variances[0] * covariances[2]))
        correlations[1] = np.sqrt((covariances[0] * covariances[2]) / (variances[1] * covariances[1]))
        correlations[2] = np.sqrt((covariances[1] * covariances[2]) / (variances[2] * covariances[0]))

        print("Saving correlation coefficient", flush=True)
        cor_ds = xarray.Dataset({f"ds{i}": xarray.DataArray(correlations[i], dims=final_dims, coords=final_coords)
                                 for i in range(3)})
        cor_ds.to_netcdf(netcdf_file_path + '/corr_truth.nc')


def days_in_month(month):
    if month in [1, 3, 5, 7, 8, 10, 12]:
        return 31
    elif month in [4, 6, 9, 11]:
        return 30
    else:
        return 28


if __name__ == '__main__':
    main()