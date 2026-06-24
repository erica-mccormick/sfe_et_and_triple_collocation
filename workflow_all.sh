conda activate geo_env

### Calculate SFE
# Convert ERA5-Land net radiation to daily W/m2 with adjusted timezones
python era5land_process_Rn.py
python sfe.py


### Prepare other ET datasets for TC
# Convert ERA5-Land LE to daily ET in mm/day with adjusted timezones
python era5land_process_LE.py

# Need to resample everything to have the same spatial resolution:
# comparing to FLUXCOM, everything needs to be at a 0.5 deg

#for file in /Volumes/ToshibaDrive/gridded_originals/SFE_2024_run/SFE*.nc; do
#	gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -r bilinear -tr 0.5 0.5 -te -125 -25.5 -66.5 49.5 -et 0 -tap -of GTiff NETCDF:$file:ET SFE_ET_halfdegree/$(basename "$file" .nc)_halfdeg.tiff
#done

for file in /Volumes/ToshibaDrive/gridded_originals/SFE/SFE*.nc; do
	gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -r bilinear -tr 0.5 0.5 -te -125 -25.5 -66.5 49.5 -et 0 -tap -of GTiff NETCDF:$file:ET /Volumes/ToshibaDrive/gridded_originals/SFE_halfdegree/$(basename "$file" .nc)_halfdeg.tiff
done




# For trying out the other Gs
for file in /Volumes/ToshibaDrive/gridded_originals/SFE_G_30percent/SFE*.nc; do
	gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -r bilinear -tr 0.5 0.5 -te -125 -25.5 -66.5 49.5 -et 0 -tap -of GTiff NETCDF:$file:ET /Volumes/ToshibaDrive/gridded_originals/SFE_G30_halfdegree/$(basename "$file" .nc)_halfdeg.tiff
done


# GLEAM is currently split into two folders, so need to loop through both
for file in /Volumes/ToshibaDrive/gridded_originals/gleam_v4/E_1980_2014/*.nc; do
	gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -r bilinear -tr 0.5 0.5 -te -125 -25.5 -66.5 49.5 -et 0 -tap -of GTiff NETCDF:$file:E /Volumes/ToshibaDrive/gridded_originals/gleam_v4/gleam_e_halfdegree/$(basename "$file" .nc)_halfdeg.tiff
done

for file in /Volumes/ToshibaDrive/gridded_originals/gleam_v4/E_2015_2023/*.nc; do
	gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -r bilinear -tr 0.5 0.5 -te -125 -25.5 -66.5 49.5 -et 0 -tap -of GTiff NETCDF:$file:E /Volumes/ToshibaDrive/gridded_originals/gleam_v4/gleam_e_halfdegree/$(basename "$file" .nc)_halfdeg.tiff
done

# ERA5-Land ET is also split into two folders, so loop through both
for file in /Volumes/ToshibaDrive/gridded_originals/era5land_daily_ET/E_2015_2023/*.nc; do
	gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -r bilinear -tr 0.5 0.5 -te -125 -25.5 -66.5 49.5 -et 0 -tap -of GTiff NETCDF:$file:ET /Volumes/ToshibaDrive/gridded_originals/era5land_daily_et/era5land_et_halfdegree/$(basename "$file" .nc)_halfdeg.tiff
done

for file in /Volumes/ToshibaDrive/gridded_originals/era5land_daily_ET/*.nc; do
	gdalwarp -s_srs EPSG:4326 -t_srs EPSG:4326 -r bilinear -tr 0.5 0.5 -te -125 -25.5 -66.5 49.5 -et 0 -tap -of GTiff NETCDF:$file:ET /Volumes/ToshibaDrive/gridded_originals/era5land_daily_et/era5land_et_halfdegree/$(basename "$file" .nc)_halfdeg.tiff
done


# Run TC using SFE, GLEAM, and either FLUXCOM or ERA5. Can also choose to run for specific seasons.
# EXCLUDE WINTER
# takes 25 minutes
python tc.py -start_month 3 -end_month 10 -fig_file_path 'figs_sfe_gleam_fluxcom_month3to10' -out_netcdf_file_path 'output_sfe_gleam_fluxcom_month3to10' -ds0_name sfe -ds1_name gleam -ds2_name fluxcom
python tc.py -start_month 3 -end_month 10 -fig_file_path 'figs_sfe_gleam_era5_month3to10' -out_netcdf_file_path 'output_sfe_gleam_era5_month3to10' -ds0_name sfe -ds1_name gleam -ds2_name era5
python tc.py -start_month 3 -end_month 10 -fig_file_path 'figs_sfe_fluxcom_era5_month3to10' -out_netcdf_file_path 'output_sfe_fluxcom_era5_month3to10' -ds0_name sfe -ds1_name fluxcom -ds2_name era5
python tc.py -start_month 3 -end_month 10 -fig_file_path 'figs_fluxcom_gleam_era5_month3to10' -out_netcdf_file_path 'output_fluxcom_gleam_era5_month3to10' -ds0_name fluxcom -ds1_name gleam -ds2_name era5


# Plot results
python plot_tc_results.py output_sfe_fluxcom_era5_month3to10
python plot_tc_results.py output_sfe_gleam_fluxcom_month3to10
python plot_tc_results.py output_sfe_gleam_era5_month3to10
python plot_tc_results.py output_fluxcom_gleam_era5_month3to10



# Plot comparisons
python plot_tc_comparisons.py corr_truth sfe
python plot_tc_comparisons.py corr_truth gleam
python plot_tc_comparisons.py corr_truth era5
python plot_tc_comparisons.py corr_truth fluxcom

python plot_tc_comparisons.py rmse sfe
python plot_tc_comparisons.py rmse gleam
python plot_tc_comparisons.py rmse era5
python plot_tc_comparisons.py rmse fluxcom

# Example of other seasons:#
#python tc.py -start_month 6 -end_month 8 -fig_file_path 'figs_jja_summer' -netcdf_file_path 'output_jja_summer' -use_fluxcom True
#python tc.py -start_month 3 -end_month 5 -fig_file_path 'figs_mam_spring' -netcdf_file_path 'output_mam_spring' -use_fluxcom True
#python tc.py -start_month 9 -end_month 11 -fig_file_path 'figs_son_autumn' -netcdf_file_path 'output_son_autumn' -use_fluxcom True
#python tc.py -start_month 12 -end_month 2 -fig_file_path 'figs_djf_winter' -netcdf_file_path 'output_djf_winter' -use_fluxcom True


### Resample ancillary datasets to 0.5 degree for comparison 
# NLCD (using nearest neighbor)
gdalwarp -t_srs EPSG:4326 -tr 0.5 0.5 -te -125 -25.5 -66.5 49.5 -et 0 -tap -of GTiff -r near nlcd_2021_land_cover_l48_20230630.img nlcd_2021_halfdeg.tif


### To make figures, use
# fig_outline.ipynb


### To compress SFE files for Zenodo:
for f in SFE/*.nc; do
    nccopy -d 5 -s "$f" "SFE_compress/$(basename "$f")"
done