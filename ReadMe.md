# Description of the python codes.
## The codes will help you to detect NBP extremes and attribute to compound climate drivers.
## The short description of the code describes the aim of the code, more information is within the file.

`convert_fluxC_to_gC.py` : To convert/calculate the gC/month from g/sq.m/s; system arguments take into consideration the model configuration and variable.

`ecp_plot_diff_var_config.py`: Calculating and plotting the Difference in fluxes.

`ecp_plot_timeseries_config.py` : To plot the global timeseries of a variable (includes rolling mean).

`ssa.py` : Contains the functions to calculate trend, seasonality, and anomalies using the SSA method.

`ecp_decompose_mpi.py` : The time series decomposition is carried out by the SSA. Removing trend and MAC from timeseries yields anomalies of the variable.  (Can be run parallel!)

`ecp_concatenate_decomp_over_lats_config.py` : concatenate anomalies from 192 files created while decompostion.

`ecp_extremes_analysis.py` : To calculate extremes of a flux/variable with an option of chosing the model configuration. Allows us to compute extremes based on a given threshold value(s) as well.

`ecp_correlation_methods_pixel_win.py` : To computer the correlation cofficients when the TS is chosen differently ( cases are defined in the file as well as the docs). This version is designed to calculate for the time window of 25 years at every pixel and it also computes the statistics on the TCE and TCE_len.

`ecp_dominant_climate_driver_correlation.py`: Computes the dominant drivers of based on a p_value of 0.1 and correlation coefficient. It returns a ncfile of shape (6,18,13,192,288). 6: Ranks(0 - Most Dominant), 18 - Time windows of 25 years, 13- lag in months, 192/288: lat/lon).

`ecp_dominant_climate_driver_correlation_graphs.py` : Computes the frequency of occurance of the most dominant driver, spatial distribution, mean and standard deviation of the correlation coefficients for the whole globe and the CONUS; also outputs dataFrames.

`ecp_correlation_drivers_triggers.py` : Computes the correlation coefficients of the climate drivers and the triggers (locations of the begining of an event). The extreme event is defined as a time continuous extreme event where the max gap is 2 months and the minimum patch continuous extreme times are 3 months.

`ecp_triggers_percent_distribution_dom_drivers.py` : Plot percent distribution of dominant climate drivers for misc cases as well. It can perform calculation on the whole duration and not just triggers.

`ecp_attr_corr_triggers_plotting.py` : Plots the timeseries of the count of the dominant drivers for a particular lag for the complete time period (1850--2300).

`ecp_shaded_extremes_drivers.py` : The aim is to create a stacked graphs of gpp and climate drivers to identify the climate that led to an extreme event at that location. The TCE events are also shaded and  we could focus on a particular TCE and see the detais.

`ecp_dom_driver_RGB_tce.py` : Ploting the RGB maps from the correlation nc file of all the drivers and their correlation cofficients.

`ecp_IAV_extremes_drivers.py` : To calculate the IAV of GPP at global and pixel level, the relative IAV i.e. IAV was computed after dividing the anomalies with trend of GPP. To calculate the IAV of Climate drivers pixel level. To calculate the IAV standard deviation was computed from 1850 to 1874, 1899, 1924 , ... , 2299.

## Filenames and Description

`functions.py` : To keep all common funcions used in this project in one file 

`filepaths_to_use.sh` : To create a .csv file with all the possile locations for a particular experiment id where that variables stored. 
This code should be run at level /global/homes/b/bharat/cmip6_data/CMIP6"

`data_scope.py` : It creates a bunch of .xlxs/.csv files that summarizies the whole scope of the data.
The python file `filepaths_to_use.sh` must be run first.

`ssa_code.py` : Contains the functions to calculate trend, seasonality, and anomalies using the SSA method.

`calc_anomalies_ssa.py` : This code is designed to calculate the anomalies using the SSA. 
It is designed for parallel calculations and computed the anomalies for all lons per lat.
Hence, the ouput files are  saved with the local rank that are distributed sequentially.

`calc_anomalies_ssa_variable.py` : This code is to find anomalies for one variable.

`concate_over_lats_ano.py` : The output files of the calc_ssa_anomalies.py are concated to one file using this code.

`plot_ts_variables_cont_f.py` : To plot the yearly of 5year running mean of a variable

`plot_shaded_extremes_drivers.py`  : Plot shaded extreme anomalies

`calc_extremes.py` : To calculate the extremes of the carbon fluxes based on carbon flux anomalies in gC.
The code is fairly flexible to pass multiple filters to the code.

`calc_correlations_driver_tce.py` : to identify extremes and find correlation of the corresponding drivers (with lags). 
To compute the correlation cofficients when the TS is chosen differently (cases are defined in the file as well as the docs). 
This version is designed to calculate for the time window of 25 years at every pixel and it also computes the statistics on the TCE and TCE_len.

`calc_dominant_climate_driver_correlation_tce.py` : This code will find the most dominant climate driver based on the correlation coefficients and pvalues using the results of the individual correlation coefficients

`calc_plot_regional_analysis_drivers.py` : To calculate the spatial metrics of NBP extremes at SREX regions and TCES stats; Carbon gains and losses.

`plot_dominant_climate_driver_correlation_tce_regional_graphs.py` : It reads the individual driver's correlation nc files and uses SREX regions mask to Plot the attribution for climate drivers at SREX regions.

`plot_regional_analysis_postDF_nbp.py` : this code is to be run after the main dataframe " df_bin_wins and df_tce_wins are already created ..
Using the file "calc_plot_regional_analysis.py" ...
To calculate the spatial metrics of extemes
- Regional analysis based on SREX regions
- TCE stats of frequency and length of 
	* positive and negative extremes
- Carbon gains and losses
- Relative changes to overall changes in NBP flux
- Extremes happening during CUP and CRp


