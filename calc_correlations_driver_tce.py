# Bharat Sharma
# python 3.6
# Calculations of correlation of triggers

""" the aim of this code is to identify extremes and find correlation of the corresponding drivers (with lags)
   and doing the analysis on consecutiuve time windows
   o	Data for Extremes is a list of triggers of GPP anomalies. Data for Drivers is the anomalies of TS of all drivers except fire. The cumulative lag effects of drivers are considered by averaging the drivers from lag =1 to nth month.
   o	First, the TS of gpp anomalies and driver anomalies are normalized from 0 to 1.
   o	The patch finder function returns a list of triggers of TCE( i.e. gpp ano and dri ano). e.g., in 25 years or 300 months were have 6 TCEs, we will have a list of 6 gpp ano values and 6 driver ano averaged values.
   o	The correlation is done on these qualified values and cc, pv and slope are returned.
   o	This procedure is done for every pixel, time win of 25 years and 0 - 13 months lags.
"""
from scipy import stats
from scipy import ndimage
import glob
import sys
import netCDF4 as nc4
import numpy as np
import datetime as dt
from calendar import monthrange
import matplotlib as mpl
#mpl.use('Agg')

import matplotlib.pyplot as plt
#importing my functions

from functions import time_dim_dates, index_and_dates_slicing, geo_idx, mpi_local_and_global_index, create_seq_mat, cumsum_lagged,patch_with_gaps_and_eventsize, norm, cum_av_lagged
from timeit import default_timer as timer
from scipy.stats.stats import pearsonr
from mpi4py import MPI
import pandas as pd
import argparse
import  collections
import os
import subprocess

#Inputs:
#------
parser  = argparse.ArgumentParser()
parser.add_argument('--driver_ano'	,	'-dri_a'	, help = "Driver anomalies" 					, type= str		, default= 'pr'		) #pr
parser.add_argument('--variable'   	, 	'-var'    	, help = "Anomalies of carbon cycle variable" 	, type= str		, default= 'gpp'	)
#parser.add_argument('--var_in_gC'	, 	'-var_gC'	, help = "Unit:If the variable is gC" 			, type= str		, default= 'gC'		) # 
parser.add_argument('--source'		, 	'-src' 		, help = "Model (Source_Run)"   				, type= str		, default= 'CESM2'	) # Model Name
#parser.add_argument('--thres_type'	, 	'-th_typ' 	, help = "Type of thresholds (independent or misc)", type= str	, default= 'misc'	) # this is mainly for the misc case study filter
#parser.add_argument('--ext_type'	, 	'-ext_typ' 	, help = "Type of extreme analysis (pos or neg)", type= str		, default= 'neg'	) #pos/neg
#parser.add_argument('--member_idx'	, 	'-members' 	, help = "which member index you want to use"	, type= int		, default= 0		) #Out of n members which member index you want to choose? default = 0 or first


args = parser.parse_args()
#  -----------------------------------------------------------
# To run on the terminal:
# srun -n 64 --mpi=pmi2 python calc_correlations_driver_tce.py -src CESM2 -dri_a pr -var gpp
#runnong: run ecp_correlation_drivers_tce.py -dri_a sm  -config wo_lulcc -ext_typ neg 
#print args
variable_run	= args.variable 	# variable original
dri				= args.driver_ano
#th_type			= args.thres_type
#ext_type		= args.ext_type
source_run		= args.source
#member_idx  	= args.member_idx

# dri = 'pr'
# variable_run = 'gpp'
# source_run = 'CESM2'

# Paths for reading the main files
# --------------------------------
cori_scratch    = '/global/cscratch1/sd/bharat/'
members_list	= os.listdir(cori_scratch+"add_cmip6_data/%s/ssp585/"%source_run) 
member_run 		= members_list[0] # [changed]

paths							= {}
paths['in' ]					= {}
paths['in' ]['var']				= {}
paths['in' ]['var'][source_run]	= {}
paths['in' ]['var'][source_run][member_run] = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/"%(source_run,member_run, variable_run)

paths['in' ]['dri']				= {}
paths['in' ]['dri'][source_run] = {}
paths['in' ]['dri'][source_run][member_run] = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/"%(source_run, member_run, dri)

# reading netcdf files of variable_run like GPP,NBP
# ---------------------------------------------
nc_data									= {}
nc_data['var']							= {}
nc_data['var'][source_run]				= {}
nc_data['var'][source_run][member_run]	= nc4.Dataset( paths['in' ]['var'][source_run][member_run] + "%s_ssp585_%s_%s_anomalies_gC.nc"%(source_run,member_run,variable_run))

ano		= nc_data['var'][source_run][member_run].variables[variable_run] #Anomalies of carbon flux
lat		= nc_data['var'][source_run][member_run].variables['lat']
lon		= nc_data['var'][source_run][member_run].variables['lon']
time	= nc_data['var'][source_run][member_run].variables['time']

window 		= 25 #years
win_len     = 12 * window            #number of months in window years
nwin		= int(time.size/win_len) #number of windows

wins    = np.array([format(i,'02' ) for i in range(nwin)])
#number of months you want to use for lags : 0(during) to 12months)
max_lag		=	4 #months
lags    	= np.array([format(i,'02' ) for i in range(max_lag +1)])
dates_ar    = time_dim_dates ( base_date= dt.date(1850,1,1), total_timestamps=time.size)
start_dates	= [dates_ar[i*win_len] for i in range(nwin)]#list of start dates of 25 year window
end_dates   = [dates_ar[i*win_len+win_len -1] for i in range(nwin)]#list of end dates of the 25 year window

# Calculation of anomalies
idx_dates_win= []   #indicies of time in 30yr windows
dates_win   = []    #sel dates from time variables in win_len windows

# reading netcdf files of drivers
# -------------------------------
# >>> Renaming the file name of drivers 
try:
	name_1 = paths['in' ]['dri'][source_run][member_run] + "%s_ssp585_%s_%s_anomalies_gC.nc"%(source_run,member_run,dri)
	name_2 = paths['in' ]['dri'][source_run][member_run] + "%s_ssp585_%s_%s_anomalies.nc"%(source_run,member_run,dri)
	subprocess.call("mv %s %s"%(name_1, name_2), shell=True)
except:
	print ("The file name is correct : %s_ssp585_%s_%s_anomalies.nc"%(source_run,member_run,dri))

# Reading the information of a Climate Driver
# --------------------------------------------
nc_data['dri']							= {}
nc_data['dri'][source_run]				= {}
nc_data['dri'][source_run][member_run]	= nc4.Dataset( paths['in' ]['dri'][source_run][member_run] + "%s_ssp585_%s_%s_anomalies.nc"%(source_run,member_run,dri))

cmip6_filepath_head = '/global/cfs/cdirs/m3522/cmip6/CMIP6/'
if source_run == 'CESM2':
	nc_lf = nc4.Dataset(cmip6_filepath_head + "ScenarioMIP/NCAR/CESM2/ssp585/r1i1p1f1/fx/sftlf/gn/v20190730/sftlf_fx_CESM2_ssp585_r1i1p1f1_gn.nc").variables ['sftlf']
	nc_area = nc4.Dataset(cmip6_filepath_head + "ScenarioMIP/NCAR/CESM2/ssp585/r1i1p1f1/fx/areacella/gn/v20190730/areacella_fx_CESM2_ssp585_r1i1p1f1_gn.nc").variables ['areacella']
		
#for parallel programming: identifying only the non masked locations for better load balancing
# Creating a 2d latxlon matrix 
lt_ln_mat 	=	create_seq_mat(nlat = lat.size, nlon= lon.size)
# masking the cells that have land fraction of zero
lf = np.ma.masked_equal(nc_lf[...],0)   
# now creating a one dimensional array with only index of the lat x lon matrix
land_1d		= 	lt_ln_mat[~lf[...].mask]


# Parallel Computing task distribution
comm        =   MPI.COMM_WORLD                                                
size        =   comm.Get_size() #Number of processors I am asking for
local_rank  =   comm.Get_rank() #Rank of the current processor

# Making an array of all the file name
# ------------------------------------
win_start_years = np.arange(1850,2100,25)
exp = 'ssp585' # for calling the files


if local_rank == 0:
	corr_coeff_mat 	= np.zeros((nwin,len(lags),lat.size,lon.size))
	p_values_mat	= np.zeros((nwin,len(lags),lat.size,lon.size))
	slope_mat		= np.zeros((nwin,len(lags),lat.size,lon.size))
	
	corr_coeff_temp	= np.zeros((nwin,len(lags),lat.size,lon.size))	
	p_values_temp  	= np.zeros((nwin,len(lags),lat.size,lon.size))	
	slope_temp  	= np.zeros((nwin,len(lags),lat.size,lon.size))	

	# Receiving Data in a sequence
	# ---------------------------- 
	for i in range(1,size):
		print ("receiving from %d slave" %(i))
		#try:
		comm.Recv(corr_coeff_temp, 	source=i, tag=0)
		comm.Recv(p_values_temp, 	source=i, tag=1)
		comm.Recv(slope_temp, 		source=i, tag=2)
		print ("from slave %d ..."%i, np.nansum(corr_coeff_temp))
		corr_coeff_mat	= corr_coeff_mat 	+ corr_coeff_temp
		p_values_mat	= p_values_mat 		+ p_values_temp
		slope_mat		= slope_mat 		+ slope_temp
		print ("................................. received from %d slave" %(i))

	land_masking_ar 	 = np.ma.masked_all((nwin,len(lags),lat.size,lon.size))
	land_masking_ar[:,:] = lf[...]
	
	corr_coeff_mat 	= np.ma.masked_array(corr_coeff_mat	, mask = land_masking_ar.mask)
	p_values_mat 	= np.ma.masked_array(p_values_mat	, mask = land_masking_ar.mask)
	slope_mat 		= np.ma.masked_array(slope_mat		, mask = land_masking_ar.mask)
	print (".............++++++++++++++++++++test")
	#Saving as nc file
	# Check if the directory 'path_TCE' already exists? If not, then create one:
	path_corr = cori_scratch + 'add_cmip6_data/%s/%s/%s/%s/Correlations/'%(source_run,exp,member_run,dri)
	if os.path.isdir(path_corr) == False:
		os.makedirs(path_corr)      
	with nc4.Dataset(path_corr + 'corr_coff_cumlag_%s_%s.nc'%(variable_run,dri), mode="w") as dset:
		dset        .createDimension( "win" ,size = nwin)
		dset        .createDimension( "lag" ,size = lags.size)
		dset        .createDimension( "lat"	, size = lat.size)
		dset        .createDimension( "lon"	, size = lon.size)
		w   =   dset.createVariable(varname = "win"  ,datatype = float, dimensions = ("win") ,fill_value = np.nan)                       
		t   =   dset.createVariable(varname = "lag"  ,datatype = float, dimensions = ("lag") ,fill_value = np.nan)
		y   =   dset.createVariable(varname = "lat"  ,datatype = float, dimensions = ("lat") ,fill_value = np.nan)
		x   =   dset.createVariable(varname = "lon"  ,datatype = float, dimensions = ("lon") ,fill_value = np.nan)
		z   =   dset.createVariable(varname = "corr_coeff" 	,datatype = float, dimensions = ("win","lag","lat","lon"),fill_value=np.nan)
		zz  =  dset.createVariable(varname = "p_value"     	,datatype = float, dimensions = ("win","lag","lat","lon"),fill_value=np.nan)
		zzz =  dset.createVariable(varname = "slope"     	,datatype = float, dimensions = ("win","lag","lat","lon"),fill_value=np.nan)
		w.axis  =   "T"
		t.axis  =   "T"
		x.axis  =   "X"
		y.axis  =   "Y"
		w[...]  =   wins
		t[...]  =   lags
		x[...]  =   lon[...]
		y[...]  =   lat[...]
		z[...]  =   corr_coeff_mat
		zz[...] =   p_values_mat
		zzz[...]=   slope_mat
		z.missing_value =   np.nan
		z.stardard_name =   "Correlation coefficient of %s_anomalies with %s TCEs"%(dri,variable_run)
		z.setncattr         ("cell_methods",'Stats.linregress correlation coefficient')
		z.units         =   "no units"
		zz.missing_value =   np.nan
		zz.stardard_name =   "p_value for %s anomalies with %s TCEs "%(dri,variable_run)
		zz.setncattr        ("cell_methods",'p-value for testing correlation')
		zz.units        =   "no units"
		zzz.missing_value =   np.nan
		zzz.stardard_name =   "slope for %s anomalies with %s negative extreme triggers "%(dri,variable_run)
		zzz.setncattr        ("cell_methods",'slope for attribution correlation')
		zzz.units        =   "no units"
		x.units         =   lat.units
		x.missing_value =   np.nan
		x.setncattr         ("standard_name",lat.standard_name)
		y.units         =   lon.units
		y.missing_value =   np.nan
		y.setncattr         ("standard_name",lon.standard_name)
		t.units         =   "lag by months"
		w.units         =   "25 year wins"
		w.setncattr         ("standard_name","starting from 1850-01-15 i.e. win=0: 1850-01-15 to 1874-12-15")
	
	
	#Saving the table of gobal average and median of corelation coeff
	cc_table_av	= np.zeros((len(lags),nwin))
	pv_table_av	= np.zeros((len(lags),nwin))
	cc_table_md	= np.zeros((len(lags),nwin))
	pv_table_md	= np.zeros((len(lags),nwin))
	
	#Saving table only for CONUS
	top     = 49.345    +.5
	bottom  = 24.743    -.5
	left    = 360-124.78-.5
	right   = 360-66.95 +.5
	cc_table_conus_av	= np.zeros((len(lags),nwin))
	pv_table_conus_av	= np.zeros((len(lags),nwin))
	
#	np.savetxt('temp/'+'table_start_.csv', (0,1), delimiter = ',')
	
	
	for w in range(nwin):
		for l in range(len(lags)):
			cc_table_av[l,w] 		= np.nanmean(corr_coeff_mat[w,l])
			pv_table_av[l,w] 		= np.nanmean(p_values_mat[w,l])
			cc_table_md[l,w] 		= np.nanmedian(corr_coeff_mat[w,l])
			pv_table_md[l,w] 		= np.nanmedian(p_values_mat[w,l])
			cc_table_conus_av[l,w] 	= np.nanmean(corr_coeff_mat[w,l,geo_idx(bottom,lat[...]):geo_idx(top,lat[...]),geo_idx(left,lon[...]):geo_idx(right,lon[...])])
			pv_table_conus_av[l,w] 	= np.nanmean(p_values_mat[w,l, geo_idx(bottom,lat[...]):geo_idx(top,lat[...]),geo_idx(left,lon[...]):geo_idx(right,lon[...])])

	#np.savetxt('temp/'+'table_end_.csv', (0,1), delimiter = ',')
	
	df_cc_av 		=  pd.DataFrame(data = cc_table_av)
	df_pv_av 		=  pd.DataFrame(data = pv_table_av)
	df_cc_md 		=  pd.DataFrame(data = cc_table_md)
	df_pv_md 		=  pd.DataFrame(data = pv_table_md)
	df_cc_conus_av 	=  pd.DataFrame(data = cc_table_conus_av)
	df_pv_conus_av 	=  pd.DataFrame(data = pv_table_conus_av)

	in_yr   = 1850
	win_yr  = [str(in_yr+i*25) + '-'+str(in_yr +(i+1)*25-1)[2:] for i in np.array(wins,dtype =int)]

	col_names 				= win_yr#['win'+ format(i+1,'02') for i in range(nwin)]
	df_cc_av.columns 		= col_names
	df_pv_av.columns 		= col_names
	df_cc_md.columns 		= col_names
	df_pv_md.columns 		= col_names
	df_cc_conus_av.columns 	= col_names
	df_pv_conus_av.columns 	= col_names

	id_names 	= ['lag'+ format(i,'02') for i in range(len(lags))]
	df_cc_av.index 			= id_names
	df_pv_av.index 			= id_names
	df_cc_md.index 			= id_names
	df_pv_md.index 			= id_names
	df_cc_conus_av.index 	= id_names
	df_pv_conus_av.index 	= id_names

	df_cc_av.to_csv 		(path_corr + 'cc_table_av_cumlag_%s_%s.csv'	%(variable_run,dri)	)
	df_pv_av.to_csv 		(path_corr + 'pv_table_av_cumlag_%s_%s.csv'	%(variable_run,dri)	)
	df_cc_md.to_csv 		(path_corr + 'cc_table_md_cumlag_%s_%s.csv'	%(variable_run,dri)	)
	df_pv_md.to_csv 		(path_corr + 'pv_table_md_cumlag_%s_%s.csv'	%(variable_run,dri)	)
	df_cc_conus_av.to_csv 	(path_corr + 'cc_table_conus_av_cumlag_%s_%s.csv' %(variable_run,dri)	)
	df_pv_conus_av.to_csv 	(path_corr + 'pv_table_conus_av_cumlag_%s_%s.csv' %(variable_run,dri)	)

else:
	#chunk size or delta
	local_n		=	int(np.ceil(len(land_1d)/(size-1))) #size -1 because we are sending information to the root

	#calculating the range for every parallel process
	begin_idx	=	(local_rank-1)*local_n #local_rank-1 because the rank starts from 1 and we will miss the index 0 if we do not do this ammendment
	end_idx		=	begin_idx+local_n
	
	#Defining empty arrays
	corr_coeff_ar	= 	np.zeros((nwin,len(lags),lat.size,lon.size))
	p_values_ar		= 	np.zeros((nwin,len(lags),lat.size,lon.size))
	slope_ar		= 	np.zeros((nwin,len(lags),lat.size,lon.size))

	if begin_idx <=	len(land_1d)-1:
		if local_n + begin_idx >len(land_1d)-1:
			end_idx = len(land_1d)-1


		loc_pp		=	land_1d[begin_idx:end_idx]		#locations per processor

		#for multiple windows:
		dates_ar	=	time_dim_dates(base_date=dt.date(1850,1,1), total_timestamps=time.size)
		win_len		=	25 *12 #for 25 years
		start_dates	=	[dates_ar[i*win_len] for i in range(nwin)] #list of the start dates of all 25 year windows
		end_dates	=	[dates_ar[i*win_len+win_len-1] for i in range(nwin)] #list of the end dates of all 25 year windows

 
		for win in range(nwin):
			# idx with store all the time indexs b/w start and end dates from the input array
			idx_loc,_	= index_and_dates_slicing(dates_ar,start_dates[win],end_dates[win])
			ano_loc 	= ano[idx_loc[0]:idx_loc[-1]+1,:,:]
			dri_loc		= nc_data['dri'][source_run][member_run][dri] [idx_loc[0]:idx_loc[-1]+1,:,:]
			start_yr	= win_start_years[win]
			for lag in np.asarray(lags, dtype = int):
				#print ("calculating on rank_%d_for %s for window %d and lag %d......"%(local_rank,dri,win,lag))
				for pixel in loc_pp:
					lt,ln = np.argwhere(lt_ln_mat == pixel)[0]
					lag_idx_bin = -1
					ext_type = 'neg'
					bin_tce_1s   = nc4.Dataset(cori_scratch + 'add_cmip6_data/%s/%s/%s/%s_TCE/bin_TCE_1s_%s_%d.nc'%(
									source_run,exp,member_run,variable_run,ext_type,start_yr))['%s_TCE_1s'%variable_run] [lag_idx_bin,:,lt,ln]
					bin_tce_01s  = nc4.Dataset(cori_scratch + 'add_cmip6_data/%s/%s/%s/%s_TCE/bin_TCE_01s_%s_%d.nc'%(
									source_run,exp,member_run,variable_run,ext_type,start_yr))['%s_TCE_01s'%variable_run] [lag_idx_bin,:,lt,ln]
					ext_type = 'pos'
					bin_tce_1s_alt   = nc4.Dataset(cori_scratch + 'add_cmip6_data/%s/%s/%s/%s_TCE/bin_TCE_1s_%s_%d.nc'%(
										source_run,exp,member_run,variable_run,ext_type,start_yr))['%s_TCE_1s'%variable_run] [lag_idx_bin,:,lt,ln]
					bin_tce_01s_alt  = nc4.Dataset(cori_scratch + 'add_cmip6_data/%s/%s/%s/%s_TCE/bin_TCE_01s_%s_%d.nc'%(
										source_run,exp,member_run,variable_run,ext_type,start_yr))['%s_TCE_01s'%variable_run] [lag_idx_bin,:,lt,ln]

					#TCE
					larray,narray   = ndimage.label(bin_tce_1s,structure = np.ones(3))
					locations       = ndimage.find_objects(larray)
					#TCE _alt
					larray_alt,narray_alt   = ndimage.label(bin_tce_1s_alt,structure = np.ones(3))
					locations_alt       = ndimage.find_objects(larray_alt)
						
					if (narray > 1 and narray_alt > 1):  #CHECK  great than 1 because this package return 1 event if all are masked or 0's and with only 1 tce and other 0's
						print ("Check Passed")
						# Normalized data
						data_norm = {}
						data_norm [dri] 		= norm (nc_data['dri'][source_run][member_run] [dri][win*win_len:(win+1)*win_len, lt, ln])
						data_norm ['TCE']		= larray
						data_norm ['TCE_alt']	= larray_alt
						data_norm [variable_run]	= norm(nc_data['var'][source_run][member_run].variables[variable_run][win*win_len:(win+1)*win_len, lt, ln])
						df_norm	= pd.DataFrame(data_norm, index = dates_ar[win*win_len:(win+1)*win_len])
					
						# Correlation Formulae
						cors = 'tce_cum_av_lag_b'
						ts_tce	= df_norm['TCE']
						idxs    = np.asarray(np.arange(len(ts_tce)), dtype = int)
						tce_idxs= np.array([])
						for loc in locations: 
							tce_idxs = np.append(tce_idxs, idxs[loc])
						tce_idxs = np.asarray(tce_idxs, dtype = int)
						#alt:
						ts_tce_alt	= df_norm['TCE_alt']
						idxs_alt    = np.asarray(np.arange(len(ts_tce_alt)), dtype = int)
						tce_idxs_alt= np.array([])
						for loc in locations_alt: 
							tce_idxs_alt = np.append(tce_idxs_alt, idxs[loc])
						tce_idxs_alt = np.asarray(tce_idxs_alt, dtype = int)
						tce_idxs_concate	= np.concatenate((tce_idxs, tce_idxs_alt))
						tce_idxs_concate.sort() # TCE are correlated for complete timeseries of TCE
						if lag == 0:
							corr_coeff_ar 	[win,lag,lt,ln]	= np.nan
							p_values_ar 	[win,lag,lt,ln]	= np.nan
							slope_ar 		[win,lag,lt,ln]	= np.nan
						else:
							if len(tce_idxs_concate) <= len(lags):
								corr_coeff_ar 	[win,lag,lt,ln]	= np.nan
								p_values_ar 	[win,lag,lt,ln]	= np.nan
								slope_ar 		[win,lag,lt,ln]	= np.nan
							elif lag<=tce_idxs_concate[0]:
						 		ts_dri_tce_b	= np.array([cum_av_lagged(df_norm[dri],ignore_t0 = True, lag = lag)[i] for i in (tce_idxs_concate)])
							 	ts_var_tce_b    = np.array([np.array(df_norm[variable_run])[i] for i in tce_idxs_concate])
							 	slope,c,corr_coeff,p_value,stderr = stats.linregress(ts_dri_tce_b, ts_var_tce_b)
							 	corr_coeff_ar   [win,lag,lt,ln]     = corr_coeff
							 	p_values_ar     [win,lag,lt,ln]     = p_value
						 		slope_ar        [win,lag,lt,ln]     = slope
							 	print ("error_1rank_win%s_lag%s_lt%s_ln%s"%(str(win), str(lag), str(lt), str(ln)))
		print ("Sending the information from rank : %d" %local_rank)
		print ("sending cc from rank %d ... "%local_rank ,  np.nansum(corr_coeff_ar))
		comm.Send(corr_coeff_ar	, dest = 0, tag=0)
		comm.Send(p_values_ar	, dest = 0, tag=1)
		comm.Send(slope_ar		, dest = 0, tag=2)
		#df_temp = pd.DataFrame(data = corr_coeff_ar[win,lag])
		#df_temp.to_csv('temp/cc_%d.csv'%local_rank, sep=',')

		print ("Data Sent from rank : %d" %local_rank)
	
	else:
		print ("Sending blank information from rank : %d" %local_rank)
		comm.Send(corr_coeff_ar	, dest = 0, tag=0)
		comm.Send(p_values_ar	, dest = 0, tag=1)	
		comm.Send(slope_ar		, dest = 0, tag=2)

