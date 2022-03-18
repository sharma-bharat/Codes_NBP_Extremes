# Bharat Sharma
#python 3.6

"""
	This code will find the most dominant climate driver based on the correlation coefficients and pvalues using
	the results of the individual correlation coefficients
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
from functions import time_dim_dates, index_and_dates_slicing, geo_idx, mpi_local_and_global_index, create_seq_mat, cumsum_lagged,patch_with_gaps_and_eventsize, norm
from timeit import default_timer as timer
from scipy.stats.stats import pearsonr
import pandas as pd
import argparse
import  collections
import os

parser  = argparse.ArgumentParser()
#parser.add_argument('--driver_ano'  ,   '-dri_a'    , help = "Driver anomalies"                     , type= str     , default= 'pr'     ) #pr
parser.add_argument('--variable'    ,   '-var'      , help = "Anomalies of carbon cycle variable"   , type= str     , default= 'gpp'    )
parser.add_argument('--source'      ,   '-src'      , help = "Model (Source_Run)"                   , type= str     , default= 'CESM2'  ) # Model Name
parser.add_argument('--member_idx'  ,   '-m_idx'    , help = "Member Index"                   		, type= int     , default= 0  		) # Index of the member

args = parser.parse_args()

# run calc_dominant_climate_driver_correlation_tce.py -var gpp -src CESM2 -dri_a 'pr-tas-mrso'
print (args)

variable    	= args.variable 
#drivers_string  = args.driver_ano
source_run      = args.source
member_idx      = args.member_idx

# List of the drivers that will be considered and their names for Plotting
# -----------
if source_run == 'CESM2':
	driver_consider = 4
	drivers			= np.array(['pr','mrso','tas','fFireAll']) [:driver_consider]
	drivers_names 	= np.array(['Prcp','Soil Moisture', 'TAS','Fire']) [:driver_consider]
	drivers_code    = np.array([  10,   20,   30,   40])  [:driver_consider]
else:
	driver_consider = 3
	drivers			= np.array(['pr','mrso','tas']) [:driver_consider]
	drivers_names 	= np.array(['Prcp','Soil Moisture', 'TAS']) [:driver_consider]
	drivers_code    = np.array([  10,   20,   30])  [:driver_consider]

# Paths for reading the main files
# --------------------------------
cori_scratch    = '/global/cscratch1/sd/bharat/'
members_list    = os.listdir(cori_scratch+"add_cmip6_data/%s/ssp585/"%source_run)
member_run      = members_list[member_idx]

# Storing the file name and abr of the drivers to be considered
# -------------------------------------------------------------
features            	= {}
features['abr']			= drivers
features['filenames']   = {}

features['filenames'][variable]	= {}	# Creating a empty directory for storing multi members if needed
features['filenames'][variable][member_run]	= cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_anomalies_gC.nc"%(source_run,member_run, variable,member_run,variable)

# Creating a empty directory for storing multi members if needed; 
# At the moment only we will run only one member per code run
for dri in drivers:
	features['filenames'][dri] = {}
for dri in drivers:
	features['filenames'][dri][member_run] = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Correlations/corr_coff_cumlag_%s_%s.nc"%(source_run,member_run, dri,variable,dri)

# The name with which the variables are stored in the nc files:
features['Names']			 	= {}
features['Names']['pr']	 		= 'pr'
features['Names']['mrso']	 	= 'mrso'
features['Names']['tas']	 	= 'tas'
if source_run == 'CESM2':	
	features['Names']['fFireAll']	= 'Fire'
#features['Names']['tasmax']	 	= 'tasmax'


# Reading time, lat, lon
time 	= nc4.Dataset(features['filenames'][variable][member_run],mode = 'r').variables['time']
lat		= nc4.Dataset(features['filenames'][variable][member_run],mode = 'r').variables['lat']
lon		= nc4.Dataset(features['filenames'][variable][member_run],mode = 'r').variables['lon']

# Reading the NC files
features['ncfiles']			 = {}
for dr in features['abr']: 
	features['ncfiles'][dr] = {}  # Creating a empty directory for storing multi members if needed
for dr in features['abr']: 
	print (dr)
	features['ncfiles'][dr][member_run] = nc4.Dataset(features['filenames'][dr][member_run],mode = 'r')

#Checks:
p_value_th				= .1 #or 90% confidence
#p_value_th				= 1.01 # without p-value filter [changed]

#creating the empty arrays
(nwin,nlag,nlat,nlon)	= features['ncfiles']['tas'][member_run].variables['corr_coeff'].shape

# The add the dimension of all the drivers that are considered
dom_dri_corr_coeff		= np.ma.masked_all((driver_consider,nwin,nlag,nlat,nlon))
dom_dri_ids 			= np.ma.copy(dom_dri_corr_coeff,True)

corr_dri_mask			= np.ma.ones((nwin,nlag,nlat,nlon))
for dr in features['abr']: 
	corr_dri_mask.mask 	= features['ncfiles'][dr][member_run].variables['corr_coeff'][...].mask +corr_dri_mask.mask

# the array below will add all the values of different time-periods and lags to a 2-d array thus giving us the mask of a 2-d array where calculations should be performed.
# Check this section
# ------------------
""" It seems like I am only performing the computation at places where there is some correlation of drivers and GPP
need more clarity"""

corr_dri_mask_2d		= np.ma.copy(corr_dri_mask, True)
corr_dri_mask_2d[corr_dri_mask_2d.mask == True] = 0
corr_dri_mask_2d		= np.ma.sum(corr_dri_mask_2d,axis = 0) #4d to 3d
corr_dri_mask_2d		= np.ma.sum(corr_dri_mask_2d,axis = 0) #3d to 2d
corr_dri_mask_2d		= np.ma.masked_equal(corr_dri_mask_2d, 0)

lt_ln_mat   			= create_seq_mat(nlat=lat.size, nlon=lon.size)
corr_dri_mask_1d     	= lt_ln_mat[~corr_dri_mask_2d[...].mask]

wins 	= np.array(range(nwin))
lags 	= np.array(range(nlag))

for win in wins:
	for lg in lags:
		for pixel in corr_dri_mask_1d:
			lt,ln=np.argwhere(lt_ln_mat == pixel)[0]
			dri_id = []
			dri_cc = []
			dri_pv = []
			for idx,dr in enumerate (features['abr']):
				dri_cc.append(features['ncfiles'][dr][member_run].variables['corr_coeff'][win,lg,lt,ln])
				dri_id.append(drivers_code[idx])
				dri_pv.append(features['ncfiles'][dr][member_run].variables['p_value'][win,lg,lt,ln])
			dri_id	= np.array(dri_id)
			dri_cc 	= np.array(dri_cc)
			dri_pv	= np.array(dri_pv)
			if (np.unique(dri_pv < p_value_th, return_counts= 1)[0][0] == False and np.unique(dri_pv < p_value_th, return_counts= 1)[1][0] == driver_consider):
				dri_dom_cc = np.ma.masked_all(driver_consider)
				dri_dom_id = np.ma.masked_all(driver_consider)
			elif sum(dri_cc) == 0: # masking all the places with zero correlation coefficients
				dri_dom_cc = np.ma.masked_all(driver_consider)
				dri_dom_id = np.ma.masked_all(driver_consider)
			else:
				dri_cp 		= dri_cc[dri_pv < p_value_th]				# driver's corr coeff with confidence
				dri_idp 	= dri_id[dri_pv < p_value_th]				# driver's id codes with confidence
				dri_cp_abs 	= np.abs(dri_cp)							# absolute driver's corr coeff with confidence
				dri_cp_des 	= dri_cp_abs[np.argsort(-dri_cp_abs)]	# absolute driver's corr coeff with confidence in desending order
				dri_dom_cc 	= []
				dri_dom_id	= []
				for c in  dri_cp_des:
					dri_dom_cc.	append(dri_cp[np.argwhere(dri_cp_abs == c)[0][0]])
					dri_dom_id.	append(dri_idp[np.argwhere(dri_cp_abs == c)[0][0]])

			dri_dom_cc_ma 		= np.ma.masked_all(driver_consider)
			dri_dom_id_ma 		= np.ma.masked_all(driver_consider)

			dri_dom_cc_ma[:len(dri_dom_cc)]	= dri_dom_cc
			dri_dom_id_ma[:len(dri_dom_id)]	= dri_dom_id
			dom_dri_corr_coeff	[:,win,lg,lt,ln]= dri_dom_cc_ma
			dom_dri_ids			[:,win,lg,lt,ln]= dri_dom_id_ma

exp= 'ssp585'
path_corr = cori_scratch + 'add_cmip6_data/%s/%s/%s/%s/Correlations/'%(source_run,exp,member_run,variable)
if os.path.isdir(path_corr) == False:
	os.makedirs(path_corr)

#with nc4.Dataset(path_corr + 'dominant_driver_correlation_%s_np2.nc'%(variable), mode="w") as dset:  # [Changed]
with nc4.Dataset(path_corr + 'dominant_driver_correlation_%s.nc'%(variable), mode="w") as dset:
	dset        .createDimension( "rank" ,size = drivers.size) 															##################
	dset        .createDimension( "win" ,size = nwin) 															##################
	dset        .createDimension( "lag" ,size = nlag) 															##################
	dset        .createDimension( "lat" ,size = lat.size) 															##################
	dset        .createDimension( "lon" ,size = lon.size)
	t   =   dset.createVariable(varname = "rank"  		,datatype = float	, dimensions = ("rank") 			,fill_value = np.nan)
	v   =   dset.createVariable(varname = "win"  		,datatype = float	, dimensions = ("win") 				,fill_value = np.nan)
	w   =   dset.createVariable(varname = "lag"  		,datatype = float	, dimensions = ("lag") 				,fill_value = np.nan)
	y   =   dset.createVariable(varname = "lat"  		,datatype = float	, dimensions = ("lat") 				,fill_value = np.nan)
	x   =   dset.createVariable(varname = "lon"  		,datatype = float	, dimensions = ("lon") 				,fill_value = np.nan)
	z   =   dset.createVariable(varname = "dri_id"  	,datatype = float	, dimensions = ("rank","win","lag","lat","lon")	,fill_value = np.nan)
	zz  =   dset.createVariable(varname = "dri_coeff"  	,datatype = float	, dimensions = ("rank","win","lag","lat","lon")	,fill_value = np.nan)
	t.axis  =   "T"
	v.axis  =   "V"
	w.axis  =   "W"
	x.axis  =   "X"
	y.axis  =   "Y"
	t[...]	=	range(len(drivers_code))
	v[...]	=	np.array([format(i,'002') for i in range(nwin)]) 
	w[...]  =   np.array([format(i,'002') for i in range(nlag)])
	x[...]  =   lon[...]
	y[...]  =   lat[...]
	z[...]	= 	dom_dri_ids
	zz[...]	= 	dom_dri_corr_coeff
	t.units			=   "rank of dominant drivers"
	t.missing_value = 	np.nan
	v.units			=   "time periods(wins) 0:1850-74, 1:1875-99...."
	v.missing_value = 	np.nan
	x.units         =   lat.units
	x.missing_value =   np.nan
	x.setncattr         ("standard_name",lat.standard_name)
	y.units         =   lon.units
	y.missing_value =   np.nan
	y.setncattr         ("standard_name",lon.standard_name)
	z.missing_value =   np.nan
	z.stardard_name =   "Dominant driver ID "
	z.setncattr         ("method",'Linear regression')
	text = ""
	for i in range(driver_consider): text = text+str(drivers_code[i]) + ":" + str(drivers[i]) +","
	z.units         =   text
	zz.missing_value =   np.nan
	zz.stardard_name =   "Dominant driver Coefficents"
#zz.stardard_name =   "Dominant driver Coefficents (no p-value filter) " # [Changed]
	zz.setncattr         ("method",'Linear Regression')
	zz.units         =   "coefficients"






