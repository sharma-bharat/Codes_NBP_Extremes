# Bharat Sharma
# python 3
# To calculate the spatial metrics of extemes
# - Regional analysis based on SREX regions
# - TCE stats of frequency and length of 
# 	* positive and negative extremes
# - Carbon gains and losses
# - Relative changes to overall changes in GPP flux
# All the zeros are masked

import  matplotlib as mpl
#mpl.use('Agg')
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from functions import time_dim_dates, index_and_dates_slicing, geo_idx, mpi_local_and_global_index, create_seq_mat, cumsum_lagged,patch_with_gaps_and_eventsize, norm, Unit_Conversions
import netCDF4 as nc4
import datetime as dt
import argparse
import pandas as pd
import os
import seaborn as sns

# Plotting Libraries
# ==================
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh

parser  = argparse.ArgumentParser()
parser.add_argument('--anomalies'   ,   '-ano'      , help = "Anomalies of carbon cycle variable"   , type= str     , default= 'gpp'    )
parser.add_argument('--variable'    ,   '-var'      , help = "Original carbon cycle variable"   , type= str     , default= 'gpp'    )
parser.add_argument('--source'      ,   '-src'      , help = "Model (Source_Run)"                   , type= str     , default= 'CESM2'  ) # Model Name
parser.add_argument('--member_idx'  ,   '-m_idx'    , help = "Member Index"                   		, type= int     , default= 0  		) # Index of the member
parser.add_argument('--plot_win'    , '-pwin'       , help = "which time period to plot? 2000-24:win 06"    , type= int, default=  6        ) # 0 to 9
parser.add_argument('--driver'  	, '-dri'        , help = "Drivers (anomalies)"    				, type= str, default= 'pr'        ) # 0 to 9
args = parser.parse_args()

# run plot_regional_analysis_drivers.py -var gpp -ano gpp -src CESM2 -dri pr -m_idx 0 -pwin 6 
print (args)

variable    	= args.variable 
source_run      = args.source
member_idx      = args.member_idx
driver		= args.driver
# Paths for reading the main files
# --------------------------------
web_path        = '/global/homes/b/bharat/results/web/'
cori_scratch    = '/global/cscratch1/sd/bharat/'
members_list    = os.listdir(cori_scratch+"add_cmip6_data/%s/ssp585/"%source_run)
member_run      = members_list[member_idx]


filepaths		= {}
filepaths[source_run] = {}
filepaths[source_run][member_run] = {}
filepaths[source_run][member_run]["anomalies"] 	= cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_anomalies_gC.nc"%(
													source_run,member_run, variable,member_run,variable)
filepaths[source_run][member_run]["variable"] 	= cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_gC.nc"%(
													source_run,member_run, variable,member_run,variable)
filepaths[source_run][member_run]["dri"] 		= cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s.nc"%(
													source_run,member_run, driver,member_run, driver)
filepaths[source_run][member_run]["dri_a"] 		= cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_anomalies.nc"%(
													source_run,member_run, driver,member_run, driver)


path_bin = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s_TCE/"%(
												source_run,member_run, variable)

win_start_years = np.arange(1850,2100,25)
filepaths[source_run][member_run]["var_TCE_1s"] = {} # To save the negative bin 1s anomalies

filepaths[source_run][member_run]["var_TCE_1s"]["neg"] = {} # To save the negative bin 1s anos for multiple wins
filepaths[source_run][member_run]["var_TCE_1s"]["pos"] = {} # To save the positive bin 1s anos for multiple wins

for w_idx, wins in enumerate(win_start_years):
	filepaths[source_run][member_run]["var_TCE_1s"]["neg"][w_idx] =path_bin + "bin_TCE_1s_neg_%d.nc"%wins
	filepaths[source_run][member_run]["var_TCE_1s"]["pos"][w_idx] =path_bin + "bin_TCE_1s_pos_%d.nc"%wins

# Reading nc files
# ----------------
nc_data		= {}
nc_data [source_run] = {}
nc_data [source_run] [member_run] = {}
nc_data [source_run] [member_run] ['var'] 		=  nc4.Dataset(filepaths[source_run][member_run]["variable"]) .variables[variable]
nc_data [source_run] [member_run] ['sftlf'] 	=  nc4.Dataset(filepaths[source_run][member_run]["variable"]) .variables["sftlf"]
nc_data [source_run] [member_run] ['areacella'] =  nc4.Dataset(filepaths[source_run][member_run]["variable"]) .variables["areacella"]
nc_data [source_run] [member_run] ['lat'] 		=  nc4.Dataset(filepaths[source_run][member_run]["variable"]) .variables["lat"]
nc_data [source_run] [member_run] ['lat_bounds']=  nc4.Dataset(filepaths[source_run][member_run]["variable"]) .variables[
																			nc_data [source_run] [member_run] ['lat'].bounds]
nc_data [source_run] [member_run] ['lon'] 		=  nc4.Dataset(filepaths[source_run][member_run]["variable"]) .variables["lon"]
nc_data [source_run] [member_run] ['lon_bounds']=  nc4.Dataset(filepaths[source_run][member_run]["variable"]) .variables[
																			nc_data [source_run] [member_run] ['lon'].bounds]
nc_data [source_run] [member_run] ['time'] 		=  nc4.Dataset(filepaths[source_run][member_run]["variable"]) .variables["time"]
nc_data [source_run] [member_run] ['ano'] 		=  nc4.Dataset(filepaths[source_run][member_run]["anomalies"]) .variables[variable]

nc_data [source_run] [member_run] ['var_TCE_1s'] =  {}
nc_data [source_run] [member_run] ['var_TCE_1s'] ["neg"] = {}
nc_data [source_run] [member_run] ['var_TCE_1s'] ["pos"] = {}

nc_data [source_run] [member_run] ['dri'] 		=  nc4.Dataset(filepaths[source_run][member_run]["dri"])   .variables[driver]
nc_data [source_run] [member_run] ['dri_ano'] 	=  nc4.Dataset(filepaths[source_run][member_run]["dri_a"]) .variables[driver]

for w_idx, wins in enumerate(win_start_years):
	nc_data [source_run] [member_run] ['var_TCE_1s'] ["neg"][w_idx] = nc4.Dataset (filepaths[source_run][member_run]["var_TCE_1s"]["neg"][w_idx]).variables['%s_TCE_1s'%variable]
	nc_data [source_run] [member_run] ['var_TCE_1s'] ["pos"][w_idx] = nc4.Dataset (filepaths[source_run][member_run]["var_TCE_1s"]["pos"][w_idx]).variables['%s_TCE_1s'%variable]

# SREX regional analysis
# -----------------------
# only for CESM2 to begin and AMZ region

source_run  = "CESM2"
member_run 	= "r1i1p1f1" 

# Unit conversion for drivers:
# ----------------------------
dri_units = nc_data[source_run][member_run]['dri'].units
if driver == 'pr':
	con_factor,new_dunits = Unit_Conversions(From=dri_units, To='mm day-1')
elif driver == 'mrso':
	con_factor,new_dunits = Unit_Conversions(From=dri_units, To='mm')
elif driver == 'tas':
	con_factor,new_dunits = Unit_Conversions(From=dri_units, To='C')

import regionmask

lat 		= nc_data [source_run] [member_run] ['lat']
lon			= nc_data [source_run] [member_run] ['lon']
lat_bounds 	= nc_data [source_run] [member_run] ['lat_bounds']
lon_bounds 	= nc_data [source_run] [member_run] ['lon_bounds']
lon_edges	= np.hstack (( lon_bounds[:,0], lon_bounds[-1,-1]))
lat_edges	= np.hstack (( lat_bounds[:,0], lat_bounds[-1,-1]))

# Creating mask of the regions based on the resolution of the model
srex_mask 	= regionmask.defined_regions.srex.mask(lon[...], lat[...]).values  # it has nans
srex_mask_ma= np.ma.masked_invalid(srex_mask) # got rid of nans; values from 1 to 26

# important information:
srex_abr		= regionmask.defined_regions.srex.abbrevs
srex_names		= regionmask.defined_regions.srex.names
srex_nums		= regionmask.defined_regions.srex.numbers 
srex_centroids	= regionmask.defined_regions.srex.centroids 
srex_polygons	= regionmask.defined_regions.srex.polygons

"""
# Masking for a particular region : "AMZ"
#not_land = np.ma.masked_equal( nc_data [source_run] [member_run] ['sftlf'][...] , 0 ) 
srex_idxs 		= np.arange(len(srex_names))      
filter_region 	= np.array(srex_abr) == 'AMZ'
region_idx		= srex_idxs[filter_region][0]
region_number	= np.array(srex_nums)[filter_region][0]
region_name		= np.array(srex_names)[filter_region][0]
region_abr		= np.array(srex_abr)[filter_region][0] 

region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region
region_mask		= ~region_mask_not   # Only the regions is masked

lt_ln_mat 			= create_seq_mat(nlat=lat.size, nlon=lon.size) #2-D lat lon grid/array with unique ids
# The one dimensional array of the region of interest
region_mask_1d 		= lt_ln_mat[region_mask]
"""
# Calculation of actual area of land pixels (areacella * sftlf)
# -------------------------------------------------------------
if source_run == 'CESM2':
	lf_units = '%'
	lf_div = 100

lf	=  nc_data[source_run][member_run]['sftlf'] [...] /lf_div
area_act = nc_data[source_run][member_run]['areacella'] [...] * lf
# Creating a dataframe to capture the TCE stats
df_tce_wins = pd.DataFrame(columns = ["win_idx",'region_abr','tce_neg', 'tce_pos', 'tce_len_neg', 'tce_len_pos', 'tce_len_tot','c_gain','c_loss','tot_var','area_neg','area_pos'])
df_dri_wins	= pd.DataFrame(columns = ["win_idx",'region_abr','tot_dri_pxs','tot_dri_du','tot_dri_reg','dri_gain','dri_loss','area_act'])
#df_tce_1s = pd.DataFrame(columns = ['region_abr','tce_neg', 'tce_pos', 'tce_len_neg', 'tce_len_pos', 'tce_len_tot','c_gain','c_loss','tot_gpp'])
#df_tce_01s = pd.DataFrame(columns = ['neg_exts', 'pos_exts', 'tot_exts','c_gain','c_loss','tot_gpp'])

# Pixel wise analysis in every region
# for region_abr (this case)
from    scipy   import  ndimage
win_size = 300
count = 0 # Just to check the length of dataframe
for region_abr in srex_abr:
	print ("Looking into the Region %s"%region_abr)
# Masking for a particular region : "AMZ"
#not_land = np.ma.masked_equal( nc_data [source_run] [member_run] ['sftlf'][...] , 0 ) 
	srex_idxs 		= np.arange(len(srex_names))      
	filter_region 	= np.array(srex_abr) == region_abr
	region_idx		= srex_idxs[filter_region][0]
	region_number	= np.array(srex_nums)[filter_region][0]
	region_name		= np.array(srex_names)[filter_region][0]
	region_abr		= np.array(srex_abr)[filter_region][0] 
	region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region
	region_mask		= ~region_mask_not   # Only the regions is masked


	lt_ln_mat 			= create_seq_mat(nlat=lat.size, nlon=lon.size) #2-D lat lon grid/array with unique ids
# The one dimensional array of the region of interest
	region_mask_1d 		= lt_ln_mat[region_mask]

	# DataFrame for every region [different indexs]	
	df_tce 		= pd.DataFrame(columns = ["win_idx",'region_abr','tce_neg', 'tce_pos', 'tce_len_neg', 'tce_len_pos', 'tce_len_tot','c_gain','c_loss','tot_var','area_neg','area_pos'])
	df_dri		= pd.DataFrame(columns = ["win_idx",'region_abr','tot_dri_pxs','tot_dri_du','tot_dri_reg' ,'dri_gain','dri_loss','area_act'])
	for w_idx, wins in enumerate(win_start_years):
		lag = 1 # month
		for pixel in region_mask_1d:
			count = count +1 # check the rows of DF
			lt,ln=np.argwhere(lt_ln_mat == pixel)[0]
			# Time series of Anomalies and variable
			ts_ano	= nc_data [source_run] [member_run] ['ano'] [w_idx * win_size: (w_idx+1) * win_size, lt,ln]
			ts_var 	= nc_data [source_run] [member_run] ['var'] [w_idx * win_size: (w_idx+1) * win_size, lt,ln]
			
			# Unit conversion for drivers:
			# ---------------
			if driver in ['pr','mrso','mrsos']:
				ts_dri 		= nc_data [source_run] [member_run] ['dri']  	[w_idx * win_size: (w_idx+3) * win_size, lt,ln]	* con_factor
				ts_dri_ano 	= nc_data [source_run] [member_run] ['dri_ano'] [w_idx * win_size: (w_idx+1) * win_size, lt,ln]	* con_factor
			elif driver in ['tas']:
				ts_dri 		= nc_data [source_run] [member_run] ['dri']  	[w_idx * win_size: (w_idx+3) * win_size, lt,ln]	+ con_factor
				ts_dri_ano 	= nc_data [source_run] [member_run] ['dri_ano'] [w_idx * win_size: (w_idx+1) * win_size, lt,ln]	+ con_factor

			# Calculation of actual land area
			#area_px	= nc_data [source_run] [member_run] ['areacella'] [ lt,ln ]
			area_act_px	= (area_act [ lt,ln ]) # m2'ytick.labelsize': 'small'
	
			# Binaries
			bin_tce_1s_neg = nc_data [source_run] [member_run] ['var_TCE_1s'] ["neg"][w_idx] [lag,:,lt,ln]
			bin_tce_1s_pos = nc_data [source_run] [member_run] ['var_TCE_1s'] ["pos"][w_idx] [lag,:,lt,ln]

			
			# Finding contiguous events
			neg_mask 	= 'no'
			if (bin_tce_1s_neg.mask.size == bin_tce_1s_neg.shape[0]) or (bin_tce_1s_neg.sum()==0):  # all neg bins masked
				bin_tce_1s_neg = np.zeros(bin_tce_1s_neg.shape[0])
				neg_mask = 'yes'
			
			
			pos_mask	= 'no'
			if (bin_tce_1s_pos.mask.size == bin_tce_1s_pos.shape[0]) or (bin_tce_1s_pos.sum()==0):  # all pos bins masked
				bin_tce_1s_pos = np.zeros(bin_tce_1s_pos.shape[0])
				pos_mask = 'yes'


			# index will be created only once
			df_tce_index = "%s_%s"%(format(lt,'003'),format(ln,'003')) # based on concatenated lat-lon
			if df_tce_index not in df_tce.index:
				df_tce = df_tce.reindex(df_tce.index.values.tolist()+[df_tce_index])
		
			# index will be created only once drivers
			df_dri_index = "%s_%s"%(format(lt,'003'),format(ln,'003')) # based on concatenated lat-lon
			if df_dri_index not in df_dri.index:
				df_dri = df_dri.reindex(df_dri.index.values.tolist()+[df_dri_index])

			df_dri .loc[df_tce_index,'tot_dri_reg']     = ts_dri.sum() 
			if (neg_mask == 'yes') or (pos_mask == 'yes'):
				print ("%s"%region_abr,wins,lt,ln)
				df_tce .loc[df_tce_index,'win_idx'] 	= w_idx
				df_tce .loc[df_tce_index,'region_abr'] 	= region_abr
				df_tce .loc[df_tce_index,'c_loss'] 		= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'c_gain'] 		= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tot_var'] 	= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'area_neg'] 	= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'area_pos'] 	= np.ma.masked_equal(0,0)        

				df_tce .loc[df_tce_index,'tce_neg'] 	= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tce_pos'] 	= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tce_len_neg'] = np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tce_len_pos'] = np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tce_len_tot'] = np.ma.masked_equal(0,0)        
				
				df_dri .loc[df_tce_index,'win_idx']     = w_idx
				df_dri .loc[df_tce_index,'region_abr']  = region_abr
				df_dri .loc[df_tce_index,'tot_dri_pxs'] = np.ma.masked_equal(0,0)
				df_dri .loc[df_tce_index,'tot_dri_du']  = np.ma.masked_equal(0,0)
				df_dri .loc[df_tce_index,'dri_gain']  	= np.ma.masked_equal(0,0)
				df_dri .loc[df_tce_index,'dri_loss']  	= np.ma.masked_equal(0,0)
				df_dri .loc[df_tce_index,'area_act']  	= np.ma.masked_equal(0,0)
				continue
			

			# for non-masked TCE regions:
			# ---------------------------
			larray_neg, narray_neg   = ndimage.label(bin_tce_1s_neg,structure = np.ones(3))
			locations_neg       = ndimage.find_objects(larray_neg)

			larray_pos, narray_pos   = ndimage.label(bin_tce_1s_pos,structure = np.ones(3))
			locations_pos       = ndimage.find_objects(larray_pos)
			
			
			# local index of TS
			idxs    = np.asarray(np.arange(len(ts_ano)), dtype = int)
			
			# indexs of negative and positive TCE points and carbon losses
			tce_idxs_neg = np.array([])
			tce_ano_neg = np.array([])
			tce_dri_ano_neg = np.array([]) # to capture the driver anomalies during a neg TCE event
			tce_dri_neg = np.array([]) # to capture the driver during a neg TCE event
			for loc in locations_neg:
				tce_idxs_neg	= np.append(tce_idxs_neg, idxs[loc])
				tce_ano_neg 	= np.append(tce_ano_neg, ts_ano[loc])	
				tce_dri_ano_neg = np.append(tce_dri_ano_neg, ts_dri_ano[loc])	
				tce_dri_neg 	= np.append(tce_dri_neg, ts_dri[loc])	

			tce_idxs_neg 	= np.asarray(tce_idxs_neg, dtype = int)
			tce_ano_neg 	= np.asarray(tce_ano_neg)
			tce_dri_ano_neg = np.asarray(tce_dri_ano_neg)
			tce_dri_neg 	= np.asarray(tce_dri_neg)

			# indexs of negative and positive TCE points and carbon losses
			tce_idxs_pos 	= np.array([])
			tce_ano_pos 	= np.array([])
			tce_dri_ano_pos = np.array([])
			tce_dri_pos = np.array([])
			for loc in locations_pos:
				tce_idxs_pos 	= np.append(tce_idxs_pos, idxs[loc])
				tce_ano_pos 	= np.append(tce_ano_pos, ts_ano[loc])	
				tce_dri_ano_pos = np.append(tce_dri_ano_pos, ts_dri_ano[loc])	
				tce_dri_pos 	= np.append(tce_dri_pos, ts_dri[loc])	
			tce_idxs_pos 	= np.asarray(tce_idxs_pos, dtype = int)
			tce_ano_pos 	= np.asarray(tce_ano_pos)
			tce_dri_ano_pos = np.asarray(tce_dri_ano_pos)
			tce_dri_pos 	= np.asarray(tce_dri_pos)

			# all extremes
			tce_idxs_concate    = np.concatenate((tce_idxs_neg,tce_idxs_pos))
			df_dri .loc[df_tce_index,'tot_dri_reg']     = ts_dri.sum() if driver not in ['tas','tasmax','tasmin'] else ts_dri.mean()

			if (neg_mask == 'yes') or (pos_mask == 'yes'): # The block will not execute because of already existing continue command for this condition
				print ("%s"%region_abr,wins,lt,ln)
				df_tce .loc[df_tce_index,'win_idx'] 	= w_idx
				df_tce .loc[df_tce_index,'region_abr'] 	= region_abr
				df_tce .loc[df_tce_index,'c_loss'] 		= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'c_gain'] 		= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tot_var'] 	= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'area_neg'] 	= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'area_pos'] 	= np.ma.masked_equal(0,0)        

				df_tce .loc[df_tce_index,'tce_neg'] 	= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tce_pos'] 	= np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tce_len_neg'] = np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tce_len_pos'] = np.ma.masked_equal(0,0)        
				df_tce .loc[df_tce_index,'tce_len_tot'] = np.ma.masked_equal(0,0)        
				
				df_dri .loc[df_tce_index,'win_idx']     = w_idx
				df_dri .loc[df_tce_index,'region_abr']  = region_abr
				df_dri .loc[df_tce_index,'tot_dri_pxs'] = np.ma.masked_equal(0,0)
				df_dri .loc[df_tce_index,'tot_dri_du']  = np.ma.masked_equal(0,0)
				df_dri .loc[df_tce_index,'dri_gain']  	= np.ma.masked_equal(0,0)
				df_dri .loc[df_tce_index,'dri_loss']  	= np.ma.masked_equal(0,0)
				df_dri .loc[df_tce_index,'area_act']  	= np.ma.masked_equal(0,0)

			else:
				print ("%s"%region_abr,wins,lt,ln)
				df_tce .loc[df_tce_index,'win_idx'] 	= w_idx
				df_tce .loc[df_tce_index,'region_abr'] 	= region_abr
				df_tce .loc[df_tce_index,'c_loss'] 		= tce_ano_neg.sum()
				df_tce .loc[df_tce_index,'c_gain'] 		= tce_ano_pos.sum()
				df_tce .loc[df_tce_index,'tot_var'] 	= ts_var.sum()
				df_tce .loc[df_tce_index,'area_neg'] 	= area_act_px * len(tce_idxs_neg)
				df_tce .loc[df_tce_index,'area_pos'] 	= area_act_px * len(tce_idxs_pos)

				df_tce .loc[df_tce_index,'tce_neg'] 	= len(locations_neg)
				df_tce .loc[df_tce_index,'tce_pos'] 	= len(locations_pos)
				df_tce .loc[df_tce_index,'tce_len_neg'] = len(tce_idxs_neg)
				df_tce .loc[df_tce_index,'tce_len_pos'] = len(tce_idxs_pos)
				df_tce .loc[df_tce_index,'tce_len_tot'] = len(tce_idxs_concate)

				df_dri .loc[df_tce_index,'win_idx']     = w_idx
				df_dri .loc[df_tce_index,'region_abr']  = region_abr
				df_dri .loc[df_tce_index,'tot_dri_pxs'] = ts_dri.sum() if driver not in ['tas','tasmax','tasmin'] else ts_dri.mean()
				df_dri .loc[df_tce_index,'tot_dri_du']  = (tce_dri_neg.sum() if driver not in ['tas','tasmax','tasmin'] else tce_dri_neg.mean()+ tce_dri_pos.sum() if driver not in ['tas','tasmax','tasmin'] else tce_dri_pos.mean())
				df_dri .loc[df_tce_index,'dri_gain']  	= tce_dri_ano_pos.sum() if driver not in ['tas','tasmax','tasmin'] else tce_dri_ano_pos.mean() 
				df_dri .loc[df_tce_index,'dri_loss']  	= tce_dri_ano_neg.sum() if driver not in ['tas','tasmax','tasmin'] else tce_dri_ano_neg.mean() 
				df_dri .loc[df_tce_index,'area_act']  	= area_act_px

		# Appending all the DataFrames for different Time windows
		df_tce_wins	= df_tce_wins.append(df_tce)
		df_dri_wins	= df_dri_wins.append(df_dri)

# Saving the DataFrame
path_save	= cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Stats/"%(source_run,member_run, variable)
if os.path.isdir(path_save) == False:
	os.makedirs(path_save)
#df_tce_wins. to_csv (path_save + "%s_%s_%s_%s_TCE_Stats_Wins.csv"%(source_run,member_run,variable,driver))
df_dri_wins. to_csv (path_save + "%s_%s_%s_%s_TCE_Stats_Wins.csv"%(source_run,member_run,variable,driver))


#Main DataFrame : df_tce_wins

# Finding the basic statistics of all above metrics for every region and time window
from functions import MaskedConstant_Resolve, MaskedArray_Resolve
df_summary_wins	= pd. DataFrame(columns = ["win_idx",'region_abr','tce_neg', 'tce_pos', 'tce_len_neg', 'tce_len_pos', 'tce_len_tot','c_gain','c_loss','tot_var','area_neg','area_pos'])
df_summary_wins_dri	= pd. DataFrame(columns = ["win_idx",'region_abr','dri_gain_awm','dri_loss_awm','dri_reg_awm','dri_du_awm'])

for region_abr in srex_abr:

	df_summary		= pd. DataFrame(columns = ["win_idx",'region_abr','tce_neg', 'tce_pos', 'tce_len_neg', 'tce_len_pos', 'tce_len_tot','c_gain','c_loss','tot_var','area_neg','area_pos'])
	df_summary_dri	= pd. DataFrame(columns = ["win_idx",'region_abr', 'dri_gain_awm' ,'dri_loss_awm',  'dri_reg_awm','dri_du_awm'])
	for w_idx, wins in enumerate(win_start_years):
		
		filter_win_region 		= (df_tce_wins["win_idx"] == w_idx ) & (df_tce_wins["region_abr"] == region_abr) 
		filter_win_region_dri 	= (df_dri_wins["win_idx"] == w_idx ) & (df_dri_wins["region_abr"] == region_abr) 
		
		df_tmp		= df_tce_wins [filter_win_region]
		df_tmp_dri	= df_dri_wins [filter_win_region_dri]

		tce_stats = {}
		for col in df_tce.columns[2:]:
			tce_stats[col] 			= {}
			tce_stats[col]['mean']	= np.ma.mean(MaskedArray_Resolve(df_tmp[col]))
			tce_stats[col]['std']	= np.ma.std(MaskedArray_Resolve(df_tmp[col]))
			tce_stats[col]['max']	= np.ma.max(MaskedArray_Resolve(df_tmp[col]))
			tce_stats[col]['sum']	= np.ma.sum(MaskedArray_Resolve(df_tmp[col]))
		
		dri_stats = {}
		dri_stats['dri_loss_awm'] = {}
		dri_stats['dri_loss_awm']['mean']	= np.ma.average(MaskedArray_Resolve	(df_tmp_dri['dri_loss']), 
													weights = MaskedArray_Resolve    (df_tmp_dri['area_act']))
		dri_stats['dri_gain_awm'] = {}
		dri_stats['dri_gain_awm']['mean']	= np.ma.average(MaskedArray_Resolve	(df_tmp_dri['dri_gain']), 
														weights = MaskedArray_Resolve    (df_tmp_dri['area_act']))

		dri_stats['dri_reg_awm'] = {}
		dri_stats['dri_reg_awm']['mean']	= np.ma.average(MaskedArray_Resolve	(df_tmp_dri['tot_dri_reg']), 
														weights = MaskedArray_Resolve    (df_tmp_dri['area_act']))
		dri_stats['dri_du_awm'] = {}
		dri_stats['dri_du_awm']['mean']	= np.ma.average(MaskedArray_Resolve	(df_tmp_dri['tot_dri_du']), 
														weights = MaskedArray_Resolve    (df_tmp_dri['area_act']))

		for k in tce_stats[col].keys():
			if k not in df_summary.index:
				df_summary	= df_summary.reindex(df_summary.index.values.tolist() + [k])
		
		for kd in dri_stats['dri_du_awm'].keys():
			if kd not in df_summary_dri.index:
				df_summary_dri	= df_summary_dri.reindex(df_summary_dri.index.values.tolist() + [kd])
		
		for k in tce_stats[col].keys():
			for col in df_tce.columns[2:]:
				df_summary.loc[k,col] = tce_stats[col][k]
				df_summary.loc[k,"win_idx"] 	= w_idx
				df_summary.loc[k,"region_abr"]	= region_abr
	
		for kd in dri_stats["dri_du_awm"].keys():
			for col_dri in df_summary_dri.columns[2:]:
				df_summary_dri.loc[kd,col_dri] = dri_stats[col_dri][kd]
				df_summary_dri.loc[kd,"win_idx"] 	= w_idx
				df_summary_dri.loc[kd,"region_abr"]	= region_abr

		# Summary of results for all windows
		df_summary_wins = df_summary_wins.append(df_summary)
		df_summary_wins_dri = df_summary_wins_dri.append(df_summary_dri)

# Storing the summary
#df_summary_wins. to_csv (path_save + "%s_%s_%s_TCE_Stats_Wins_Summary_mask.csv"%(source_run,member_run,variable))
df_summary_wins_dri. to_csv (path_save + "%s_%s_%s_%s_TCE_Stats_Wins_Summary.csv"%(source_run,member_run,variable,driver))


# Plotting of the region of interest
# ----------------------------------
# Cartopy Plotting
# ----------------
import cartopy.crs as ccrs
from shapely.geometry.polygon import Polygon
import cartopy.feature as cfeature

# Fixing the error {'GeoAxesSubplot' object has no attribute '_hold'}
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
proj_trans = ccrs.PlateCarree()


# Plotting the TS of regions:
web_path	= '/global/homes/b/bharat/results/web/'
# For x-axis
in_yr   = 1850
win_yr  = [str(in_yr+i*25) + '-'+str(in_yr +(i+1)*25-1)[2:] for i in range(win_start_years.size)]

# Plotting the Timeseries of Box plot for all regions:
# ----------------------------------------------------
"""
for region_abr in srex_abr:
# TS: Box Plots of Columns:
#region_abr      = 'AMZ'
# This filter will only keep the region of interest and mask everything else
	filter_region 	= df_tce_wins['region_abr'] == region_abr

# droping the column of region_abr because it is str and I want df to be float 
	df_tmp			= df_tce_wins[filter_region].drop(columns=['region_abr']).astype(float)
	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"
	for col in df_tmp.columns[1:]:
		df_tmp[col] = df_tmp[col].mask(df_tmp[col]==0)
		fig,ax = plt.subplots(figsize=(12,5))
		ax = sns.boxplot(x="Wins", y=col, data=df_tmp)
		fig.savefig(web_path + 'Regional/%s_%s_%s_%s_box.pdf'%(source_run,member_run,region_abr,col))
		fig.savefig(path_save + '%s_%s_%s_%s_box.pdf'%(source_run,member_run,region_abr,col))
		plt.close(fig)
del df_tmp
"""

# Box Plotting
'''
for region_abr in srex_abr:
# TS: Box Plots of Columns:
#region_abr      = 'AMZ'
# This filter will only keep the region of interest and mask everything else
	filter_region 	= df_dri_wins['region_abr'] == region_abr

# droping the column of region_abr because it is str and I want df to be float 
	df_tmp			= df_dri_wins[filter_region].drop(columns=['region_abr']).astype(float)
	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"
	for col in df_tmp.columns[1:]:
		df_tmp[col] = df_tmp[col].mask(df_tmp[col]==0)
		fig,ax = plt.subplots(figsize=(12,5))
		ax = sns.boxplot(x="Wins", y=col, data=df_tmp)
		fig.savefig(web_path + 'Regional/%s_%s_%s_%s_%s_box.pdf'%(source_run,member_run,region_abr,col,driver))
		fig.savefig(path_save + '%s_%s_%s_%s_%s_box.pdf'%(source_run,member_run,region_abr,col,driver))
		plt.close(fig)
del df_tmp
'''
"""
# Creating the DataFrame of the interested stats for regions
# interested mean and sum of 
# 	* c_gain, c_loss, tot_var rel%, change in carbon_uptake

dict_carbon = {}
for r_idx, region_abr in enumerate(srex_abr):
#region_abr      = 'AMZ'
	filter_region   = df_summary_wins['region_abr'] == region_abr
	df_tmp          = df_summary_wins[filter_region].drop(columns=['region_abr']).astype(float)

	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"

# Only looking at SUM or total change in a region
	df_sum = df_tmp[df_tmp.index == 'sum']
# Checking for 'Sum'
	df_carbon = pd.DataFrame(columns = ["Uptake_Gain", "Uptake_Loss", "Uptake_Change", "%s"%variable.upper(),"Percent_Gain","Percent_Loss"])
	for win_idx in win_yr: 
		if df_tce_index not in df_carbon.index: 
			df_carbon = df_carbon.reindex(df_carbon.index.values.tolist()+[win_idx]) 
	for win_str in df_sum['Wins']:
		df_carbon.loc[win_str, "Uptake_Gain"] = (df_sum[df_sum['Wins'] == win_str]['c_gain']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Loss"] = (df_sum[df_sum['Wins'] == win_str]['c_loss']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Change"] = df_carbon.loc[win_str, "Uptake_Gain"] +  df_carbon.loc[win_str, "Uptake_Loss"]
		df_carbon.loc[win_str, "%s"%variable.upper()] = (df_sum[df_sum['Wins'] == win_str]['tot_var']/10**15).values[0]
		df_carbon.loc[win_str, "Percent_Gain"] 	= df_carbon.loc[win_str, "Uptake_Gain"]*100/df_carbon.loc[win_str, "%s"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Loss"] 	= df_carbon.loc[win_str, "Uptake_Loss"]*100/df_carbon.loc[win_str, "%s"%variable.upper()]
	df_carbon.to_csv(web_path + 'Regional/%s_%s_%s_CarbonUptake_PgC.csv'%(source_run,member_run,region_abr))
	df_carbon.to_csv(path_save + '%s_%s_%s_CarbonUptake_PgC.csv'%(source_run,member_run,region_abr))
	dict_carbon [region_abr] = df_carbon
	if r_idx == 0:
		df_carbon_all = df_carbon # df_carbon_all is for global stats of carbon uptake; intializing it
	else:
		df_carbon_all = df_carbon_all + df_carbon	
	del df_carbon

# Updating the percent Gain and Loss of carbon uptake
# ---------------------------------------------------
df_carbon_all['Percent_Gain'] = df_carbon_all['Uptake_Gain']*100/df_carbon_all['%s'%variable.upper()] 
df_carbon_all['Percent_Loss'] = df_carbon_all['Uptake_Loss']*100/df_carbon_all['%s'%variable.upper()] 
dict_carbon [ 'ALL'] = df_carbon_all
"""
# Creating the DataFrame of the interested stats of drivers for regions
# interested mean  of 
# * dri_gain, dri_loss, tot_dri rel%, change in driver_uptake
# - Driver_du_Gain :  Driver anomalies during gain in carbon uptake [pos ext]

# awm : Area weighted mean [when all the variables are averaged for a pixel and then we report awm of averages]
# awsm : Area weighted sum mean [ when all the variables are summed for a pix and then we report the awm of sum]

# Mean changes in the driver at lag = 1 month
dict_driver = {}
for r_idx, region_abr in enumerate(srex_abr):

	filter_region   = np.array(srex_abr) == region_abr     #for finding srex number
	region_number   = np.array(srex_nums)[filter_region][0] # for srex mask
	del filter_region
	region_mask_not = np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region; Mask of region is False
	region_mask     = ~region_mask_not   # Only the mask of region is True

	filter_region   = df_summary_wins_dri['region_abr'] == region_abr
	df_tmp          = df_summary_wins_dri[filter_region].drop(columns=['region_abr']).astype(float)

	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"

# Only looking at Mean change in a region
	df_mean = df_tmp[df_tmp.index == 'mean']
# Checking for 'Sum'
	df_driver = pd.DataFrame(columns = ["Driver_du_Gain", "Driver_du_Loss", "Driver_Change", "%s_reg_AWM"%driver.upper(), "%s_du_ext_awsm"%driver.upper(),  "%s_reg_awsm"%driver.upper(),"Percent_Dri_Gain","Percent_Dri_Loss"])
	for win_idx in win_yr: 
		if df_dri_index not in df_driver.index: 
			df_driver = df_driver.reindex(df_driver.index.values.tolist()+[win_idx]) 
	for w_idx, win_str in enumerate(df_mean['Wins']):
		df_driver.loc[win_str, "Driver_du_Gain"] = ((df_mean[df_mean['Wins'] == win_str]['dri_gain_awm']).values[0]) 
		df_driver.loc[win_str, "Driver_du_Loss"] = ((df_mean[df_mean['Wins'] == win_str]['dri_loss_awm']).values[0]) 
		df_driver.loc[win_str, "Driver_Change"] = df_driver.loc[win_str, "Driver_du_Gain"] +  df_driver.loc[win_str, "Driver_du_Loss"]
	
		# The following line has a masking error
		df_driver.loc[win_str, "%s_reg_AWM"%driver.upper()] =con_factor * np.ma.average(
																np.ma.mean(nc_data [source_run] [member_run] ['dri'] [w_idx * win_size: (w_idx+1) * win_size,:,:], axis=0) [region_mask],
																weights = area_act [region_mask])

		#df_driver.loc[win_str, "%s_reg"%driver.upper()] = ((df_mean[df_mean['Wins'] == win_str]['tot_dri_reg']).values[0]) * m_factor
		#df_driver.loc[win_str, "%s_du_ext"%driver.upper()] = ((df_mean[df_mean['Wins'] == win_str]['tot_dri_du']).values[0]) 
		df_driver.loc[win_str, "%s_du_ext_awsm"%driver.upper()] = ((df_mean[df_mean['Wins'] == win_str]['dri_du_awm']).values[0]) 
		df_driver.loc[win_str, "%s_reg_awsm"%driver.upper()] = ((df_mean[df_mean['Wins'] == win_str]['dri_reg_awm']).values[0])  # Area weighted sum mean
		# Percent Dri and Losses are w.r.t. to driver values during an extreme event
		df_driver.loc[win_str, "Percent_Dri_Gain"] 	= df_driver.loc[win_str, "Driver_du_Gain"]*100/df_driver.loc[win_str, "%s_du_ext_awsm"%driver.upper()]
		df_driver.loc[win_str, "Percent_Dri_Loss"] 	= df_driver.loc[win_str, "Driver_du_Loss"]*100/df_driver.loc[win_str, "%s_du_ext_awsm"%driver.upper()]
	df_driver.to_csv(web_path + 'Regional/%s_%s_%s_Driver_Change_%s.csv'%(source_run,member_run,region_abr,driver))
	df_driver.to_csv(path_save + '%s_%s_%s_Driver_Change_%s.csv'%(source_run,member_run,region_abr,driver))
	dict_driver [region_abr] = df_driver
	if r_idx == 0:
		df_driver_all = df_driver.fillna(0).copy() # df_carbon_all is for global stats of carbon uptake; intializing it
	else:
		df_driver_all = df_driver_all.copy() + df_driver.fillna(0).copy()
	#del df_driver

# Updating the percent Gain and Loss of carbon uptake
# ---------------------------------------------------
df_driver_all['Percent_Dri_Gain'] = df_driver_all['Driver_du_Gain']*100/df_driver_all['%s_du_ext_awsm'%driver.upper()] 
df_driver_all['Percent_Dri_Loss'] = df_driver_all['Driver_du_Loss']*100/df_driver_all['%s_du_ext_awsm'%driver.upper()] 
dict_driver [ 'ALL'] = df_driver_all


# Plotting the regional TS of driver and drivers anomalies during negative TCE events 
# -------------------------------------------------------------------------------------------------
x = dict_driver [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
          'legend.frameon': 'False',
          'axes.labelsize':'small',
          'ytick.labelsize': 'small',
          'xtick.labelsize': 'small',
          'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', 
            gridspec_kw={'hspace': .4, 'wspace': .4}, figsize=(6,9))
plt.title ("TS od driver %s and driver anomalies during negative TCEs"%(driver.upper()))
txt ="The left y-axis represents the area weighted mean of TS of driver %s for a region\n"%driver.upper()
txt+="The right y-axis represents the mean of sum of TS (at px) of driver %s anomalies during negative TCE or carbon loss\n"%driver.upper()
txt+="Units : %s"%new_dunits

axs     = axs.ravel()

for k_idx, key in enumerate(dict_driver.keys()):
	axs[k_idx] .plot(x, dict_driver[key]['%s_reg_AWM'%driver.upper()], 'k', linewidth = 0.6 ,label = key)
	ar= np.array([dict_driver[key]['Driver_du_Loss'].min(), 
             dict_driver[key]['Driver_du_Loss'].max()])

	ax1 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning TCE
	
	ax1 .plot(x, dict_driver[key]['Driver_du_Loss']  , 'r', linewidth = 0.4)
	ax1 .set_ylim()# ar.min()*.95,ar.max()*1.05)
	
	axs[k_idx] . legend(loc="upper left")
#for ax in axs.flat:
#    ax.label_outer()
for tick in axs[-3].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
    tick.set_rotation(45)
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12) #Caption
fig.savefig(web_path+'Regional/%s_%s_%s_driver_ano_du_neg_TCE.pdf'%(source_run,variable.upper(), driver.upper()))
fig.savefig(path_save+'%s_TS_%s_%s_driver_ano_du_neg_TCE.pdf'%(source_run,variable.upper(), driver.upper()))
plt.close(fig)

# Plotting the regional TS drivers anomalies during negative and positiveTCE events 
# -------------------------------------------------------------------------------------------------	
x = dict_driver [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
          'legend.frameon': 'False',
          'axes.labelsize':'small',
          'ytick.labelsize': 'small',
          'xtick.labelsize': 'small',
          'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', 
            gridspec_kw={'hspace': .4, 'wspace': .4}, figsize=(6,9))
plt.title ("TS od driver %s and driver anomalies during negative TCEs"%(driver.upper()))
txt ="The left y-axis represents the area weighted mean of sum of TS (at px) of driver %s for a region\n"%driver.upper()
txt+="The right y-axis represents the mean of sum of TS (at px) of driver %s anomalies during negative TCE or carbon loss\n"%driver.upper()
txt+="Units : %s"%new_dunits

axs     = axs.ravel()

for k_idx, key in enumerate(dict_driver.keys()):
	axs[k_idx] .plot(x, dict_driver[key]['Driver_du_Gain'], 'g', linewidth = 0.6 ,label = key)


	ar= np.array([abs(dict_driver[key]['Driver_du_Loss'].min()), 
             	abs(dict_driver[key]['Driver_du_Loss'].max()),
				abs(dict_driver[key]['Driver_du_Gain'].min()),
				abs(dict_driver[key]['Driver_du_Gain'].max())])

	axs[k_idx] .set_ylim( -ar.max(),ar.max())
	ax1 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning TCE
	
	ax1 .plot(x, dict_driver[key]['Driver_du_Loss']  , 'r', linewidth = 0.6)
	ax1 .set_ylim( -ar.max(),ar.max())
	
	axs[k_idx] . legend(loc="upper left")
#for ax in axs.flat:
#    ax.label_outer()
for tick in axs[-3].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
    tick.set_rotation(45)
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12) #Caption
fig.savefig(web_path+'Regional/%s_%s_%s_driver_ano_du_TCE.pdf'%(source_run,variable.upper(), driver.upper()))
fig.savefig(path_save+'%s_TS_%s_%s_driver_ano_du_TCE.pdf'%(source_run,variable.upper(), driver.upper()))
plt.close(fig)

# Plot of percent trends of driver anomalies during negative TCEs at lag =1
# ---------------------------------------------------------------
params = {'axes.labelsize':'medium',
          'ytick.labelsize': 12,
          'xtick.labelsize': 10,
          'font.size':12 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=1, ncols=1)
for k_idx, key in enumerate(dict_driver.keys()):
    if key in ['ALA','NEU','CGI']:
        continue
    dict_driver[key]['Percent_Dri_Loss'].plot(linewidth=.4, label=key)
axs = dict_driver[key]['Percent_Dri_Loss'].plot(linewidth=1, color='k', label=key)
plt.title ("Percent change in  %s anomalies w.r.t. %s during negative TCEs"%(driver.upper(), driver.upper()), fontsize=14)
txt="Colored lines represent the regions and black line represents total %s"%driver.upper()
#plt.legend()
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12) #Caption
fig.savefig(web_path+'Regional/%s_%s_%s_percent_driver_ano_du_TCE.pdf'%(source_run, variable.upper(), driver.upper()))
fig.savefig(path_save+'%s_TS_%s_%s_percent_driver_ano_du_TCE.pdf'%(source_run,variable.upper(), driver.upper()))

