# Bharat Sharma
# python 3
# This code is to be run after the main dataframe " df_bin_wins and df_tce_wins are already created ...
# ... Using the file "calc_plot_regional_analysis.py" ...
# ... and the files saved e.g. /global/cscratch1/sd/bharat/add_cmip6_data/CESM2/ssp585/r1i1p1f1/nbp/Stats/"%s_%s_%s_Bin_Stats_Wins_mask.csv"%(source_run,member_run,variable)
# To calculate the spatial metrics of extemes
# - Regional analysis based on SREX regions
# - TCE stats of frequency and length of 
# 	* positive and negative extremes
# - Carbon gains and losses
# - Relative changes to overall changes in NBP flux
# - Extremes happening during CUP and CRp

import  matplotlib as mpl
#mpl.use('Agg')
from matplotlib import cm
import numpy as np
import matplotlib.pyplot as plt
from functions import time_dim_dates, index_and_dates_slicing, geo_idx, mpi_local_and_global_index, create_seq_mat, cumsum_lagged,patch_with_gaps_and_eventsize, norm
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
#from cartopy.mpl.geoaxes import GeoAxes
#GeoAxes._pcolormesh_patched = Axes.pcolormesh

parser  = argparse.ArgumentParser()
parser.add_argument('--anomalies'    ,   '-ano'      , help = "Anomalies of carbon cycle variable"   , type= str     , default= 'gpp'    )
parser.add_argument('--variable'    ,   '-var'      , help = "Original carbon cycle variable"   , type= str     , default= 'gpp'    )
parser.add_argument('--source'      ,   '-src'      , help = "Model (Source_Run)"                   , type= str     , default= 'CESM2'  ) # Model Name
parser.add_argument('--member_idx'  ,   '-m_idx'    , help = "Member Index"                   		, type= int     , default= 0  		) # Index of the member
#parser.add_argument('--plot_win'    , '-pwin'       , help = "which time period to plot? 2000-24:win 06"    , type= int, default=  6        ) # 0 to 9
args = parser.parse_args()

# run plot_regional_analysis_postDF_nbp.py -var nbp  -ano nbp -src CESM2 -m_idx 0 
print (args)

variable    	= args.variable 
source_run      = args.source
member_idx      = args.member_idx

# Paths for reading the main files
# --------------------------------
web_path        = '/global/homes/b/bharat/results/web/'
cori_scratch    = '/global/cscratch1/sd/bharat/'
members_list    = os.listdir(cori_scratch+"add_cmip6_data/%s/ssp585/"%source_run)
member_run      = members_list[member_idx]


filepaths		= {}
filepaths[source_run] = {}
filepaths[source_run][member_run] = {}
filepaths[source_run][member_run]["anomalies"] = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_anomalies_gC.nc"%(
												source_run,member_run, variable,member_run,variable)
filepaths[source_run][member_run]["variable"] = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_gC.nc"%(
												source_run,member_run, variable,member_run,variable)
path_bin = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s_TCE/"%(
												source_run,member_run, variable)
# File paths to the binary (without TCE)
filepaths[source_run][member_run]["bin"] = {} # To save the negative bin without TCE anomalies
filepaths[source_run][member_run]["bin"]["neg"] = path_bin + '%s_%s_bin_neg.nc' %(source_run,member_run) # To save the negative bin
filepaths[source_run][member_run]["bin"]["pos"] = path_bin + '%s_%s_bin_pos.nc' %(source_run,member_run) # To save the positive bin

# Filepaths to binary of TCEs
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

# bin without TCE (same shape as gpp/ano):
nc_data [source_run] [member_run] ['bin'] = {}
nc_data [source_run] [member_run] ['bin'] ['neg'] = nc4.Dataset(filepaths[source_run][member_run]["bin"]["neg"]) .variables['%s_bin'%variable]
nc_data [source_run] [member_run] ['bin'] ['pos'] = nc4.Dataset(filepaths[source_run][member_run]["bin"]["pos"]) .variables['%s_bin'%variable]

# bin with TCEs (per win):
nc_data [source_run] [member_run] ['var_TCE_1s'] =  {}
nc_data [source_run] [member_run] ['var_TCE_1s'] ["neg"] = {}
nc_data [source_run] [member_run] ['var_TCE_1s'] ["pos"] = {}
for w_idx, wins in enumerate(win_start_years):
	nc_data [source_run] [member_run] ['var_TCE_1s'] ["neg"][w_idx] = nc4.Dataset (filepaths[source_run][member_run]["var_TCE_1s"]["neg"][w_idx]).variables['%s_TCE_1s'%variable]
	nc_data [source_run] [member_run] ['var_TCE_1s'] ["pos"][w_idx] = nc4.Dataset (filepaths[source_run][member_run]["var_TCE_1s"]["pos"][w_idx]).variables['%s_TCE_1s'%variable]


# SREX regional analysis
# -----------------------
# only for CESM2 to begin and AMZ region

source_run  = "CESM2"
member_run 	= "r1i1p1f1" 

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
Meaning of the stats:
=====================
Common to df_bin_wins and df_tce_wins:
--------------------------------------
	* win_idx 	: The index of the 25 yr time windows starting 1850
	* region_abr: The SREX region shortname or abr 

Specific to df_bin_wins:
------------------------
	* fq_neg/fq_pos		: Count of total months affect by neg/pos extremes (non-TCE)
	* c_gain/c_loss		: Total carbon uptake gain or loss due to positive or negative extremes (non -TCE)
	* reg_var			: Total GPP irrespective of if there is a bin or not
	* tot_var 			: Total GPP of the location/pixel for 25 years when at least one extreme has occured
	* tot_var_ext		: Total GPP of the location/pixel for 25 years when either a postive or negative carbon extremes as occuerd
							e.g. if a loc witnessed 30 pos and 40 neg extremes so this 'tot_var_ext' will give the gpp of 70 exts
	* area_neg/area_pos	: Total area affected by either negative or positive extreme  (Areacella * sftlf)
	* count_reg			: Count of total number of pixels if a region with non-zero carbon flux values 
	* count_px			: Count of total number pixels for every 25 years when at least one extreme has occured 

"""
# Creating a dataframe to capture the Bin stats (without TCE)
#df_bin_wins = pd.DataFrame(columns = ["win_idx",'region_abr','fq_neg', 'fq_pos', 'c_gain', 'c_loss','reg_var' ,'tot_var', 'tot_var_ext', 'area_neg', 'area_pos'])

# Creating a dataframe to capture the TCE stats
#df_tce_wins = pd.DataFrame(columns = ["win_idx",'region_abr','tce_neg', 'tce_pos', 'tce_len_neg', 'tce_len_pos', 'tce_len_tot','c_gain','c_loss', 'reg_var','tot_var', 'tot_var_ext', 'area_neg','area_pos'])

# Calculation of actual area of land pixels (areacella * sftlf)
# -------------------------------------------------------------
if source_run == 'CESM2':
    lf_units = '%'
    lf_div = 100

lf  =  nc_data[source_run][member_run]['sftlf'] [...] /lf_div
area_act = nc_data[source_run][member_run]['areacella'] [...] * lf


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
	region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region; Mask of region is False
	region_mask		= ~region_mask_not   # Only the mask of region is True

# Reading the dataframes
# ======================
path_save	= cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Stats/"%(source_run,member_run, variable)
df_tce_wins = pd.read_csv (path_save + "%s_%s_%s_TCE_Stats_Wins_mask.csv"%(source_run,member_run,variable), index_col=0)
df_bin_wins = pd.read_csv (path_save + "%s_%s_%s_Bin_Stats_Wins_mask.csv"%(source_run,member_run,variable), index_col=0)

# Converting the str of the dataframe to floats
# ---------------------------------------------
cols_tmp	= df_bin_wins.columns[2:]
for col_t in cols_tmp:
	df_bin_wins.loc[:,col_t] = pd.to_numeric(df_bin_wins.loc[:,col_t], errors = 'coerce')
del cols_tmp, col_t

cols_tmp    = df_tce_wins.columns[2:]
for col_t in cols_tmp:
    df_tce_wins.loc[:,col_t] = pd.to_numeric(df_tce_wins.loc[:,col_t], errors = 'coerce')   
del cols_tmp, col_t

#Main DataFrame : df_tce_wins, df_bin_wins


# Finding the basic statistics of all above metrics for every region and time window
from functions import MaskedConstant_Resolve, MaskedArray_Resolve

# Calculating Summary of stats per region for bin (non-TCE):
# ---------------------------------------------------------
df_bin_summary_wins = pd.DataFrame(columns = ["win_idx"	,'region_abr',
									'fq_neg'	, 'fq_pos',
								    'fq_cup_du_neg', 'fq_crp_du_neg',
									'fq_cup_du_pos', 'fq_crp_du_pos',
									'count_reg'	   , 'count_px',
									'c_gain'	, 'c_loss',
									'reg_var',
									'tot_var', 
									'tot_var_ext', 
									'area_neg'	, 'area_pos',
									'reg_cup'	, 'reg_crp',
									'tot_cup'	, 'tot_crp',
									'tot_cup_ext', 'tot_crp_ext'])

for region_abr in srex_abr:
	df_bin_summary = pd.DataFrame(columns = ["win_idx"	,'region_abr',
									'fq_neg'	, 'fq_pos',
								    'fq_cup_du_neg', 'fq_crp_du_neg',
									'fq_cup_du_pos', 'fq_crp_du_pos',
									'c_gain'	, 'c_loss',
									'reg_var',
									'tot_var', 
									'tot_var_ext', 
									'area_neg'	, 'area_pos',
									'reg_cup'	, 'reg_crp',
									'tot_cup'	, 'tot_crp',
									'tot_cup_ext', 'tot_crp_ext'])

	for w_idx, wins in enumerate(win_start_years):
		filter_win_region = (df_bin_wins["win_idx"] == w_idx ) & (df_bin_wins["region_abr"] == region_abr) 
		df_tmp	= df_bin_wins [filter_win_region]

		bin_stats = {}
		for col in df_bin_summary_wins.columns[2:]:
			bin_stats[col] 			= {}
			if col in ['count_reg']:  # for count we will report mean,max ,sum=count and std = 0
				bin_stats[col]['mean']	= df_tmp.loc[:,'reg_var'].count()
				bin_stats[col]['std']	= 0 
				bin_stats[col]['max']	= df_tmp.loc[:,'reg_var'].count() 
				bin_stats[col]['sum']	= df_tmp.loc[:,'reg_var'].count()
			elif col in ['count_px'] :
				bin_stats[col]['mean']	= df_tmp.loc[:,'tot_var'].count()
				bin_stats[col]['std']	= 0 
				bin_stats[col]['max']	= df_tmp.loc[:,'tot_var'].count() 
				bin_stats[col]['sum']	= df_tmp.loc[:,'tot_var'].count()
			else:
				bin_stats[col]['mean']	= np.ma.mean(MaskedArray_Resolve(df_tmp[col]))
				bin_stats[col]['std']	= np.ma.std(MaskedArray_Resolve(df_tmp[col]))
				bin_stats[col]['max']	= np.ma.max(MaskedArray_Resolve(df_tmp[col]))
				bin_stats[col]['sum']	= np.ma.sum(MaskedArray_Resolve(df_tmp[col]))
		
		for k in bin_stats[col].keys():
			if k not in df_bin_summary.index:
				df_bin_summary	= df_bin_summary.reindex(df_bin_summary.index.values.tolist() + [k])
		
		for k in bin_stats[col].keys():
			for col in df_bin_summary_wins.columns[2:]:
				df_bin_summary.loc[k,col] = bin_stats[col][k]
				df_bin_summary.loc[k,"win_idx"] 	= w_idx
				df_bin_summary.loc[k,"region_abr"]	= region_abr
		del df_tmp, filter_win_region
		# Summary of results for all windows
		df_bin_summary_wins = df_bin_summary_wins.append(df_bin_summary)

# Storing the summary of the binary stats
df_bin_summary_wins. to_csv (path_save + "%s_%s_%s_Bin_Stats_Wins_Summary_mask.csv"%(source_run,member_run,variable))

# CALCULATION of Summary of stats per region for bin_TCE (with TCE):
# ------------------------------------------------------------------
df_tce_summary_wins	= pd. DataFrame(columns = ["win_idx",'region_abr','tce_neg', 'tce_pos', 'tce_len_neg', 'tce_len_pos', 'tce_len_tot','c_gain','c_loss','reg_var' ,'tot_var', 'tot_var_ext' ,'area_neg','area_pos'])

for region_abr in srex_abr:
	df_summary		= pd. DataFrame(columns = ["win_idx",'region_abr','tce_neg', 'tce_pos', 'tce_len_neg', 'tce_len_pos', 'tce_len_tot','c_gain','c_loss', 'reg_var' ,'tot_var','tot_var_ext' ,'area_neg','area_pos'])
	for w_idx, wins in enumerate(win_start_years):
		filter_win_region = (df_tce_wins["win_idx"] == w_idx ) & (df_tce_wins["region_abr"] == region_abr) 
		df_tmp	= df_tce_wins [filter_win_region]

		tce_stats = {}
		for col in df_tce_summary_wins.columns[2:]:
			tce_stats[col] 			= {}
			tce_stats[col]['mean']	= np.ma.mean(MaskedArray_Resolve(df_tmp[col]))
			tce_stats[col]['std']	= np.ma.std(MaskedArray_Resolve(df_tmp[col]))
			tce_stats[col]['max']	= np.ma.max(MaskedArray_Resolve(df_tmp[col]))
			tce_stats[col]['sum']	= np.ma.sum(MaskedArray_Resolve(df_tmp[col]))
		
		for k in tce_stats[col].keys():
			if k not in df_summary.index:
				df_summary	= df_summary.reindex(df_summary.index.values.tolist() + [k])
		
		for k in tce_stats[col].keys():
			for col in df_tce_summary_wins.columns[2:]:
				df_summary.loc[k,col] = tce_stats[col][k]
				df_summary.loc[k,"win_idx"] 	= w_idx
				df_summary.loc[k,"region_abr"]	= region_abr

		# Summary of results for all windows
		df_tce_summary_wins = df_tce_summary_wins.append(df_summary)

# Storing the summary
df_tce_summary_wins. to_csv (path_save + "%s_%s_%s_TCE_Stats_Wins_Summary_mask.csv"%(source_run,member_run,variable))


# Plotting of the region of interest
# ----------------------------------
# Regional analysis
# -----------------
import regionmask
# Cartopy Plotting
# ----------------
import cartopy.crs as ccrs
from shapely.geometry.polygon import Polygon
import cartopy.feature as cfeature



# Plotting the TS of regions:

region_abr 		= 'CAM'
filter_region   = np.array(srex_abr) == region_abr
region_idx      = srex_idxs[filter_region][0]
region_number   = np.array(srex_nums)[filter_region][0]
region_name     = np.array(srex_names)[filter_region][0]

filter_region = df_tce_summary_wins ['region_abr'] == region_abr 

web_path	= '/global/homes/b/bharat/results/web/'

# For x-axis
in_yr   = 1850
win_yr  = [str(in_yr+i*25) + '-'+str(in_yr +(i+1)*25-1)[2:] for i in range(win_start_years.size)]
# Plotting the Timeseries of Box plot for all regions:
# ----------------------------------------------------
for region_abr in srex_abr:
# TS: Box Plots of Columns:
#region_abr      = 'AMZ'
# This filter will only keep the region of interest and mask everything else
	filter_region 	= df_tce_wins['region_abr'] == region_abr

# droping the column of region_abr because it is str and I want df to be float 
	df_tmp			= df_tce_wins[filter_region].drop(columns=['region_abr'])
	col_tmp 		= df_tmp.columns[1:]					
	for col_t in col_tmp:
		df_tmp[col_t] = pd.to_numeric(df_tmp[col_t],errors='coerce')
	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp.loc[:,col][df_tmp.loc[:,col] == idx] = w_str  #improved version of the line below
#df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"
	for col in df_tmp.columns[1:]:
		df_tmp[col] = df_tmp[col].mask(df_tmp[col]==0)
		fig,ax = plt.subplots(figsize=(12,5))
		ax = sns.boxplot(x="Wins", y=col, data=df_tmp)
		fig.savefig(web_path + 'Regional/%s_%s_%s_%s_box.pdf'%(source_run,member_run,region_abr,col))
		fig.savefig(path_save + '%s_%s_%s_%s_box.pdf'%(source_run,member_run,region_abr,col))
		plt.close(fig)
	del df_tmp

# Creating the DataFrame of the interested stats for regions
# interested mean and sum of 
# 	* c_gain, c_loss, tot_var rel%, change in carbon_uptake
# ==========================================================

# Saving carbon loss and gains for every region based on Bin (non-TCE) Stats :
# ----------------------------------------------------------------------------
dict_carbon_bin = {}
for r_idx, region_abr in enumerate(srex_abr):
	filter_region   = np.array(srex_abr) == region_abr     #for finding srex number
	region_number   = np.array(srex_nums)[filter_region][0] # for srex mask
	del filter_region
	region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region; Mask of region is False
	region_mask		= ~region_mask_not   # Only the mask of region is True

	filter_region   = df_bin_summary_wins['region_abr'] == region_abr
	df_tmp          = df_bin_summary_wins[filter_region].drop(columns=['region_abr']).astype(float)

	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"

# Only looking at SUM or total change in a region
	df_sum = df_tmp[df_tmp.index == 'sum']
# Checking for 'Sum'
	df_carbon = pd.DataFrame(columns = ["Uptake_Gain", "Uptake_Loss", "Uptake_Change",
										"Regional_%s"%variable.upper() , "%s"%variable.upper(),"%s_du_Ext"%variable.upper(),
										"Percent_Gain_du","Percent_Loss_du", "Percent_Gain_reg", "Percent_Loss_reg", "Percent_Gain_px", "Percent_Loss_px"])
	for win_str in win_yr: 
		if win_str not in df_carbon.index: 
			df_carbon = df_carbon.reindex(df_carbon.index.values.tolist()+[win_str]) 
	for w_idx,win_str in enumerate(df_sum['Wins']):
		df_carbon.loc[win_str, "Uptake_Gain"] = (df_sum[df_sum['Wins'] == win_str]['c_gain']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Loss"] = (df_sum[df_sum['Wins'] == win_str]['c_loss']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Change"] = df_carbon.loc[win_str, "Uptake_Gain"] +  df_carbon.loc[win_str, "Uptake_Loss"]
		
		df_carbon.loc[win_str, "Regional_%s"%variable.upper()] = np.sum(nc_data [source_run] [member_run] ['var'] [w_idx * win_size: (w_idx+1) * win_size,:,:]* np.array([region_mask]*300))/10**15
		df_carbon.loc[win_str, "%s"%variable.upper()] = (df_sum[df_sum['Wins'] == win_str]['tot_var']/10**15).values[0]
		df_carbon.loc[win_str, "%s_du_Ext"%variable.upper()] = (df_sum[df_sum['Wins'] == win_str]['tot_var_ext']/10**15).values[0]

#	print (region_abr, df_carbon.loc[win_str, "Regional_%s"%variable.upper()], df_carbon.loc[win_str, "%s"%variable.upper()])
		df_carbon.loc[win_str, "Percent_Gain_du"] 	= df_carbon.loc[win_str, "Uptake_Gain"]*100/df_carbon.loc[win_str, "%s_du_Ext"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Loss_du"] 	= df_carbon.loc[win_str, "Uptake_Loss"]*100/df_carbon.loc[win_str, "%s_du_Ext"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Gain_reg"] 	= df_carbon.loc[win_str, "Uptake_Gain"]*100/df_carbon.loc[win_str, "Regional_%s"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Loss_reg"] 	= df_carbon.loc[win_str, "Uptake_Loss"]*100/df_carbon.loc[win_str, "Regional_%s"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Gain_px"] 	= df_carbon.loc[win_str, "Uptake_Gain"]*100/df_carbon.loc[win_str, "%s"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Loss_px"] 	= df_carbon.loc[win_str, "Uptake_Loss"]*100/df_carbon.loc[win_str, "%s"%variable.upper()]

	df_carbon.to_csv(web_path + 'Regional/%s_%s_%s_CarbonUptake_PgC.csv'%(source_run,member_run,region_abr))
	df_carbon.to_csv(path_save + '%s_%s_%s_CarbonUptake_PgC.csv'%(source_run,member_run,region_abr))
	dict_carbon_bin [region_abr] = df_carbon
	if r_idx == 0:
		df_bin_carbon_all = df_carbon.fillna(0) # df_bin_carbon_all is for global stats of carbon uptake; intializing it
	else:
		df_bin_carbon_all = df_bin_carbon_all + df_carbon.fillna(0)
	del df_carbon,df_tmp, filter_region 

# Updating the percent Gain and Loss of carbon uptake of all regions
# -------------------------------------------------------------------
df_bin_carbon_all['Percent_Gain_du'] 	= df_bin_carbon_all['Uptake_Gain']*100/df_bin_carbon_all['%s_du_Ext'%variable.upper()] 
df_bin_carbon_all['Percent_Loss_du'] 	= df_bin_carbon_all['Uptake_Loss']*100/df_bin_carbon_all['%s_du_Ext'%variable.upper()] 
df_bin_carbon_all['Percent_Gain_reg'] 	= df_bin_carbon_all['Uptake_Gain']*100/df_bin_carbon_all['Regional_%s'%variable.upper()] 
df_bin_carbon_all['Percent_Loss_reg'] 	= df_bin_carbon_all['Uptake_Loss']*100/df_bin_carbon_all['Regional_%s'%variable.upper()] 
df_bin_carbon_all['Percent_Gain_px'] 	= df_bin_carbon_all['Uptake_Gain']*100/df_bin_carbon_all['%s'%variable.upper()] 
df_bin_carbon_all['Percent_Loss_px'] 	= df_bin_carbon_all['Uptake_Loss']*100/df_bin_carbon_all['%s'%variable.upper()] 
dict_carbon_bin [ 'ALL'] = df_bin_carbon_all

"""
Meaning of the stats:
=====================
Common to dict_carbon_freq_bin
--------------------------------------
	* index  			:  'Wins' or Str of window range
	* key 				:region_abr or The SREX region shortname or abr 

Specific to dict_carbon_freq_bin for every 25 year time window for 'key' regions:
--------------------------------------------------------------------------------
	* Uptake_Gain		: Sum of anomalies in C flux during positive extremes
	* Uptake_Loss		: Sum of anomalies in C flux during negative extremes
	* Uptake_Change		: Uptake_Gain + Uptake_Loss
	* Regional_{C-flux}	: Sum of all {C-Flux} irrespective with or without an extreme
	* Count_Reg			: Total Count of total number of pixels with non-zero carbon flux values
	* {C-flux}			: Sum of all {C-Flux} at pixels  where at least one extreme has occured
	* Count_px			: Total Count of total number of pixels where at least one extreme has occured
	* {C-flux}_du_Ext	: Sum of all {C-Flux} at pixels and times i.e. same filter or space and time of extremes..
							... where either or both postive or negative carbon extremes as occuerd ...
							... e.g. if a loc witnessed 30 pos and 40 neg extremes so the ...
							... '{C-flux}_du_Ext' will give the total gpp of during these 70 exts.
	* Count_Neg_Ext		: Count of total months affect by neg extremes (non-TCE)
	* Count_Pos_Ext		: Count of total months affect by pos extremes (non-TCE)

"""
# Saving carbon loss and gains for every region based on Bin (non-TCE) Stats, with Frequency and count of cells:
# --------------------------------------------------------------------------------------------------------------
dict_carbon_freq_bin = {} 
for r_idx, region_abr in enumerate(srex_abr):
	filter_region   = np.array(srex_abr) == region_abr     #for finding srex number
	region_number   = np.array(srex_nums)[filter_region][0] # for srex mask
	del filter_region
	region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region; Mask of region is False
	region_mask		= ~region_mask_not   # Only the mask of region is True

	filter_region   = df_bin_summary_wins['region_abr'] == region_abr
	df_tmp          = df_bin_summary_wins[filter_region].drop(columns=['region_abr']).astype(float)

	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"

# Only looking at SUM or total change in a region
	df_sum = df_tmp[df_tmp.index == 'sum']
# Checking for 'Sum'
	df_carbon = pd.DataFrame(columns = ["Uptake_Gain", "Uptake_Loss", "Uptake_Change",
										"Regional_%s"%variable.upper() ,"Count_Reg", "%s"%variable.upper(), "Count_px","%s_du_Ext"%variable.upper(),
										"Count_Neg_Ext","Count_Pos_Ext"])
	for win_str in win_yr: 
		if win_str not in df_carbon.index: 
			df_carbon = df_carbon.reindex(df_carbon.index.values.tolist()+[win_str]) 
	for w_idx,win_str in enumerate(df_sum['Wins']):
		df_carbon.loc[win_str, "Uptake_Gain"] 	= (df_sum[df_sum['Wins'] == win_str]['c_gain']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Loss"] 	= (df_sum[df_sum['Wins'] == win_str]['c_loss']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Change"] = df_carbon.loc[win_str, "Uptake_Gain"] +  df_carbon.loc[win_str, "Uptake_Loss"]
		
		df_carbon.loc[win_str, "Regional_%s"%variable.upper()] 	= np.sum(nc_data [source_run] [member_run] ['var'] [w_idx * win_size: (w_idx+1) * win_size,:,:]* np.array([region_mask]*300))/10**15
		df_carbon.loc[win_str, "%s"%variable.upper()] 			= (df_sum[df_sum['Wins'] == win_str]['tot_var']/10**15).values[0]
		df_carbon.loc[win_str, "%s_du_Ext"%variable.upper()] 	= (df_sum[df_sum['Wins'] == win_str]['tot_var_ext']/10**15).values[0]


		df_carbon.loc[win_str, "CUP_reg"] 		= (df_sum[df_sum['Wins'] == win_str]['reg_cup']/10**15).values[0]
		df_carbon.loc[win_str, "CRP_reg"] 		= (df_sum[df_sum['Wins'] == win_str]['reg_crp']/10**15).values[0]

		df_carbon.loc[win_str, "CRP_px"] 		= (df_sum[df_sum['Wins'] == win_str]['tot_crp']/10**15).values[0]
		df_carbon.loc[win_str, "CUP_px"] 		= (df_sum[df_sum['Wins'] == win_str]['tot_cup']/10**15).values[0]
		
		df_carbon.loc[win_str, "CUP_du_Ext"] 	= (df_sum[df_sum['Wins'] == win_str]['tot_cup_ext']/10**15).values[0]
		df_carbon.loc[win_str, "CRP_du_Ext"] 	= (df_sum[df_sum['Wins'] == win_str]['tot_crp_ext']/10**15).values[0]

		df_carbon.loc[win_str, "Count_Reg"] 	= (df_sum[df_sum['Wins'] == win_str]['count_reg'])	.values[0]
		df_carbon.loc[win_str, "Count_px"]  	= (df_sum[df_sum['Wins'] == win_str]['count_px'	])	.values[0]
		df_carbon.loc[win_str, "Count_Neg_Ext"] = (df_sum[df_sum['Wins'] == win_str]['fq_neg'	])	.values[0]
		df_carbon.loc[win_str, "Count_Pos_Ext"] = (df_sum[df_sum['Wins'] == win_str]['fq_pos'	])	.values[0]
		
		df_carbon.loc[win_str, "Count_CUP_du_Neg_Ext"] = (df_sum[df_sum['Wins'] == win_str]['fq_cup_du_neg'	])	.values[0]
		df_carbon.loc[win_str, "Count_CRP_du_Neg_Ext"] = (df_sum[df_sum['Wins'] == win_str]['fq_crp_du_neg'	])	.values[0]

		df_carbon.loc[win_str, "Count_CUP_du_Pos_Ext"] = (df_sum[df_sum['Wins'] == win_str]['fq_cup_du_pos'	])	.values[0]
		df_carbon.loc[win_str, "Count_CRP_du_Pos_Ext"] = (df_sum[df_sum['Wins'] == win_str]['fq_crp_du_pos'	])	.values[0]

	df_carbon.to_csv(web_path + 'Regional/%s_%s_%s_CarbonUptake_Freq.csv'%(source_run,member_run,region_abr))
	df_carbon.to_csv(path_save + '%s_%s_%s_CarbonUptake_Freq.csv'%(source_run,member_run,region_abr))
	dict_carbon_freq_bin [region_abr] = df_carbon
	if r_idx == 0:
		df_bin_carbon_all = df_carbon.fillna(0) # df_bin_carbon_all is for global stats of carbon uptake; intializing it
	else:
		df_bin_carbon_all = df_bin_carbon_all + df_carbon.fillna(0)
	del df_carbon,df_tmp, filter_region 


# Updating the percent Gain and Loss of carbon uptake
# ---------------------------------------------------
dict_carbon_freq_bin [ 'ALL'] = df_bin_carbon_all



# Saving carbon loss and gains for every region based on TCE  Stats, with Frequency and count of cells:
# --------------------------------------------------------------------------------------------------------------
dict_carbon_freq_tce = {} 
for r_idx, region_abr in enumerate(srex_abr):
	filter_region   = np.array(srex_abr) == region_abr     #for finding srex number
	region_number   = np.array(srex_nums)[filter_region][0] # for srex mask
	del filter_region
	region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region; Mask of region is False
	region_mask		= ~region_mask_not   # Only the mask of region is True

	filter_region   = df_tce_summary_wins['region_abr'] == region_abr
	df_tmp          = df_tce_summary_wins[filter_region].drop(columns=['region_abr']).astype(float)

	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"

# Only looking at SUM or total change in a region
	df_sum = df_tmp[df_tmp.index == 'sum']
# Checking for 'Sum'
	df_carbon = pd.DataFrame(columns = ["Uptake_Gain", "Uptake_Loss", "Uptake_Change",
                                        "Len_Neg_TCE", "Len_Pos_TCE", "Count_Neg_TCE", "Count_Pos_TCE"])
	for win_str in win_yr: 
		if win_str not in df_carbon.index: 
			df_carbon = df_carbon.reindex(df_carbon.index.values.tolist()+[win_str]) 
	for w_idx,win_str in enumerate(df_sum['Wins']):
		df_carbon.loc[win_str, "Uptake_Gain"] 	= (df_sum[df_sum['Wins'] == win_str]['c_gain']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Loss"] 	= (df_sum[df_sum['Wins'] == win_str]['c_loss']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Change"] = df_carbon.loc[win_str, "Uptake_Gain"] +  df_carbon.loc[win_str, "Uptake_Loss"]
									
		df_carbon.loc[win_str, "Len_Neg_TCE"] 	= (df_sum[df_sum['Wins'] == win_str]['tce_len_neg'])	.values[0]
		df_carbon.loc[win_str, "Len_Pos_TCE"]  	= (df_sum[df_sum['Wins'] == win_str]['tce_len_pos'])	.values[0]
		df_carbon.loc[win_str, "Count_Neg_TCE"] = (df_sum[df_sum['Wins'] == win_str]['tce_neg'	])	.values[0]
		df_carbon.loc[win_str, "Count_Pos_TCE"] = (df_sum[df_sum['Wins'] == win_str]['tce_pos'	])	.values[0]
		
						
	df_carbon.to_csv(web_path + 'Regional/%s_%s_%s_CarbonUptake_Freq_tce.csv'%(source_run,member_run,region_abr))
	df_carbon.to_csv(path_save + '%s_%s_%s_CarbonUptake_Freq_tce.csv'%(source_run,member_run,region_abr))
	dict_carbon_freq_tce [region_abr] = df_carbon
	if r_idx == 0:
		df_tce_carbon_all = df_carbon.fillna(0) # df_bin_carbon_all is for global stats of carbon uptake; intializing it
	else:
		df_tce_carbon_all = df_tce_carbon_all + df_carbon.fillna(0)
	del df_carbon,df_tmp, filter_region 


# Updating the percent Gain and Loss of carbon uptake TCE
# ---------------------------------------------------
dict_carbon_freq_tce [ 'ALL'] = df_tce_carbon_all







# Saving carbon loss and gains for every region based on TCE Stats :
dict_carbon_tce = {}
for r_idx, region_abr in enumerate(srex_abr):
#region_abr      = 'AMZ'
	filter_region   = np.array(srex_abr) == region_abr     #for finding srex number  
	region_number   = np.array(srex_nums)[filter_region][0] # for srex mask
	del filter_region
	region_mask_not = np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region; Mask of region is False
	region_mask     = ~region_mask_not   # Only the mask of region is True

	filter_region   = df_tce_summary_wins['region_abr'] == region_abr
	df_tmp          = df_tce_summary_wins[filter_region].drop(columns=['region_abr']).astype(float)

	for idx,w_str in enumerate(win_yr):
		col = df_tmp.columns[0]
		df_tmp[col][df_tmp[col]==idx] = w_str
	df_tmp.rename(columns={'win_idx':'Wins'}, inplace=True) # Changing the win_idx : "Wins"

	# Only looking at SUM or total change in a region
	df_sum = df_tmp[df_tmp.index == 'sum']
	# Checking for 'Sum'
	df_carbon = pd.DataFrame(columns = ["Uptake_Gain", "Uptake_Loss", "Uptake_Change",
										"Regional_%s"%variable.upper() , "%s"%variable.upper(),"%s_du_Ext"%variable.upper(),
										"Percent_Gain_du","Percent_Loss_du", "Percent_Gain_reg", "Percent_Loss_reg","Percent_Gain_px", "Percent_Loss_px"])

	for win_str in win_yr: 
		if win_str not in df_carbon.index: 
			df_carbon = df_carbon.reindex(df_carbon.index.values.tolist()+[win_str]) 
	for win_str in df_sum['Wins']:
		df_carbon.loc[win_str, "Uptake_Gain"] = (df_sum[df_sum['Wins'] == win_str]['c_gain']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Loss"] = (df_sum[df_sum['Wins'] == win_str]['c_loss']/10**15).values[0]
		df_carbon.loc[win_str, "Uptake_Change"] = df_carbon.loc[win_str, "Uptake_Gain"] +  df_carbon.loc[win_str, "Uptake_Loss"]

		df_carbon.loc[win_str, "Regional_%s"%variable.upper()] = np.sum(nc_data [source_run] [member_run] ['var'] [w_idx * win_size: (w_idx+1) * win_size,:,:]* np.array([region_mask]*300))/10**15
		df_carbon.loc[win_str, "%s"%variable.upper()] = (df_sum[df_sum['Wins'] == win_str]['tot_var']/10**15).values[0]
		df_carbon.loc[win_str, "%s_du_Ext"%variable.upper()] = (df_sum[df_sum['Wins'] == win_str]['tot_var_ext']/10**15).values[0]
	
		df_carbon.loc[win_str, "Percent_Gain_du"] 	= df_carbon.loc[win_str, "Uptake_Gain"]*100/df_carbon.loc[win_str, "%s_du_Ext"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Loss_du"] 	= df_carbon.loc[win_str, "Uptake_Loss"]*100/df_carbon.loc[win_str, "%s_du_Ext"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Gain_reg"] 	= df_carbon.loc[win_str, "Uptake_Gain"]*100/df_carbon.loc[win_str, "Regional_%s"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Loss_reg"] 	= df_carbon.loc[win_str, "Uptake_Loss"]*100/df_carbon.loc[win_str, "Regional_%s"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Gain_px"] 	= df_carbon.loc[win_str, "Uptake_Gain"]*100/df_carbon.loc[win_str, "%s"%variable.upper()]
		df_carbon.loc[win_str, "Percent_Loss_px"] 	= df_carbon.loc[win_str, "Uptake_Loss"]*100/df_carbon.loc[win_str, "%s"%variable.upper()]

	df_carbon.to_csv(web_path + 'Regional/%s_%s_%s_CarbonUptake_PgC.csv'%(source_run,member_run,region_abr))
	df_carbon.to_csv(path_save + '%s_%s_%s_CarbonUptake_PgC.csv'%(source_run,member_run,region_abr))
	dict_carbon_tce [region_abr] = df_carbon
	if r_idx == 0:
		# .fillna(0) will replace the NaN to 0 so that the addition operation could result in non-zero numbers
		df_tce_carbon_all = df_carbon.fillna(0) # df_tce_carbon_all is for global stats of carbon uptake; intializing it
	else:
		df_tce_carbon_all = df_tce_carbon_all + df_carbon.fillna(0)
	del df_carbon

# Updating the percent Gain and Loss of carbon uptake
# ---------------------------------------------------
df_tce_carbon_all['Percent_Gain_du'] 	= df_tce_carbon_all['Uptake_Gain']*100/df_tce_carbon_all['%s_du_Ext'%variable.upper()] 
df_tce_carbon_all['Percent_Loss_du'] 	= df_tce_carbon_all['Uptake_Loss']*100/df_tce_carbon_all['%s_du_Ext'%variable.upper()] 
df_tce_carbon_all['Percent_Gain_reg'] 	= df_tce_carbon_all['Uptake_Gain']*100/df_tce_carbon_all['Regional_%s'%variable.upper()] 
df_tce_carbon_all['Percent_Loss_reg'] 	= df_tce_carbon_all['Uptake_Loss']*100/df_tce_carbon_all['Regional_%s'%variable.upper()] 
df_tce_carbon_all['Percent_Gain_px'] 	= df_tce_carbon_all['Uptake_Gain']*100/df_tce_carbon_all['%s'%variable.upper()] 
df_tce_carbon_all['Percent_Loss_px'] 	= df_tce_carbon_all['Uptake_Loss']*100/df_tce_carbon_all['%s'%variable.upper()] 
dict_carbon_tce [ 'ALL'] = df_tce_carbon_all


"""
# ploting the Normalized and GPP of all regions for TCE Stats
# -----------------------------------------------------------
x = dict_carbon_tce [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
		  'legend.frameon': 'False',
		  'axes.labelsize':'small',
		  'ytick.labelsize': 'small',
		  'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', sharey='row',
			gridspec_kw={'hspace': 0, 'wspace': 0})
axs 	= axs.ravel()

for k_idx, key in enumerate(dict_carbon_tce.keys()):
	axs[k_idx] .plot(x, norm(dict_carbon_tce[key]['%s'%variable.upper()]), 'k', linewidth = 0.6 ,label = key)
	axs[k_idx] .plot(x, norm(abs(dict_carbon_tce[key]['Percent_Loss']))  , 'r--', linewidth = 0.4)
	axs[k_idx] .plot(x, norm(abs(dict_carbon_tce[key]['Percent_Gain']))  , 'g--', linewidth = 0.4)
	axs[k_idx] . legend(loc="upper left")
for ax in axs.flat:
    ax.label_outer()

for tick in axs[-3].get_xticklabels():
	tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
	tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
	tick.set_rotation(45)

fig.savefig(web_path+'Regional/%s_Norm_%s.pdf'%(source_run,variable.upper()))
fig.savefig(path_save+'%s_TS_Norm_%s_Regions.pdf'%(source_run,variable.upper()))
plt.close(fig)
"""
"""
# Plotting the real GPP and Percent Change in Carbon Uptake during TCE
# --------------------------------------------------------------------
x = dict_carbon_tce [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
          'legend.frameon': 'False',
          'axes.labelsize':'small',
          'ytick.labelsize': 'small',
          'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', 
            gridspec_kw={'hspace': .4, 'wspace': .4}, figsize=(6,9))
plt.title ("%s and Percent Carbon Uptake and Loss"%(variable.upper()))
txt ="The left y-axis denotes the total %s in the region per 25 years; Units: PgC\n"%variable.upper()
txt+="The right y-axis denotes the percent carbon uptake w.r.t. to total %s\n"%variable.upper()
txt+="Red and Green represents Percent Loss and Gain in Carbon Uptake\n"
txt+="The carbon uptake is calclated during TCEs"
axs     = axs.ravel()

for k_idx, key in enumerate(dict_carbon_tce.keys()):
    axs[k_idx] .plot(x, dict_carbon_tce[key]['%s'%variable.upper()], 'k', linewidth = 0.6 ,label = key)
    ar= np.array([abs(dict_carbon_tce[key]['Percent_Loss'].min()), 
             abs(dict_carbon_tce[key]['Percent_Loss'].max()),
             abs(dict_carbon_tce[key]['Percent_Gain'].max()),
             abs(dict_carbon_tce[key]['Percent_Gain'].min())])
    ax1 = axs[k_idx] .twinx()
    ax2 = axs[k_idx] .twinx()
    ax1 .plot(x, abs(dict_carbon_tce[key]['Percent_Loss'])  , 'r--', linewidth = 0.4)
    ax2 .plot(x, abs(dict_carbon_tce[key]['Percent_Gain'])  , 'g--', linewidth = 0.4)
    ax1.set_ylim(ar.min(),ar.max())
    ax2.set_ylim(ar.min(),ar.max())
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
fig.savefig(web_path+'Regional/%s_%s_per_Uptake_TCE.pdf'%(source_run,variable.upper()))
fig.savefig(path_save+'%s_TS_%s_per_Uptake_Regions_TCE.pdf'%(source_run,variable.upper()))
plt.close(fig)
"""


# Plotting the real GPP and Percent Change in Carbon Uptake for TCE + Bin w.r.t. Total Regional GPP 
# -------------------------------------------------------------------------------------------------
x = dict_carbon_bin [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
          'legend.frameon': 'False',
          'axes.labelsize':'small',
          'ytick.labelsize': 'small',
          'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', 
            gridspec_kw={'hspace': .4, 'wspace': .4}, figsize=(6,9))
txt ="The left y-axis denotes the total %s in the region per 25 years; Units: PgC\n"%variable.upper()
txt+="The right y-axis denotes the percent carbon uptake w.r.t. to total regional %s\n"%variable.upper()
txt+="Red and Green represents Percent Loss and Gain in Carbon Uptake\n"
txt+="The carbon uptake during TCE and bin extremes is shown by dashed and solid lines"
axs     = axs.ravel()

for k_idx, key in enumerate(dict_carbon_bin.keys()):
	axs[k_idx] .plot(x, dict_carbon_bin[key]['%s'%variable.upper()], 'k', linewidth = 0.6 ,label = key)
	ar= np.array([abs(dict_carbon_bin[key]['Percent_Loss_reg'].min()), 
             abs(dict_carbon_bin[key]['Percent_Loss_reg'].max()),
             abs(dict_carbon_bin[key]['Percent_Gain_reg'].max()),
             abs(dict_carbon_bin[key]['Percent_Gain_reg'].min()),
			 abs(dict_carbon_tce[key]['Percent_Loss_reg'].min()), 
             abs(dict_carbon_tce[key]['Percent_Loss_reg'].max()),
             abs(dict_carbon_tce[key]['Percent_Gain_reg'].max()),
			 abs(dict_carbon_tce[key]['Percent_Gain_reg'].min())])

	ax1 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning TCE
	ax2 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning TCE
	ax3 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning BIN
	ax4 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning BIN
	
	ax1 .plot(x, abs(dict_carbon_tce[key]['Percent_Loss_reg'])  , 'r--', linewidth = 0.4)
	ax2 .plot(x, abs(dict_carbon_tce[key]['Percent_Gain_reg'])  , 'g--', linewidth = 0.4)
	ax1 .set_ylim(ar.min()*.95,ar.max()*1.05)
	ax2 .set_ylim(ar.min()*.95,ar.max()*1.05)
	
	ax3 .plot(x, abs(dict_carbon_bin[key]['Percent_Loss_reg'])  , 'r', linewidth = 0.4)
	ax4 .plot(x, abs(dict_carbon_bin[key]['Percent_Gain_reg'])  , 'g', linewidth = 0.4)
	ax3 .set_ylim(ar.min()*.95,ar.max()*1.05)
	ax4 .set_ylim(ar.min()*.95,ar.max()*1.05)
	
	axs[k_idx] . legend(loc="upper left")
#for ax in axs.flat:
#    ax.label_outer()
for tick in axs[-3].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
    tick.set_rotation(45)
plt.suptitle ("Percent Uptake in carbon w.r.t. Total Regional %s"%variable.upper())
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=7) #Caption
fig.savefig(web_path+'Regional/%s_%s_per_Uptake_regional.pdf'%(source_run,variable.upper()))
fig.savefig(path_save+'%s_TS_%s_per_Uptake_Regional.pdf'%(source_run,variable.upper()))
plt.close(fig)

# Plotting the real GPP and Percent Change in Carbon Uptake for TCE + Bin w.r.t. Total GPP during extremes for every region
# -------------------------------------------------------------------------------------------------------------------------
x = dict_carbon_bin [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
          'legend.frameon': 'False',
          'axes.labelsize':'small',
          'ytick.labelsize': 'small',
          'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', 
            gridspec_kw={'hspace': .4, 'wspace': .4}, figsize=(6,9))
txt ="The left y-axis denotes the total %s in the region per 25 years; Units: PgC\n"%variable.upper()
txt+="The right y-axis denotes the percent carbon uptake w.r.t. to %s during extremes\n"%variable.upper()
txt+="Red and Green represents Percent Loss and Gain in Carbon Uptake\n"
txt+="The carbon uptake during TCE and bin extremes is shown by dashed and solid lines"
axs     = axs.ravel()

for k_idx, key in enumerate(dict_carbon_bin.keys()):
	axs[k_idx] .plot(x, dict_carbon_bin[key]['%s'%variable.upper()], 'k', linewidth = 0.6 ,label = key)
	ar= np.array([abs(dict_carbon_bin[key]['Percent_Loss_du'].min()), 
	             abs(dict_carbon_bin[key]['Percent_Loss_du'].max()),
    	         abs(dict_carbon_bin[key]['Percent_Gain_du'].max()),
        	     abs(dict_carbon_bin[key]['Percent_Gain_du'].min()),
				 abs(dict_carbon_tce[key]['Percent_Loss_du'].min()), 
	             abs(dict_carbon_tce[key]['Percent_Loss_du'].max()),
    	         abs(dict_carbon_tce[key]['Percent_Gain_du'].max()),
				 abs(dict_carbon_tce[key]['Percent_Gain_du'].min())])

	ax1 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning TCE
	ax2 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning TCE
	ax3 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning BIN
	ax4 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning BIN
	
	ax1 .plot(x, abs(dict_carbon_tce[key]['Percent_Loss_du'])  , 'r--', linewidth = 0.4)
	ax2 .plot(x, abs(dict_carbon_tce[key]['Percent_Gain_du'])  , 'g--', linewidth = 0.4)
	ax1 .set_ylim(ar.min()*.95,ar.max()*1.05)
	ax2 .set_ylim(ar.min()*.95,ar.max()*1.05)
	
	ax3 .plot(x, abs(dict_carbon_bin[key]['Percent_Loss_du'])  , 'r', linewidth = 0.4)
	ax4 .plot(x, abs(dict_carbon_bin[key]['Percent_Gain_du'])  , 'g', linewidth = 0.4)
	ax3 .set_ylim(ar.min()*.95,ar.max()*1.05)
	ax4 .set_ylim(ar.min()*.95,ar.max()*1.05)
	
	axs[k_idx] . legend(loc="upper left")
#for ax in axs.flat:
#    ax.label_outer()
for tick in axs[-3].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
    tick.set_rotation(45)
plt.suptitle ("Total Carbon Loss or Gain during %s TCEs"%variable.upper())
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8) #Caption
fig.savefig(web_path+'Regional/%s_%s_per_Uptake_regional_du_ext.pdf'%(source_run,variable.upper()))
fig.savefig(path_save+'%s_TS_%s_per_Uptake_Regional_du_ext.pdf'%(source_run,variable.upper()))
plt.close(fig)

# Plotting the real GPP and Percent Change in Carbon Uptake for TCE + Bin w.r.t. Total GPP during extremes for every region
# -------------------------------------------------------------------------------------------------------------------------
x = dict_carbon_bin [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
          'legend.frameon': 'False',
          'axes.labelsize':'small',
          'ytick.labelsize': 'small',
          'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', 
            gridspec_kw={'hspace': .4, 'wspace': .4}, figsize=(6,9))
txt ="The left y-axis denotes the total %s in the region per 25 years; Units: PgC\n"%variable.upper()
txt+="The right y-axis denotes the percent carbon uptake w.r.t. to %s during extremes\n"%variable.upper()
txt+="Red and Green represents Loss and Gain in Carbon Uptake\n"
txt+="The carbon uptake during TCE and bin extremes is shown by dashed and solid lines"
axs     = axs.ravel()

for k_idx, key in enumerate(dict_carbon_bin.keys()):
	axs[k_idx] .plot(x, dict_carbon_bin[key]['%s'%variable.upper()], 'k', linewidth = 0.6 ,label = key)
	ar= np.array([abs(dict_carbon_bin[key]['Uptake_Loss'].min()), 
	             abs(dict_carbon_bin[key]['Uptake_Loss'].max()),
    	         abs(dict_carbon_bin[key]['Uptake_Gain'].max()),
        	     abs(dict_carbon_bin[key]['Uptake_Gain'].min()),
				 abs(dict_carbon_tce[key]['Uptake_Loss'].min()), 
	             abs(dict_carbon_tce[key]['Uptake_Loss'].max()),
    	         abs(dict_carbon_tce[key]['Uptake_Gain'].max()),
				 abs(dict_carbon_tce[key]['Uptake_Gain'].min())])

	ax1 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning TCE
	ax2 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning TCE
	ax3 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning BIN
	ax4 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning BIN
	
	ax1 .plot(x, abs(dict_carbon_tce[key]['Uptake_Loss'])  , 'r--', linewidth = 0.4)
	ax2 .plot(x, abs(dict_carbon_tce[key]['Uptake_Gain'])  , 'g--', linewidth = 0.4)
	ax1 .set_ylim(ar.min()*.95,ar.max()*1.05)
	ax2 .set_ylim(ar.min()*.95,ar.max()*1.05)
	
	ax3 .plot(x, abs(dict_carbon_bin[key]['Uptake_Loss'])  , 'r', linewidth = 0.4)
	ax4 .plot(x, abs(dict_carbon_bin[key]['Uptake_Gain'])  , 'g', linewidth = 0.4)
	ax3 .set_ylim(ar.min()*.95,ar.max()*1.05)
	ax4 .set_ylim(ar.min()*.95,ar.max()*1.05)
	
	axs[k_idx] . legend(loc="upper left")
#for ax in axs.flat:
#    ax.label_outer()
for tick in axs[-3].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
    tick.set_rotation(45)
plt.suptitle ("Uptake in Carbon %s"%variable.upper())
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8) #Caption
fig.savefig(web_path+'Regional/%s_%s_Total_Uptake_PgC.pdf'%(source_run,variable.upper()))
fig.savefig(path_save+'%s_TS_%s_Total_Uptake_Regions_PgC.pdf'%(source_run,variable.upper()))
plt.close(fig)

# Plotting the GPP during extremes and for all times in pixels with atleast one extreme for every region
# -------------------------------------------------------------------------------------------------------------------------
x = dict_carbon_bin [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
          'legend.frameon': 'False',
          'axes.labelsize':'small',
          'ytick.labelsize': 'small',
          'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', 
            gridspec_kw={'hspace': .4, 'wspace': .4}, figsize=(6,9))

plt.suptitle ("%s in Regions during extremes"%variable.upper())
txt ="The left y-axis denotes the total %s in the region per 25 years when pixel have at least one extreme\n"%variable.upper()
txt+="The right y-axis denotes the total %s in the region per 25 years in pixels during extremes\n"%variable.upper()
txt+="Blue and green represents TCE and Binary extremes\n"
txt+="The dashed and solid lines represent %s during extreme and for all times in a pixel with atleast one extreme "%variable.upper()
axs     = axs.ravel()

for k_idx, key in enumerate(dict_carbon_bin.keys()):
#axs[k_idx] .plot(x, dict_carbon_bin[key]['Regional_%s'%variable.upper()], 'k', linewidth = 0.6 ,label = key) #Regional GPP
	ar0 = np.array([abs(dict_carbon_bin[key]['%s_du_Ext'%variable.upper()].max()),
					abs(dict_carbon_bin[key]['%s_du_Ext'%variable.upper()].min()),
					abs(dict_carbon_tce[key]['%s_du_Ext'%variable.upper()].max()),
					abs(dict_carbon_tce[key]['%s_du_Ext'%variable.upper()].min())])

	ar= np.array([abs(dict_carbon_bin[key]['%s'%variable.upper()].min()), 
	             abs(dict_carbon_bin[key]['%s'%variable.upper()].max()),
				 abs(dict_carbon_tce[key]['%s'%variable.upper()].min()), 
	             abs(dict_carbon_tce[key]['%s'%variable.upper()].max())])

	ax1 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning TCE
	ax2 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning TCE

	#ax3 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning BIN
	#ax4 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning BIN
	
	ax1 .plot(x, abs(dict_carbon_tce[key]['%s_du_Ext'%variable.upper()])  , 'b--', linewidth = 0.4)
	ax2 .plot(x, abs(dict_carbon_bin[key]['%s_du_Ext'%variable.upper()])  , 'g--', linewidth = 0.4)
	ax1 .set_ylim(ar0.min()*.95,ar0.max()*1.05)
	ax2 .set_ylim(ar0.min()*.95,ar0.max()*1.05)
	
	axs[k_idx].plot(x, abs(dict_carbon_tce[key]['%s'%variable.upper()])  , 'b', linewidth = 0.4, label = key)
	axs[k_idx] .plot(x, abs(dict_carbon_bin[key]['%s'%variable.upper()])  , 'g', linewidth = 0.4)
	axs[k_idx] .set_ylim(ar.min()*.95,ar.max()*1.05)
	axs[k_idx] .set_ylim(ar.min()*.95,ar.max()*1.05)
	
	axs[k_idx] . legend(loc="upper left")
#ax3 .plot(x, dict_carbon_bin[key]['Regional_%s'%variable.upper()], 'k', linewidth = 0.6 ,label = key) #Regional GPP
#for ax in axs.flat:
#    ax.label_outer()
for tick in axs[-3].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
    tick.set_rotation(45)
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8) #Caption
fig.savefig(web_path+'Regional/%s_%s_GPP_du_Extremes_PgC.pdf'%(source_run,variable.upper()))
fig.savefig(path_save+'%s_TS_%s_Total_GPP_du_Extremes_PgC.pdf'%(source_run,variable.upper()))
plt.close(fig)

# Plotting the Carbon Uptake losses and gains for with TCE and without TCEs
# -------------------------------------------------------------------------------------------------------------------------
x = dict_carbon_bin [ 'ALL'].index
import pylab as plot
params = {'legend.fontsize': 6,
          'legend.handlelength': 1,
          'legend.frameon': 'False',
          'axes.labelsize':'small',
          'ytick.labelsize': 'small',
          'font.size':5 }
plot.rcParams.update(params)
fig, axs = plt.subplots(nrows=9, ncols=3, sharex='col', 
            gridspec_kw={'hspace': .4, 'wspace': .4}, figsize=(6,9))

plt.suptitle ("%s Uptake loss and gain with TCE and BIN extremes"%variable.upper())
txt ="The left y-axis denotes the total %s Uptake loss and gains during binary extremes\n"%variable.upper()
txt+="The right y-axis denotes the total %s Uptake loss and gains during TCE extremes\n"%variable.upper()
txt+="Red and green represents uptakes losses and gains; units PgC \n"
txt+="The dashed and solid lines represent TCE and Bin extremes"
axs     = axs.ravel()

for k_idx, key in enumerate(dict_carbon_bin.keys()):
#axs[k_idx] .plot(x, dict_carbon_bin[key]['Regional_%s'%variable.upper()], 'k', linewidth = 0.6 ,label = key) #Regional GPP
	ar = np.array([abs(dict_carbon_bin[key]['Uptake_Gain'].max()),
					abs(dict_carbon_bin[key]['Uptake_Gain'].min()),
					abs(dict_carbon_bin[key]['Uptake_Loss'].max()),
					abs(dict_carbon_bin[key]['Uptake_Loss'].min())])

	ar0 = np.array([abs(dict_carbon_tce[key]['Uptake_Gain'].max()),
					abs(dict_carbon_tce[key]['Uptake_Gain'].min()),
					abs(dict_carbon_tce[key]['Uptake_Loss'].max()),
					abs(dict_carbon_tce[key]['Uptake_Loss'].min())])


	ax1 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning TCE
	ax2 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning TCE

	#ax3 = axs[k_idx] .twinx() 	# for representing Percent carbon gain durning BIN
	#ax4 = axs[k_idx] .twinx()	# for representing Percent carbon loss durning BIN
	
	ax1 .plot(x, abs(dict_carbon_tce[key]['Uptake_Gain'])  , 'g--', linewidth = 0.4)
	ax2 .plot(x, abs(dict_carbon_tce[key]['Uptake_Loss'])  , 'r--', linewidth = 0.4)
	ax1 .set_ylim(ar0.min()*.95,ar0.max()*1.05)
	ax2 .set_ylim(ar0.min()*.95,ar0.max()*1.05)
	
	axs[k_idx] .plot(x, abs(dict_carbon_bin[key]['Uptake_Gain'])  , 'g', linewidth = 0.4, label = key)
	axs[k_idx] .plot(x, abs(dict_carbon_bin[key]['Uptake_Loss'])  , 'r', linewidth = 0.4)
	axs[k_idx] .set_ylim(ar.min()*.95,ar.max()*1.05)
	axs[k_idx] .set_ylim(ar.min()*.95,ar.max()*1.05)
	
	axs[k_idx] . legend(loc="upper left")
#ax3 .plot(x, dict_carbon_bin[key]['Regional_%s'%variable.upper()], 'k', linewidth = 0.6 ,label = key) #Regional GPP
#for ax in axs.flat:
#    ax.label_outer()
for tick in axs[-3].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
    tick.set_rotation(45)
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=8) #Caption
fig.savefig(web_path+'Regional/%s_%s_Uptake_Losses_and_Gains_PgC.pdf'%(source_run,variable.upper()))
fig.savefig(path_save+'%s_TS_%s_Uptake_Losses_and_Gains_PgC.pdf'%(source_run,variable.upper()))
plt.close(fig)

# Stacked Bar plot to highlight the Statistics of C-Flux and Freq of extremes
# ---------------------------------------------------------------------------

# Spatial Plot of Integrated NBPs during extremes
# -----------------------------------------------
# Common to rest of the spatial figures:
# ======================================
import colorsys as cs
val         = 0.8
Rd          = cs.rgb_to_hsv(1,0,0)
Rd          = cs.hsv_to_rgb(Rd[0],Rd[1],val)                     
Gn          = cs.rgb_to_hsv(0,1,0)
Gn          = cs.hsv_to_rgb(Gn[0],Gn[0],val)
RdGn        = {'red'  : ((0.0,  0.0,    Rd[0]),
                         (0.5,  1.0,    1.0  ),
                         (1.0,  Gn[0],  0.0  )),
               'green': ((0.0,  0.0,    Rd[1]),
                         (0.5,  1.0,    1.0  ),
                         (1.0,  Gn[1],  0.0  )),
               'blue' : ((0.0,  0.0,    Rd[2]),
                         (0.5,  1.0,    1.0  ),
                         (1.0,  Gn[2],  0.0  ))}
plt.register_cmap(name  = 'RdGn',data = RdGn)

# The index of the region in region mask
#srex_idx = srex_idxs[np.array(srex_abr) == reg_abr][0]
Wins_to_Plot = ['1850-74', '1900-24', '1950-74', '2000-24', '2050-74', '2075-99']
sub_fig_text = ['(a)', '(b)', '(c)',
               '(d)', '(e)', '(f)']

# Plotting of "NBP_du_Ext"
# =====================================
values_range = []
sign = {}
for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,'NBP_du_Ext'])
        if dict_carbon_freq_bin[r].loc[wi,'NBP_du_Ext'] > 0:
            sign[r][wi] = '+' 
        elif dict_carbon_freq_bin[r].loc[wi,'NBP_du_Ext'] < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())
levels = np.arange(-6,6,2)
print (levels)
# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi,'NBP_du_Ext'] # for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] ['NBP_du_Ext'] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()
fig = plt.figure(figsize = (12,9), dpi = 200)

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]]['NBP_du_Ext'], 
                                    transform=ccrs.PlateCarree(),vmax=6,vmin=-6,cmap='RdYlGn')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]]['NBP_du_Ext'], 
                                    transform=ccrs.PlateCarree(),vmax=6,vmin=-6,cmap='RdYlGn')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
#plt.colorbar(h, orientation='horizontal', pad=0.04);
ax[1].set_title("Integrated NBP during NBP Extreme Events (PgC)", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_Du_Exts.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_Du_Exts.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_Du_Exts.pdf")
plt.close(fig)

# Plotting of "Uptake_Change"
# =====================================
values_range = []
sign = {}
Col_Name = 'Uptake_Change'
for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,Col_Name])
        if dict_carbon_freq_bin[r].loc[wi, Col_Name] > 0:
            sign[r][wi] = '+' 
        elif dict_carbon_freq_bin[r].loc[wi, Col_Name] < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())


ymax = 1
ymin = -1
print (ymax, ymin)

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name] # for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] [Col_Name] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()

fig = plt.figure(figsize = (12,9), dpi=400)
plt.style.use("classic")

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax= ymax,vmin= ymin, cmap='RdYlGn')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap='RdYlGn')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);

ax[1].set_title ("Net Uptake Change in NBP extremes (PgC)", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_Uptake_Change.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_Uptake_Change.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_Uptake_Change.pdf")
plt.close(fig)

# Plotting of "NBP" spatial filter of extremes
# ===========================================
values_range = []
sign = {}
Col_Name = 'NBP'
for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,Col_Name])
        if dict_carbon_freq_bin[r].loc[wi, Col_Name] > 0:
            sign[r][wi] = '+' 
        elif dict_carbon_freq_bin[r].loc[wi, Col_Name] < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())


ymax = 20
ymin = -20
print (ymax, ymin)

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name] # for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] [Col_Name] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()
fig = plt.figure(figsize = (12,9), dpi=200)

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax= ymax,vmin= ymin, cmap='RdYlGn')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap='RdYlGn')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
ax[1].set_title ("Integrated NBP for pixels affected with atleast one NBP Extreme Event (PgC)", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_px.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_px.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_px.pdf")
plt.close(fig)

# Plotting of "Regional_NBP"
# =====================================
values_range = []
sign = {}
Col_Name = 'Regional_NBP'
for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,Col_Name])
        if dict_carbon_freq_bin[r].loc[wi, Col_Name] > 0:
            sign[r][wi] = '+' 
        elif dict_carbon_freq_bin[r].loc[wi, Col_Name] < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())


ymax = 20
ymin = -20
print (ymax, ymin)

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name] # for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] [Col_Name] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()
fig = plt.figure(figsize = (12,9), dpi=200)

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax= ymax,vmin= ymin, cmap='RdYlGn')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap='RdYlGn')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
ax[1].set_title ("Regional Integrated NBP (PgC)", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_regional.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_regional.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_regional.pdf")
plt.close(fig)

# Plotting of "NBP" Count of Negative extremes
# as count of negative extrems: 
# ==============================================================
values_range = []
sign = {}
Col_Name = 'Count_Neg_Ext'
for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,Col_Name])
        if dict_carbon_freq_bin[r].loc[wi, Col_Name] > 0:
            sign[r][wi] = '+' 
        elif dict_carbon_freq_bin[r].loc[wi, Col_Name] < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())


ymax = 20000
ymin = 0
print (ymax, ymin)

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name] # for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] [Col_Name] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()
fig = plt.figure(figsize = (12,9), dpi=200)

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax= ymax,vmin= ymin, cmap='autumn_r')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    color = 'blue', fontweight = 'bold',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap='autumn_r')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
#cbar = ax.cax.colorbar(h)
#cbar = grid.cbar_axes[0].colorbar(h)
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
#plt.colorbar(h, orientation='horizontal', pad=0.04);
ax[1].set_title ("Freq of Negative NBP extremes", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_neg_exts.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_neg_exts.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_fq_neg_exts.pdf")
plt.close(fig)

# Plotting of "NBP" Count of Positive extremes
# as count of negative extrems: 
# ==============================================================
values_range = []
sign = {}
Col_Name = 'Count_Pos_Ext'
for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,Col_Name])
        if dict_carbon_freq_bin[r].loc[wi, Col_Name] > 0:
            sign[r][wi] = '+' 
        elif dict_carbon_freq_bin[r].loc[wi, Col_Name] < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())


ymax = 20000
ymin = 0
print (ymax, ymin)

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name] # for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] [Col_Name] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()
fig = plt.figure(figsize = (12,9), dpi=200)

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax= ymax,vmin= ymin, cmap='autumn_r')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    color = 'blue', fontweight = 'bold',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name], 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap='autumn_r')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
#cbar = ax.cax.colorbar(h)
#cbar = grid.cbar_axes[0].colorbar(h)
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
#plt.colorbar(h, orientation='horizontal', pad=0.04);
ax[1].set_title ("Freq of Positive NBP extremes", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_pos_exts.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_pos_exts.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_fq_pos_exts.pdf")
plt.close(fig)

# Plotting of "NBP" Count of Negative and Positive extremes
# as count of positive - negative extrems: 
# ==============================================================


values_range = []
sign = {}
Col_Name1 = 'Count_Pos_Ext'
Col_Name2 = 'Count_Neg_Ext'
# anaylsis do Col_Name1 - Col_Name2
for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,Col_Name1] - dict_carbon_freq_bin[r].loc[wi,Col_Name2])
        if (dict_carbon_freq_bin[r].loc[wi,Col_Name1] - dict_carbon_freq_bin[r].loc[wi,Col_Name2]) > 0:
            sign[r][wi] = '+' 
        elif (dict_carbon_freq_bin[r].loc[wi,Col_Name1] - dict_carbon_freq_bin[r].loc[wi,Col_Name2]) < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())


ymax = 2000
ymin = -2000
print (ymax, ymin)

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = (dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name1] -
                           dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name2]  )# for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] [Col_Name1+'-'+Col_Name2] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()

fig = plt.figure(figsize = (12,9), dpi =400)
plt.style.use("classic")

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name1+'-'+Col_Name2], 
                                    transform=ccrs.PlateCarree(),vmax= ymax,vmin= ymin, cmap='PuOr')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name1+'-'+Col_Name2], 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap='PuOr')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
ax[1].set_title (r"Count Positive $-$ Negative of NBP extremes", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_pos_neg_ext.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_pos_neg_ext.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_fq_pos_neg_ext.pdf")
plt.close(fig)

# Plotting of "NBP" Count of CRP or CUP During Negative extremes
# as CUP - CRP : counts of CUP 'x' higher than CRP during negative extremes
# ==============================================================

values_range = []
sign = {}
Col_Name1 = 'Count_CUP_du_Neg_Ext'
Col_Name2 = 'Count_CRP_du_Neg_Ext'
for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,Col_Name1] - dict_carbon_freq_bin[r].loc[wi,Col_Name2])
        if (dict_carbon_freq_bin[r].loc[wi,Col_Name1] - dict_carbon_freq_bin[r].loc[wi,Col_Name2]) > 0:
            sign[r][wi] = '+' 
        elif (dict_carbon_freq_bin[r].loc[wi,Col_Name1] - dict_carbon_freq_bin[r].loc[wi,Col_Name2]) < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())


ymax = 5000
ymin = -5000
print (ymax, ymin)

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = (dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name1] -
                           dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name2]  )# for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] [Col_Name1+'-'+Col_Name2] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()
fig = plt.figure(figsize = (12,9), dpi =200)

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name1+'-'+Col_Name2], 
                                    transform=ccrs.PlateCarree(),vmax= ymax,vmin= ymin, cmap='PuOr')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]][Col_Name1+'-'+Col_Name2], 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap='PuOr')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
ax[1].set_title (r"Count CUP $-$ CRP during Negative Extremes", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_CUP_CRP_neg_ext.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_CUP_CRP_neg_ext.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_fq_CUP_CRP_neg_ext.pdf")
plt.close(fig)

# Count of CUP and CRP during all Extremes
# ======================================

values_range = []
sign = {}
Col_Name1 = 'Count_CUP_du_Pos_Ext'
Col_Name2 = 'Count_CUP_du_Neg_Ext'
Col_Name3 = 'Count_CRP_du_Pos_Ext'
Col_Name4 = 'Count_CRP_du_Neg_Ext'

for r in srex_abr:
    sign[r] = {}
    for wi in Wins_to_Plot:
        values_range.append((dict_carbon_freq_bin[r].loc[wi,Col_Name1] + dict_carbon_freq_bin[r].loc[wi,Col_Name2])
                            - (dict_carbon_freq_bin[r].loc[wi,Col_Name3] + dict_carbon_freq_bin[r].loc[wi,Col_Name4])
                           )
        if ((dict_carbon_freq_bin[r].loc[wi,Col_Name1] + dict_carbon_freq_bin[r].loc[wi,Col_Name2])
                            - (dict_carbon_freq_bin[r].loc[wi,Col_Name3] + dict_carbon_freq_bin[r].loc[wi,Col_Name4])
                           ) > 0:
            sign[r][wi] = '+' 
        elif ((dict_carbon_freq_bin[r].loc[wi,Col_Name1] + dict_carbon_freq_bin[r].loc[wi,Col_Name2])
                            - (dict_carbon_freq_bin[r].loc[wi,Col_Name3] + dict_carbon_freq_bin[r].loc[wi,Col_Name4])
                           ) < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = '*'

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())


ymax = 10000
ymin = -10000
print (ymax, ymin)

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for wi in Wins_to_Plot:
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = ((dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi,Col_Name1] + dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi,Col_Name2])
                            - (dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi,Col_Name3] + dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi,Col_Name4])
                           )# for 1850-74
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] ['Count CUP - CRP du Extremes'] = all_masked
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()
fig = plt.figure(figsize = (12,9), dpi=200)

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]]['Count CUP - CRP du Extremes'], 
                                    transform=ccrs.PlateCarree(),vmax= ymax,vmin= ymin, cmap='PuOr')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    color = 'blue', fontweight = 'bold',
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  ploting_stats[Wins_to_Plot[plot_idx]]['Count CUP - CRP du Extremes'], 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap='PuOr')
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'blue', fontweight = 'bold',
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    gl[plot_idx] = ax[plot_idx].gridlines(crs=ccrs.PlateCarree(), draw_labels=False,
                                           linewidth=.5, color='gray', alpha=0.5, linestyle='--')
    ax[plot_idx].text(-90, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 9)
    
gl[3].xlabels_bottom = True
gl[4].xlabels_bottom = True
gl[5].xlabels_bottom = True
gl[3].xformatter = LONGITUDE_FORMATTER
gl[4].xformatter = LONGITUDE_FORMATTER
gl[5].xformatter = LONGITUDE_FORMATTER
gl[0].ylabels_left = True
gl[3].ylabels_left = True
gl[0].yformatter = LATITUDE_FORMATTER
gl[3].yformatter = LATITUDE_FORMATTER
plt.subplots_adjust(wspace=0.02,hspace=-.695)

cax = plt.axes([0.92, 0.335, 0.015, 0.34])
plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
ax[1].set_title (r"Count CUP $-$ CRP during NBP Extremes", fontsize = 16)
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_CUP_CRP_du_ext.pdf")
fig.savefig(web_path + "Spatial_Maps/Spatial_NBP_fq_CUP_CRP_du_ext.png")
fig.savefig(path_save + "Spatial/Spatial_NBP_fq_CUP_CRP_du_ext.pdf")
plt.close(fig)



# The following plots are made for NERSC Jupyter Notebook
# =======================================================

# Block 1
# --------
reg_abr = 'SSA'
dict_carbon_freq_bin[reg_abr]


# Block 2
# --------
#reg_abr = 'NEB'
ddd = dict_carbon_freq_bin[reg_abr].filter(['Regional_NBP','NBP','NBP_du_Ext','Uptake_Change'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
ddd.plot.bar(stacked =False, 
              figsize=(15,6), 
              fontsize = 14,
              grid='--')
plt.title('NBP Stats for %s'%reg_abr, loc='left',fontsize =22)
plt.legend(loc='upper right', bbox_to_anchor=(1,1.2), fontsize=14, ncol=1)


# Block 3
# --------
ddd = dict_carbon_freq_bin[reg_abr].filter(['Count_Pos_Ext','Count_Neg_Ext'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
ddd.plot.bar(stacked =False, 
              figsize=(15,4), 
              fontsize = 14,
              grid='--')
plt.title('Months Count Stats for %s'%reg_abr, loc='left', fontsize =22)
plt.legend(loc='upper right', bbox_to_anchor=(1,1.2), fontsize=14, ncol=1)

# Block 4
# --------
ddd = dict_carbon_freq_bin[reg_abr].filter(['CUP_du_Ext','CRP_du_Ext','NBP_du_Ext'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
ddd.plot.bar(stacked =False, 
              figsize=(15,4), 
              fontsize = 14,
              grid='--')
plt.title('CUP/CRP during Extremes for %s'%reg_abr, loc='left', fontsize =22)
plt.legend(loc='upper right', bbox_to_anchor=(1,1.2), fontsize=14, ncol=1)


# Block 5
# --------
ddd = dict_carbon_freq_bin[reg_abr].filter(['CUP_px','CRP_px','NBP'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
ddd.plot.bar(stacked =False, 
              figsize=(15,4), 
              fontsize = 14,
              grid='--')
plt.title('Integrated CUP/CRP for %s with alteast one extreme (PgC)'%reg_abr, loc='left', fontsize =22)
plt.legend(loc='upper right', bbox_to_anchor=(1,1.2), fontsize=14, ncol=1)


# Block 6
# --------
ddd = dict_carbon_freq_bin[reg_abr].filter(['Count_CUP_du_Neg_Ext','Count_CRP_du_Neg_Ext'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
ddd.plot.bar(stacked =False, 
              figsize=(15,4), 
              fontsize = 14,
              grid='--')
plt.title('Count of CUP/CRP during Extremes Negative Extremes for %s'%reg_abr, loc='left', fontsize =22)
plt.legend(loc='upper right', bbox_to_anchor=(1,1.2), fontsize=14, ncol=1)


# Block 7
# --------
ddd = dict_carbon_freq_bin[reg_abr].filter(['Count_CUP_du_Pos_Ext','Count_CRP_du_Pos_Ext'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
ddd.plot.bar(stacked =False, 
              figsize=(15,4), 
              fontsize = 14,
              grid='--')
plt.title('Count of CUP/CRP during Extremes Positive Extremes for %s'%reg_abr, loc='left', fontsize =22)
plt.legend(loc='upper right', bbox_to_anchor=(1,1.2), fontsize=14, ncol=1)


# Block 8
# --------
ddd = dict_carbon_freq_bin[reg_abr].filter(['Count_CUP_du_Pos_Ext','Count_CRP_du_Pos_Ext',
                                            'Count_CUP_du_Neg_Ext','Count_CRP_du_Neg_Ext'], axis =1)
ddd_curp = ddd.copy(deep=True) # to capture the fq of cup or crp
ddd_curp['Count_CUP_du_Ext'] = ddd.loc[:,'Count_CUP_du_Pos_Ext'] + ddd.loc[:,'Count_CUP_du_Neg_Ext']
ddd_curp['Count_CRP_du_Ext'] = ddd.loc[:,'Count_CRP_du_Pos_Ext'] + ddd.loc[:,'Count_CRP_du_Neg_Ext']
ddd_tmp = ddd_curp.filter(['Count_CUP_du_Ext','Count_CRP_du_Ext'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
ddd_tmp.plot.bar(stacked =False, 
              figsize=(15,4), 
              fontsize = 14,
              grid='--')
plt.legend(loc='upper right', bbox_to_anchor=(1,1.2), fontsize=14, ncol=1)
plt.title('Total Count of CUP/CRP during Extremes for %s'%reg_abr, loc='left',fontsize =22)


# Block 9
# --------
ddd = dict_carbon_freq_bin[reg_abr].filter(['Count_Pos_Ext','Count_Neg_Ext'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
plt.style.use("classic")
ddd.plot.bar(stacked =False, 
              figsize=(9,4), 
              fontsize = 14,
              color = ['royalblue','darkorange'])

plt.legend(['Positive Extremes', 'Negative Extremes'], loc='upper right', 
            bbox_to_anchor=(1.,1.15), fontsize=12, ncol=1)
plt.title(f'Duration of {variable.upper()} Extremes at {reg_abr}', loc='left',fontsize=16)
#plt.title('Months Count Stats for %s'%reg_abr, loc='left', fontsize =16)
plt.grid (which='both', ls='--', lw='.5', alpha=.4 )
plt.xlabel ("Time", fontsize=14)
plt.ylabel ("Total Duration (Months)", fontsize=14)
plt.xticks (fontsize=12, rotation=60)
plt.yticks (fontsize=12, rotation=0)
plt.savefig(web_path + f"Regional/{source_run.upper()}_{reg_abr}_duration_{variable}_extremes.pdf",
           edgecolor="w", bbox_inches="tight")
plt.savefig(web_path + f"Regional/{source_run.upper()}_{reg_abr}_duration_{variable}_extremes.png",
           edgecolor="w", bbox_inches="tight")
fig.savefig(path_save + f"{source_run.upper()}_{reg_abr}_duration_{variable}_extremes.pdf",
           edgecolor="w", bbox_inches="tight")

# Block 10
# --------
ddd = dict_carbon_freq_bin[reg_abr].filter(['Count_CRP_du_Pos_Ext','Count_CRP_du_Neg_Ext'], axis =1)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
plt.style.use("classic")
ddd.plot.bar(stacked =False, 
              figsize=(9,4), 
              fontsize = 14,
              color = ['royalblue','darkorange'])
#plt.title('Count of Neg/Pos Extremes during CRP %s'%reg_abr, loc='left', fontsize =16)
plt.title(f'Duration of Carbon Release Periods during {variable.upper()} Extremes', 
          loc='left', fontsize =16)
plt.legend(['Positive Extremes', 'Negative Extremes'],
           loc='upper right', bbox_to_anchor=(1,1.), fontsize=12, ncol=1)
plt.xlabel ("Time", fontsize=14)
plt.ylabel ("Total Duration (Months)", fontsize=14)
plt.xticks (fontsize=12, rotation=60)
plt.yticks (fontsize=12, rotation=0)
plt.grid (which='both', ls='--', lw='.5', alpha=.4 )
plt.savefig(web_path + f"Regional/{source_run.upper()}_{reg_abr}_duration_{variable}_extremes_du_CRP.pdf",
           edgecolor="w", bbox_inches="tight")
plt.savefig(web_path + f"Regional/{source_run.upper()}_{reg_abr}_duration_{variable}_extremes_du_CRP.png",
           edgecolor="w", bbox_inches="tight")
fig.savefig(path_save + f"{source_run.upper()}_{reg_abr}_duration_{variable}_extremes_du_CRP.pdf",
           edgecolor="w", bbox_inches="tight")


# 
# =====================================================
# Calculating the number of regions per time window...
# ... that are dominanted by Positive or Negative ...
# ... extremes in NBP anomalies!
values_range = []
sign_count_ext = {}
Col_Name1 = 'Count_Pos_Ext'
Col_Name2 = 'Count_Neg_Ext'
# anaylsis do Col_Name1 - Col_Name2
for r in srex_abr:
    sign_count_ext[r] = {}
    for wi in win_yr:
        values_range.append(dict_carbon_freq_bin[r].loc[wi,Col_Name1] - 
                            dict_carbon_freq_bin[r].loc[wi,Col_Name2])
        if (dict_carbon_freq_bin[r].loc[wi,Col_Name1] - dict_carbon_freq_bin[r].loc[wi,Col_Name2]) > 0:
            sign_count_ext[r][wi] = '+' 
        elif (dict_carbon_freq_bin[r].loc[wi,Col_Name1] - dict_carbon_freq_bin[r].loc[wi,Col_Name2]) < 0:
            sign_count_ext[r][wi] = u"\u2212"
        else:
            sign_count_ext[r][wi] = '*'

stats_neg_pos_extremes_regions = {}
for wi in win_yr:
    stats_neg_pos_extremes_regions[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        stats_neg_pos_extremes_regions[wi][srex_abr[s_idx]] = (
                dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name1] -
                dict_carbon_freq_bin[srex_abr[s_idx]].loc[wi, Col_Name2]  )
        
counts_ext = {}
for wi in win_yr: 
    counts_ext[wi] = {}
    (counts_ext[wi]['pos'],
     counts_ext[wi]['neg']) = ((np.array(list(stats_neg_pos_extremes_regions[wi].values()))>0).sum(),
                               (np.array(list(stats_neg_pos_extremes_regions[wi].values()))<0).sum()
    )

# Creating the data frame of the counts
df_counts = pd.DataFrame.from_dict(counts_ext,orient='index')

# Plot of the counts of neg/pos

import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plt.style.use("classic")
plot.rcParams.update(params)
df_counts.plot.bar(stacked =False, 
              figsize=(8,4), 
              fontsize = 14,
              color = ['royalblue','darkorange'])
plt.legend(['Positive Extremes', 'Negative Extremes'], 
           loc='upper left', fontsize=12, ncol=1)
#plt.legend(['Positive', 'Negative'], loc='upper right', bbox_to_anchor=(1.2,.6), fontsize=14, ncol=1)
plt.ylim([6,20])
plt.title('Count of Regions Dominated by Negative or Positive Extremes in NBP\n', loc='left',fontsize =14)
plt.ylabel ("Count of Regions", fontsize=14)
plt.xlabel ("Time", fontsize=14)
plt.yticks (fontsize=12)
plt.xticks (fontsize=12, rotation=45)
plt.grid (which='both', ls='--', lw='.5', alpha=.4 )
plt.text(3.5,19,"Total Regions: 26", fontsize=14, fontweight='bold', color='brown')
for w_idx,win in enumerate(win_yr):
    plt.text(w_idx+.025,df_counts.loc[win,'pos']+1.8,f"{int(np.round(df_counts.loc[win,'pos']/.26))}%",
             ha='right', va='top',color='k',rotation=90)
    plt.text(w_idx+.025,df_counts.loc[win,'neg']+1.8,f"{int(np.round(df_counts.loc[win,'neg']/.26))}%",
             ha='left', va='top',color='k',rotation=90)
plt.savefig(web_path + "Spatial_Maps/Bar_plot_count_regions_and_exts.pdf", bbox_inches='tight')
plt.savefig(web_path + "Spatial_Maps/Bar_plot_count_regions_and_exts.png", bbox_inches='tight')
plt.savefig(path_save + "Spatial/Bar_plot_count_regions_and_exts.pdf", bbox_inches='tight')


# To find which Regions in tropics dominated by negative extremes:
Tropics = np.array(["CAM", "AMZ", "NEB","WAF","EAF", "SAF", "SAS","SEA","NAU"])
print ("Regions in tropics dominated by negative extremes:")
print ("==================================================\n")
for w_str in win_yr:
    regions_neg_ext = []
    for key in stats_neg_pos_extremes_regions[w_str].keys():
        val = stats_neg_pos_extremes_regions[w_str][key]
        if val < 0:
            regions_neg_ext.append(key)
    print( w_str, list(set(Tropics).intersection(regions_neg_ext)), len(list(set(Tropics).intersection(regions_neg_ext))))

# =============================================

# Calculating the total Uptake loss (c-loss) for all regions and windows BIN (NON-TCE)
tmp_array_closs = np.zeros((len(srex_abr),len(win_yr))) 
for s_idx,s_name in enumerate(srex_abr):
    for w_idx,w_name in enumerate(win_yr):
        tmp_array_closs[s_idx,w_idx] = dict_carbon_bin[s_name].loc[w_name,'Uptake_Loss']

df_c_loss_regions_bin = pd.DataFrame(data=tmp_array_closs, index= srex_abr, columns=win_yr)

df_c_loss_regions_bin.to_csv(web_path + 'Regional/%s_%s_Uptake_loss_bin_all_wins-regions_PgC.csv'%(source_run,member_run))
df_c_loss_regions_bin.to_csv(path_save + '%s_%s_Uptake_loss_bin_all_wins-regions_PgC.csv'%(source_run,member_run))


# Calculating the total Uptake loss (c-loss) for all regions and windows - TCE
tmp_array_closs = np.zeros((len(srex_abr),len(win_yr)))
for s_idx,s_name in enumerate(srex_abr):
    for w_idx,w_name in enumerate(win_yr):
        tmp_array_closs[s_idx,w_idx] = dict_carbon_tce[s_name].loc[w_name,'Uptake_Loss']

df_c_loss_regions_tce = pd.DataFrame(data=tmp_array_closs, index= srex_abr, columns=win_yr)

df_c_loss_regions_tce.to_csv(web_path + 'Regional/%s_%s_Uptake_loss_tce_all_wins-regions_PgC.csv'%(source_run,member_run))
df_c_loss_regions_tce.to_csv(path_save + '%s_%s_Uptake_loss_tce_all_wins-regions_PgC.csv'%(source_run,member_run))


