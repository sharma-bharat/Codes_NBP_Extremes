# Bharat Sharma
# python 3.6
""" Input:
	------
	It reads the individual driver's correlation nc files
	
	Also uses regional masks of SREX regions to find dominant drivers regionally
	Output:
	-------
	* Timeseries of the percent distribution of dominant drivers at different lags

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
import xarray as xr


#1- Hack to fix missing PROJ4 env var for Basemaps Error
import os

"""
import conda
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
#-1 Hack end

from    mpl_toolkits.basemap    import Basemap
from    matplotlib import cm
import matplotlib.patches as patches
"""
parser  = argparse.ArgumentParser()
#parser.add_argument('--driver_ano'  ,   '-dri_a'    , help = "Driver anomalies"                     , type= str     , default= 'pr'     ) #pr
parser.add_argument('--variable'    ,   '-var'      , help = "Anomalies of carbon cycle variable"   , type= str     , default= 'gpp'    )
parser.add_argument('--source'      ,   '-src'      , help = "Model (Source_Run)"                   , type= str     , default= 'CESM2'  ) # Model Name
parser.add_argument('--member_idx'  ,   '-m_idx'    , help = "Member Index"                   		, type= int     , default= 0  		) # Index of the member
#parser.add_argument ('--cum_lag'    ,'-lag'     , help = 'cum lag months? (multiple lag optional) use , to add multiple'        , type = str    , default = '01,02,03'  )

args = parser.parse_args()

# run plot_dominant_climate_driver_correlation_tce_regional_graphs.py -var gpp -src CESM2
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

# The name with which the variables are stored in the nc files:
features['Names']			 	= {}
features['Names']['pr']	 		= 'pr'
features['Names']['mrso']	 	= 'mrso'
features['Names']['tas']	 	= 'tas'
if source_run == 'CESM2':	
	features['Names']['fFireAll']	= 'Fire'
#features['Names']['tasmax']	 	= 'tasmax'

features['filenames'][variable]	= {}	# Creating a empty directory for storing multi members if needed
features['filenames'][variable][member_run]	= cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_anomalies_gC.nc"%(source_run,member_run, variable,member_run,variable)

# Reading the Correlations Data
# -----------------------------
exp= 'ssp585'
path_corr	= cori_scratch + 'add_cmip6_data/%s/%s/%s/%s/Correlations/'%(source_run,exp,member_run,variable)
nc_corr		= nc4.Dataset(path_corr + 'dominant_driver_correlation_%s.nc'%(variable))

# Reading the variables from the variable (gpp) anomalies file
# ------------------------------------------------------------
nc_var	= nc4.Dataset(features['filenames'][variable][member_run])
time	= nc_var .variables['time']

# Reading the variables from the correlation file
# -----------------------------------------------
ranks		= nc_corr .variables['rank'		]
wins		= nc_corr .variables['win'		]
lags		= nc_corr .variables['lag'		]
dom_dri_ids	= nc_corr .variables['dri_id'	]
dom_dri_cc	= nc_corr .variables['dri_coeff']

# Grids:
# -------
lat			= nc_var .variables ['lat']
lon			= nc_var .variables ['lon']
lat_bounds  = nc_var .variables [nc_var.variables['lat'].bounds ]
lon_bounds  = nc_var .variables [nc_var.variables['lon'].bounds ]
lon_edges	= np.hstack (( lon_bounds[:,0], lon_bounds[-1,-1]))
lat_edges	= np.hstack (( lat_bounds[:,0], lat_bounds[-1,-1]))

# Creating mask of the regions based on the resolution of the model
import regionmask
srex_mask 	= regionmask.defined_regions.srex.mask(lon[...], lat[...]).values  # it has nans
srex_mask_ma= np.ma.masked_invalid(srex_mask) # got rid of nans; values from 1 to 26

# important regional information:
srex_abr		= regionmask.defined_regions.srex.abbrevs
srex_names		= regionmask.defined_regions.srex.names
srex_nums		= regionmask.defined_regions.srex.numbers 
srex_centroids	= regionmask.defined_regions.srex.centroids 
srex_polygons	= regionmask.defined_regions.srex.polygons


# Organizing time
# ---------------
window      = 25 #years
win_len     = 12 * window            #number of months in window years
nwin        = int(time.size/win_len) #number of windows

#wins    = np.array([format(i,'02' ) for i in range(nwin)])
dates_ar    = time_dim_dates ( base_date= dt.date(1850,1,1), total_timestamps=time.size)
start_dates = [dates_ar[i*win_len] for i in range(nwin)]#list of start dates of 25 year window
end_dates   = [dates_ar[i*win_len+win_len -1] for i in range(nwin)]#list of end dates of the 25 year window

# String
# ------
wins_str = [format(int(i),'02') for i in wins[...]] 
lags_str = [format(int(i),'02') for i in lags[...]] 
ranks_str = [format(int(i),'02') for i in ranks[...]] 

# Regional masks
# --------------
import regionmask

# To store all the DataFrames of counts of dominant climate drivers in a dictionnary for every region
DataFrames_counts = {}
#format>>> DataFrames [regions] [wins] [lags] [ranks]
# range: regions: 26
# wins : 10 
# lags : 1
# ranks: 1
for region_abr in srex_abr:
	DataFrames_counts[region_abr] = {}
	for w in wins_str:
		DataFrames_counts[region_abr][w] = {}
		for l in lags_str[1:3]:
			DataFrames_counts[region_abr][w][l] = {}

save_path = "/global/cscratch1/sd/bharat/add_cmip6_data/%s/ssp585/%s/%s/Correlations/Regional/DataFrames/"%(
					source_run, member_run, variable)

if os.path.isdir(save_path) == False:
    os.makedirs(save_path)

# Storing the dataframes for regions,win,lag, rk
# ----------------------------------------------
dict_counts = {}
for region_abr in srex_abr:
	dict_counts[region_abr] = {}
	for win in np.asarray(wins[...], dtype =int):
		dict_counts[region_abr][win] = {}
		for lg in np.asarray(lags[...] [1:],dtype = int):
			dict_counts[region_abr][win][lg] = {}
			for rk in np.asarray(ranks[...][0:1], dtype = int):
				dict_counts[region_abr][win][lg][rk] = {}


# Computing the DataFrames
# ------------------------	
for region_abr in srex_abr: #testing for AMZ only
	srex_idxs 		= np.arange(len(srex_names))      
	filter_region 	= np.array(srex_abr) == region_abr
	region_idx		= srex_idxs[filter_region][0]
	region_number	= np.array(srex_nums)[filter_region][0]
	region_name		= np.array(srex_names)[filter_region][0]
	region_abr		= np.array(srex_abr)[filter_region][0] 
	region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region
	region_mask		= ~region_mask_not   # Only the regions is masked

	for win in np.asarray(wins[...], dtype =int):
		for lg in np.asarray(lags[...] [1:3],dtype = int):  # interested in lag = 1 month i.e. index = 1
			for rk in np.asarray(ranks[...][0:1], dtype = int): # interested in the dominant driver only

				counts          = np.unique( np.ma.masked_equal( np.ma.masked_invalid( 
									dom_dri_ids[rk,win,lg,:,:][region_mask]),0), 
									return_counts=True)  
				# there are np.nans and 0's in the array that have to be masked
				counts_drivers	= np.array([counts[1][i] for i in range(counts[1].size)])
				#since many drivers were not dominant for most part so only limiting the plot to the relevant ones
				print ("counts for dom rank %s and lag %s...:"%(format(rk,'002'), format(lg,'002')))
				tmp_drivers_code    = np.copy(drivers_code)
				for d in counts[0].data:
					tmp_drivers_code = np.ma.masked_equal (tmp_drivers_code, d)
				df_counts       = pd.DataFrame({'Counts':counts_drivers[:-1]})  #the last value corresponds to the masks

				df_counts.index = drivers [tmp_drivers_code.mask]
				perc = [round(i*100./sum(df_counts['Counts'].values),2) for i in df_counts['Counts'].values]
				df_counts['percentage']=perc
				
				#Calculating the mean and std of the climate drivers
				mean_cc = []
				std_cc  = []
				for code_id in drivers_code[tmp_drivers_code.mask]:
					#print "code_ID...", code_id
					mean_cc.append(np.ma.mean(dom_dri_cc[rk,win,lg,:,:][~np.ma.masked_not_equal(dom_dri_ids[rk,win,lg,:,:],code_id).mask]))
					std_cc.append(np.ma.std(dom_dri_cc[rk,win,lg,:,:][~np.ma.masked_not_equal(dom_dri_ids[rk,win,lg,:,:],code_id).mask]))
				df_counts['mean_coeff'] = mean_cc
				df_counts['std_coeff']  = std_cc
				# Saving the Data Frame in a dic:
				DataFrames_counts[ region_abr] [wins_str[win]] [lags_str[lg]] [ranks_str[rk]] = df_counts #since the numbers are indexs are same	
				print ('dataframe_win_%s_lag_%s_and_rank_%s.csv'%(format(win,'02'),format(lg,'02'),format(rk,'02')))
				df_counts .to_csv(save_path +  'df_reg_%s_win_%s_lag_%s_and_rank_%s.csv'%(region_abr, format(win,'02'),format(lg,'02'),format(rk,'02')),sep=',')

				# Regional dominant driver to dictionary
				# --------------		
				df_counts_t = df_counts[df_counts.loc[:,'percentage'] == df_counts.loc[:,'percentage'].max()]
				if df_counts_t.size > 0:
					dict_counts[region_abr][win][lg][rk] ['Dri_Name'] 	= df_counts_t.index[0]
					dict_counts[region_abr][win][lg][rk] ['Corr_Coeff'] = df_counts_t['mean_coeff'][0]
					dict_counts[region_abr][win][lg][rk] ['Dri_Code']   = drivers_code[drivers == df_counts_t.index[0]][0]
				elif df_counts_t.size == 0:
					dict_counts[region_abr][win][lg][rk] ['Dri_Name'] 	= np.nan
					dict_counts[region_abr][win][lg][rk] ['Corr_Coeff'] = np.nan
					dict_counts[region_abr][win][lg][rk] ['Dri_Code']	= np.nan

#df_counts .to_csv(path_corr +  'dataframes/dataframe_win_%s_lag_%s_and_rank_%s_np2.csv'%(format(win,'02'),format(lg,'02'),format(rk,'02')),sep=',') # [Changed] No pvalue filter
#print(breakit)
"""		
# =============================================================
# based on " ecp_triggers_percent_distribution_dom_drivers.py "
# =============================================================

# Plotting the timeseries of dominant drivers
# -------------------------------------------

in_yr   = 1850
win_yr  = [str(in_yr+i*25) + '-'+str(in_yr +(i+1)*25-1)[2:] for i in range(wins.size)]
plot_lags = ['01','02','03']
data_percent = np.zeros((len(win_yr), len(drivers_names)))
print ("data_percent shape: ", data_percent.shape)

data_lag = {}
for LAG in plot_lags : 
	data_percent = np.zeros((len(win_yr), len(drivers_names)))
	print ("data_percent shape: ", data_percent.shape)
	print ("data shape", np.transpose(DataFrames_counts[w] [LAG] [ranks_str[rk]]['percentage']).shape)
	df = pd.DataFrame( data_percent , index = win_yr, columns = drivers_names) #main dataframe
	
	for w in wins_str:
		data = DataFrames_counts [w] [LAG] [ranks_str[rk]]
		drivers = data.iloc[:,0]
		data_df	= pd.DataFrame( DataFrames_counts[w] [LAG] [ranks_str[rk]]['percentage'].values.reshape(1,len(drivers)),index =  [win_yr[int(w)]], columns = drivers) # dataframe for a particuar window

		for idx,dr in enumerate (drivers):
			df.loc[data_df.index,drivers_names[idx]] = data_df[dr]
	data_lag [LAG] = df

# Plotting Subplots
# -----------------
if source_run == 'CESM2':
	color_list      = ['b','b','g','r']
	linestyle_list  = ['--','-','-','-']
else:
	color_list      = ['b','g','r'] 
	linestyle_list  = ['-','-','-']
fig,ax  = plt.subplots(nrows=3,ncols= 1,gridspec_kw = {'wspace':0, 'hspace':0.02},tight_layout = True, figsize = (7,8.5), dpi = 400)
ax      = ax.ravel()
for lag_idx,LAG in enumerate(plot_lags):
	for dri_idx in range(len(drivers_names)):
		ax[lag_idx].plot( range(wins.size), data_lag[LAG].iloc[:,dri_idx], label = drivers_names[dri_idx], color=color_list[dri_idx],linestyle=linestyle_list[dri_idx], linewidth = 1)
	ax[lag_idx].set_xticks(range(wins.size))
	ax[lag_idx].tick_params(axis="x",direction="in")
	ax[lag_idx].set_xticklabels([])
	ax[lag_idx].set_ylabel("Lag: %s"%(LAG))
	ax[lag_idx].set_ylim([0,50])
	ax[lag_idx].grid(which='major', linestyle=':', linewidth='0.3', color='gray')
ax[lag_idx].set_xticklabels(df.index,fontsize =9)
for tick in ax[lag_idx].get_xticklabels():
	tick.set_rotation(90)
ax[lag_idx].set_xlabel('Time ($25-yr)$ wins',fontsize =14)
fig.text(0.03, 0.5, 'Percent Distribution of Climate Drivers', va='center', ha='center', rotation='vertical', fontsize=14)
ax[0].legend(loc = 'upper center',ncol=len(drivers_names), bbox_to_anchor=(.44,1.15),frameon =False,fontsize=9,handletextpad=0.1)
plt.gcf().subplots_adjust(bottom=0.1)
fig.savefig(path_corr +  'per_dom/percent_dominance_multilag_123_rank_%s.pdf'%(ranks_str[rk]))
#fig.savefig(path_corr +  'per_dom/percent_dominance_multilag_123_rank_%s_np2.pdf'%(ranks_str[rk])) # [changed] no p-value filter
"""
# Plotting Subplots
# -----------------
if source_run == 'CESM2':
	color_list      = ['b','b','g','r']
	linestyle_list  = ['--','-','-','-']
else:
	color_list      = ['b','g','r'] 
	linestyle_list  = ['-','-','-']
web_path 	= '/global/homes/b/bharat/results/web/Regional/Attribution/'
path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/"%(source_run,member_run, variable)      

# x -  axis
in_yr   = 1850
win_yr  = [str(in_yr+i*25) + '-'+str(in_yr +(i+1)*25-1)[2:] for i in range(wins.size)]

# Initializing the dataframe
data_percent =  np.zeros((len(win_yr), len(drivers_names)))

# Choose the lag
# -------------
lg =1

#Creating an empty dict for storing the dataframes:
# ------------------------------------------------
dict_dataframe  = {}
for r_idx, region_abr in enumerate(srex_abr):
	df = pd.DataFrame( data_percent , index = win_yr, columns = drivers) #main dataframe
	for w_idx, w in enumerate (wins_str):
		data = DataFrames_counts[region_abr][w][lags_str[lg]] [ranks_str[rk]]
		drivers_tmp = data.iloc[:,0]
		for col in df.columns :
			try:
				df .loc[win_yr[w_idx],col] = data.loc[col,'percentage']
			except:
				df .loc[win_yr[w_idx],col] = 0
	dict_dataframe[region_abr] = df.copy(deep = True)
	del df

# Plotting the dominant driver distribution for all the regions:
# --------------------------------------------------------------
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
plt.suptitle ("TS of dominant drivers during TCEs (lag:%d)"%lg, fontsize = 14)
txt ="The left y-axis represents the percent count of drivers in that region"
axs     = axs.ravel()
for k_idx, key in enumerate(dict_dataframe.keys()):
	df = dict_dataframe[key]
	for dri_idx in range(len(drivers)):
		axs[k_idx].plot( range(wins.size), df.iloc[:,dri_idx], label = drivers_names[dri_idx], color=color_list[dri_idx],linestyle=linestyle_list[dri_idx], linewidth = 0.8)
	#axs[k_idx].set_xticks(range(wins.size))
	axs[k_idx].set_ylabel("%s"%key)
	axs[k_idx].grid(which='major', linestyle=':', linewidth='0.3', color='gray')
for tick in axs[-3].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-2].get_xticklabels():
    tick.set_rotation(45)
for tick in axs[-1].get_xticklabels():
    tick.set_rotation(45)

#axs[1].legend(loc = 'upper center',ncol=len(drivers_names), bbox_to_anchor=(.5,1.15),frameon =False,fontsize=9,handletextpad=0.1)
plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12) #Caption
fig.savefig(web_path + 'percent_dom_%s_%s_lag_%s_regions_%s.pdf'%(source_run, member_run, format(lg,'02'),variable.upper()) ) 

# Common Information for spatial plots
# ====================================
sub_fig_text = ['(a)', '(b)', '(c)',
               '(d)', '(e)', '(f)']
Wins_to_Plot = ['1850-74', '1900-24', '1950-74', '2000-24', '2050-74', '2075-99']
Wins_to_Plot_idxs = [0,2,4,6,8,9]
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid
proj_trans = ccrs.PlateCarree()
proj_output = ccrs.PlateCarree()

# Plotting individual drivers
# ===========================
# Spatial plot of individual driver correlatons
# for idx, dri in enumerate (drivers_names):
sub_fig_text = ['(a)', '(b)', '(c)',
               '(d)', '(e)', '(f)']
Wins_to_Plot = ['1850-74', '1900-24', '1950-74', '2000-24', '2050-74', '2075-99']
Wins_to_Plot_idxs = [0,2,4,6,8,9]
ymax = 1
ymin = -1
import cartopy.crs as ccrs
from matplotlib.axes import Axes
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid
proj_trans = ccrs.PlateCarree()
proj_output = ccrs.PlateCarree()

for dri_idx, dri in enumerate (drivers_names):
    fig = plt.figure(figsize = (12,9), dpi = 200)

    #pwin = Wins_to_Plot_idxs[0]
    plag = 1

    ax = {}
    gl = {}
    for plot_idx, win_idx in enumerate(Wins_to_Plot_idxs):
    #plot_idx = 0 #
        gl[plot_idx] = 0
        if plot_idx == 0:
            ax[plot_idx] = fig.add_subplot(
                        2, 3, plot_idx+1, projection= proj_output
                    )
			# Mean Correlation Coefficient of the Selected climate Drivers at any rank
            plot_data =  np.ma.mean(np.ma.masked_array(data=dom_dri_cc[:,win_idx,plag,:,:], 
                                                    mask = np.ma.masked_not_equal(dom_dri_ids[:,win_idx,plag,:,:], 
                                                    drivers_code[dri_idx]) .mask),axis = 0)
            h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  plot_data, 
                                        transform=ccrs.PlateCarree(), vmax=ymax, vmin=ymin, cmap='PuOr')
            for srex_idx,abr in enumerate (srex_abr):
                ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)

        elif plot_idx>0:
            ax[plot_idx] = fig.add_subplot(
                        2, 3, plot_idx+1, projection= proj_output,
                        sharex=ax[0], sharey=ax[0]
                    )
			# Mean Correlation Coefficient of the Selected climate Drivers at any rank
            plot_data =  np.ma.mean(np.ma.masked_array(data=dom_dri_cc[:,win_idx,plag,:,:], 
                                                    mask = np.ma.masked_not_equal(dom_dri_ids[:,win_idx,plag,:,:], 
                                                    drivers_code[dri_idx]) .mask),axis = 0)
            h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  plot_data, 
                                        transform=ccrs.PlateCarree(), vmax=ymax, vmin=ymin, cmap='PuOr')
            for srex_idx,abr in enumerate (srex_abr):
                ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)

    for plot_idx in range(len(Wins_to_Plot)):
        ax[plot_idx].coastlines(alpha=0.75)
        ax[plot_idx].text(-85, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
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
    ax[1].set_title("Correlation Coefficient of %s with %s extremes"%(dri,variable.upper()), fontsize=14)
    fig.savefig(web_path + "Spatial_Corr_%s_%s_lag_%d.pdf"%(variable,dri,plag),
                bbox_inches = "tight", edgecolor="w")
    fig.savefig(web_path + "Spatial_Corr_%s_%s_lag_%d.png"%(variable,dri,plag),
                bbox_inches = "tight", edgecolor="w")
    fig.savefig(path_save + "Correlations/Spatial_Maps/Spatial_Corr_%s_%s_lag_%d.pdf"%(variable,dri,plag),
                bbox_inches = "tight", edgecolor="w")
    del fig




# Dominant Driver spatial plot at lag =1 month
# ===========================================
# Spatial plot of Dominant driver correlatons
# for idx, dri in enumerate (drivers_names):
ymax = 45
ymin = 5

rk = 0 #Dominant driver
plag = 1 # lag =1 month

fig = plt.figure(figsize = (12,9), dpi = 200)
ax = {}
gl = {}
for plot_idx, win_idx in enumerate(Wins_to_Plot_idxs):
#plot_idx = 0 #
    gl[plot_idx] = 0
    if plot_idx == 0:
        ax[plot_idx] = fig.add_subplot(
                    2, 3, plot_idx+1, projection= proj_output
                )
        plot_data = np.ma.masked_equal(np.ma.masked_invalid(dom_dri_ids[rk,win_idx,plag,:,:]),0)


        cmap    = plt.get_cmap('rainbow', drivers_code.size)
        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  plot_data,
                                    transform=ccrs.PlateCarree(), vmax=ymax, vmin=ymin, cmap=cmap)
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
    elif plot_idx>0:
        ax[plot_idx] = fig.add_subplot(
                    2, 3, plot_idx+1, projection= proj_output,
                    sharex=ax[0], sharey=ax[0]
                    )
        plot_data = np.ma.masked_equal(np.ma.masked_invalid(dom_dri_ids[rk,win_idx,plag,:,:]),0)
        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  plot_data,
                                    transform=ccrs.PlateCarree(), vmax=ymax, vmin=ymin, cmap= cmap)
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)

for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    ax[plot_idx].text(-85, -10, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
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
cbar    = plt.colorbar(h, cax=cax, ticks = range(drivers_code[0],drivers_code[-1]+1,10))
cbar    .ax.set_yticklabels(drivers_names)
#plt.colorbar( h, cax=cax, orientation='vertical', pad=0.04, shrink=0.95);
ax[1].set_title("Dominant Drivers of %s extremes"%(variable.upper()), fontsize=14)
fig.savefig(web_path + "Spatial_Dominant_Driver_%s_lag_%d.pdf"%(variable,plag), 
            bbox_inches = "tight", edgecolor="w")
fig.savefig(web_path + "Spatial_Dominant_Driver_%s_lag_%d.png"%(variable,plag),
            bbox_inches = "tight", edgecolor="w")
fig.savefig(path_save + "Correlations/Spatial_Maps/Dominant_Driver_%s_lag_%d.pdf"%(variable,plag),
            bbox_inches = "tight", edgecolor="w")
del fig

# Plotting of "Regional Dominance"
# =====================================
#dict_counts[region_abr][win][lg][rk] ['Dri_Name']
#dict_counts[region_abr][win][lg][rk] ['Corr_Coeff'] 
rk=0
lg=1
plag=1

values_range = []
sign = {}
for r in srex_abr:
    sign[r] = {}
    for win_idx, wi in enumerate(Wins_to_Plot):
        values_range.append(dict_counts[r][Wins_to_Plot_idxs[win_idx]][lg][rk] ['Corr_Coeff'])
        #print(win_idx,dict_counts[r][Wins_to_Plot_idxs[win_idx]][lg][rk] ['Corr_Coeff'] )
        if dict_counts[r][Wins_to_Plot_idxs[win_idx]][lg][rk] ['Corr_Coeff'] > 0:
            sign[r][wi] = '+' 
        elif dict_counts[r][Wins_to_Plot_idxs[win_idx]][lg][rk] ['Corr_Coeff'] < 0:
            sign[r][wi] = u"\u2212"
        else:
            sign[r][wi] = ' '

print ("To check for the range of values")
print (np.array(values_range).min())
print (np.array(values_range).max())
ymax = 45
ymin = 5

# Creating the NBP Values for 1850-74 for all regions for NBP du Ext
ploting_stats = {}
for win_idx, wi in enumerate(Wins_to_Plot):
    ploting_stats[wi] = {}
    
    all_masked = np.ma.masked_equal(np.ma.zeros(srex_mask_ma.shape),0)
    for s_idx in srex_idxs:
        tmp = np.ma.masked_equal(srex_mask_ma,s_idx+ 1).mask  # +1 because srex_idxs start from 1
        all_masked[tmp] = dict_counts[srex_abr[s_idx]][Wins_to_Plot_idxs[win_idx]][lg][rk]  ['Dri_Code']
        del tmp

    all_masked = np.ma.masked_array(all_masked, mask = srex_mask_ma.mask)
    ploting_stats[wi] ['Dri_Codes'] = np.ma.masked_equal(np.ma.masked_invalid(all_masked),0)
# test plot
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from mpl_toolkits.axes_grid1 import AxesGrid

proj_trans = ccrs.PlateCarree()
#proj_output = ccrs.Robinson(central_longitude=0)
proj_output = ccrs.PlateCarree()
fig = plt.figure(figsize = (12,9), dpi = 400)
plt.style.use("classic")

ax = {}
gl = {}

for plot_idx in range(len(Wins_to_Plot)):
    gl[plot_idx] = 0
    if plot_idx == 0 :

        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output
        )
        plot_data = np.ma.masked_equal(np.ma.masked_invalid(ploting_stats[Wins_to_Plot[plot_idx]]['Dri_Codes']),0)
        cmap    = plt.get_cmap('rainbow', drivers_code.size)
        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  plot_data, 
                                    transform=ccrs.PlateCarree(),vmax=ymax, vmin=ymin,cmap= cmap)
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], sign[abr][Wins_to_Plot[plot_idx]],
                    horizontalalignment='center',
                    color = 'white', fontweight = 'bold',fontsize=10,
                    transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
        
    
    elif plot_idx>0:
            
        ax[plot_idx] = fig.add_subplot(
            2, 3, plot_idx+1, projection= proj_output,
            sharex=ax[0], sharey=ax[0]
            )
        plot_data = np.ma.masked_equal(np.ma.masked_invalid(ploting_stats[Wins_to_Plot[plot_idx]]['Dri_Codes']),0)

        h = ax[plot_idx].pcolormesh(lon_edges[...],lat_edges[...],  plot_data, 
                                    transform=ccrs.PlateCarree(),vmax=ymax,vmin=ymin,cmap= cmap)
        for srex_idx,abr in enumerate (srex_abr):
            ax[plot_idx].text (  srex_centroids[srex_idx][0], srex_centroids[srex_idx][-1], 
                               sign[abr][Wins_to_Plot[plot_idx]],
                               horizontalalignment='center',
                               color = 'white', fontweight = 'bold',fontsize=10,
                               transform = proj_trans)

            ax[plot_idx].add_geometries([srex_polygons[srex_idx]], crs = proj_trans, facecolor='none',  edgecolor='black', alpha=0.4)
            


for plot_idx in range(len(Wins_to_Plot)):
    ax[plot_idx].coastlines(alpha=0.75)
    ax[plot_idx].text(80, -60, sub_fig_text[plot_idx] + ' '+ Wins_to_Plot[plot_idx],
                     horizontalalignment="right",
                     verticalalignment='center',
                     fontsize = 12)
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
cbar    = plt.colorbar(h, cax=cax, ticks = range(drivers_code[0],drivers_code[-1]+1,10))
drivers_names_plotting = np.array(['Prcp', 'SM','TAS','Fire'])
cbar    .ax.set_yticklabels(drivers_names_plotting)
# cbar    .ax.set_yticklabels(drivers_names)
#plt.colorbar(h, orientation='horizontal', pad=0.04);
ax[1].set_title("Regional Distribution of Dominant Drivers of %s extremes \n"%(variable.upper()), fontsize=14)
fig.savefig(web_path + "Spatial_Regional_Dominant_Driver_%s_lag_%d.pdf"%(variable,plag), 
            edgecolor = "w", bbox_inches = "tight")
fig.savefig(web_path + "Spatial_Regional_Dominant_Driver_%s_lag_%d.png"%(variable,plag),
            bbox_inches = "tight")
fig.savefig(path_save + "Correlations/Spatial_Maps/Dominant_Regional_Driver_%s_lag_%d.pdf"%(variable,plag), 
            edgecolor = "w", bbox_inches = "tight")


# Calculation of the count of pixels of different regions...
# ...with positive and negative correlation coefficients!
# ========================================================
# For MRSO
# --------
dri_idx = 1 #for MRSO
plag = 1

# Dict to store the counts of pos/neg extremes
# --------------------------------------------
dict_mrso_cc_count = {}
for region_abr in srex_abr: 
    dict_mrso_cc_count[region_abr] = {}
    for win_idx, win_str in enumerate(win_yr):
        dict_mrso_cc_count[region_abr][win_str] = {}
del region_abr,win_idx, win_str

# Calculation of counts:
for region_abr in srex_abr: 
    for win_idx, win_str in enumerate(win_yr):
        driver_cc_win_tmp = np.ma.masked_array(data=dom_dri_cc[:,win_idx,plag,:,:], 
                    mask = np.ma.masked_not_equal(dom_dri_ids[:,win_idx,plag,:,:], 
                    drivers_code[dri_idx]) .mask)


        filter_region 	= np.array(srex_abr) == region_abr
        region_idx		= srex_idxs[filter_region][0]
        region_number	= np.array(srex_nums)[filter_region][0]
        region_name		= np.array(srex_names)[filter_region][0]
        region_abr		= np.array(srex_abr)[filter_region][0] 
        region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region
        region_mask		= ~region_mask_not   # Only the regions is masked

        cc_values_tmp = driver_cc_win_tmp[np.array([region_mask]*4)][driver_cc_win_tmp[np.array([region_mask]*4)].mask ==False]
        dict_mrso_cc_count[region_abr][win_str]['pos'] = (cc_values_tmp > 0).sum()
        dict_mrso_cc_count[region_abr][win_str]['neg'] = (cc_values_tmp < 0).sum()
        
del region_abr,win_idx, win_str,cc_values_tmp,region_mask


# For TAS
# --------
dri_idx = 2 #for TAS
plag = 1

# Dict to store the counts of pos/neg extremes
# --------------------------------------------
dict_tas_cc_count = {}
for region_abr in srex_abr: 
    dict_tas_cc_count[region_abr] = {}
    for win_idx, win_str in enumerate(win_yr):
        dict_tas_cc_count[region_abr][win_str] = {}
del region_abr,win_idx, win_str

# Calculation of counts:
for region_abr in srex_abr: 
    for win_idx, win_str in enumerate(win_yr):
        driver_cc_win_tmp = np.ma.masked_array(data=dom_dri_cc[:,win_idx,plag,:,:], 
                    mask = np.ma.masked_not_equal(dom_dri_ids[:,win_idx,plag,:,:], 
                    drivers_code[dri_idx]) .mask)

        filter_region 	= np.array(srex_abr) == region_abr
        region_idx		= srex_idxs[filter_region][0]
        region_number	= np.array(srex_nums)[filter_region][0]
        region_name		= np.array(srex_names)[filter_region][0]
        region_abr		= np.array(srex_abr)[filter_region][0] 
        region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region
        region_mask		= ~region_mask_not   # Only the regions is masked

        cc_values_tmp = driver_cc_win_tmp[np.array([region_mask]*4)][driver_cc_win_tmp[np.array([region_mask]*4)].mask ==False]
        dict_tas_cc_count[region_abr][win_str]['pos'] = (cc_values_tmp > 0).sum()
        dict_tas_cc_count[region_abr][win_str]['neg'] = (cc_values_tmp < 0).sum()
        
del region_abr,win_idx, win_str,cc_values_tmp,region_mask

# Analysis and presentation of data on correlation coefficient:
# -------------------------------------------------------------
# MRSO
df_mrso_cc = {}
for region_abr in srex_abr:
    df_mrso_cc[region_abr] = pd.DataFrame.from_dict(dict_mrso_cc_count[region_abr], orient='index')
    df_mrso_cc[region_abr].loc[:,"%pos"] = (df_mrso_cc[region_abr].loc[:,"pos"]*100/(
                                            df_mrso_cc[region_abr].loc[:,"pos"] + 
                                            df_mrso_cc[region_abr].loc[:,"neg"])
                                          ).round(decimals=1)
    df_mrso_cc[region_abr].loc[:,"%neg"] = (df_mrso_cc[region_abr].loc[:,"neg"]*100/(
                                            df_mrso_cc[region_abr].loc[:,"pos"] + 
                                            df_mrso_cc[region_abr].loc[:,"neg"])
                                          ).round(decimals=1)
del region_abr

#TAS
df_tas_cc = {}
for region_abr in srex_abr:
    df_tas_cc[region_abr] = pd.DataFrame.from_dict(dict_tas_cc_count[region_abr], orient='index')
    df_tas_cc[region_abr].loc[:,"%pos"] = (df_tas_cc[region_abr].loc[:,"pos"]*100/(
                                            df_tas_cc[region_abr].loc[:,"pos"] + 
                                            df_tas_cc[region_abr].loc[:,"neg"])
                                          ).round(decimals=1)
    df_tas_cc[region_abr].loc[:,"%neg"] = (df_tas_cc[region_abr].loc[:,"neg"]*100/(
                                            df_tas_cc[region_abr].loc[:,"pos"] + 
                                            df_tas_cc[region_abr].loc[:,"neg"])
                                          ).round(decimals=1)
del region_abr

# Ploting in Jupyter Notebook
# ---------------------------
# Percent count of pixels that are positively...
# ...or negatively correlated with MRSO
region_abr = srex_abr[2]

import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
df_mrso_cc[region_abr].iloc[2:,2:].plot.bar(stacked =False, 
              figsize=(9,4), 
              fontsize = 14,
              grid='--')
plt.legend(loc='upper right', bbox_to_anchor=(1.25,.6), fontsize=14, ncol=1)
plt.ylim([0,100])
plt.title(f"Percent count of the pixel with pos/neg correlation with TAS for {region_abr}", 
            loc='left',fontsize =15)
#plt.text(0,18,"Total Regions: 26", fontsize=14, fontweight='bold', color='brown')

# The number 10 or y axis represents the number of pixels 
for w_idx,w_str in enumerate(win_yr[2:]):
    plt.text(w_idx,10,f"{int(np.round(df_mrso_cc[region_abr].loc[w_str,'pos']))}",
             ha='right', va='top',color='white',rotation=90,fontsize=10,weight='bold')
    plt.text(w_idx,10,f"{int(np.round(df_mrso_cc[region_abr].loc[w_str,'neg']))}",
             ha='left', va='top',color='white',rotation=90,fontsize=10,weight='bold')

# Percent count of pixels that are positively...
# ...or negatively correlated with TAS

# The Srex_index for NAS is 17
region_abr = srex_abr[17]
#fig1 = plt.figure(figsize = (9,5), dpi = 400)
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
plt.style.use("classic")
df_tas_cc[region_abr].iloc[2:,2:].plot.bar(stacked =False, 
              figsize=(9,4), 
              fontsize = 14,
              color = ['royalblue','darkorange'])
plt.legend(['Positive Correlation', 'Negative Correlation'], 
           loc='upper right', fontsize=12, ncol=1)
plt.ylim([0,100])
plt.title(f"Correlation of {variable.upper()} Extremes with TAS for {region_abr}",
            fontsize =16)
#plt.text(0,18,"Total Regions: 26", fontsize=14, fontweight='bold', color='brown')
plt.ylabel ("Percent Count of Grid-cells", fontsize=14)
plt.xlabel ("Time", fontsize=14)
plt.yticks (fontsize=12)
plt.xticks (fontsize=12, rotation=60)
plt.grid (which='both', ls='--', lw='.5', alpha=.4 )
# The number 10 or y axis represents the number of pixels 
for w_idx,w_str in enumerate(win_yr[2:]):
    plt.text(w_idx+.04,10,f"{int(np.round(df_tas_cc[region_abr].loc[w_str,'pos']))}",
             ha='right', va='top',color='white',rotation=90,fontsize=12)
    plt.text(w_idx+.04,10,f"{int(np.round(df_tas_cc[region_abr].loc[w_str,'neg']))}",
             ha='left', va='top',color='white',rotation=90,fontsize=12)
plt.savefig(web_path + f"Change_in_Corr_of_{variable}_for_{region_abr}_lag_{plag}.pdf", 
            edgecolor = "w", bbox_inches = "tight")
plt.savefig(web_path + f"Change_in_Corr_of_{variable}_for_{region_abr}_lag_{plag}.png", 
            edgecolor = "w", bbox_inches = "tight")
plt.savefig(path_save + f"Change_in_Corr_of_{variable}_for_{region_abr}_lag_{plag}.pdf",
            edgecolor = "w", bbox_inches = "tight")


# Finding the locations of the extremes TCEs and correlations with TAS in NAS
save_txt_tas_nas = 'n'
if save_txt_tas_nas in ['y','Y','yes']:
    import sys
    stdout_fileno = sys.stdout
    # Redirect sys.stdout to the file
    cli_var = 'tas'
    path_tas_nas = f"{cori_scratch}add_cmip6_data/{source_run}/ssp585/{member_run}/{cli_var}/Correlations/Region_NAS/"
    if os.path.isdir(path_tas_nas) == False:
        os.makedirs(path_tas_nas)

    sys.stdout = open(path_tas_nas+'loc_nbp_tas_nas.txt', 'w')
    sys.stdout.write (f"win_idx,lt_idx,ln_idx,lag1,lag2,lag3,lag4],Dom-T/F,Dom-T/F,Dom-T/F,Dom-T/F" + '\n')
    # Dom-T/F: if True indicates the Dominant Drivers
    # TAS is often dominant at lag 2

    locs_tas_nas = {}
    list_all_wins_tas_nas = []
    for win_idx in range(len(win_yr)):
        # Correlation coefficients for lag = 1 for 'NAS'
        CC_TAS_all_ranks = np.ma.masked_array(data=dom_dri_cc[:,win_idx,plag,:,:], 
                                                        mask = np.ma.masked_not_equal(dom_dri_ids[:,win_idx,plag,:,:], 
                                                        drivers_code[dri_idx]) .mask)
        # figure out how to read only the non masked lat_lon for NAS
        tas_mask_true = (np.max(abs(CC_TAS_all_ranks),0).mask )
        lt_ln_mat               = create_seq_mat(nlat=lat.size, nlon=lon.size)
        # list of all location_ids in the global with a valid cc of TAS : 
        tas_global_1d_locs = lt_ln_mat[~tas_mask_true]
        # list of all location_ids in the SREX region:
        region_1d_locs = lt_ln_mat[region_mask]
        # list of location_ids in a region with a valid tas cc
        tas_region_1d_locs = np.intersect1d ( tas_global_1d_locs,region_1d_locs )
        list_locs_tmp = []


        for pixel in tas_region_1d_locs:
            lt,ln = np.argwhere(lt_ln_mat == pixel)[0]
            #print (win_idx, lt,ln, CC_TAS_all_ranks[:,lt,ln].data,CC_TAS_all_ranks[:,lt,ln].mask)
            tmp_text=  (f"{win_idx},{lt},{ln},{CC_TAS_all_ranks[:,lt,ln].data[0]},"
                       f"{CC_TAS_all_ranks[:,lt,ln].data[1]},{CC_TAS_all_ranks[:,lt,ln].data[2]},"
                        + f"{CC_TAS_all_ranks[:,lt,ln].data[3]},{CC_TAS_all_ranks[:,lt,ln].mask[0]},"
                        + f"{CC_TAS_all_ranks[:,lt,ln].mask[1]},{CC_TAS_all_ranks[:,lt,ln].mask[2]},"
                        + f"{CC_TAS_all_ranks[:,lt,ln].mask[3]}")
            list_locs_tmp.append(f"{lt}_{ln}")
            list_all_wins_tas_nas.append(f"{lt}_{ln}")

            # Prints to the redirected stdout (Output.txt)
            sys.stdout.write(tmp_text + '\n')
            # Prints to the actual saved stdout handler
            stdout_fileno.write(tmp_text + '\n')
            locs_tas_nas[win_idx] = np.array(list_locs_tmp)

    # List and count of the locations with correlation coefficients
    tas_nas_unique_locs,tas_nas_counts= np.unique(np.array(list_all_wins_tas_nas), return_counts=1)

    # Saving the Common locationa and count of the occurance for all wins
    tas_nas_unique_locs,tas_nas_counts= np.unique(np.array(list_all_wins_tas_nas), return_counts=1)
    stdout_fileno = sys.stdout
    sys.stdout = open(path_tas_nas+'locs_count_nbp_tas_nas.txt', 'w')
    sys.stdout.write (f"locs, count" + '\n')
    for idx in range(len(tas_nas_unique_locs)):
        tmp_text = f"{tas_nas_unique_locs[idx]},{tas_nas_counts[idx]}"
        sys.stdout.write(tmp_text + '\n')
        stdout_fileno.write(tmp_text + '\n')


# Analysis of the locations are done in an Excel sheet in the Document/cmip6/Region_NAS

# Calculating the change in TAS at different quantiles
# ====================================================
Calculate_quantile = 'n'
if Calculate_quantile in ['y','Y','yes']:
    # Calculating the quantiles of climate variable at pixel levels
    import xarray as xr
    cli_var = 'tas'
    path_cli_var = f"{cori_scratch}add_cmip6_data/{source_run}/ssp585/{member_run}/{cli_var}"
    file_cli_var = f"{path_cli_var}/{source_run}_ssp585_{member_run}_{cli_var}.nc"

    # Reading the tas
    nc_cli = xr.open_dataset(file_cli_var) # reading nc file
    tas = nc_cli.tas   # tas as object

    # extracting data for a location
    # -------------------------------
    lat_idx = 167
    lon_idx = 90

    # cli variable for a pixel
    tas_px = tas.isel(lat=lat_idx,lon=lon_idx).data

    # extracting data for a location
    # -------------------------------
    lat_idx = 157
    lon_idx = 52

    # cli variable for a pixel
    tas_px = tas.isel(lat=lat_idx,lon=lon_idx).data

    Quantiles = np.arange(0.1,1,.1)

    # Saving the quantiles and tas in Celsius
    tas_quant_px = {}
    for Quant in Quantiles:
        tas_quant_px[Quant] = {}

    # Finding the lowest and highest temperatures of tas:
    tas_low = 0
    tas_high = 0
    for Quant in Quantiles:
        for w_idx in range(10):
            tas_px_win = tas_px[w_idx*300:(w_idx+1)*300] - 273.15
            tas_q_px = np.quantile(tas_px_win,Quant)
            tas_quant_px[Quant][w_idx] = tas_q_px

            if tas_q_px < tas_low:
                tas_low = tas_q_px
            if tas_q_px > tas_high:
                tas_high = tas_q_px

    # Dataframe from dict of quantiles of a pixel
    df_quant_px = pd.DataFrame.from_dict(tas_quant_px)
    # the columns are the quantiles and the rows are the window index


    # rate of increase of Tas per window
    slope_px = []
    for Quant in Quantiles:
        slope_px.append(stats.linregress(range(10),list(tas_quant_px[Quant].values())))

    df_quant_px = pd.DataFrame.from_dict(tas_quant_px)
    #q = .1

    fig = plt.figure()
    ax = plt.subplot(111)

    for q in Quantiles:
        ax.plot(range(10),df_quant_px.loc[:,q], label= f"{q:.1f}")

    # text of slope of rise in temperature per window
    ax.text(8,df_quant_px.loc[7,.1], f"{slope_px[0][0]:.2f}" ) # Quant = .1
    ax.text(8,df_quant_px.loc[7,.5], f"{slope_px[4][0]:.2f}" ) # Quant = .5
    ax.text(8,df_quant_px.loc[7,.9], f"{slope_px[-1][0]:.2f}" ) # Quant = .9

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * .9])
    # Put a legend below current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5),
              fancybox=True, shadow=True, ncol=1,
             title='Quantiles')

    #ax.set_ylim(np.floor(tas_low)-1,np.ceil(tas_high)+1)

    # Show duplicate y-axis:
    plt.tick_params(labeltop=False, labelright=True)
    # Show grid
    ax.grid (which='both', ls='--', lw='.5', alpha=.4 )
    ax.set_ylabel ("Temperature (Celsius)", fontsize=14)
    #ax.set_yticklabels(fontsize= 10)
    ax.set_xticklabels(win_yr)
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    ax.set_xlabel ("Time", fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
    plt.title (f"TAS at lat={lat_idx},lon={lon_idx}", fontsize = 14)


    # Area weighted mean and quantile tas distribution of the region of TAS
    # =====================================================================
    # To calculate the area-weighted average of temperature:
    # Reading files/data
    tas = nc_cli.tas   # tas as object
    area = nc_cli.areacella # area as object

    #mask of the region
    region_abr = srex_abr[5] # for NAS
    filter_region 	= np.array(srex_abr) == region_abr # for NAS
    region_number	= np.array(srex_nums)[filter_region][0]
    region_name		= np.array(srex_names)[filter_region][0]
    region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region
    region_mask		= ~region_mask_not   # Only the regions is True or 1s

    # Masking the area to only the region of interest
    area_region= np.ma.masked_array(area,mask=region_mask_not)

    # mean area weighted average for every window
    print (f"Region: {srex_abr[5]}")
    print ("Area Weighted Mean Temperature")
    print ("---------")
    for w_idx in range(10):
        tas_awm_region = np.ma.average(np.mean(tas[w_idx*300:(w_idx+1)*300].data,axis=0),  weights=area_region) - 273.15
        print (f"for window {win_yr[w_idx]}, AWM: {tas_awm_region:.2f} Celsius")

    # Quantiles for the region of NAS
    # Saving the quantiles and tas in Celsius
    tas_quant_reg = {}
    for Quant in Quantiles:
        tas_quant_reg[Quant] = {}

    tas_reg = tas.data * np.array([region_mask]*tas.shape[0])
    # Finding the lowest and highest temperatures of tas:
    tas_low_reg = 0
    tas_high_reg = 0
    for Quant in Quantiles:
        for w_idx in range(10):

            tas_reg_win = tas_reg[w_idx*300:(w_idx+1)*300] 
            tas_q_reg = np.quantile(tas_reg_win[tas_reg_win!=0],Quant) # finding quantiles for non-zero outside the domain values
            tas_quant_reg[Quant][w_idx] = tas_q_reg - 273.15

            if tas_q_reg < tas_low_reg:
                tas_low_reg = tas_q_reg
            if tas_q_reg > tas_high_reg:
                tas_high_reg = tas_q_reg

    # Dataframe from dict of quantiles of a region
    df_quant_reg = pd.DataFrame.from_dict(tas_quant_reg)
    # the columns are the quantiles and the rows are the window index

    # rate of increase of Tas per window
    slope_reg = []
    for Quant in Quantiles:
        slope_reg.append(stats.linregress(range(10),list(tas_quant_reg[Quant].values())))


    # Plot of Quantilies of regions
    fig1 = plt.figure()
    ax = plt.subplot(111)
    color_list = []
    for q in Quantiles:
        p = ax.plot(range(10),df_quant_reg.loc[:,q], label= f"{q:.1f}")
        color_list.append(p[0].get_color())

    # text of slope of rise in temperature per window
    ax.text(6.5,df_quant_reg.loc[7,.1], f"{slope_reg[0][0]:.2f}", color=color_list[0] ) # Quant = .1
    ax.text(6.5,df_quant_reg.loc[7,.5], f"{slope_reg[4][0]:.2f}", color=color_list[4] ) # Quant = .5
    ax.text(6.5,df_quant_reg.loc[7,.9], f"{slope_reg[-1][0]:.2f}", color=color_list[-1] ) # Quant = .9

    # Shrink current axis's height by 10% on the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * .9])
    # Put a legend below current axis
    ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.5),
              fancybox=True, shadow=True, ncol=1,
             title='Quantiles')

    #ax.set_ylim(np.floor(tas_low)-1,np.ceil(tas_high)+1)

    # Show duplicate y-axis:
    plt.tick_params(labeltop=False, labelright=True)
    # Show grid
    ax.grid (which='both', ls='--', lw='.5', alpha=.4 )
    ax.set_ylabel ("Temperature (Celsius)", fontsize=14)
    #ax.set_yticklabels(fontsize= 10)
    ax.set_xticklabels(win_yr)
    for tick in ax.get_xticklabels():
        tick.set_rotation(60)
    ax.set_xlabel ("Time", fontsize=14)
    ax = plt.gca()
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
    #plt.title (f"TAS at {region_abr}", fontsize = 14)
    plt.savefig(web_path + f"TAS_quantiles_at_{region_abr}.pdf", 
                edgecolor = "w", bbox_inches = "tight")
    plt.savefig(web_path + f"TAS_quantiles_at_{region_abr}.png", 
                edgecolor = "w", bbox_inches = "tight")


# Senisitivity test: First order of TAS on NBP (regional) Similar to Pan 2020
# ===========================================================================
Sensitivity_test = 'n'
if Sensitivity_test in ['y','Y','yes']:
    file_nbp = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_gC.nc"%(
                source_run,member_run, variable,member_run,variable)
    nc_nbp = xr.open_dataset(file_nbp)
    nbp = nc_nbp.nbp  # nbp
    lf = nc_nbp.sftlf # land fraction

    #mask of the region
    region_abr = srex_abr[5] # for NAS
    filter_region 	= np.array(srex_abr) == region_abr # for NAS
    region_number	= np.array(srex_nums)[filter_region][0]
    region_name		= np.array(srex_names)[filter_region][0]
    region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region
    region_mask		= ~region_mask_not   # Only the regions is masked, True: region

    # area of the regions
    area_region= np.ma.masked_array(area, mask=region_mask_not)

    # mean area weighted average of TAS,NBP for NAS
    nas_awm_tas = []
    nas_awm_nbp = []
    nas_90q_tas = []
    nas_10q_tas = []
    for idx in range(tas.shape[0]):
        tas_awm_region = np.ma.average(tas[idx].data,  weights=area_region) - 273.15
        nbp_awm_region = np.ma.average(np.ma.masked_invalid(nbp[idx].data),  weights=area_region * lf)
        nas_90q_region = np.quantile(tas[idx].data,.9) - 273.15
        nas_10q_region = np.quantile(tas[idx].data,.1) - 273.15
        nas_awm_tas.append(tas_awm_region)
        nas_awm_nbp.append(nbp_awm_region)
        nas_90q_tas.append(nas_90q_region)
        nas_10q_tas.append(nas_10q_region)
    del tas_awm_region, nbp_awm_region,nas_90q_region
    #    print (f"for window {win_yr[w_idx]}, AWM: {tas_awm_region:.2f} Celsius")

    # rolling mean of a month of NBP (10 years)
    #pd_con_var_global_yr_tot_awm    = pd.Series (con_var_global_yr_tot_awm)
    #con_var_global_rm5yr_tot_awm    = pd_con_var_global_yr_tot_awm.rolling(window=5,center = False).mean()# 5 year rolling mean

    pd_nas_tas_awm = pd.Series (nas_awm_tas)
    pd_nas_tas_awm_rm_10yrs = pd_nas_tas_awm.rolling(window=10*12,center = True).mean()
    pd_nas_nbp_awm = pd.Series (nas_awm_nbp)
    pd_nas_nbp_awm_rm_10yrs = pd_nas_nbp_awm.rolling(window=10*12,center = True).mean()
    pd_nas_tas_90q = pd.Series (nas_90q_tas)
    pd_nas_tas_90q_rm_10yrs = pd_nas_tas_90q.rolling(window=10*12,center = True).mean()
    pd_nas_tas_10q = pd.Series (nas_10q_tas)
    pd_nas_tas_10q_rm_10yrs = pd_nas_tas_10q.rolling(window=10*12,center = True).mean()

    # Detrended anomalies of TAS and NBP ignoring the first 5 and last 5 years
    nas_awm_tas_detrend = nas_awm_tas [60:-59] - pd_nas_tas_awm_rm_10yrs[60:-59]
    nas_awm_nbp_detrend = nas_awm_nbp [60:-59] - pd_nas_nbp_awm_rm_10yrs[60:-59]
    nas_90q_tas_detrend = nas_90q_tas [60:-59] - pd_nas_tas_90q_rm_10yrs[60:-59]
    nas_10q_tas_detrend = nas_10q_tas [60:-59] - pd_nas_tas_10q_rm_10yrs[60:-59]


    # Sensitivity of the variable_ detrended
    step_size = 120 # 10 years
    Sensitivity_detrend = [region_abr]
    for i in range (int(nas_awm_nbp_detrend.size/step_size)):
        frame = { 'nbp': nas_awm_nbp_detrend[i*step_size:(i+1)*step_size], 'tas': nas_awm_tas_detrend[i*step_size:(i+1)*step_size] }  
        df_nbp_tas_detrend = pd.DataFrame(frame)

        lm = smf.ols(formula='nbp ~ tas', data=df_nbp_tas_detrend).fit()
        #print(lm.params)
        Sensitivity_detrend.append(np.format_float_scientific(lm.params[-1], exp_digits=2))

    print ("Sensitivity Test Result: ",Sensitivity_detrend )

    # Sensitivity at a pixel
    # ----------------------
    nas_awm_tas = []
    nas_awm_nbp = []
    nas_90q_tas = []

    lat_idx = 150
    lon_idx = 91
    tas_px = tas.isel(lat=lat_idx,lon=lon_idx).data - 273.15
    nbp_px = nbp.isel(lat=lat_idx,lon=lon_idx).data

    pd_nas_tas_px = pd.Series(tas_px)
    pd_nas_tas_px_rm_10yrs = pd_nas_tas_px.rolling(window=10*12,center = True).mean()
    pd_nas_nbp_px = pd.Series(nbp_px)
    pd_nas_nbp_px_rm_10yrs = pd_nas_nbp_px.rolling(window=10*12,center = True).mean()

    # Detrended anomalies of TAS and NBP ignoring the first 5 and last 5 years
    nas_tas_px_detrend = tas_px [60:-59] - pd_nas_tas_px_rm_10yrs[60:-59]
    nas_nbp_px_detrend = nbp_px [60:-59] - pd_nas_nbp_px_rm_10yrs[60:-59]

    # Sensitivity of the variable_ detrended
    step_size = 60 # 10 years
    Sensitivity_detrend_px = [f"{lat_idx}_{lon_idx}"]
    for i in range (int(nas_nbp_px_detrend.size/step_size)):
        frame = { 'nbp': nas_nbp_px_detrend[i*step_size:(i+1)*step_size], 'tas': nas_tas_px_detrend[i*step_size:(i+1)*step_size] }  
        df_nbp_tas_detrend = pd.DataFrame(frame)

        lm = smf.ols(formula='nbp ~ tas', data=df_nbp_tas_detrend).fit()
        #print(lm.params)
        Sensitivity_detrend_px.append(np.format_float_scientific(lm.params[-1], exp_digits=2))

    print ("Sensitivity Test Result: ",Sensitivity_detrend_px )

Concurrency_test = "no"
if Concurrency_test in ['y','Y','yes']:
    # To find the concurrency of the higher quantile Tas and NBP (lower quantile) at a pixel
    # ======================================================================================
    w_idx = 7
    lat_idx = 150
    lon_idx = 91
    per_tas = .9 #percentile of tas
    per_nbp = .1 # percentile of nbp
    lag = 3     # lag months for lag>1

    tas_px = tas.isel(lat=lat_idx,lon=lon_idx).data - 273.15
    nbp_px = nbp.isel(lat=lat_idx,lon=lon_idx).data

    pd_nas_tas_px = pd.Series(tas_px)
    pd_nas_tas_px_rm_10yrs = pd_nas_tas_px.rolling(window=10*12,center = True).mean()
    pd_nas_nbp_px = pd.Series(nbp_px)
    pd_nas_nbp_px_rm_10yrs = pd_nas_nbp_px.rolling(window=10*12,center = True).mean()

    # Detrended anomalies of TAS and NBP ignoring the first 5 and last 5 years
    nas_tas_px_detrend = tas_px [60:-59] - pd_nas_tas_px_rm_10yrs[60:-59]
    nas_nbp_px_detrend = nbp_px [60:-59] - pd_nas_nbp_px_rm_10yrs[60:-59]

    tas_win_px = tas_px[w_idx*300:(w_idx+1)*300]
    nbp_win_px = nbp_px[w_idx*300:(w_idx+1)*300]

    nas_tas_win_px_detrend = nas_tas_px_detrend[w_idx*300:(w_idx+1)*300]
    nas_nbp_win_px_detrend = nas_nbp_px_detrend[w_idx*300:(w_idx+1)*300]

    print ("What is the concurrency of higher quantile of TAS and lower quantile of NBP?")
    print ("or checking the concurrency of hot temperatures and negative NBPs")
    print ("-----------------")
    print (f"Window: {win_yr[w_idx]}")
    print(f"At location Lat: {round(nbp.lat.data[lat_idx],2)}, Lon: {round(nbp.lon.data[lon_idx],2)}")
    print(f"Lag: {lag} months")
    print (f"Quantile of TAS: {per_tas}")
    print (f"Quantile of NBP: {per_nbp}")
    print (f"Total Pixels : {round(tas_win_px.size*(1-per_tas))}")
    print (f"Concurrency : {((tas_win_px>np.quantile(tas_win_px,per_tas))[:-lag] * (nbp_win_px<np.quantile(nbp_win_px,per_nbp))[lag:]).sum()}")
    print ("Hot Months :", np.array(['1:Jan','2:Feb','3:Mar','4:Apr','5:May','6:Jun','7:July','8:Aug','9:Sep','10:Oct','11:Nov','12:Dec']*25)[
                            (tas_win_px>np.quantile(tas_win_px,per_tas))])    


# Sensitivity Analysis
# Senisitivity test: First order of TAS on NBP (regional) Similar to Pan 2020
Sensivity_test_reg = 'no'
if Sensivity_test_reg in ['yes', 'y', 'Y']:
    file_nbp = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_gC.nc"%(
                    source_run,member_run, variable,member_run,variable)

    nc_nbp = xr.open_dataset(file_nbp)
    nbp = nc_nbp.nbp  # nbp
    lf = nc_nbp.sftlf # land fraction

    #mask of the region
    region_abr = srex_abr[5] # for NAS
    filter_region 	= np.array(srex_abr) == region_abr # for NAS
    region_number	= np.array(srex_nums)[filter_region][0]
    region_name		= np.array(srex_names)[filter_region][0]
    region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region
    region_mask		= ~region_mask_not   # Only the regions is masked, True: region

    # area of the regions
    area_region= np.ma.masked_array(area, mask=region_mask_not)

    # mean area weighted average of TAS,NBP for NAS
    nas_awm_tas = []
    nas_awm_nbp = []
    nas_90q_tas = []
    nas_10q_tas = []
    for idx in range(tas.shape[0]):
        tas_awm_region = np.ma.average(tas[idx].data,  weights=area_region) - 273.15
        nbp_awm_region = np.ma.average(np.ma.masked_invalid(nbp[idx].data),  weights=area_region * lf)
        nas_90q_region = np.quantile(tas[idx].data,.9) - 273.15
        nas_10q_region = np.quantile(tas[idx].data,.1) - 273.15
        nas_awm_tas.append(tas_awm_region)
        nas_awm_nbp.append(nbp_awm_region)
        nas_90q_tas.append(nas_90q_region)
        nas_10q_tas.append(nas_10q_region)
    del tas_awm_region, nbp_awm_region,nas_90q_region
    #    print (f"for window {win_yr[w_idx]}, AWM: {tas_awm_region:.2f} Celsius")

    # rolling mean of a month of NBP (10 years)
    #pd_con_var_global_yr_tot_awm    = pd.Series (con_var_global_yr_tot_awm)
    #con_var_global_rm5yr_tot_awm    = pd_con_var_global_yr_tot_awm.rolling(window=5,center = False).mean()# 5 year rolling mean

    pd_nas_tas_awm = pd.Series (nas_awm_tas)
    pd_nas_tas_awm_rm_10yrs = pd_nas_tas_awm.rolling(window=10*12,center = True).mean()
    pd_nas_nbp_awm = pd.Series (nas_awm_nbp)
    pd_nas_nbp_awm_rm_10yrs = pd_nas_nbp_awm.rolling(window=10*12,center = True).mean()
    pd_nas_tas_90q = pd.Series (nas_90q_tas)
    pd_nas_tas_90q_rm_10yrs = pd_nas_tas_90q.rolling(window=10*12,center = True).mean()
    pd_nas_tas_10q = pd.Series (nas_10q_tas)
    pd_nas_tas_10q_rm_10yrs = pd_nas_tas_10q.rolling(window=10*12,center = True).mean()

    # Detrended anomalies of TAS and NBP ignoring the first 5 and last 5 years
    nas_awm_tas_detrend = nas_awm_tas [60:-59] - pd_nas_tas_awm_rm_10yrs[60:-59]
    nas_awm_nbp_detrend = nas_awm_nbp [60:-59] - pd_nas_nbp_awm_rm_10yrs[60:-59]
    nas_90q_tas_detrend = nas_90q_tas [60:-59] - pd_nas_tas_90q_rm_10yrs[60:-59]
    nas_10q_tas_detrend = nas_10q_tas [60:-59] - pd_nas_tas_10q_rm_10yrs[60:-59]

    # Sensitivity of the variable
    import statsmodels.formula.api as smf
    step_size = 120 # 10 years
    Sensitivity = [region_abr]
    for i in range (int(nas_awm_nbp_detrend.size/step_size)):
        frame = { 'nbp': nas_awm_nbp[i*step_size:(i+1)*step_size], 'tas': nas_awm_tas[i*step_size:(i+1)*step_size] }  
        df_nbp_tas_detrend = pd.DataFrame(frame)

        lm = smf.ols(formula='nbp ~ tas', data=df_nbp_tas_detrend).fit()
        #print(lm.params)
        Sensitivity.append(np.format_float_scientific(lm.params[-1], exp_digits=2))

    print (Sensitivity)


    # Sensitivity of the variable_ detrended
    step_size = 120 # 10 years
    Sensitivity_detrend = [region_abr]
    for i in range (int(nas_awm_nbp_detrend.size/step_size)):
        frame = { 'nbp': nas_awm_nbp_detrend[i*step_size:(i+1)*step_size], 'tas': nas_awm_tas_detrend[i*step_size:(i+1)*step_size] }  
        df_nbp_tas_detrend = pd.DataFrame(frame)

        lm = smf.ols(formula='nbp ~ tas', data=df_nbp_tas_detrend).fit()
        #print(lm.params)
        Sensitivity_detrend.append(np.format_float_scientific(lm.params[-1], exp_digits=2))

    print (Sensitivity_detrend)

    # Sensitivity at a pixel
    nas_awm_tas = []
    nas_awm_nbp = []
    nas_90q_tas = []

    lat_idx = 150
    lon_idx = 91
    tas_px = tas.isel(lat=lat_idx,lon=lon_idx).data - 273.15
    nbp_px = nbp.isel(lat=lat_idx,lon=lon_idx).data

    pd_nas_tas_px = pd.Series(tas_px)
    pd_nas_tas_px_rm_10yrs = pd_nas_tas_px.rolling(window=10*12,center = True).mean()
    pd_nas_nbp_px = pd.Series(nbp_px)
    pd_nas_nbp_px_rm_10yrs = pd_nas_nbp_px.rolling(window=10*12,center = True).mean()

    # Detrended anomalies of TAS and NBP ignoring the first 5 and last 5 years
    nas_tas_px_detrend = tas_px [60:-59] - pd_nas_tas_px_rm_10yrs[60:-59]
    nas_nbp_px_detrend = nbp_px [60:-59] - pd_nas_nbp_px_rm_10yrs[60:-59]

    # Sensitivity of the variable_ detrended
    step_size = 60 # 10 years
    Sensitivity_detrend_px = [f"{lat_idx}_{lon_idx}"]
    for i in range (int(nas_nbp_px_detrend.size/step_size)):
        frame = { 'nbp': nas_nbp_px_detrend[i*step_size:(i+1)*step_size], 'tas': nas_tas_px_detrend[i*step_size:(i+1)*step_size] }  
        df_nbp_tas_detrend = pd.DataFrame(frame)

        lm = smf.ols(formula='nbp ~ tas', data=df_nbp_tas_detrend).fit()
        #print(lm.params)
        Sensitivity_detrend_px.append(np.format_float_scientific(lm.params[-1], exp_digits=2))

    print (Sensitivity_detrend_px)

# Sensitivity of latitudes
# working (...)
Sensivity_test_lat = 'no'
if Sensivity_test_lat in ['yes', 'y', 'Y']:
    file_nbp = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_gC.nc"%(
                    source_run,member_run, variable,member_run,variable)

    nc_nbp = xr.open_dataset(file_nbp)
    nbp = nc_nbp.nbp  # nbp
    lf = nc_nbp.sftlf # land fraction
    area = nc_nbp.areacella # area

    # Creating a dictionary of Sensitivity based on lat index
    Sensitivity_lat_band = {}
    for lat_idx in range(lat.size):
        Sensitivity_lat_band[lat_idx] = {}
        
    detrended_nbp_lat = []
    detrended_nbp_tas = [] 
    for lat_idx in range(lat.size):
        print(f"Lat Idx : {lat_idx}")
        # Check for all zeros:
        if np.all((nbp[:,lat_idx,:].values==0)):
            detrended_nbp_lat.append(nbp_lat_detrend)
            detrended_nbp_tas.append(tas_lat_detrend)
            continue
        # Average over all lons
        nbp_lat_tmp = pd.DataFrame(nbp[:,lat_idx,:].values).replace(0,np.NaN).mean(axis=1)
        pd_nbp_lat_rm_10yrs = nbp_lat_tmp.rolling(window=10*12,center = True).mean()
        # Detrended anomalies of TAS and NBP ignoring the first 5 and last 5 years
        nbp_lat_detrend = nbp_lat_tmp [60:-59] - pd_nbp_lat_rm_10yrs[60:-59]
        
        tas_lat_tmp = pd.DataFrame(tas[:,lat_idx,:].values).replace(0,np.NaN).mean(axis=1)
        pd_tas_lat_rm_10yrs = tas_lat_tmp.rolling(window=10*12,center = True).mean()
        tas_lat_detrend = tas_lat_tmp [60:-59] - pd_tas_lat_rm_10yrs[60:-59]
        
        print(f"Lat Idx : {lat_idx}")
        detrended_nbp_lat.append(nbp_lat_detrend)
        detrended_nbp_tas.append(tas_lat_detrend)
        
        
        