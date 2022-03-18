# Bharat Sharma
# python 3.7
# The aim is to create a stacked graphs of gpp, gpp anomalies and climate drivers,
# to identify the drivers that lead to an carbon cycle extreme event at that location/region
# The  TCE events are also shaded and  we could focus on a particular TCE and see the detais 
# Highlights negative and positive extremes at the same location

import numpy as np
import pandas as pd
import  matplotlib as mpl
#mpl.use('Agg')
from functions import time_dim_dates, index_and_dates_slicing, geo_idx,norm
import datetime as dt
import netCDF4 as nc4
import  matplotlib.pyplot as plt
import argparse
#from    mpl_toolkits.basemap import Basemap
from    matplotlib import cm

#1- Hack to fix missing PROJ4 env var for Basemaps Error                                          
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
from    mpl_toolkits.basemap import Basemap
#-1 Hack end

parser  = argparse.ArgumentParser()
parser.add_argument ('--lat_idx'    ,'-lat_idx' , help = ' Lat coordinate of the location'                  , type= int	, default = 105 )
parser.add_argument ('--lon_idx'    ,'-lon_idx' , help = ' Lon coordinate of the location'                  , type= int , default = 29  )
#parser.add_argument('--variables',   '-vars'  , help = "Type all the variables you want to plot( separated by '-' minus sign" , type= str, default= "prcp-sm-tmax-col_fire_closs")
parser.add_argument('--plot_win'    , '-pwin'       , help = "which time period to plot? 2000-24:win 06"	, type= int	, default=  6   )
#parser.add_argument('--plot_ano'    , '-pano'       , help = "Do you want to plot the anomalies also"    	, type= str	, default=  'n' )
parser.add_argument('--source'      ,   '-src'      , help = "Model (Source_Run)"                   		, type= str , default= 'CESM2'  ) # Model Name
parser.add_argument('--member_idx'  ,   '-m_idx'    , help = "Member Index"                         		, type= int , default= 0    ) # Index of the member
parser.add_argument('--variable'    ,   '-var'      , help = "Anomalies of carbon cycle variable"   , type= str     , default= 'gpp'    )
args = parser.parse_args()

# running the code:
# run plot_shaded_extremes_drivers.py  -pwin 6 -src CESM2 -lat_idx 102 -lon_idx 26

# Inputs
# ------
source_run      = args.source       
member_idx      = args.member_idx
pwin    		= args.plot_win
lat_in  		= args.lat_idx
lon_in  		= args.lon_idx
variable        = args.variable
#plt_ano 		= args.plot_ano
 
# List of the drivers that will be considered and their names for Plotting
# -----------                                                                                                
if source_run == 'CESM2':
    driver_consider = 4
    drivers         = np.array(['pr','mrso','tas','fFireAll']) [:driver_consider]
    drivers_names   = np.array(['Prcp','Soil Moisture', 'TAS','Fire']) [:driver_consider]
else:
    driver_consider = 3
    drivers         = np.array(['pr','mrso','tas']) [:driver_consider]
    drivers_names   = np.array(['Prcp','Soil Moisture', 'TAS']) [:driver_consider]

# Paths for reading the main files
# --------------------------------
cori_scratch    = '/global/cscratch1/sd/bharat/'
members_list    = os.listdir(cori_scratch+"add_cmip6_data/%s/ssp585/"%source_run)
member_run      = members_list[member_idx]


# Storing the file name and abr of the drivers to be considered
# -------------------------------------------------------------
features                = {}
features['abr']         = drivers
features['filenames_ano']   = {}

# The name with which the variables are stored in the nc files:
features['Names']               = {}
features['Names']['pr']         = 'pr'
features['Names']['mrso']       = 'mrso'
features['Names']['tas']        = 'tas'
if source_run == 'CESM2':
    features['Names']['fFireAll']   = 'Fire'
#features['Names']['tasmax']        = 'tasmax'

features['filenames_ano'][variable] = {}    # Creating a empty directory for storing multi members if needed
features['filenames_ano'][variable][member_run] = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_anomalies_gC.nc"%(source_run,member_run, variable,member_run,variable)

for dri in drivers:
	features['filenames_ano'][dri] = {}
	features['filenames_ano'][dri][member_run] = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/CESM2_ssp585_%s_%s_anomalies.nc"%(source_run,member_run, dri ,member_run,dri)


# Reading the variables from the variable (gpp) anomalies file
# ------------------------------------------------------------
nc_var	= nc4.Dataset(features['filenames_ano'][variable][member_run])
time    = nc_var .variables['time']
lat		= nc_var .variables['lat']
lon    	= nc_var .variables['lon']
var_ano = nc_var.variables[variable] # anomalies of the variable

win_len     = 25*12

dates_ar    = time_dim_dates(base_date=dt.date(1850,1,1), total_timestamps=time.size)                                             
start_dates = [dates_ar[i*win_len] for i in range(int(time.size/win_len))] #list of the start dates of all 25 year windows
end_dates   = [dates_ar[i*win_len+win_len-1] for i in range(int(time.size/win_len))] #list of the end dates of all 25 year windows

# The start years of the time periods:
# ------------------------------------
win_start_years = np.arange(1850,2100,25)

# Reading the Binary of 01s and 1s at a pixel for extreme events for selected time window:
# ----------------------------------------------------------------------------------------
file_bin_tce_01s_neg  = cori_scratch + 'add_cmip6_data/%s/ssp585/%s/gpp_TCE/bin_TCE_01s_neg_%s.nc'%(source_run, member_run, win_start_years [pwin])
file_bin_tce_01s_pos  = cori_scratch + 'add_cmip6_data/%s/ssp585/%s/gpp_TCE/bin_TCE_01s_pos_%s.nc'%(source_run, member_run, win_start_years [pwin])

px_bin_tce_01s_neg  = nc4.Dataset(file_bin_tce_01s_neg).variables['gpp_TCE_01s'] [0,:,lat_in,lon_in]
px_bin_tce_01s_pos  = nc4.Dataset(file_bin_tce_01s_pos).variables['gpp_TCE_01s'] [0,:,lat_in,lon_in]

file_bin_tce_1s_neg  = cori_scratch + 'add_cmip6_data/%s/ssp585/%s/gpp_TCE/bin_TCE_1s_neg_%s.nc'%(source_run, member_run, win_start_years [pwin])
file_bin_tce_1s_pos  = cori_scratch + 'add_cmip6_data/%s/ssp585/%s/gpp_TCE/bin_TCE_1s_pos_%s.nc'%(source_run, member_run, win_start_years [pwin])

px_bin_tce_1s_neg  = nc4.Dataset(file_bin_tce_1s_neg).variables['gpp_TCE_1s'] [0,:,lat_in,lon_in]
px_bin_tce_1s_pos  = nc4.Dataset(file_bin_tce_1s_pos).variables['gpp_TCE_1s'] [0,:,lat_in,lon_in]

# Carbon loss gain due to TCEs [1s]
carbon_loss_1s  = (px_bin_tce_1s_neg * var_ano [pwin*300:(pwin+1)*300,lat_in,lon_in]).sum()
carbon_gain_1s  = (px_bin_tce_1s_pos * var_ano [pwin*300:(pwin+1)*300,lat_in,lon_in]).sum()

#Total Carbon lost or gained
filter_tot_car_loss = var_ano [pwin*300:(pwin+1)*300,lat_in,lon_in] <0
tot_carbon_loss		= var_ano [pwin*300:(pwin+1)*300,lat_in,lon_in] [filter_tot_car_loss].sum()

filter_tot_car_gain = var_ano [pwin*300:(pwin+1)*300,lat_in,lon_in] >0
tot_carbon_gain		= var_ano [pwin*300:(pwin+1)*300,lat_in,lon_in] [filter_tot_car_gain].sum()



# Basic information of the location:
# ----------------------------------
#print ("The area of pixel: \t %f sq.km\n"%nc4.Dataset('/home/ud4/CESM1-BGC/gpp/without_land_use_change/cesm1bgc_pftcon_gpp_gC.nc')['area'][lat_in,lon_in])
print ("The Carbon loss during TCE/ total are : \t%.2f/%.2f TgC"%(carbon_loss_1s/10**12, tot_carbon_loss/10**12))
print ("The Carbon gain during TCE/ total are : \t%.2f/%.2f TgC"%(carbon_gain_1s/10**12, tot_carbon_gain/10**12))
print ("The lat coordinates is: \t%f deg North"%(lat[lat_in]))
print ("The lon coordinates is: \t%f deg East\n"%(lon[lon_in]))

# Plotting the shaded TS
# ----------------------
from    scipy   import  ndimage
larray_neg,narray_neg   = ndimage.label(px_bin_tce_1s_neg,structure = np.ones(3))
locations_neg       = ndimage.find_objects(larray_neg)
larray_pos,narray_pos   = ndimage.label(px_bin_tce_1s_pos,structure = np.ones(3))
locations_pos       = ndimage.find_objects(larray_pos)

neg_ext_locs = []
for loc in locations_neg:
	start   = dates_ar[pwin*win_len:(pwin+1)*win_len][loc[0]][0]
	end     = dates_ar[pwin*win_len:(pwin+1)*win_len][loc[0]][-1]+dt.timedelta(1)
	neg_ext_locs.append((start,end))

pos_ext_locs = []
for loc in locations_pos:
	start   = dates_ar[pwin*win_len:(pwin+1)*win_len][loc[0]][0]
	end     = dates_ar[pwin*win_len:(pwin+1)*win_len][loc[0]][-1]+dt.timedelta(1)
	pos_ext_locs.append((start,end))

path_save = "/global/cscratch1/sd/bharat/add_cmip6_data/CESM2/ssp585/r1i1p1f1/gpp/Correlations/compound_drivers/win_06/"

plt_ano = 'y'
# Plot the timeseries of the Anomalies of the Climate Drivers:
# ------------------------------------------------------------
if plt_ano in ['y','Y','yes']:
	fig_ano,ax  = plt.subplots(nrows=(len(drivers)+1),ncols= 1,gridspec_kw = {'wspace':0, 'hspace':0.05},tight_layout = True, figsize = (10,7), dpi = 400)
	fig_ano.suptitle ('Normalized Anomalies of the Climate Drivers: %s'%source_run)
	ax      = ax.ravel()
	norm_dri = {}
	for idx, dri in enumerate (drivers):
		norm_dri[dri] = norm (nc4.Dataset(features['filenames_ano'][dri][member_run]).variables[dri][pwin*win_len:(pwin+1)*win_len, lat_in, lon_in])
		ax[idx] .plot(dates_ar[pwin*win_len:(pwin+1)*win_len], norm_dri [dri], label = features['Names'][dri])
		print (dri)
		ax[idx].set_xticklabels([])
		ax[idx].tick_params(axis="x",direction="in")
		ax[idx].set_ylabel(" %s"%(features['Names'][dri]))
		for (start,end) in neg_ext_locs:
			ax[idx].axvspan(start,end,alpha = .3, color = 'red')
		for (start_alt,end_alt) in pos_ext_locs:
			ax[idx].axvspan(start_alt,end_alt,alpha = .3, color = 'green')

	# Plotting Variable -ORI
	#ax[-2].plot(dates_ar[pwin*win_len:(pwin+1)*win_len], norm(nc_gpp[pwin*win_len:(pwin+1)*win_len, lat_in, lon_in]), label = 'gpp')
	#ax[-2].set_ylabel("GPP")
	#for (start,end) in ext_locs:
		#ax[-2].axvspan(start,end,alpha = .3, color = 'red')
	#for (start_alt,end_alt) in ext_locs_alt:
		#ax[-2].axvspan(start_alt,end_alt,alpha = .3, color = 'green')

	#ax[-2].set_xticklabels([])
	#ax[-2].tick_params(axis="x",direction="in")

	# Plotting Variable -Anomalies
	ax[-1].plot(dates_ar[pwin*win_len:(pwin+1)*win_len], norm(var_ano[pwin*win_len:(pwin+1)*win_len, lat_in, lon_in]), label = variable)
	ax[-1].set_xlabel('Time',fontsize =14)
	ax[-1].set_ylabel("%s Anomalies"%(variable.upper()))
	for (start,end) in neg_ext_locs:
		ax[-1].axvspan(start,end,alpha = .3, color = 'red')
	for (start_alt,end_alt) in pos_ext_locs:
		ax[-1].axvspan(start_alt,end_alt,alpha = .3, color = 'green')
	#ax[0].legend(loc = 'upper center',ncol=len(variables_list)+1, bbox_to_anchor=(.1,1.15),frameon =False,fontsize=9,handletextpad=0.1)
	i = 0
	for (start,end) in neg_ext_locs:
		a,b = start,end
		c	= (b-a)/2
		x	= a+c
		y 	= norm_dri[drivers[0]].max()
		i	= i+1
		ax[0].text(x,y*1.1,'N%d'%i,horizontalalignment='center')
	del (i)
	i = 0
	for (start,end) in pos_ext_locs:
		a,b = start,end
		c	= (b-a)/2
		x	= a+c
		y 	= norm_dri[drivers[0]].max()
		i 	= i+1
		ax[0].text(x,y*1.1,'P%d'%i,horizontalalignment='center')
	del(i)
	fig_ano.savefig(cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Correlations/compound_drivers/win_%s/shaded_TCE_anomalies_%d_%d.pdf" %(source_run, member_run,variable,format(pwin,'02'),lat_in,lon_in))

loc_on_map = np.ma.masked_all((lat.size,lon.size))
loc_on_map[int(lat_in),int(lon_in)] = 1
lat_deg = lat[int(lat_in)]
lon_deg = lon[int(lon_in)]

# Spatial Map of the location of the selected Pixel
# -------------------------------------------------
fig_map,ax = plt.subplots(figsize = (5,2.8),tight_layout=True,dpi=500)
bmap    = Basemap(  projection  =   'eck4',
                    lon_0       =   0.,                                                                                                      
                    resolution  =   'c')
x,y     = bmap(lon_deg,lat_deg)
bmap    . plot(x,y,'b*',markersize=6)
LAT,LON = np.meshgrid(lat[...], lon[...],indexing ='ij')
ax      = bmap.pcolormesh(LON,LAT,np.ma.masked_invalid(loc_on_map),latlon=True,cmap= cm.afmhot_r)#,vmax= ymax, vmin= ymin)
bmap    .drawparallels(np.arange(-90., 90., 30.),fontsize=14, linewidth = .2)
bmap    .drawmeridians(np.arange(0., 360., 60.),fontsize=14, linewidth = .2)
bmap    .drawcoastlines(linewidth = .2)
plt.title("2000-24; The location!")
fig_map.savefig(cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Correlations/compound_drivers/win_%s/shaded_TCE_location_%d_%d.pdf" %(source_run, member_run,variable,format(pwin,'02'),lat_in,lon_in))
plt.close(fig_map)

# Spatial Map of the pixel using cartopy
# --------------------------------------
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# test 2-d data
data = var_ano[0,:,:]
mid_lon	= int( lon[int(lon[...].size/2)] ) 
proj	= ccrs.PlateCarree(central_longitude= mid_lon)
fig, ax = plt.subplots( figsize=(5,2.8), 
						subplot_kw = proj)


# The SREX Regions bounds
# ------------------------
file_srex_bounds = '/global/cscratch1/sd/bharat/add_cmip6_data/Observed/SREX_referenceRegions.xls'

df_srex_bounds = pd.read_excel(file_srex_bounds)
srex_idx = 20 # CAS
filter_srex_idx = df_srex_bounds.loc[:,'SRES index'] == np.float(srex_idx)
Code_srex		= df_srex_bounds.loc[:,"Code"][filter_srex_idx]

# Need to parse the lat and lon points and find all lat-lon pairs that fall inside the region
