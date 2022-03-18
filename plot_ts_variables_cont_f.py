"""
Bharat Sharma
python 3.7
:> to plot the variables that are selected
check/run following before running this file:
	- filepaths_to_use.sh (returns the paths of variables that are avaiable)
	- data_scope.py		  (creates a dataframe with hierarchy of data structure)
	- data_processing/df_data_selected.csv	(.csv output of data_scope.py)
The ploting of SSP585 and Historical will be done based on the raw files stored in the repository
"""

import	pandas	as	pd
import	numpy 	as 	np
import	matplotlib.pyplot	as plt
import 	netCDF4	as 	nc4
import	re
import	seaborn	as	sns
import 	cftime
import 	os
from 	functions	import Unit_Conversions, time_dim_dates,index_and_dates_slicing
import 	datetime as dt

#1- Hack to fix missing PROJ4 env var
import os
#import conda
"""
conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
#-1 Hack end 


from    mpl_toolkits.basemap    import Basemap
"""
# Reading the dataframe of the selected files
# -------------------------------------------
#web_path	= '/project/projectdirs/m2467/www/bharat/'
web_path    	= '/global/homes/b/bharat/results/web/'
in_path		= '/global/homes/b/bharat/results/data_processing/'
cmip6_filepath_head = '/global/cfs/cdirs/m3522/cmip6/CMIP6/'
#cmip6_filepath_head = '/global/homes/b/bharat/cmip6_data/CMIP6/'
df_files	=	pd.read_csv(in_path + 'df_data_selected.csv')
cori_scratch    = '/global/cscratch1/sd/bharat/'    # where the anomalies per slave rank are saved   

hierarcy_str= [	'activity_id','institution_id','source_id','experiment_id',
				'member_id','table_id','variable_id','grid_label','version','filenames']

# Source_id or Model Names:
# -------------------------
source_ids		= np.unique(df_files['source_id'])

# Experiments
# -----------
experiments_ids	= np.unique(df_files['experiment_id'])

# Variable names:
# --------------
variable_ids	= np.unique(df_files['variable_id'])


#Create a DataFrame to store the essential informtation of the model and variables
col_names 	= [ 'source_id','experiment_id','member_id','variable_id',
				'grid_label','version','time_units','time_calendar',
				'lat','lon','var_units','area_units']
df_model_info 	= pd.DataFrame(columns = col_names)
#def create_df_info (s = source_run,e = exp, m=member, v=variable_run, g=grid_run, ver = ver_run, tu=time.units, tc=time.calendar,lt = lat.size, ln = lon.size, vu=var.units, au =area.units, ignore_index = True):
def create_df_info (i=np.nan, s = np.nan,e = np.nan, m=np.nan, v=np.nan, 
					g=np.nan, ver_h = np.nan, ver_s = np.nan, tu=np.nan, 
					tc=np.nan,lt = np.nan ,ln = np.nan, vu=np.nan, au =np.nan):
	d = {'source_id'	: pd.Series([s],index=[i]),
		 'experiment_id': pd.Series([e],index=[i]),
		 'member_id'	: pd.Series([m],index=[i]),
		 'variable_id'	: pd.Series([v],index = [i]),
		 'grid_label'	: pd.Series([g],index = [i]), 
		 'version_historial': pd.Series([ver_h],index = [i]),
		 'version_ssp587': pd.Series([ver_s],index = [i]), 
		 'time_units'	: pd.Series([tu],index = [i]), 
		 'time_calendar': pd.Series ([tc],index = [i]), 
		 'lat'			: pd.Series([lt], index = [i]), 
		 'lon'			: pd.Series([ln],index= [i]),
		 'var_units'	: pd.Series([vu],index = [i]),
		 'area_units'	: pd.Series([au],index = [i])}
	df = pd.DataFrame(d)
	return df
# -----------------------
#creating a copy of the df_files
temp			= df_files.copy(deep = True)

#creating the filters based on mpodel and variable
# model for this run
source_run		= 'CanESM5'
variable_run	= 'ra'

# The Models that have gpp for historical and ssp585 experiments:
source_selected = ['CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1']
source_selected = ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1','EC-Earth3-Veg','UKESM1-0-LL']
source_selected = ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1'] # for GPP // No areacella in : 'EC-Earth3-Veg','UKESM1-0-LL'
#source_selected = ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1'] # for NBP // no NBP in BCC //No areacella in : 'EC-Earth3-Veg','UKESM1-0-LL'
# Select which model you want to run:
# ===================================
#source_selected = ['CNRM-CM6-1'     ]
#source_selected = ['BCC-CSM2-MR'    ]
#source_selected = ['CNRM-ESM2-1'    ]
#source_selected = ['IPSL-CM6A-LR'   ]
#source_selected    = ['CanESM5'        ]
source_selected    = ['CESM2'          ]

# The abriviation of the models that will be analyzed:
source_code	= { 'cesm'	: 'CESM2',
				'can'	: 'CanESM5',
				'ipsl'	: 'IPSL-CM6A-LR',
				'bcc'	: 'BCC-CSM2-MR',
				'cnrn-e': 'CNRM-ESM2-1',
				'cnrn-c': 'CNRM-CM6-1' }

import argparse
parser  = argparse.ArgumentParser()
parser.add_argument('--sources'			,'-src'			, help = "Which model(s) to analyse?"		, type= str,	default= 'all'		)
parser.add_argument('--variable'        ,'-var'         , help = "variable? gpp/npp/nep/nbp,,,,"    , type= str,    default= 'gpp'      )
args = parser.parse_args()

# The inputs:
src			= str	(args.sources)
variable_run= str   (args.variable)

# Model(s) to analyze:
# --------------------
source_selected	= []
if len(src.split('-')) >1:
	source_selected = src.split('-') 
elif src in ['all', 'a']:
	source_selected = list(source_code.values() )
elif len(src.split('-')) == 1:
	if src in source_code.keys():
		source_selected = [source_code[src]]
	else:
		print (" Enter a valid source id")

#running :  run plot_ts_variables_cont_f.py -src cesm -var ra

# Creating a lis to Unique colors for multiple models:
# ---------------------------------------------------
NUM_COLORS = len(source_selected)
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)

sns.reset_orig()  # get default matplotlib styles back
clrs = sns.color_palette('husl', n_colors=NUM_COLORS)

# To Save the ts of every source in a dict for ploting
# -----------------------------------------------------
ts_yr_source_var_member				= {}
ts_rm5yr_source_var_member			= {}
ts_av_rm5yr_source_var_member		= {}
# Savings a dictionary of common members per source id
# ----------------------------------------------------
var_ar_gC	= {} # to store the variable array
for s_idx, source_run in enumerate(source_selected):
	var_ar_gC [source_run] = {}
	
member_ids_common_source			= {}

for s_idx, source_run in enumerate(source_selected):
	filters			= (temp['source_id'] == source_run) & (temp['variable_id'] == variable_run)
	filters_area	= (temp['source_id'] == source_run) & (temp['variable_id'] == 'areacella')
	filters_lf		= (temp['source_id'] == source_run) & (temp['variable_id'] == 'sftlf')
	ts_yr_source_var_member[source_run] 		= {}
	ts_rm5yr_source_var_member[source_run] 		= {}
	ts_av_rm5yr_source_var_member[source_run]	= {}
#passing the filters to the dataframe
	df_tmp			= temp[filters]
	df_tmp_area		= temp[filters_area]
	df_tmp_lf		= temp[filters_lf]

# grid of the filtered dataframe
	grid_run		= np.unique(df_tmp['grid_label'])[0]

#list of member ids
	filters_exp 		= {} #dic to store filters of experiments
	member_ids_exp		= {} #dic to store member ids of experiments
	filters_mem			= {} #dic to store member ids of experiments
#checking and using the latest version
	vers				= {}
	df_ver_tmp			= {}
	filters_ver			= {}

	ix = 1 #initiating the idx for the dataframe for info
	for exp in experiments_ids:
		vers [exp]			= {} # initiating a version dic per member
		df_ver_tmp [exp]	= {}
		filters_ver[exp]	= {}
		filters_mem[exp]	= (df_tmp['experiment_id'] == exp )
		member_ids_exp[exp] = np.unique(df_tmp[filters_mem[exp]]['member_id'])
	member_ids_common   	= np.intersect1d(member_ids_exp['historical'] , member_ids_exp['ssp585'])
	
# Checking for the latest version of the members
	for member in member_ids_common:
		for exp in experiments_ids:
			df_ver_tmp	[exp][member]	= df_tmp[filters_mem[exp]]
			filters_ver	[exp][member]	= (df_tmp['experiment_id'] == exp ) & (df_tmp['member_id'] == member )
			vers_tmp			= np.unique(df_tmp[filters_ver[exp][member]]['version']) # all versions in str
			vers_tmp_int    	= np.array([int(v[1:]) for v in vers_tmp])      # all versions in int
			tmp_idx         	= np.where(vers_tmp_int == np.max(vers_tmp_int))# index of max number version
			vers[exp][member] 	= vers_tmp[tmp_idx[0][0]]                       # newest for this run

# Saving the common members to a dict for plotting purposes
	if (source_run == 'CESM2') and (variable_run == 'tasmax'): member_ids_common = ['r1i1p1f1']
	member_ids_common_source[source_run] = member_ids_common
	for member in member_ids_common_source[source_run]:
		var_ar_gC [source_run][member] = {}	
#Check if the files are in chunks of time
	num_chunk_time	= {}
	for exp in experiments_ids:
		if (source_run == 'CESM2') and (variable_run == 'tasmax'):
			num_chunk_time[exp] = 1
		else:	
			num_chunk_time[exp]		= len(df_tmp[filters_ver[exp][member]][df_tmp['version'] == vers[exp][member]])  
			print ("Number of chunks of time in of model %s under experiment '%s' are: %d"%(source_run ,exp, num_chunk_time[exp]))

# Creating a dictionary for storing the nc data
	nc_data = {}

	for exp in experiments_ids:
		nc_data [exp] = {}
	
	filepath_areacella 	= {}
	filepath_sftlf		= {}	
	for member in member_ids_common:
# Pointing to the selective files that i need for this plot
		print ("Source ID :%s, Member ID :%s"%(source_run,member))
		filepaths_cont = {}

		for exp in experiments_ids:
			filepaths_cont[exp] = []
			member_id_tmp	= member
			nc_data [exp][member_id_tmp] = {}	# reading members separately per experiment

			print ("==============================================================")

# This is when the versions are saved over multiple time chunks
			if num_chunk_time[exp] >= 1:
				if (source_run == 'CESM2') and (variable_run == 'tasmax'):
					pass
				else:	
					filepath_ar = np.array(df_ver_tmp[exp][member][filters_ver[exp][member]][df_ver_tmp[exp][member]['version'] == vers[exp][member]]) 
				for chunk_idx in range(num_chunk_time[exp]):
					if (source_run == 'CESM2') and (variable_run == 'tasmax') and (exp == 'historical'):
						filepaths_cont[exp] = ["/global/cscratch1/sd/bharat/add_cmip6_data/CESM2/extra_cmip6_data/tasmax_Amon_CESM2_historical_r1i1p1f1_gn_185001-201412.nc"]
					elif (source_run == 'CESM2') and (variable_run == 'tasmax') and (exp == 'ssp585'):
						filepaths_cont[exp] = ["/global/cscratch1/sd/bharat/add_cmip6_data/CESM2/extra_cmip6_data/tasmax_Amon_CESM2_ssp585_r1i1p1f1_gn_201501-210012.nc"]

					else:
						filepaths_cont[exp].append (cmip6_filepath_head + "/".join(filepath_ar[chunk_idx]))

			if source_run == 'BCC-CSM2-MR':
				filepath_area 	= "/global/homes/b/bharat/extra_cmip6_data/areacella_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc"
				filepath_lf		= "/global/homes/b/bharat/extra_cmip6_data/sftlf_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc"
			else:
				filters_area    = (temp['variable_id'] == 'areacella') & (temp['source_id'] == source_run)
				filters_lf      = (temp['variable_id'] == 'sftlf') & (temp['source_id'] == source_run)
				filepath_area   = cmip6_filepath_head + "/".join(np.array(temp[filters_area].iloc[-1]))
				filepath_lf     = cmip6_filepath_head + "/".join(np.array(temp[filters_lf].iloc[-1]))

# Check chunk_idx nc_data[exp][member_id_tmp][chunk_idx]
		for exp in experiments_ids:
			for chunk_idx in range(num_chunk_time[exp]):
				nc_data[exp][member_id_tmp][chunk_idx] = nc4.Dataset(filepaths_cont[exp][chunk_idx])
		
#nc_data[member_id_tmp] = nc4.MFDataset([filepaths_cont['historical'], filepaths_cont['ssp585']])
		var			= nc_data['historical'][member_id_tmp][0].variables[variable_run]
		lat 		= nc_data['historical'][member_id_tmp][0].variables['lat']
		lon			= nc_data['historical'][member_id_tmp][0].variables['lon']
		time		= nc_data['historical'][member_id_tmp][0].variables['time']
		lat_bounds	= nc_data['historical'][member_id_tmp][0].variables[lat.bounds]
		lon_bounds	= nc_data['historical'][member_id_tmp][0].variables[lon.bounds]
		print ("Time Size: ", time.size, "no. of lats: ", lat.size, "no. of lons: ", lon.size)	
# Concatenating the variables under consideration
# Since the time can start on 1850 or 2015, so it is important to use cftime and read the units along with it
# The same calculation is applied to time_bounds
		var_data		= nc_data['historical'][member_id_tmp][0].variables[variable_run][...]
		time_datetime	= cftime.num2date	(times 		= nc_data['historical'][member_id_tmp][0].variables['time'][...], 
											 units		= nc_data['historical'][member_id_tmp][0].variables['time'].units,
											 calendar	= nc_data['historical'][member_id_tmp][0].variables['time'].calendar  )

		time_bounds_datetime	= cftime.num2date (	times	= nc_data['historical'][member_id_tmp][0].variables[time.bounds][...], 
													units	= nc_data['historical'][member_id_tmp][0].variables['time'].units,
													calendar= nc_data['historical'][member_id_tmp][0].variables['time'].calendar )

# concatenating the variables under consideration
# the aim is to make one variable for the whole time duration from 1850 -- 2100

		for exp in experiments_ids:
			for chunk_idx in range(num_chunk_time[exp]):
				if (exp == 'historical') and (chunk_idx == 0):
					continue
				print (exp)
				if (source_run == 'CESM2') and (variable_run == 'tasmax') and (exp=='ssp585'):
					cesm2_tasmax_bias = 6.433
					var_data 		= np.concatenate(  (var_data, 
													nc_data[exp][member_id_tmp][chunk_idx].variables[variable_run][...] - cesm2_tasmax_bias ),
													axis =0) # units: kg m-2 s-1
				else:
					var_data 		= np.concatenate(  (var_data, 
													nc_data[exp][member_id_tmp][chunk_idx].variables[variable_run][...]),
													axis =0) # units: kg m-2 s-1

				time_datetime	= np.concatenate(  (time_datetime, 
													cftime.num2date(times		= nc_data[exp][member_id_tmp][chunk_idx].variables['time'][...],
																	units		= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].units,
																	calendar	= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].calendar)),
													axis = 0)
				try:
					time_bounds_datetime	= np.concatenate(	(time_bounds_datetime, 
															 cftime.num2date(times		= nc_data[exp][member_id_tmp][chunk_idx].variables[time.bounds][...],
																			 units		= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].units,
																			 calendar	= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].calendar))
															,axis = 0)
				except:
					time_bounds_datetime	= np.concatenate(	(time_bounds_datetime, 
															 cftime.num2date(times		= nc_data[exp][member_id_tmp][chunk_idx].variables['time_bnds'][...],
																			 units		= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].units,
																			 calendar	= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].calendar))
															,axis = 0)

				print (exp)

# Masking the values again to avoid errors arising due to masking
		var_data = np.ma.masked_equal(var_data, var.missing_value)

# saving datetime time dates
# now the units are the same for the time bounds as "TIME_units"
		time_bounds	= nc_data['historical'][member_id_tmp][0].variables[time.bounds]
		TIME_units	= 'days since 1850-01-01 00:00:00'
		time_floats	= cftime.date2num	(dates		= time_datetime, 
										 units		= TIME_units,
										 calendar	= nc_data['historical'][member_id_tmp][0].variables['time'].calendar)
		time_bounds_floats = cftime.date2num	(dates		= time_bounds_datetime, 
												 units		= TIME_units,
												 calendar	= nc_data['historical'][member_id_tmp][0].variables['time'].calendar)		

		try:	
			area	= nc4.Dataset(filepath_areacella['historical']).variables['areacella']
			lf		= nc4.Dataset(filepath_sftlf['historical']).variables['sftlf']
		except:
			if source_run == 'BCC-CSM2-MR':
				area	= nc4.Dataset("/global/homes/b/bharat/extra_cmip6_data/areacella_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc").variables['areacella']
				lf		= nc4.Dataset("/global/homes/b/bharat/extra_cmip6_data/sftlf_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc").variables['sftlf']
			else:
				area	= nc4.Dataset(filepath_area).variables["areacella"]
				lf		= nc4.Dataset(filepath_lf).variables["sftlf"]
#convert "kg m-2 s-1" to "gC"
		if lf.units == '%': 
			lf 	= lf[...]/100 # converting the land fraction percentage to the fraction.
			area_act	= area[...] * lf #area_act (m2) is the effective or actual area of that pixels
			
		if variable_run in ['gpp','npp','nep','nbp','fFireAll','ra','rh', 'fHarvest', 'fDeforestToAtmos', 'fLulccAtmLut','cTotFireLut']:
			time_days   = [int(time_bounds_floats[i][1]-time_bounds_floats[i][0]) for i in range(time_bounds_floats.shape[0])]  
			time_sec    = np.array(time_days)*24*3600
			vol_m2s     = time_sec[:,np.newaxis,np.newaxis] * area_act  # units vol: m^2*s
			if variable_run in ['fLulccAtmLut', 'cTotFireLut']:
				var_gC		= vol_m2s * np.sum(var_data,axis=1) * 1000 	# gC/mon  for due to Primary and Secondary
			else:
				var_gC		= vol_m2s * var_data * 1000 	# gC/mon
			var_ar_gC [source_run] [member] ['var_gC'] = var_gC
			var_ar_gC [source_run] [member] ['lat'] 	= lat[...]
			var_ar_gC [source_run] [member] ['lon'] 	= lon[...]
			var_ar_gC [source_run] [member] ['lat_bounds'] 	= lat_bounds[...]
			var_ar_gC [source_run] [member] ['lon_bounds'] 	= lon_bounds[...]


			var_gC_global_mon_tot	= np.ma.array([np.ma.sum(var_gC[i,:,:]) for i in range(var_data.shape[0])]) #g/mon
			var_gC_global_yr_tot	= np.ma.array([np.ma.sum(var_gC_global_mon_tot[i*12:(i*12)+12]) for i in range(len(var_gC_global_mon_tot)//12)]) #g/y
			pd_var_gC_global_yr_tot = pd.Series(var_gC_global_yr_tot) #g/y
			var_gC_global_rm5yr_tot	= pd_var_gC_global_yr_tot.rolling(window=5,center = False).mean()# 5 year rolling mean

			ts_yr_source_var_member[source_run][member] 	= var_gC_global_yr_tot #g/y
			ts_rm5yr_source_var_member[source_run][member] 	= var_gC_global_rm5yr_tot #g/y
			df_tmp	= create_df_info(	s = source_run,e = exp, m=member, v=variable_run, 
										g=grid_run, ver_h = vers['historical'],ver_s = vers['ssp585'], 
										tu=time.units, tc=time.calendar,lt = lat.size, 
										ln = lon.size, vu=var.units, au =area.units,i=ix)
			df_model_info	= df_model_info.append(df_tmp, sort = True)	
			ix = ix +1

# Calculations in case of climate drivers :
# -----------------------------------------
		if variable_run in ['pr','mrso','mrsos']:
# Conversted Variable in the desired units of [mm day-1]	
			if variable_run == 'pr'		: des_units = 'mm day-1'# Desired units of precipitation
			if variable_run == 'mrso'	: des_units = 'mm' 		# Desired units of soil moisture
			if variable_run == 'mrsos'	: des_units = 'mm' 		# Desired units of soil moisture
			con_factor	= Unit_Conversions (From=var.units, To= des_units)[0] # Conversion Factor
			con_var		= var_data * con_factor
			con_var_units	= Unit_Conversions (From=var.units, To= des_units)[1] # Converted Units
			
# Area weighted averages
			con_var_global_mon_awm		= np.ma.array([np.ma.average(con_var[i,:,:],
										  weights = area_act ) for i in range(con_var.shape[0])])
			con_var_global_yr_tot_awm	= np.ma.array([np.ma.average(np.ma.sum(con_var[i*12:(i*12+12),:,:],axis =0),
										  weights = area_act)  for i in range ( con_var.shape[0]//12)])
			con_var_global_yr_av_awm	= np.ma.array([np.ma.average(np.ma.mean(con_var[i*12:(i*12+12),:,:],axis =0),
										  weights = area_act)  for i in range ( con_var.shape[0]//12)])
# Calculation the moving averages
			pd_con_var_global_yr_tot_awm	= pd.Series (con_var_global_yr_tot_awm)
			con_var_global_rm5yr_tot_awm	= pd_con_var_global_yr_tot_awm.rolling(window=5,center = False).mean()# 5 year rolling mean

			pd_con_var_global_yr_av_awm	= pd.Series (con_var_global_yr_av_awm)
			con_var_global_rm5yr_av_awm	= pd_con_var_global_yr_av_awm.rolling(window=5,center = False).mean()# 5 year rolling mean

			ts_yr_source_var_member[source_run][member] 	= con_var_global_yr_tot_awm 		# con units
			ts_rm5yr_source_var_member[source_run][member] 	= con_var_global_rm5yr_tot_awm	# con units
			ts_av_rm5yr_source_var_member[source_run][member] 	= con_var_global_rm5yr_av_awm	# con units
		
		if variable_run in ['tas','tasmax','tasmin']:
# Conversted Variable in the desired units of [mm day-1]	
			if variable_run in ['tas','tasmax','tasmin']  : des_units = 'C'# Desired units of precipitation
			con_factor	= Unit_Conversions (From=var.units, To= des_units)[0] # Conversion Factor
			con_var		= var_data + con_factor
			con_var_units	= Unit_Conversions (From=var.units, To= des_units)[1] # Converted Units
			
# Area weighted averages
			con_var_global_mon_awm		= np.ma.array([np.ma.average(con_var[i,:,:],
										  weights = area_act ) for i in range(con_var.shape[0])])
			con_var_global_yr_tot_awm	= np.ma.array([np.ma.average(np.ma.sum(con_var[i*12:(i*12+12),:,:],axis =0),
										  weights = area_act)  for i in range ( con_var.shape[0]//12)])
			con_var_global_yr_av_awm	= np.ma.array([np.ma.average(np.ma.mean(con_var[i*12:(i*12+12),:,:],axis =0),
										  weights = area_act)  for i in range ( con_var.shape[0]//12)])
# Calculation the moving averages

			pd_con_var_global_yr_tot_awm	= pd.Series (con_var_global_yr_tot_awm)
			con_var_global_rm5yr_tot_awm	= pd_con_var_global_yr_tot_awm.rolling(window=5,center = False).mean()# 5 year rolling mean

			pd_con_var_global_yr_av_awm	= pd.Series (con_var_global_yr_av_awm)
			con_var_global_rm5yr_av_awm	= pd_con_var_global_yr_av_awm.rolling(window=5,center = False).mean()# 5 year rolling mean

			ts_yr_source_var_member[source_run][member] 	= con_var_global_yr_tot_awm 		# con units
			ts_rm5yr_source_var_member[source_run][member] 	= con_var_global_rm5yr_tot_awm	# con units
			ts_av_rm5yr_source_var_member[source_run][member] 	= con_var_global_rm5yr_av_awm	# con units

			
	if variable_run in ['gpp','npp','nep','nbp','fFireAll','ra','rh', 'fHarvest', 'fDeforestToAtmos', 'fLulccAtmLut','cTotFireLut']:
# Plotting Total Global Yearly GPP/NPP/NEP/NBP
# --------------------------------------------
		time_x					= np.arange(1850,2101)

		fig_1		= plt.figure(tight_layout = True, dpi = 400)
		for member in member_ids_common:
			if (variable_run == 'nep') and (source_run == 'CESM2'):
				plt.plot    (time_x , -ts_yr_source_var_member[source_run][member]/(10**15) , label = member, linewidth = .3)
			else:
				plt.plot	(time_x , ts_yr_source_var_member[source_run][member]/(10**15) , label = member, linewidth = .3)
			plt.title	("%s - %s: Total Global Yearly"%(source_run, variable_run))
			plt.ylabel	("PgC/year")
			plt.xlabel	("Time")
			plt.grid  	(True, linestyle='--',linewidth = .5)
			plt.legend  (loc='upper center', bbox_to_anchor=(0.5, -0.06),
						fancybox=True, shadow=True, ncol=7,fontsize=6)
		try:	os.remove (web_path + "TS_%s_%s_%s_gC_tot_global_yr.pdf"%(source_run, member, variable_run))
		except: print("The fig1 does not already exist")
		fig_1	.savefig (web_path + "TS_%s_%s_%s_gC_tot_global_yr.pdf"%(source_run, member, variable_run))
		# Saving the plots
		path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/TimeSeries/"%(source_run,member, variable_run)
		if os.path.isdir(path_save) == False:
		    os.makedirs(path_save)
		fig_1	.savefig (path_save + "TS_%s_%s_%s_gC_tot_global_yr.pdf"%(source_run, member, variable_run))
		plt.close (fig_1)	

		
# Plotting 5yr Running mean Total Global Yearly GPP/NPP/NEP/NBP
# -------------------------------------------------------------
		fig_2		= plt.figure(tight_layout = True, dpi = 400)
		for member in member_ids_common:
			if (variable_run == 'nep') and (source_run == 'CESM2'):
				plt.plot    (time_x , -ts_rm5yr_source_var_member[source_run][member]/(10**15), label = member, linewidth = .3)
			else:
				plt.plot	(time_x , ts_rm5yr_source_var_member[source_run][member]/(10**15), label = member, linewidth = .3)
			plt.title	("%s - %s: 5 year Moving Average Total Global Yearly"%(source_run, variable_run))
			plt.ylabel	("PgC/year")
			plt.xlabel	("Time")
			plt.grid  	(True, linestyle='--',linewidth = .5)
			plt.legend (loc='upper center', bbox_to_anchor=(0.5, -0.06),
						fancybox=True, shadow=True, ncol=7,fontsize=6)
		try: os.remove (web_path + "%s_%s_gC_tot_global_rm5yr.pdf"%(source_run,variable_run))
		except: print("The fig2 does not already exist")
		fig_2	.savefig (web_path + "%s_%s_%s_gC_tot_global_rm5yr.pdf"%(source_run,member, variable_run))
		# Saving the plots
		path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/TimeSeries/"%(source_run,member, variable_run)
		if os.path.isdir(path_save) == False:
		    os.makedirs(path_save)
		fig_2	.savefig (path_save + "%s_%s_%s_gC_tot_global_rm5yr.pdf"%(source_run,member, variable_run))
		plt.close (fig_2)	

	if variable_run in ['pr','mrso','mrsos','tas','tasmax']:
# Climate Drivers

		time_x					= np.arange(1850,2101)

# Plotting yr Running mean Total Global AWM Yearly
# -------------------------------------------------------------

		fig_11		= plt.figure(tight_layout = True, dpi = 400)
		for member in member_ids_common:
			plt.plot    (time_x , ts_yr_source_var_member[source_run][member] , label = member, linewidth = .3)
			plt.title	("%s - %s: Total Global AWM Yearly"%(source_run, variable_run))
			plt.ylabel	(con_var_units)
			plt.xlabel	("Time")
			plt.grid  	(True, linestyle='--',linewidth = .5)
			plt.legend  (loc='upper center', bbox_to_anchor=(0.5, -0.06),
						fancybox=True, shadow=True, ncol=7,fontsize=6)
		try:	os.remove (web_path + "%s_%s_tot_awm_global_yr.pdf"%(source_run,variable_run))
		except: print("The fig1 does not already exist")
		fig_11	.savefig (web_path + "%s_%s_%s_tot_awm_global_yr.pdf"%(source_run,member, variable_run))
		# Saving the plots
		path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/TimeSeries/"%(source_run,member, variable_run)
		if os.path.isdir(path_save) == False:
		    os.makedirs(path_save)
		fig_11	.savefig (path_save + "%s_%s_%s_tot_awm_global_yr.pdf"%(source_run,member, variable_run))
		plt.close (fig_11)	

# Plotting 5yr Running mean Total Global AWM Yearly
# -------------------------------------------------------------
		fig_12		= plt.figure(tight_layout = True, dpi = 400)
		for member in member_ids_common:
			plt.plot	(time_x , ts_rm5yr_source_var_member[source_run][member], label = member, linewidth = .3)
			plt.title	("%s - %s: 5 year Moving Average Total Global AWM"%(source_run, variable_run))
			plt.ylabel	(con_var_units)
			plt.xlabel	("Time")
			plt.grid  	(True, linestyle='--',linewidth = .5)
			plt.legend (loc='upper center', bbox_to_anchor=(0.5, -0.06),
						fancybox=True, shadow=True, ncol=7,fontsize=6)
		try: os.remove (web_path + "%s_%s_tot_global_awm_rm5yr.pdf"%(source_run,variable_run))
		except: print("The fig2 does not already exist")
		fig_12	.savefig (web_path + "%s_%s_%s_tot_global_awm_rm5yr.pdf"%(source_run, member, variable_run))
		# Saving the plots
		path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/TimeSeries/"%(source_run,member, variable_run)
		if os.path.isdir(path_save) == False:
		    os.makedirs(path_save)
		fig_12	.savefig (path_save + "%s_%s_%s_tot_global_awm_rm5yr.pdf"%(source_run, member, variable_run))
		plt.close (fig_12)	


# Plotting yr Running mean Average Global AWM Yearly
# -------------------------------------------------------------

		fig_13		= plt.figure(tight_layout = True, dpi = 400)
		for member in member_ids_common:
			plt.plot    (time_x , ts_av_rm5yr_source_var_member[source_run][member] , label = member, linewidth = .3)
			plt.title	("%s - %s: Average Global AWM Yearly"%(source_run, variable_run))
			plt.ylabel	(con_var_units)
			plt.xlabel	("Time")
			plt.grid  	(True, linestyle='--',linewidth = .5)
			plt.legend  (loc='upper center', bbox_to_anchor=(0.5, -0.06),
						fancybox=True, shadow=True, ncol=7,fontsize=6)
		try:	os.remove (web_path + "%s_%s_av_awm_global_yr.pdf"%(source_run,variable_run))
		except: print("The fig13 does not already exist")
		fig_13	.savefig (web_path + "%s_%s_%s_av_awm_global_yr.pdf"%(source_run, member, variable_run))
		# Saving the plots
		path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/TimeSeries/"%(source_run,member, variable_run)
		if os.path.isdir(path_save) == False:
		    os.makedirs(path_save)
		fig_13	.savefig (path_save + "%s_%s_%s_av_awm_global_yr.pdf"%(source_run, member, variable_run))
		plt.close (fig_13)	



if variable_run in ['gpp','npp','nep','nbp','fFireAll','ra','rh', 'fHarvest', 'fDeforestToAtmos', 'fLulccAtmLut','cTotFireLut']:
	fig_3,ax = plt.subplots(nrows=1,ncols=1,tight_layout = True, dpi = 400)
	plt.title   ("Multi-Model %s: Total Global Yearly"%(variable_run.upper()))
	plt.ylabel  ("PgC/year")
	plt.xlabel  ("Time")
	plt.grid    (True, linestyle='--',linewidth = .5)
	for s_idx, source_run in enumerate(source_selected):
		if (variable_run == 'nep') and (source_run == 'CESM2'):
			mm_mean   = ax.plot    (time_x , -np.array(pd.DataFrame(ts_yr_source_var_member[source_run]).mean(axis = 1))/(10**15) , label = source_run, linewidth = 1)
		else:
			mm_mean   = ax.plot    (time_x , np.array(pd.DataFrame(ts_yr_source_var_member[source_run]).mean(axis = 1))/(10**15) , label = source_run, linewidth = 1)
		mm_mean[0]	.set_color(clrs[s_idx])
		for m_idx,member in enumerate(member_ids_common_source[source_run]):
			if (variable_run == 'nep') and (source_run == 'CESM2'):
				lines   = ax.plot    (time_x , -ts_yr_source_var_member[source_run][member]/(10**15) , label = member, linewidth = .3)
			else: 
				lines	= ax.plot    (time_x , ts_yr_source_var_member[source_run][member]/(10**15) , label = member, linewidth = .3)
			lines[0].set_color(clrs[s_idx])
			lines[0].set_linestyle(LINE_STYLES[m_idx%NUM_STYLES])
		plt.legend (loc='upper center', bbox_to_anchor=(0.5, -0.06),
				fancybox=True, shadow=True, ncol=7,fontsize=6)
	try: os.remove (web_path + "MultiModel_%s_gC_tot_global_yr.pdf"%(variable_run))
	except: print("The fig3 does not already exist")

	fig_3.savefig (web_path + "MultiModel_%s_gC_tot_global_yr.pdf"%(variable_run))

	fig_4,ax = plt.subplots(nrows=1,ncols=1,tight_layout = True, dpi = 400, figsize=(9,5))
	plt.title   ("Multi-Model %s: 5 Year Moving Average Total Global "%(variable_run.upper()))
	#plt.ylabel  ("PgC/year")
	plt.ylabel  ("Total NBP (PgC/year)", fontsize = 14)
	plt.xlabel  ("Time", fontsize = 14)
	plt.xticks(fontsize = 12) #
	plt.grid    (True, linestyle='--',linewidth = .5)
	for s_idx, source_run in enumerate(source_selected):
		if (variable_run == 'nep') and (source_run == 'CESM2'):
			mm_mean   = ax.plot    (time_x , -np.array(pd.DataFrame(ts_rm5yr_source_var_member[source_run]).mean(axis = 1))/(10**15) , label = source_run, linewidth = 1)
		else:
			mm_mean   = ax.plot    (time_x , np.array(pd.DataFrame(ts_rm5yr_source_var_member[source_run]).mean(axis = 1))/(10**15) , label = source_run, linewidth = 1)
		mm_mean[0]	.set_color(clrs[s_idx])
		for m_idx,member in enumerate(member_ids_common_source[source_run]):
			if (variable_run == 'nep') and (source_run == 'CESM2'):
				lines   = ax.plot    (time_x , -ts_rm5yr_source_var_member[source_run][member]/(10**15) , linewidth = .3)
			else:
				lines   = ax.plot    (time_x , ts_rm5yr_source_var_member[source_run][member]/(10**15) , linewidth = .3)
			lines[0].set_color(clrs[s_idx])
			lines[0].set_linestyle(LINE_STYLES[m_idx%NUM_STYLES])
		plt.legend (loc='upper center', bbox_to_anchor=(0.5, -0.06),
			fancybox=True, shadow=True, ncol=3,fontsize=10)
	try: os.remove(web_path + "MultiModel_%s_gC_tot_global_rm5yr.pdf"%(variable_run))
	except: print("The fig4 does not already exist")
	fig_4.savefig (web_path + "MultiModel_%s_gC_tot_global_rm5yr.pdf"%(variable_run))

	# Figure for the paper for NBP and CESM2
	fig_411,ax = plt.subplots(nrows=1,ncols=1,tight_layout = True, dpi = 400, figsize=(9,5))
	plt.title   (f"Multi-Model {variable_run.upper()}: 5 Year Moving Average Total Global \n")
	#plt.ylabel  ("PgC/year")
	plt.ylabel  ("Total NBP (PgC/year)", fontsize = 14)
	plt.xlabel  ("Time", fontsize = 14)
	plt.xticks(fontsize = 12) #
	plt.yticks(fontsize = 12) #
	plt.grid    (True, linestyle='--',linewidth = .5)
	for s_idx, source_run in enumerate(source_selected):
		if (variable_run == 'nbp') and (source_run == 'CESM2'):
			mm_mean   = ax.plot    (time_x , np.array(pd.DataFrame(ts_rm5yr_source_var_member[source_run]).mean(axis = 1))/(10**15) , label = source_run, linewidth = 1, color='k')
		for m_idx,member in enumerate(member_ids_common_source[source_run]):
			if (variable_run == 'nbp') and (source_run == 'CESM2'):
				lines   = ax.plot    (time_x , ts_rm5yr_source_var_member[source_run][member]/(10**15) , linewidth = .3, color='gray')
			lines[0].set_linestyle(LINE_STYLES[m_idx%NUM_STYLES])
	try: os.remove(web_path + "MultiModel_%s_gC_tot_global_rm5yr.pdf"%(variable_run))
	except: print("The fig4 does not already exist")
	fig_411.savefig (web_path + "MultiModel_%s_gC_tot_global_rm5yr.pdf"%(variable_run))


# Multimodal Climate drivers
# --------------------------
if variable_run in ['pr','mrso','mrsos','tas','tasmax','tasmin']:
	fig_13,ax = plt.subplots(nrows=1,ncols=1,tight_layout = True, dpi = 400)
	plt.title   ("Multi-Model %s: Total Global AWM Yearly"%(variable_run.upper()))
	plt.ylabel  (con_var_units)
	plt.xlabel  ("Time")
	plt.grid    (True, linestyle='--',linewidth = .5)
	for s_idx, source_run in enumerate(source_selected):
		mm_mean   = ax.plot    (time_x , np.array(pd.DataFrame(ts_yr_source_var_member[source_run]).mean(axis = 1)) , label = source_run, linewidth = 1)
		mm_mean[0]	.set_color(clrs[s_idx])
		for m_idx,member in enumerate(member_ids_common_source[source_run]):
			lines	= ax.plot    (time_x , ts_yr_source_var_member[source_run][member] , label = member, linewidth = .3)
			lines[0].set_color(clrs[s_idx])
			lines[0].set_linestyle(LINE_STYLES[m_idx%NUM_STYLES])
		plt.legend (loc='upper center', bbox_to_anchor=(0.5, -0.06),
				fancybox=True, shadow=True, ncol=7,fontsize=6)
	try: os.remove (web_path + "MultiModel_%s_tot_global_awm_yr.pdf"%(variable_run))
	except: print("The fig3 does not already exist")

	fig_13.savefig (web_path + "MultiModel_%s_tot_global_awm_yr.pdf"%(variable_run))

	fig_14,ax = plt.subplots(nrows=1,ncols=1,tight_layout = True, dpi = 400)
	plt.title   ("Multi-Model %s: 5 Year Moving Average Total Global AWM "%(variable_run.upper()))
	plt.ylabel  (con_var_units)
	plt.xlabel  ("Time")
	plt.grid    (True, linestyle='--',linewidth = .5)
	for s_idx, source_run in enumerate(source_selected):
		mm_mean   = ax.plot    (time_x , np.array(pd.DataFrame(ts_rm5yr_source_var_member[source_run]).mean(axis = 1)) , label = source_run, linewidth = 1)
		mm_mean[0]	.set_color(clrs[s_idx])
		for m_idx,member in enumerate(member_ids_common_source[source_run]):
			lines   = ax.plot    (time_x , ts_rm5yr_source_var_member[source_run][member] , linewidth = .3)
			lines[0].set_color(clrs[s_idx])
			lines[0].set_linestyle(LINE_STYLES[m_idx%NUM_STYLES])
		plt.legend (loc='upper center', bbox_to_anchor=(0.5, -0.06),
				fancybox=True, shadow=True, ncol=3,fontsize=10)
	try: os.remove(web_path + "MultiModel_%s_tot_global_awm_rm5yr.pdf"%(variable_run))
	except: print("The fig4 does not already exist")
	fig_14.savefig (web_path + "MultiModel_%s_tot_global_awm_rm5yr.pdf"%(variable_run))

	fig_15,ax = plt.subplots(nrows=1,ncols=1,tight_layout = True, dpi = 400)
	plt.title   ("Multi-Model %s: 5 Year Moving Average Mean Global AWM "%(variable_run.upper()))
	plt.ylabel  (con_var_units)
	plt.xlabel  ("Time")
	plt.grid    (True, linestyle='--',linewidth = .5)
	for s_idx, source_run in enumerate(source_selected):
		mm_mean   = ax.plot    (time_x , np.array(pd.DataFrame(ts_av_rm5yr_source_var_member[source_run]).mean(axis = 1)) , label = source_run, linewidth = 1)
		mm_mean[0]	.set_color(clrs[s_idx])
		for m_idx,member in enumerate(member_ids_common_source[source_run]):
			lines   = ax.plot    (time_x , ts_av_rm5yr_source_var_member[source_run][member] , linewidth = .3)
			lines[0].set_color(clrs[s_idx])
			lines[0].set_linestyle(LINE_STYLES[m_idx%NUM_STYLES])
		plt.legend (loc='upper center', bbox_to_anchor=(0.5, -0.06),
				fancybox=True, shadow=True, ncol=3,fontsize=10)
	try: os.remove(web_path + "MultiModel_%s_av_global_awm_rm5yr.pdf"%(variable_run))
	except: print("The fig4 does not already exist")
	fig_15.savefig (web_path + "MultiModel_%s_av_global_awm_rm5yr.pdf"%(variable_run))


# Difference of Fluxes
""" This is added here because the code to concatenate and have the units same for all the files is long...
	The diff of extremes is done in 'calc_extremes.py'
	The function is the same as used in 'calc_extremes.py'
"""
# Arranging Time Array for plotting and calling
# --------------------------------------------
window		= 25 #years
win_len     = 12 * window             #number of months in window years
total_years	= 251 #years from 1850 to 2100
total_months= total_years * 12

dates_ar    = time_dim_dates( base_date = dt.date(1850,1,1), 
							  total_timestamps = 3012 )
start_dates = np.array(	[dates_ar[i*win_len] for i in range(int(total_months/win_len))])    #list of start dates of 25 year window
end_dates   = np.array( [dates_ar[i*win_len+win_len -1] for i in range(int(total_months/win_len))]) #list of end dates of the 25 year window
# Initiation:
# -----------
def TS_Dates_and_Index (dates_ar = dates_ar,start_dates = start_dates, end_dates=end_dates ):
    """
    Returns the TS of the dates and index of consecutive windows of len 25 years

    Parameters:
    -----------
    dates_ar :  an array of dates in datetime.date format
                the dates are chosen from this array
    start_dates: an array of start dates, the start date will decide the dates and index of the first entry for final time series for that window
    end_dates: similar to start_dates but for end date

    Returns:
    --------
    dates_win: a 2-d array with len of start dates/  total windows and each row containing the dates between start and end date
    idx_dates_win : a 2-d array with len of start dates/  total windows and each row containing the index of dates between start and end date
    """
    idx_dates_win   = []    #indicies of time in 25yr windows
    dates_win       = []    #sel dates from time variables in win_len windows
    for i in range(len(start_dates)):
        idx_loc, dates_loc  = index_and_dates_slicing(dates_ar,start_dates[i],end_dates[i]) # see functions.py
        idx_dates_win       . append    (idx_loc)
        dates_win           . append    (dates_loc)
    return np.array(dates_win), np.array(idx_dates_win)

# Calling the function "ts_dates_and_index"; Universal for rest of the code
dates_win, idx_dates_win = TS_Dates_and_Index ()


def Sum_and_Diff_of_Fluxes_perWin(ano_gC, bin_ar = None, data_type = 'ext', diff_ref_yr = 1850):
	"""
	returns a 2-d array sum of fluxes and difference of the sum of fluxes with reference to the ref yr
start_dates
	Parameters:
	----------
	bin_ar: the binary array of extremes (pos/neg) 
	ano_gC : the array which will use the mask or binary arrays to calc the carbon loss/gain
	diff_ref_yr : the starting year of the reference time window for differencing
	data_type : do you want to calculate the sum and difference of extremes or original fluxes? ...
				'ext' is for extremes and will mask based on the 'bin_ar' in calculation ... 
				otherwise it will not multiply by bin_ar and the original flux difference will be calculated.
				'ext' will calculate the extremes and anything else with calc on original flux diff

	Universal:
	----------
	start_dates :  the start_dates of every 25 year window, size = # wins

	Returns:
	--------
	sum_flux : shape (# wins, nlat,nlon), sum of fluxes per window
	diff_flux : shape (# wins, nlat,nlon), difference of sum of fluxes per window and reference window
	"""

	if data_type != 'ext': bin_ar = np.ma.ones(ano_gC.shape)
	sum_ext 	= []
	for i in range(len(start_dates)):
		ext_gC = bin_ar[idx_dates_win[i][0] : idx_dates_win [i][-1]+1,:,:] * ano_gC[idx_dates_win[i][0] : idx_dates_win [i][-1]+1,:,:]
		sum_ext . append (np.ma.sum(ext_gC, axis = 0))
	sum_ext 	= np.ma.asarray(sum_ext)
	
	#to calculate the index of the reference year starting window:
	for i,date in enumerate(start_dates):
		if date.year in [diff_ref_yr]:
			diff_yr_idx = i
	diff_ext	= []
	for i in range(len(start_dates)):
		diff	= sum_ext[i] - sum_ext[diff_yr_idx]
		diff_ext . append (diff)
	diff_ext	= np.ma.asarray(diff_ext) 
	return sum_ext , diff_ext

# -------------------------------
# Saving the data
# -------------------------------
Results	= {}
for source_run in source_selected:
	Results [source_run] = {}
	for member_run in member_ids_common_source [source_run]:
		Results [source_run] [member_run] = {}

if variable_run in ['gpp','npp','nep','nbp','fFireAll','ra','rh', 'fHarvest', 'fDeforestToAtmos', 'fLulccAtmLut','cTotFireLut']:
	for source_run in source_selected:
		for member_run in member_ids_common_source [source_run]:
			#Negative Flux/Ori
			sum_ori	, diff_ori = Sum_and_Diff_of_Fluxes_perWin (bin_ar = None,
																ano_gC = var_ar_gC [source_run] [member_run]['var_gC'],
																data_type = 'ori',
																diff_ref_yr = 1850)
			Results[source_run][member_run]['sum_ori']	= sum_ori
			Results[source_run][member_run]['diff_ori']	= diff_ori

# Registering a color map
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

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
"""
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh
"""
proj_trans = ccrs.PlateCarree()

def Plot_Diff_Plot(diff_array, datatype='neg_ext', analysis_type='independent', savefile_head = ""):
	""" Plotting the change or difference maps for the given data and passing the following filters for plotting
	
	Parameters:
	-----------
	diff_array: the difference array of shape  (nwin,nlat,nlon)
	datatype : 'ext' or 'var_ori'
	analysis type: 'independent or 'rel_av_all' or 'rel_pos_wo_wulcc'

	"""
	text = []
	for i in start_dates:
		a = i.year
		b = a+24
		c = str(a)+'-'+str(b)[2:]
		text.append(c)
	#to calculate the index of the reference year starting window:
	diff_ref_yr = 1850
	for i,date in enumerate(start_dates):
		if date.year in [diff_ref_yr]:
			diff_yr_idx = i

	if (variable_run =='gpp' and datatype=='ori' and analysis_type=='independent'): 
		if source_run == 'CESM2'		:  ymax = 800 ; ymin = -ymax
		if source_run == 'BCC-CSM2-MR'	:  ymax = 1000; ymin = -ymax
		if source_run == 'CNRM-CM6-1'	:  ymax = 2000; ymin = -ymax
		if source_run == 'CNRM-ESM2-1'	:  ymax = 1500; ymin = -ymax
		if source_run == 'CanESM5'		:  ymax = 7000; ymin = -ymax
		if source_run == 'IPSL-CM6A-LR' :  ymax = 2000; ymin = -ymax

		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	if (variable_run=='gpp' and datatype=='neg_ext'):
		if source_run == 'CESM2':  ymax = 100; ymin = -ymax

		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	if (variable_run=='gpp' and datatype=='pos_ext'): 
		ymin = -14
		ymax = 14
		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	
	if (variable_run =='nbp' and datatype=='ori' and analysis_type=='independent'): 
		if source_run == 'CESM2'		:  ymax = 5 ; ymin = -ymax  #5th or 95th percentile values
		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	if (variable_run =='ra' and datatype=='ori' and analysis_type=='independent'): 
		if source_run == 'CESM2'		:  ymax = 32 ; ymin = -ymax  #5th or 95th percentile values
		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	if (variable_run =='rh' and datatype=='ori' and analysis_type=='independent'): 
		if source_run == 'CESM2'		:  ymax = 17 ; ymin = -ymax  #5th or 95th percentile values
		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	if (variable_run =='fFireAll' and datatype=='ori' and analysis_type=='independent'): 
		if source_run == 'CESM2'		:  ymax = 1 ; ymin = -ymax  #3rd or 97th percentile values
		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	if (variable_run =='fHarvest' and datatype=='ori' and analysis_type=='independent'): 
		if source_run == 'CESM2'		:  ymax = 1 ; ymin = -ymax  #3rd or 97th percentile values
		ts_div_factor	= 10**12
		y_label_text	= 'TgC'


	for i, data in enumerate (diff_array):
		print ("Plotting %s %s for the win %d"%(datatype,analysis_type,i))
		fig4 = plt.figure(figsize=[12, 5])
		ax = fig4.add_subplot(1, 1, 1, projection=ccrs.Robinson(central_longitude=0))
		Lat_Bounds = var_ar_gC [source_run] [member] ['lat_bounds']
		Lon_Bounds = var_ar_gC [source_run] [member] ['lon_bounds']
		lat_edges   = np.hstack (( Lat_Bounds[:,0], Lat_Bounds[-1,-1]))
		lon_edges   = np.hstack (( Lon_Bounds[:,0], Lon_Bounds[-1,-1]))
		h 	= ax.pcolormesh(lon_edges[...],lat_edges[...], np.ma.masked_invalid(data/ts_div_factor), transform = proj_trans, cmap="RdGn", vmax= ymax, vmin= ymin) 
		cbar = plt.colorbar(h)
		ax.coastlines(linewidth = .4)
		cbar    .ax.set_ylabel(y_label_text)
		gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=2, color='gray', alpha=0.5, linestyle='--')
		gl.xlabels_top = False
		gl.ylabels_right = False
		gl.xformatter = LONGITUDE_FORMATTER
		gl.yformatter = LATITUDE_FORMATTER
		
		plt.title("%s for model %s - %s minus %s" %(variable_run,source_run, text[i], text[diff_yr_idx]))
		#plt.title ("%s Ano for model %s and win %d"%(variable_run,source_run,i))

		fig4.savefig(web_path + 'Spatial_Maps/%s_%s.png'%(savefile_head, format(i,'02')))
		# Saving the plots
		path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Spatial_Maps/"%(source_run,member, variable_run)
		if os.path.isdir(path_save) == False:
		    os.makedirs(path_save)
		fig4	.savefig (path_save + "%s_%s_Difference_of_int_%s_win_%s.pdf"%(source_run, member, variable_run,format(i,'02')))

		plt.close(fig4)
		"""
		Basemap stuff:
		--------------
		fig4,ax	= plt.subplots(figsize = (7,2.8),tight_layout=True,dpi=500)
#plt.title ("%s Ano for model %s and win %d"%(variable_run,source_run,i))
		bmap    = Basemap(  projection  =   'eck4',
							lon_0       =   0.,
							resolution  =   'c')
		lat		= var_ar_gC [source_run] [member_run] ['lat']
		lon		= var_ar_gC [source_run] [member_run] ['lon']
		LAT,LON = np.meshgrid(lat[...], lon[...],indexing ='ij')
		ax      = bmap.pcolormesh(LON,LAT,np.ma.masked_invalid(data/ts_div_factor),latlon=True,cmap= 'RdGn',vmax= ymax, vmin= ymin)
		cbar    = plt.colorbar(ax)
		cbar    .ax.set_ylabel(y_label_text)
		bmap    .drawparallels(np.arange(-90., 90., 30.),fontsize=14, linewidth = .2)
		bmap    .drawmeridians(np.arange(0., 360., 60.),fontsize=14, linewidth = .2)
		bmap    .drawcoastlines(linewidth = .2)
		plt.title("%s for model %s - %s minus %s" %(variable_run,source_run, text[i], text[diff_yr_idx]))
		fig4.savefig(web_path + 'Spatial_Maps/%s_%s.png'%(savefile_head, format(i,'02')))
		plt.close(fig4i)
		"""

plot_diff = "no"
if plot_diff in ["y", "yes"]:
	if variable_run in ['gpp','npp','nep','nbp','fFireAll','ra','rh', 'fHarvest', 'fDeforestToAtmos', 'fLulccAtmLut','cTotFireLut']:
		for s_idx, source_run in enumerate(source_selected):
			for m_idx, member_run in enumerate( member_ids_common_source [source_run]):
				text = "plot_diff_%s_%s"%(variable_run, source_run)
				print ( source_run,member_run)
				Plot_Diff_Plot(diff_array = Results[source_run][member_run]['diff_ori']    , datatype='ori',  analysis_type='independent', savefile_head = text)
				break



# Regional variable Timeseries
import regionmask
# Cartopy Plotting
# ----------------
import cartopy.crs as ccrs
from shapely.geometry.polygon import Polygon
import cartopy.feature as cfeature

srex_mask 	= regionmask.defined_regions.srex.mask(lon[...], lat[...]).values  # it has nans
srex_mask_ma= np.ma.masked_invalid(srex_mask) # got rid of nans; values from 1 to 26

# important information:
srex_abr		= regionmask.defined_regions.srex.abbrevs
srex_names		= regionmask.defined_regions.srex.names
srex_nums		= regionmask.defined_regions.srex.numbers 
srex_centroids	= regionmask.defined_regions.srex.centroids 
srex_polygons	= regionmask.defined_regions.srex.polygons


# Calculation of NBP timeseries of the Regions
# --------------------------------------------
ts_nbp_reg_yr_PgC = {}
ts_nbp_reg_rm5yr_PgC = {}
ts_nbp_reg_rm30yr_PgC = {}
ts_nbp_reg_rm10yr_PgC = {}
for region_abr in srex_abr:
    print ("Looking into the Region %s"%region_abr)
    srex_idxs 		= np.arange(len(srex_names))      
    filter_region 	= np.array(srex_abr) == region_abr
    region_idx		= srex_idxs[filter_region][0]
    region_number	= np.array(srex_nums)[filter_region][0]
    region_name		= np.array(srex_names)[filter_region][0]
    region_abr		= np.array(srex_abr)[filter_region][0] 
    region_mask_not	= np.ma.masked_not_equal(srex_mask_ma, region_number).mask   # Masked everthing but the region; Mask of region is False
    region_mask		= ~region_mask_not   # Only the mask of region is True
    nbp_mat_region  = var_gC * np.array([region_mask]*3012)
    var_gC_reg_mon_tot	= np.ma.array([np.ma.sum(nbp_mat_region[i,:,:]) for i in range(var_data.shape[0])]) 
    var_gC_reg_yr_tot	= np.ma.array([np.ma.sum(var_gC_reg_mon_tot[i*12:(i*12)+12]) 
                                   for i in range(len(var_gC_reg_mon_tot)//12)]) #g/y
    pd_var_gC_reg_yr_tot = pd.Series(var_gC_reg_yr_tot) #g/y
    var_gC_reg_rm5yr_tot = pd_var_gC_reg_yr_tot.rolling(window=5,center = True).mean()# 5 year rolling mean
    var_gC_reg_rm5yr_tot.index = np.arange(1850,2101,1)
    var_gC_reg_rm30yr_tot = pd_var_gC_reg_yr_tot.rolling(window=30,center = True).mean()# 30 year rolling mean
    var_gC_reg_rm30yr_tot.index = np.arange(1850,2101,1)
    var_gC_reg_rm10yr_tot = pd_var_gC_reg_yr_tot.rolling(window=10,center = True).mean()# 10 year rolling mean
    var_gC_reg_rm10yr_tot.index = np.arange(1850,2101,1)

    ts_nbp_reg_yr_PgC[region_abr] = pd_var_gC_reg_yr_tot/10**15
    ts_nbp_reg_rm5yr_PgC[region_abr] = var_gC_reg_rm5yr_tot/10**15
    ts_nbp_reg_rm30yr_PgC[region_abr] = var_gC_reg_rm30yr_tot/10**15
    ts_nbp_reg_rm10yr_PgC[region_abr] = var_gC_reg_rm10yr_tot/10**15

# Plotting the Integrated NBP Timeseries for a Region
# Plotting for "CAM"

reg_idx = 5 # index of the region you want to look at?
reg_abr = srex_abr[reg_idx]
reg_name = srex_names [reg_idx]
print (reg_abr)
#fig1 = plt.figure()
plt.style.use("classic")
ts_nbp_reg_rm5yr_PgC[reg_abr].plot(color='purple',lw=2)
ts_nbp_reg_rm30yr_PgC[reg_abr].plot(stacked =False, 
              figsize=(8,4), 
              fontsize = 14,
                lw=2, color = 'k')
plt.grid(	which='both', linestyle='--', linewidth='1', color='gray',alpha=.3)
#plt.title(f"{reg_name}")
import pylab as plot
params = {'legend.fontsize': 20,
          'legend.handlelength': 2}
plot.rcParams.update(params)
plt.title(f"Time series of Integrated {variable_run} for {reg_abr}\n", loc='center', fontsize =16)
#plt.style.use('bmh')
plt.legend(['Moving average 5-years', 'Moving average 30-years'], 
           loc='upper left', 
		   fontsize=12, 
		   ncol=1, 
		   framealpha=.1)
plt.ylabel(f"{variable_run.upper()} (PgC/month)", fontsize=14)
plt.xlabel("Time", fontsize=14)
#plt.legend(loc='upper right', bbox_to_anchor=(1,1.2), fontsize=14, ncol=1)
plt.savefig(web_path + f"Regional/{source_run.upper()}_{reg_abr}_TS_{variable_run}_rm5yr.pdf",
           edgecolor="w", bbox_inches="tight")
plt.savefig(web_path + f"Regional/{source_run.upper()}_{reg_abr}_TS_{variable_run}_rm5yr.png",
           edgecolor="w", bbox_inches="tight")
plt.savefig(path_save + f"{source_run.upper()}_{reg_abr}_TS_{variable_run}_rm5yr.pdf",
           edgecolor="w", bbox_inches="tight")
