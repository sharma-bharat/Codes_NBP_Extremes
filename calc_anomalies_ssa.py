"""
Bharat Sharma
python 3.7
Non- MPI version
:> to plot the variables that are selected
check/run following before running this file:
	- filepaths_to_use.sh (returns the paths of variables that are avaiable)
	- data_scope.py		  (creates a dataframe with hierarchy of data structure)
	- data_processing/df_data_selected.csv	(.csv output of data_scope.py)

Aim:- This code is designed to calculate the anomalies using the SSA.
	- It is not designed for parallel calculations and computed the anomalies for all lons per lat.
	- Hence, the ouput files are  saved with the local rank that are distributed sequentially.
	- you also have an option to save the concatenated variable/flux

"""

import	pandas	as	pd
import	numpy 	as 	np
import	matplotlib.pyplot	as plt
import 	netCDF4	as 	nc4
import	re
import	seaborn	as	sns
import 	cftime
import 	os
from 	functions	import	mpi_local_and_global_index
import	collections
import 	ssa_code   as ssa
# Reading the dataframe of the selected files
# -------------------------------------------
#web_path	= '/project/projectdirs/m2467/www/bharat/' # Error in saving files "Disk quota full/exceeded"
web_path	= '/global/homes/b/bharat/results/web/' 
in_path		= '/global/homes/b/bharat/results/data_processing/'
#cmip6_filepath_head = '/global/homes/b/bharat/cmip6_data/CMIP6/'
cmip6_filepath_head = '/global/cfs/cdirs/m3522/cmip6/CMIP6/'  # new filehead
add_cmip6_path		= '/global/homes/b/bharat/add_cmip6_data/'
cori_scratch	= '/global/cscratch1/sd/bharat/'
path_cdirs  = '/global/cfs/cdirs/m2467/bharat/analysis_data/' 

# The spreadsheet with all the available data of cmip 6
# -----------------------------------------------------
df_files	=	pd.read_csv(in_path + 'df_data_selected.csv')

hierarcy_str= ['activity_id','institution_id','source_id','experiment_id','member_id',
				'table_id','variable_id','grid_label','version','filenames']

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
col_names = ['source_id','experiment_id','member_id','variable_id','grid_label',
			'version','time_units','time_calendar','lat','lon','var_units','area_units']
df_model_info 	= pd.DataFrame(columns = col_names)

def create_df_info (i=np.nan, s = np.nan,e = np.nan, m=np.nan, v=np.nan, 
					g=np.nan, ver_h = np.nan, ver_s = np.nan, tu=np.nan, 
					tc=np.nan,lt = np.nan ,ln = np.nan, vu=np.nan, au =np.nan):
	d = {'source_id': pd.Series([s],index=[i]),
		 'experiment_id': pd.Series([e],index=[i]),
		 'member_id': pd.Series([m],index=[i]),
		 'variable_id': pd.Series([v],index = [i]),
		 'grid_label': pd.Series([g],index = [i]), 
		 'version_historial': pd.Series([ver_h],index = [i]),
		 'version_ssp585': pd.Series([ver_s],index = [i]), 
		 'time_units': pd.Series([tu],index = [i]), 
		 'time_calendar': pd.Series ([tc],index = [i]), 
		 'lat': pd.Series([lt], index = [i]), 
		 'lon': pd.Series([ln],index= [i]),
		 'var_units': pd.Series([vu],index = [i]),
		 'area_units': pd.Series([au],index = [i])
		}
	df = pd.DataFrame(d)
	return df
# -----------------------
#creating a copy of the df_files
temp			= df_files.copy(deep = True)

#creating the filters based on mpodel and variable
# model for this run
source_run		= 'CESM2'
variable_run	= 'cTotFireLut'

# The Models that have gpp for historical and ssp585 experiments:
source_selected = ['CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1']
#source_selected = ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1','EC-Earth3-Veg','UKESM1-0-LL']
#source_selected = ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1'] # for GPP // No areacella in : 'EC-Earth3-Veg','UKESM1-0-LL'
#source_selected = ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1'] # for NBP // no NBP in BCC //No areacella in : 'EC-Earth3-Veg','UKESM1-0-LL'

# Select which model you want to run and Source index:
# ===================================

source_selected	= ['CESM2'			] 	# idx : 0
source_selected	= ['CanESM5'		] 	# idx : 1
source_selected	= ['IPSL-CM6A-LR'	] 	# idx : 2
source_selected	= ['CNRM-ESM2-1'	]	# idx : 3
source_selected	= ['CNRM-CM6-1'		] 	# idx : 4
source_selected	= ['BCC-CSM2-MR'	] 	# idx : 5

import	argparse
parser	= argparse.ArgumentParser()
parser. add_argument ('--source_idx', '-src',	help="The model name",	type = int, default = 0)
parser. add_argument ('--variable', '-var',	help="The variable name",	type = str, default = 'gpp')

args = parser.parse_args()
src 	= int (args.source_idx)
variable_run = args.variable
# For selecting the index of the model from the list of models in the following list
# ----------------------------------------------------------------------------------
models 	= ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','CNRM-CM6-1','BCC-CSM2-MR']
# Here the index starts from 1: CESM2, 2:CanESM5, ..., 6:CNRM-CM6-1
if src in range(len(models)):
	source_selected	= []
	source_selected . append ( models[src])
# --------------------------------------------------------------------
# To run on the terminal
# > python calc_anomalies_ssa.py -src 0 -var nbp  # To calculate anomalies of NBP of CEMS2
# --------------------------------------------------------------------

# Savings a dictionary of common members per source id for anomaly calculations:
# ------------------------------------------------------------------------------

print (" >>> Calculating anomalies for the variable %s and model %s <<< "%(variable_run, source_selected))
concatenated_data = {}
member_ids_common_source			= {}
for s_idx, source_run in enumerate(source_selected):
	concatenated_data[source_run] = {}  	# Creating the nested dic for storing concatenated variable
	filters			= (temp['source_id'] == source_run) & (temp['variable_id'] == variable_run)
#passing the filters to the dataframe
	df_tmp			= temp[filters]

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
			vers_tmp					= np.unique(df_tmp[filters_ver[exp][member]]['version']) # all versions in str
			vers_tmp_int   			 	= np.array([int(v[1:]) for v in vers_tmp])      # all versions in int
			tmp_idx         			= np.where(vers_tmp_int == np.max(vers_tmp_int))# index of max number version
			vers[exp][member] 			= vers_tmp[tmp_idx[0][0]]                       # newest for this run

# Saving the common members to a dict for plotting purposes
	member_ids_common_source[source_run] = member_ids_common	
#Check if the files are in chunks of time
	num_chunk_time	= {}
	for exp in experiments_ids: 
		num_chunk_time[exp]		= len(df_tmp[filters_ver[exp][member]][df_tmp['version'] == vers[exp][member]])  
		print ("Number of chunks of time in of model %s under experiment '%s' are: %d"%(source_run ,exp, num_chunk_time[exp]))

# Creating a dictionary for storing the nc data
	nc_data = {}

	for exp in experiments_ids:
		nc_data [exp] = {}
	
	filepath_areacella 	= {}
	filepath_sftlf		= {}	
	for member in member_ids_common:
		concatenated_data[source_run][member] = {} # Creating the nested dic for storing cont var
# Pointing to the selective files that i need for this plot
		print ("Source ID :%s, Member ID :%s"%(source_run,member))
		filepaths_cont = {}

		for exp in experiments_ids:
			filepaths_cont[exp] = []
			member_id_tmp	= member
			nc_data [exp][member_id_tmp] = {}	# reading members separately per experiment

# This tmp file is just for fx
			filepath_tmp	= "/".join(np.array(df_ver_tmp[exp][member][filters_ver[exp][member]][df_ver_tmp[exp][member]['version'] == vers[exp][member] ].iloc[0]))
			
			if num_chunk_time[exp] >= 1:
				
				filepath_ar = np.array(df_ver_tmp[exp][member][filters_ver[exp][member]][df_ver_tmp[exp][member]['version'] == vers[exp][member]]) 
				for chunk_idx in range(num_chunk_time[exp]):
					filepaths_cont[exp].append (cmip6_filepath_head + "/".join(filepath_ar[chunk_idx]))
				
			"""
#The path of the areacella is at a different location so:
			filepath_fx				= re.sub("Lmon|Amon|Emon","fx", cmip6_filepath_head + filepath_tmp) 	# replacing Lmon with fx
			filepath_areacella_tmp	= re.sub("|".join(variable_ids),"areacella"	,filepath_fx)	# replacing var name with areacella
			filepath_sftlf_tmp		= re.sub("|".join(variable_ids),"sftlf"		,filepath_fx)	# replacing var name with sftlf
# the original filepaths have a time duration in it which needs to be replaced for reading the areacella and lf
			time_duration			= '_' + filepath_areacella_tmp.split('/')[-1].split('_')[-1].split('.')[0]
			filepath_areacella [exp]	= re.sub(time_duration,"",filepath_areacella_tmp) 
			filepath_sftlf 	   [exp]	= re.sub(time_duration,"",filepath_sftlf_tmp) 
			"""
			if source_run == 'BCC-CSM2-MR':
				filepath_area   = "/global/homes/b/bharat/extra_cmip6_data/areacella_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc"
				filepath_lf     = "/global/homes/b/bharat/extra_cmip6_data/sftlf_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc"
			else:
				filters_area 	= (temp['variable_id'] == 'areacella') & (temp['source_id'] == source_run)	
				filters_lf 		= (temp['variable_id'] == 'sftlf') & (temp['source_id'] == source_run)	
				filepath_area	= cmip6_filepath_head + "/".join(np.array(temp[filters_area].iloc[0]))
				filepath_lf		= cmip6_filepath_head + "/".join(np.array(temp[filters_lf].iloc[0]))

# The above commented section is the older version of 
# Check chunk_idx nc_data[exp][member_id_tmp][chunk_idx]
		for exp in experiments_ids:
			for chunk_idx in range(num_chunk_time[exp]):
				nc_data[exp][member_id_tmp][chunk_idx] = nc4.Dataset(filepaths_cont[exp][chunk_idx])
		
#nc_data[member_id_tmp] = nc4.MFDataset([filepaths_cont['historical'], filepaths_cont['ssp585']])
		var			= nc_data['historical'][member_id_tmp][0].variables[variable_run]
		lat 		= nc_data['historical'][member_id_tmp][0].variables['lat']
		lon			= nc_data['historical'][member_id_tmp][0].variables['lon']
		time		= nc_data['historical'][member_id_tmp][0].variables['time']
		
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

# Concatenating the variables under consideration
# The aim is to make one variable for the whole time duration from 1850 -- 2100

		for exp in experiments_ids:
			for chunk_idx in range(num_chunk_time[exp]):
				if (exp == 'historical') and (chunk_idx == 0):
					continue
				var_data 		= np.concatenate(  (var_data, 
													nc_data[exp][member_id_tmp][chunk_idx].variables[variable_run][...]),
													axis =0) # units: kg m-2 s-1
				time_datetime	= np.concatenate(  (time_datetime, 
													cftime.num2date(times		= nc_data[exp][member_id_tmp][chunk_idx].variables['time'][...],
																	units		= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].units,
																	calendar	= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].calendar)),
													axis = 0)
				time_bounds_datetime	= np.concatenate(	(time_bounds_datetime, 
															 cftime.num2date(times		= nc_data[exp][member_id_tmp][chunk_idx].variables[time.bounds][...],
																			 units		= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].units,
																			 calendar	= nc_data[exp][member_id_tmp][chunk_idx].variables['time'].calendar))
														,axis = 0)
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
			landfrac		= nc4.Dataset(filepath_sftlf['historical']).variables['sftlf']
		except:
			if source_run == 'BCC-CSM2-MR':
				area	= nc4.Dataset("/global/homes/b/bharat/extra_cmip6_data/areacella_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc").variables['areacella']
				landfrac	= nc4.Dataset("/global/homes/b/bharat/extra_cmip6_data/sftlf_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc").variables['sftlf']
			else:
				area	= nc4.Dataset(filepath_area).variables["areacella"]
				landfrac		= nc4.Dataset(filepath_lf).variables["sftlf"]
#convert "kg m-2 s-1" to "gC"
		if landfrac.units == '%': 
			lf 	= landfrac[...]/100 # converting the land fraction percentage to the fraction.
		else:
			lf = landfrac[...]
		if variable_run in ['gpp','npp','nep','nbp','ra','rh','fFireAll', 'fHarvest', 'fLulccAtmLut','cTotFireLut','fDeforestToAtmos']:
			area_act	= area[...] * lf[...] #area_act (m2) is the effective or actual area of that pixels
			time_days   = [int(time_bounds_floats[i][1]-time_bounds_floats[i][0]) for i in range(time_bounds_floats.shape[0])]  
			time_sec    = np.array(time_days)*24*3600
			vol_m2s     = time_sec[:,np.newaxis,np.newaxis] * area_act  # units vol: m^2*s
			if variable_run in ['fLulccAtmLut', 'cTotFireLut']:
				var_gC		= vol_m2s * np.sum(var_data, axis=1) * 1000 	# gC/mon
			else:
				var_gC		= vol_m2s * var_data * 1000 	# gC/mon
			concatenated_data[source_run][member] = var_gC
		else:
			concatenated_data[source_run][member] = var_data
# Variables that should be saved
# ------------------------------
#Lon bounds
#if np.array_equal (nc_data['ssp585'][member_id_tmp][0].variables['lon_bnds'][...], nc_data['historical'][member_id_tmp][0].variables['lon_bnds'][...]) == True:
#lon_bnds = np.array_equal (nc_data['ssp586'][member_id_tmp][0].variables['lon_bnds'][...]

# Saving the original concatenated files 
# --------------------------------------------------
Save_Concate = 'y'
if Save_Concate in ['y', 'yes', 'Y']:
	for member_run in member_ids_common:
		save_var	= cori_scratch + 'add_cmip6_data/%s/%s/%s/%s/'%(source_run,exp,member_run,variable_run)
		if os.path.isdir(save_var) == False:
			os.makedirs(save_var)
		if variable_run in ['gpp','npp','nep','nbp', 'ra','rh','fFireAll', 
							'fHarvest', 'fLulccAtmLut','fDeforestToAtmos','cTotFireLut']:
			out_filename = save_var + '%s_%s_%s_%s_gC.nc'%(source_run,exp,member_run,variable_run)
		else:
			out_filename = save_var + '%s_%s_%s_%s.nc'%(source_run,exp,member_run,variable_run)
		with nc4.Dataset(out_filename,mode="w") as dset:
			dset        .createDimension( "time",size = time_datetime.size)
			dset        .createDimension( "lat" ,size = lat.size)
			dset        .createDimension( "lon" ,size = lon.size)
			t   =   dset.createVariable(varname = "time" ,datatype = time.datatype, dimensions = ("time"))
			y   =   dset.createVariable(varname = "lat"  ,datatype = lat.datatype, dimensions = ("lat") )
			x   =   dset.createVariable(varname = "lon"  ,datatype = lon.datatype, dimensions = ("lon") )
			z   =   dset.createVariable(varname = variable_run  ,datatype = var.datatype, dimensions = ("time","lat","lon"),fill_value = 1e36)
			t.axis  =   "T"
			x.axis  =   "X"
			y.axis  =   "Y"
			t[...]  =   time_floats
			x[...]  =   lon[...]
			y[...]  =   lat[...]
			z[...]  =   concatenated_data[source_run][member_run]
			z.missing_value =   1e36
			if variable_run in ['gpp','npp','nep','nbp','ra','rh','fFireAll', 'fHarvest', 'fLulccAtmLut','fDeforestToAtmos','cTotFireLut']:
				z.setncattr         ("units",'g mon-1') # [Changed: The variable units are modified]
			else:
				z.setncattr ("units", var.units)
			t.units         =   TIME_units
			t.setncattr         ("calendar",time.calendar)

			# creating the dimension to capture bounds	
			toexclude = ['time','lat','lon'] 	
			for dname, the_dim in nc_data['historical'][member_id_tmp][0].dimensions.items():
				if dname not in toexclude:
					print(dname,the_dim.size)
					dset .createDimension(dname, the_dim.size if not the_dim.isunlimited() else None )

			# creating the variable of time bounds
			v_name	= time.bounds
			varin     = nc_data['historical'][member_id_tmp][0].variables[v_name]
			out_Var   =	dset.createVariable(varname = v_name,
											datatype =  varin.datatype, 
											dimensions =  varin.dimensions)
			out_Var[:]			=	time_bounds_floats
			out_Var.units 		=   TIME_units
			out_Var.setncattr   	("calendar",time.calendar)

			#copying the attributed of "variable_run"
			v_name = variable_run
			toexclude = ['_FillValue','missing_value','units']
			for k in var.ncattrs():
				if k not in toexclude:
					z.setncatts({k:var.getncattr(k)})

			#copying the attributed of "lat"
			v_name = "lat"
			toexclude = ['_FillValue','missing_value']
			for k in lat.ncattrs():
				if k not in toexclude:
					y.setncatts({k:lat.getncattr(k)})

			#copying the attributed of "lon"
			v_name = "lon"
			toexclude = ['_FillValue','missing_value']
			for k in lon.ncattrs():
				if k not in toexclude:
					x.setncatts({k:lon.getncattr(k)})

			#copying the attributed of "time"
			v_name = "time"
			toexclude = ['_FillValue','missing_value','units','calendar']
			for k in time.ncattrs():
				if k not in toexclude:
					t.setncatts({k:time.getncattr(k)})

			# creating the variable "areacella"
			v_name = "areacella"
			out_Var   =	dset.createVariable(varname = v_name,
											datatype =  area.datatype, 
											dimensions =  area.dimensions,
											fill_value = 1e36)
			out_Var[:]	= area[...]
			toexclude = ['_FillValue','missing_value']
			for k in area.ncattrs():
				if k not in toexclude:
					out_Var.setncatts({k:area.getncattr(k)})

			# creating the variable "sftlf"
			v_name = "sftlf"
			out_Var   =	dset.createVariable(varname = v_name,
											datatype =  landfrac.datatype, 
											dimensions =  landfrac.dimensions,
											fill_value = 1e36)
			out_Var[:]	= landfrac[...]
			toexclude = ['_FillValue','missing_value']
			for k in area.ncattrs():
				if k not in toexclude:
					out_Var.setncatts({k:landfrac.getncattr(k)})

			# creating the variables "lat and lon bounds"
			try:
				for v_name in [lat.bounds, lon.bounds]:
					varin	  = nc_data['historical'][member_id_tmp][0].variables[v_name]
					out_Var   =	dset.createVariable(varname = v_name,
												datatype =  varin.datatype, 
												dimensions =  varin.dimensions,
												fill_value = 1e36)
					out_Var.setncatts({k:varin.getncattr(k) for k in varin.ncattrs()})
					out_Var[:] = varin[:]
			except:
				print("\t\t\t **********\tLat and Lon bounds do not exist\t*************")



print ("Only saving the concatenated variables, not running the anomalies")

print (Breakit) # to break the code at this point!
# Concatenated file saved
# -----------------XXX ----

# Until here we get the continuous ts of variables in the units of  gC/month	
# From here on, we work on the path of finding anomalies of gpp and other fluxes
#SSA tool
window  =   120     #for trend removal of 10 years and above
# Parallel Computing task distribution but constaining to non-MPI
# ---------------------------------------------------------------
size = 1 	# non-mpi
local_rank = 0 # non-mpi
local_n     =   int(lat.size/size)
# calculating the range for every parallel process:
begin_idx   =   local_rank*local_n
end_idx     =   begin_idx+local_n
print ("Time Size: ", time_floats.size, " Local rank :", local_n, "no. of lats: ", lat.size, "no. of lons: ", lon.size)
print (type(time_floats.size),type(local_n),type(lon.size))
resids      =   np.zeros((time_datetime.size,local_n,lon.size))
lats_pp     =   lat[...][begin_idx:end_idx]     #lons per processor
loc_idx_n_lats_pp   =   mpi_local_and_global_index(begin_idx = begin_idx,local_n=local_n) #local idx and global lon index per processor
# mask all land fractions equal to zero
lf = np.ma.masked_less_equal(lf[...],0) 
print (" >>> Calculating anomalies for the variable %s and model %s <<< "%(variable_run, source_run))
# Defining the save path for the anomalies:
# ----------------------------------------

#for member_run in [member_ids_common[-1]]: # [changed: to run the last common member]
for member_run in member_ids_common:
#if member_run == 'r2i1p1f1': continue # incase you want to do or not do it by one member_run
	save_ano	= cori_scratch + 'add_cmip6_data/%s/%s/%s/%s/'%(source_run,exp,member_run,variable_run)
	if os.path.isdir(save_ano) == False: #Check if the directory already exists?
		os.makedirs(save_ano)
	np.savetxt ( cori_scratch+ 'add_cmip6_data/common_members/%s_%s_common_members.csv'%(source_run,exp), member_ids_common, delimiter=',',fmt="%s")
	for k,i in loc_idx_n_lats_pp:
		for j in range(lon.size):
			tmp_ts  =   concatenated_data[source_run][member] [:,i,j]
			if lf.mask[i,j]   ==  True:           #mask non land pixels
				resids[:,k,j]           =   np.array([np.nan]*time_datetime.size)
			elif tmp_ts.mask.sum()      == time_datetime.size: # incase all the gpp is masked
				resids[:,k,j]           =   np.array([np.nan]*time_datetime.size)
			elif collections.Counter(tmp_ts)[0] ==  len(tmp_ts):    #where values are all zero print all zeros too
				resids[:,k,j]           =   np.array([0]*time_datetime.size)
			else:
				y                       =   np.array(tmp_ts)
				resids[:,k,j]           =   ssa.GetResidual(y,window)
		resids = np.ma.masked_invalid(resids)
		print (k,i)
		if variable_run in ['gpp','npp','nep','nbp', 'ra','rh','fFireAll', 
							'fHarvest', 'fLulccAtmLut','fDeforestToAtmos','cTotFireLut']:
			out_filename = save_ano + '%s_%s_%s_%s_anomalies_gC.nc'%(source_run,exp,member_run,variable_run)
		else:
			out_filename = save_ano + '%s_%s_%s_%s_anomalies.nc'%(source_run,exp,member_run,variable_run)
#np.savetxt( save_ano + '%s_%s_%s_%s_test.txt'%(source_run,exp,member_run,variable_run), ["The File is getting stored"], fmt='%s') # to check if the files are saved at the desired destination
		with nc4.Dataset(out_filename,mode="w") as dset:
			dset        .createDimension( "time",size = time_datetime.size)
			dset        .createDimension( "lat" ,size = lat.size)
			dset        .createDimension( "lon" ,size = lon.size)
			t   =   dset.createVariable(varname = "time" ,datatype = float, dimensions = ("time"))
			y   =   dset.createVariable(varname = "lat"  ,datatype = float, dimensions = ("lat") )
			x   =   dset.createVariable(varname = "lon"  ,datatype = float, dimensions = ("lon") )
			z   =   dset.createVariable(varname = variable_run  ,datatype = float, dimensions = ("time","lat","lon"),fill_value = 1e36)
			t.axis  =   "T"
			x.axis  =   "X"
			y.axis  =   "Y"
			t[...]  =   time_floats
			x[...]  =   lon[...]
			y[...]  =   lat[...]
			z[...]  =   resids
			z.missing_value =   1e36
			z.standard_name =   "anomalies of "+var.standard_name
			z.setncattr         ("long_name",'Anomalies of '+var.long_name)
			if variable_run in ['gpp','npp','nep','nbp', 'ra','rh','fFireAll', 'fHarvest', 'fLulccAtmLut','fDeforestToAtmos','cTotFireLut']:
				z.setncattr         ("units",'g mon-1') # [Changed: The variable units are modified]
			else:
				z.setncattr ("units", var.units)
			t.units         =   TIME_units
			t.setncattr         ("calendar",time.calendar)
			
			# creating the dimension to capture bounds	
			toexclude = ['time','lat','lon'] 	
			for dname, the_dim in nc_data['historical'][member_id_tmp][0].dimensions.items():
				if dname not in toexclude:
					print(dname,the_dim.size)
					dset .createDimension(dname, the_dim.size if not the_dim.isunlimited() else None )

			# creating the variable of time bounds
			v_name	= time.bounds
			varin     = nc_data['historical'][member_id_tmp][0].variables[v_name]
			out_Var   =	dset.createVariable(varname = v_name,
											datatype =  varin.datatype, 
											dimensions =  varin.dimensions)
			out_Var[:]			=	time_bounds_floats
			out_Var.units 		=   TIME_units
			out_Var.setncattr   	("calendar",time.calendar)

			#copying the attributed of "variable_run"
			v_name = variable_run
			toexclude = ['_FillValue','missing_value','units','standard_name','long_name']
			for k in var.ncattrs():
				if k not in toexclude:
					z.setncatts({k:var.getncattr(k)})

			#copying the attributed of "lat"
			v_name = "lat"
			toexclude = ['_FillValue','missing_value']
			for k in lat.ncattrs():
				if k not in toexclude:
					y.setncatts({k:lat.getncattr(k)})

			#copying the attributed of "lon"
			v_name = "lon"
			toexclude = ['_FillValue','missing_value']
			for k in lon.ncattrs():
				if k not in toexclude:
					x.setncatts({k:lon.getncattr(k)})

			#copying the attributed of "time"
			v_name = "time"
			toexclude = ['_FillValue','missing_value','units','calendar']
			for k in time.ncattrs():
				if k not in toexclude:
					t.setncatts({k:time.getncattr(k)})
			# creating the variable "areacella"
			v_name = "areacella"
			out_Var   =	dset.createVariable(varname = v_name,
											datatype =  area.datatype, 
											dimensions =  area.dimensions,
											fill_value = 1e36)
			out_Var[:]	= area[...]
			toexclude = ['_FillValue','missing_value']
			for k in area.ncattrs():
				if k not in toexclude:
					out_Var.setncatts({k:area.getncattr(k)})
			
			# creating the variable "sftlf"
			v_name = "sftlf"
			out_Var   =	dset.createVariable(varname = v_name,
											datatype =  landfrac.datatype, 
											dimensions =  landfrac.dimensions,
											fill_value = 1e36)
			out_Var[:]	= landfrac[...]
			toexclude = ['_FillValue','missing_value']
			for k in area.ncattrs():
				if k not in toexclude:
					out_Var.setncatts({k:landfrac.getncattr(k)})

			# creating the variables "lat and lon bounds"
			try:
				for v_name in [lat.bounds, lon.bounds]:
					varin	  = nc_data['historical'][member_id_tmp][0].variables[v_name]
					out_Var   =	dset.createVariable(varname = v_name,
												datatype =  varin.datatype, 
												dimensions =  varin.dimensions,
												fill_value = 1e36)
					out_Var.setncatts({k:varin.getncattr(k) for k in varin.ncattrs()})
					out_Var[:] = varin[:]
			except:
				print("\t\t\t **********\tLat and Lon bounds do not exist\t*************")



# Finish Message:
# ---------------
print ("The code ran successfully for model: %s at the local rank: %s"%(source_run, format(local_rank,'03')))

# Creating temp files to check how many slave nodes have responded 
#np.savetxt ( cori_scratch+ 'add_cmip6_data/temp/%s_%s.txt'%(source_run, format(local_rank,'03')),['%s %s'%(source_run, format(local_rank,'03'))],fmt="%s")
	
