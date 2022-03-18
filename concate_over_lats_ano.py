# Bharat Sharma
# Python 3.7
"""
	The predecessor file's (calc_ssa_anomalies.py) output will be concatenated with this file
	The individual files will be deleted as well
"""

import 	netCDF4	as nc4
import	numpy 	as np
import 	pandas	as pd
import	glob
import	os

# Reading the dataframe of the selected files
# -------------------------------------------
cori_scratch    = '/global/cscratch1/sd/bharat/' 	# where the anomalies per slave rank are saved
in_path     = '/global/homes/b/bharat/results/data_processing/' # to read the filters
cmip6_filepath_head = '/global/homes/b/bharat/cmip6_data/CMIP6/' 

# The spreadsheet with all the available data of cmip 6
# -----------------------------------------------------
df_files	= pd.read_csv(in_path + 'df_data_selected.csv')
temp		= df_files.copy(deep = True)       

hierarcy_str= ['activity_id','institution_id','source_id','experiment_id','member_id','table_id','variable_id','grid_label','version','filenames']

# List of models that are selected as of now
source_selected = ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1'] # for GPP 

# Select which model you want to run:
# ===================================
source_selected    = ['BCC-CSM2-MR'    ] # Ran
source_selected    = ['CNRM-CM6-1'     ] # Ran
source_selected    = ['CNRM-ESM2-1'    ]
source_selected    = ['IPSL-CM6A-LR'   ] # Ran
source_selected    = ['CanESM5'        ] # Ran
source_selected    = ['CESM2'          ] # Ran

import  argparse
parser  = argparse.ArgumentParser()
parser. add_argument ('--source_idx', '-src',   help="The model name",  type = int, default = 0)
args = parser.parse_args()
src     = int (args.source_idx)
models  = ['CESM2','CanESM5','IPSL-CM6A-LR','CNRM-ESM2-1','BCC-CSM2-MR','CNRM-CM6-1']                                        
if src in range(len(models)):
	source_selected = []
	source_selected . append ( models[src])

# inputs
source_run	= source_selected[0]
exp			= 'ssp585'
variable_run    = 'fFireAll'

# -----------------------------------------------------------
# To run on the terminal
# > python concate_over_lats_ano.py -src 0
#		e.g. src = 0 is for CESM2
# -----------------------------------------------------------


# Common members per model
common_members		= {}
common_members [source_run] = pd.read_csv	(cori_scratch + 'add_cmip6_data/common_members/%s_%s_common_members.csv'%(source_run,exp),
										 	 header=None)
#member_run	= common_members[source_run].loc[1,0]

for member_run in common_members[source_run].loc[:,0]:
	try:
		print ("Source : %s \t Member : %s"%(source_run, member_run))
# Readiing the saved anomalies 
# ----------------------------
		saved_ano	= cori_scratch + 'add_cmip6_data/%s/%s/%s/%s/'%(source_run,exp,member_run,variable_run)
		files_ano	= glob.glob (saved_ano + '%s_*.nc'%(source_run.lower()) )

# Reading the main variable file:
# -------------------------------
		filters	= (   (temp['source_id'] == source_run) 
					& (temp['variable_id'] == variable_run) 
					& (temp['member_id'] ==member_run) 
					& (temp['experiment_id'] == 'ssp585') )

# This is the original file name of the variable run
		var_filename = temp[filters].iloc[:,-1].iloc[0]

# using this file name as a filter we can find the complete path of the variable
		nc_data	= nc4.Dataset(cmip6_filepath_head+"/".join(temp[temp['filenames'] == var_filename].iloc[0]))
		#nc_data = nc4.Dataset('/global/homes/b/bharat/extra_cmip6_data/CESM2/tasmax_Amon_CESM2_r1i1p1f1_gn_185001-210012.nc') # [changed for 'taxmax'] 
		var		= nc_data.variables[variable_run]  
		lat		=  nc_data.variables['lat']

# in order to read the anomalies variable and time dimension let us read one of the local anomaly file
		var_ano_tmp	= nc4.Dataset(files_ano[0]).variables[variable_run]
		lon_ano		= nc4.Dataset(files_ano[0]).variables['lon']
		time_ano	= nc4.Dataset(files_ano[0]).variables['time']

# Hence, the shape of anomalies should be "time_ano","lat","lon_ano"
# Creating a new 3d array to store the anomalies in one file
# -----------------------------------------------------------------

# Sorting the file names
		text		= files_ano[0][:-6] 
		ll			= [format(i,'003') for i in range(len(files_ano))]
		filenames = [text+ll[i]+'.nc' for i in range(len(ll))]

# initializing with the empty dataframe
		anomalies	= np.ma.masked_all ((time_ano.size, lat.size, lon_ano.size))

		i= 0
		for fname in filenames:
			lat_ano_i	=	nc4.Dataset(fname).variables['lat']
			lat_tmp_size	= lat_ano_i.size

			anomalies[:,i:i+lat_tmp_size,:] = nc4.Dataset(fname,'r').variables[variable_run][...]
			i	= i + lat_tmp_size

# Saving the anomalies as one file at the same place
# --------------------------------------------------
		if variable_run in ['gpp','npp','nep','nbp','fFireAll']:
			out_filename	= saved_ano + '%s_%s_%s_%s_anomalies_gC.nc'%(source_run,exp,member_run,variable_run)
		elif variable_run in ['pr','tas','tasmax','mrso']:
			out_filename	= saved_ano + '%s_%s_%s_%s_anomalies.nc'%(source_run,exp,member_run,variable_run)
			
		with nc4.Dataset(out_filename,mode="w") as dset:
			dset        .createDimension( "time",size = time_ano.size)
			dset        .createDimension( "lat" ,size = lat.size)
			dset        .createDimension( "lon" ,size = lon_ano.size)
			t   =   dset.createVariable(varname = "time" ,datatype = float, dimensions = ("time"))
			y   =   dset.createVariable(varname = "lat"  ,datatype = float, dimensions = ("lat") )
			x   =   dset.createVariable(varname = "lon"  ,datatype = float, dimensions = ("lon") )
			z   =   dset.createVariable(varname = variable_run  ,datatype = float, dimensions = ("time","lat","lon"),fill_value    = 1e36)
			t.axis  =   "T"
			x.axis  =   "X"
			y.axis  =   "Y"
			t[...]  =   time_ano[...]
			x[...]  =   lon_ano [...]
			y[...]  =   lat[...]
			z[...]  =   anomalies
			z.missing_value =   1e36
			z.standard_name =   var_ano_tmp.standard_name
			z.units         =   var_ano_tmp.units
			z.setncattr         ("long_name",var_ano_tmp.long_name)
			x.units         =   lon_ano.units
			x.standard_name =   lon_ano.standard_name
			y.units         =   lat.units
			y.standard_name =   lat.standard_name
			t.units         =   time_ano.units
			t.setncattr         ("calendar",time_ano.calendar)
			t.setncattr         ("bounds",time_ano.bounds)
			t.standard_name =   time_ano.standard_name

		for fname in filenames:
			os.remove(fname)
		print (a)
	except:
		print ("Some members are different")
# Finish Message:
# ---------------
print ("The code ran successfully for model: %s "%(source_run))
