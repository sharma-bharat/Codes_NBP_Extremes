"""
Bharat Sharma
python 3.7

Aim: To find the anomalies of one variable
==========================================
The concatenation is not embedded in the code, so it has to be done prior
Note:- This code is designed to calculate the anomalies using the SSA.
	- It is designed for parallel calculations and computed the anomalies for all lons per lat.
	- Hence, the ouput files are  saved with the local rank that are distributed sequentially.
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
from 	mpi4py	import	MPI
import	collections
import 	ssa_code   as ssa
# Reading the dataframe of the selected files
# -------------------------------------------
web_path	= '/project/projectdirs/m2467/www/bharat/'
in_path		= '/global/homes/b/bharat/results/data_processing/'
cmip6_filepath_head = '/global/homes/b/bharat/cmip6_data/CMIP6/'
add_cmip6_path		= '/global/homes/b/bharat/add_cmip6_data/'
cori_scratch	= '/global/cscratch1/sd/bharat/'

# Basic Details:
source_run		= 'CESM2'
variable_run	= 'tasmax'
member_run		= 'r1i1p1f1'
exp 			= 'ssp585'

print (".....")
# Running the file
# > srun -n 192 --mpi=pmi2 python calc_anomalies_ssa_variable.py 

print (" >>> Calculating anomalies for the variable %s and model %s <<< "%(variable_run, source_run))

filepath 	= "/global/homes/b/bharat/extra_cmip6_data/CESM2/"
filename 	= "tasmax_Amon_CESM2_r1i1p1f1_gn_185001-210012.nc"
nc_data 	= nc4.Dataset(filepath+filename)

var			= nc_data.variables[variable_run] 
lat			= nc_data.variables['lat'		] 
lon			= nc_data.variables['lon'		]			 
time		= nc_data.variables['time'		] 
area		= nc_data.variables['area'		]
landfrac	= nc_data.variables['landfrac'	]
landmask	= nc_data.variables['landmask'	]


time_datetime   = cftime.num2date( 	times 	= time[...],
									units 	= time.units,
									calendar= time.calendar) 
time_bounds_datetime = cftime.num2date (times   = nc_data.variables[time.bounds][...],
										units	= time.units,
										calendar= time.calendar)
# Variable Data
var_data	= np.ma.masked_equal(var, var.missing_value )

if landfrac.units == '%':
	landfrac = landfrac/100.
lf = landfrac

#SSA tool
window  =   120     #for trend removal of 10 years and above
# Parallel Computing task distribution
# ------------------------------------
comm        =   MPI.COMM_WORLD
size        =   comm.Get_size() #Number of processors I am asking for
local_rank  =   comm.Get_rank() #Rank of the current processor
# chunk size or delta:
local_n     =   int(lat.size/size)
# calculating the range for every parallel process:
begin_idx   =   local_rank*local_n
end_idx     =   begin_idx+local_n
print ("Time Size: ", time.size, " Local rank :", local_n, "no. of lats: ", lat.size, "no. of lons: ", lon.size)
print (type(time.size),type(local_n),type(lon.size))
resids      =   np.zeros((time_datetime.size,local_n,lon.size))
lats_pp     =   lat[...][begin_idx:end_idx]     #lons per processor
loc_idx_n_lats_pp   =   mpi_local_and_global_index(begin_idx = begin_idx,local_n=local_n) #local idx and global lon index per processor
# mask all land fractions equal to zero
lf = np.ma.masked_less_equal(lf[...],0) 
print (" >>> Calculating anomalies for the variable %s and model %s <<< "%(variable_run, source_run))
# Defining the save path for the anomalies:
# ----------------------------------------

save_ano	= cori_scratch + 'add_cmip6_data/%s/%s/%s/%s/'%(source_run,exp,member_run,variable_run)
if os.path.isdir(save_ano) == False:
	os.makedirs(save_ano)
for k,i in loc_idx_n_lats_pp:
	for j in range(lon.size):
		tmp_ts  =   var [:,i,j]
		if lf.mask[i,j]   ==  True:           #mask non land pixels
			resids[:,k,j]           =   np.array([np.nan]*time_datetime.size)
		elif tmp_ts.mask.sum()      == time_datetime.size: # incase all the gpp is masked
			resids[:,k,j]           =   np.array([np.nan]*time_datetime.size)
		elif collections.Counter(tmp_ts)[0] ==  len(tmp_ts):    #where values are all zero print all zeros too
			resids[:,k,j]           =   np.array([0]*time_datetime.size)
		else:
			y                       =   np.array(tmp_ts)
			resids[:,k,j]           =   ssa.GetResidual(y,window)
			print (i,j)
	resids = np.ma.masked_invalid(resids)
	print (k,i)
	out_filename = save_ano + '%s_%s_%s_%s_anomalies_gC_%s.nc'%(source_run.lower(),exp,member_run,variable_run,format(local_rank,'03'))

	with nc4.Dataset(out_filename,mode="w") as dset:
		dset        .createDimension( "time",size = time_datetime.size)
		dset        .createDimension( "lat" ,size = local_n)
		dset        .createDimension( "lon" ,size = lon.size)
		t   =   dset.createVariable(varname = "time" ,datatype = float, dimensions = ("time"))
		y   =   dset.createVariable(varname = "lat"  ,datatype = float, dimensions = ("lat") )
		x   =   dset.createVariable(varname = "lon"  ,datatype = float, dimensions = ("lon") )
		z   =   dset.createVariable(varname = variable_run  ,datatype = float, dimensions = ("time","lat","lon"),fill_value = 1e36)
		t.axis  =   "T"
		x.axis  =   "X"
		y.axis  =   "Y"
		t[...]  =   time[...]
		x[...]  =   lon[...]
		y[...]  =   lat[i]
		z[...]  =   resids
		z.missing_value =   1e36
		z.standard_name =   "anomalies of "+var.standard_name
		z.units         =   var.units
		z.setncattr         ("long_name",'Anomalies of '+var.long_name)
		x.units         =   lon.units
		x.standard_name =   lon.standard_name
		y.units         =   lat.units
		y.standard_name =   lat.standard_name
		t.units         =   TIME_units
		t.setncattr         ("calendar",time.calendar)
		t.setncattr         ("bounds",time.bounds)
		t.standard_name =   time.standard_name

# Finish Message:
# ---------------
print ("The code ran successfully for model: %s at the local rank: %s"%(source_run, format(local_rank,'03')))

# Creating temp files to check how many slave nodes have responded 
np.savetxt ( cori_scratch+ 'add_cmip6_data/temp/%s_%s.txt'%(source_run, format(local_rank,'03')),['%s %s'%(source_run, format(local_rank,'03'))],fmt="%s")
	


