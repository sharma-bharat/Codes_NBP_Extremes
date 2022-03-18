# Bharat Sharma
# python 3.7
"""
	To calculate the extremes of the carbon fluxes based on carbon flux anomalies in gC.
	The code is fairly flexible to pass multiple filters to the code.

	Output:
		* Saving the binarys of extremes
		* Saving the TCE binaries at multiple lags [0-4 months)

"""
import 	os
import 	netCDF4		as nc4
import 	numpy 		as np
import	pandas		as pd
import	datetime	as dt
import	seaborn		as sns
import	argparse
from 	scipy 		import stats
from 	functions	import time_dim_dates, index_and_dates_slicing, norm, geo_idx, patch_with_gaps_and_eventsize
""" Arguments to input while running the python file
--percentile (-per)	: percentile under consideration
                  	  looking at the negative/positive  tail of gpp events: {eg.1,5,10,90,95,99}
--th_type			: Thresholds can be computed at each tail i.e. 'ind' or 'common'. 
					  'common' means that total number of events greater that the modulus of anomalies represent 'per' percentile
--sources (-src)	: the models that you want to analyze, separated by hyphens or 'all' for all the models
--variable (-var)	: the variable to analyze gpp/nep/npp/nbp
--window (wsize)	: time window size in years
# Running: run calc_extremes.py -src cesm -var gpp
"""
print ("Last edit on May 08, 2020")
# The abriviation of the models that will be analyzed:
source_code	= { 'cesm'	: 'CESM2',
				'can'	: 'CanESM5',
				'ipsl'	: 'IPSL-CM6A-LR',
				'bcc'	: 'BCC-CSM2-MR',
				'cnrn-e': 'CNRM-ESM2-1',
				'cnrn-c': 'CNRM-CM6-1' }

parser  = argparse.ArgumentParser()
parser.add_argument('--percentile'      ,'-per'         , help = "Threshold Percentile?"            , type= int,    default= 5          )
parser.add_argument('--th_type'      	,'-th'        	, help = "Threshold Percentile?"            , type= str,    default= 'common'   )
parser.add_argument('--sources'			,'-src'			, help = "Which model(s) to analyse?"		, type= str,	default= 'all'		)
parser.add_argument('--variable'        ,'-var'         , help = "variable? gpp/npp/nep/nbp,,,,"    , type= str,    default= 'gpp'      )
parser.add_argument('--window'          ,'-wsize'       , help = "window size (25 years)?"          , type= int,    default= 25         )
args = parser.parse_args()

# The inputs:
per			= int	(args.percentile)
th_type		= str	(args.th_type)
src			= str	(args.sources)
variable_run= str   (args.variable)
window		= int	(args.window)


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

#running :  run calc_extremes.py -per 5 -var nbp -src cesm

# Reading the dataframe of the selected files 
# -------------------------------------------
cori_scratch    = '/global/cscratch1/sd/bharat/'    # where the anomalies per slave rank are saved   
in_path     	= '/global/homes/b/bharat/results/data_processing/' # to read the filters
#cmip6_filepath_head = '/global/homes/b/bharat/cmip6_data/CMIP6/' 
cmip6_filepath_head = '/global/cfs/cdirs/m3522/cmip6/CMIP6/' 
#web_path    	= '/project/projectdirs/m2467/www/bharat/'
web_path    	= '/global/homes/b/bharat/results/web/'
# exp is actually 'historical + ssp585' but saved as 'ssp585'
exp	= 'ssp585'

# Common members per model
# ------------------------
common_members		= {}
for source_run in source_selected:
	common_members [source_run] = pd.read_csv	(cori_scratch + 'add_cmip6_data/common_members/%s_%s_common_members.csv'%(source_run,exp),
											 	 header=None).iloc[:,0]

# The spreadsheet with all the available data of cmip 6
# -----------------------------------------------------
df_files    = pd.read_csv(in_path + 'df_data_selected.csv')
temp        = df_files.copy(deep = True)

# Saving the path of area and lf
filepath_areacella  = {} 
filepath_sftlf      = {}

for s_idx, source_run in enumerate(source_selected):
	filters         = (temp['source_id'] == source_run) & (temp['variable_id'] == variable_run)	# original Variable
	filters_area    = (temp['source_id'] == source_run) & (temp['variable_id'] == 'areacella')  # areacella
	filters_lf      = (temp['source_id'] == source_run) & (temp['variable_id'] == 'sftlf')		# land fraction
#passing the filters to the dataframe
	df_tmp          = temp[filters]
	df_tmp_area     = temp[filters_area]
	df_tmp_lf       = temp[filters_lf]

	for member_run in common_members [source_run]:		
		if source_run == 'BCC-CSM2-MR':
			filepath_area   = "/global/homes/b/bharat/extra_cmip6_data/areacella_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc"
			filepath_lf     = "/global/homes/b/bharat/extra_cmip6_data/sftlf_fx_BCC-CSM2-MR_hist-resIPO_r1i1p1f1_gn.nc"
		else:
			filters_area    = (temp['variable_id'] == 'areacella') & (temp['source_id'] == source_run)
			filters_lf      = (temp['variable_id'] == 'sftlf') & (temp['source_id'] == source_run)
			filepath_area   = cmip6_filepath_head + "/".join(np.array(temp[filters_area].iloc[-1]))
			filepath_lf     = cmip6_filepath_head + "/".join(np.array(temp[filters_lf].iloc[-1]))
		filepath_areacella	[source_run] = filepath_area
		filepath_sftlf		[source_run] = filepath_lf

# Extracting the area and land fractions of different models
# ==========================================================
data_area	= {}
data_lf	= {}
for source_run in source_selected:
	data_area [source_run] = nc4.Dataset (filepath_areacella[source_run]) . variables['areacella']
	data_lf	  [source_run] = nc4.Dataset (filepath_sftlf	[source_run]) . variables['sftlf']

# Saving the paths of anomalies
# hier. :  source_id > member_id
# ------------------------------------
paths				= {}
for source_run in source_selected:
	paths[source_run] = {}
for source_run in source_selected:
	for member_run in common_members [source_run]:
		saved_ano   = cori_scratch + 'add_cmip6_data/%s/%s/%s/%s/'%(source_run,exp,member_run,variable_run)
		paths[source_run][member_run] = saved_ano
		del saved_ano

# Reading and saving the data:
# ----------------------------
nc_ano				= {}
for source_run in source_selected:
	nc_ano[source_run] = {}
for source_run in source_selected:
	for member_run in common_members [source_run]:
		nc_ano[source_run][member_run] = nc4.Dataset(paths[source_run][member_run] + '%s_%s_%s_%s_anomalies_gC.nc'%(source_run,exp,member_run,variable_run))


# Arranging Time Array for plotting and calling
# --------------------------------------------
win_len     = 12 * window             #number of months in window years
total_years	= 251 #years from 1850 to 2100
total_months= total_years * 12

dates_ar    = time_dim_dates( base_date = dt.date(1850,1,1), 
							  total_timestamps = 3012 )
start_dates = np.array(	[dates_ar[i*win_len] for i in range(int(total_months/win_len))])    #list of start dates of 25 year window
end_dates   = np.array( [dates_ar[i*win_len+win_len -1] for i in range(int(total_months/win_len))]) #list of end dates of the 25 year window

idx_yr_2100 = 3012 # upper open index 2100 from the year 1850 if the data is monthly i.e. for complete TS write ts[:3012]
idx_yr_2014 = 1980 # upper open index 2014 from the year 1850 if the data is monthly i.e. for complete TS write ts[:1980]
idx_yr_2099 = 3000 # upper open index 2099 from the year 1850 if the data is monthly i.e. for complete TS write ts[:3000]

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

# The saving the results in a dictionary
# --------------------------------------
Results			= {}
for source_run in source_selected:
	Results[source_run] = {}
	for member_run in common_members [source_run]:
		Results[source_run][member_run] = {}

# Calculation of thresholds (rth percentile at each tail):
# ------------------------------------------------------------
def Threshold_and_Binary_Ar(data = nc_ano[source_run][member_run].variables[variable_run][...], per = per):
	"""
	In this method the 1 percentile threshold is calculated are both tails of the pdf of anomalies...
	i.e. the same number of values are selected on either tails.

	returns the global percentile based thresholds and binary arrays of consecutive windows

	Parameters:
	-----------
	data : The anomalies whose threshold you want to calculate

	Universal:
	---------
	start_dates, idx_dates_win, per
	Returns:
	--------
	threshold_neg:  the threshold for negative extremes; size = # windows
	threshold_pos:  the threshold for positive extremes; size = # windows
	bin_ext_neg:    the binary array 1's are extremes based on the threshold_neg; shape = same as data
	bin_ext_pos:    the binary array 1's are extremes based on the threshold_pos; shape = same as data
	"""
	thresholds_1= []    #thresholds for consecutive windows of defined size for a 'per' percentile
	thresholds_2= []    #thresholds for consecutive windows of defined size for a '100-per' percentile
	bin_ext_neg = np.ma.zeros((data.shape)) #3d array to capture the True binaray extmalies w.r.t. gpp loss events
	bin_ext_pos = np.ma.zeros((data.shape)) #3d array to capture the True binaray extmalies w.r.t. gpp gain events

	for i in range(len(start_dates)):
		ano_loc         = data[idx_dates_win[i][0]:idx_dates_win[i][-1]+1,:,:]
		threshold_loc_1 = np.percentile(ano_loc[ano_loc.mask == False],per) # calculation of threshold for the local anomalies
		thresholds_1    . append(threshold_loc_1)
		threshold_loc_2 = np.percentile(ano_loc[ano_loc.mask == False],(100-per))
		thresholds_2    . append(threshold_loc_2)
# Binary arrays:
		if per <=50:
			bin_ext_neg[idx_dates_win[i][0]:idx_dates_win[i][-1]+1,:,:] = ano_loc < threshold_loc_1
			bin_ext_pos[idx_dates_win[i][0]:idx_dates_win[i][-1]+1,:,:] = ano_loc > threshold_loc_2
		else:
			bin_ext_pos[idx_dates_win[i][0]:idx_dates_win[i][-1]+1,:,:] = ano_loc > threshold_loc_1
			bin_ext_neg[idx_dates_win[i][0]:idx_dates_win[i][-1]+1,:,:] = ano_loc < threshold_loc_2
# Thresholds for consecutive windows:
		if per < 50:
			threshold_neg = np.ma.array(thresholds_1)
			threshold_pos = np.ma.array(thresholds_2)
		elif per > 50:
			threshold_neg = np.ma.array(thresholds_2)
			threshold_pos = np.ma.array(thresholds_1)

	return threshold_neg, threshold_pos, bin_ext_neg, bin_ext_pos

# Calculation of thresholds (rth percentile combines for both tails):
# ------------------------------------------------------------
def Threshold_and_Binary_Ar_Common(data = nc_ano[source_run][member_run].variables[variable_run][...], per = per ):
	"""
	In this method the rth percentile threshold is calculated at sum of both tails of the pdf of anomalies...
	i.e. total number of elements on left and right tail make up for rth percentile (jakob 2014, anex A2)...
	This can be done by taking a modulus of anomalies and then calcuate the rth percentile th = q

	Negative extremes: anomalies < -q
	Positive extremes: anomalies >  q


	Returns the global percentile based thresholds and binary arrays of consecutive windows

	Parameters:
	-----------
	data : The anomalies whose threshold you want to calculate

	Universal:
	---------
	start_dates, idx_dates_win, per
	Returns:
	--------
	threshold_neg:  the threshold for negative extremes; size = # windows
	threshold_pos:  the threshold for positive extremes; size = # windows
	bin_ext_neg:    the binary array 1's are extremes based on the threshold_neg; shape = same as data
	bin_ext_pos:    the binary array 1's are extremes based on the threshold_pos; shape = same as data
	"""
	thresholds_p= []    #thresholds for consecutive windows of defined size for a 'per' percentile
	thresholds_n= []    #thresholds for consecutive windows of defined size for a '100-per' percentile
	bin_ext_neg = np.ma.zeros((data.shape)) #3d array to capture the True binaray extmalies w.r.t. gpp loss events
	bin_ext_pos = np.ma.zeros((data.shape)) #3d array to capture the True binaray extmalies w.r.t. gpp gain events

	assert per <50, "Percentile must be less than 50" 

	for i in range(len(start_dates)):
		ano_loc         = data[idx_dates_win[i][0]:idx_dates_win[i][-1]+1,:,:]
		threshold_loc 	= np.percentile(np.abs(ano_loc[ano_loc.mask == False]), (100-per) ) # calculation of threshold for the local anomalies
# The (100-per) is used because after taking the modulus negative extremes fall along positive on the right hand 

		thresholds_p	. append(threshold_loc)
		thresholds_n	. append(-threshold_loc)

# Binary arrays:
# --------------
		bin_ext_neg[idx_dates_win[i][0]:idx_dates_win[i][-1]+1,:,:] = ano_loc < -threshold_loc
		bin_ext_pos[idx_dates_win[i][0]:idx_dates_win[i][-1]+1,:,:] = ano_loc >  threshold_loc
# Thresholds for consecutive windows:
# -----------------------------------
	threshold_neg = np.ma.array(thresholds_n)
	threshold_pos = np.ma.array(thresholds_p)

	return threshold_neg, threshold_pos, bin_ext_neg, bin_ext_pos



limits = {}
limits ['min'] = {}
limits ['max'] = {}
limits ['min']['th_pos'] = 0
limits ['max']['th_pos'] = 0
limits ['min']['th_neg'] = 0
limits ['max']['th_neg'] = 0

p =0
for source_run in source_selected:
	for member_run in common_members [source_run]:
		p = p+1
# threshold at each tail
		if th_type == 'ind':
			A,B,C,D	= Threshold_and_Binary_Ar(data = nc_ano[source_run][member_run].variables[variable_run][...], per = per )
		if th_type == 'common':
			A,B,C,D	= Threshold_and_Binary_Ar_Common(data = nc_ano[source_run][member_run].variables[variable_run][...], per = per )
		Results[source_run][member_run]['th_neg'] 		= A
		Results[source_run][member_run]['th_pos']		= B
		Results[source_run][member_run]['bin_ext_neg']	= C
		Results[source_run][member_run]['bin_ext_pos']	= D
		Results[source_run][member_run]['ts_th_neg']	= np.array([np.array([A[i]]*win_len) for i in range(len(A))]).flatten()     
		Results[source_run][member_run]['ts_th_pos']	= np.array([np.array([B[i]]*win_len) for i in range(len(B))]).flatten()



# Checking
		if p%3	== 0:	print ("Calculating Thresholds ......")
		elif p%3 == 1:	print ("Calculating Thresholds ....")
		else:			print ("Calculating Thresholds ..")
del A,B,C,D
# Saving the binary data
# ----------------------
save_binary_common = 'n'

if save_binary_common in ['y','yy','Y','yes']:
	"""
	To save the binary matrix of the so that the location and duration of the extremes can be identified.

	If you want to save the binary matrix of extremes as nc files
	this was done so that this coulld be used as input the attribution analysis
	"""
	for source_run in source_selected:
		for member_run in common_members [source_run]:
			path_TCE = cori_scratch + 'add_cmip6_data/%s/%s/%s/%s_TCE/'%(source_run,exp,member_run,variable_run)
			# Check if the directory 'path_TCE' already exists? If not, then create one:
			if os.path.isdir(path_TCE) == False:
				os.makedirs(path_TCE)	
			for ext_type in ['neg','pos']:
				print("Saving the binary matrix for %s,%s,%s"%(source_run,member_run,ext_type))
				with nc4.Dataset( path_TCE + '%s_%s_bin_%s.nc'%(source_run,member_run,ext_type), mode = 'w') as dset:
					dset        .createDimension( "time" ,size = nc_ano[source_run][member_run].variables['time'].size)
					dset        .createDimension( "lat" ,size = nc_ano[source_run][member_run].variables['lat'].size)
					dset        .createDimension( "lon" ,size = nc_ano[source_run][member_run].variables['lon'].size)
					t   =   dset.createVariable(varname = "time" ,datatype = float, dimensions = ("time"), fill_value = 1e+36)
					x   =   dset.createVariable(varname = "lon"  ,datatype = float, dimensions = ("lon") , fill_value = 1e+36)
					y   =   dset.createVariable(varname = "lat"  ,datatype = float, dimensions = ("lat") , fill_value = 1e+36)
					z   =   dset.createVariable(varname = variable_run +'_bin'  ,datatype = float, dimensions = ("time","lat","lon"),fill_value = 1e+36) #varible = gpp_bin_ext
					t.axis  =   "T"
					x.axis  =   "X"
					y.axis  =   "Y"
					t[...]  =   nc_ano[source_run][member_run].variables['time'] [...]
					x[...]  =   nc_ano[source_run][member_run].variables['lon'][...]
					y[...]  =   nc_ano[source_run][member_run].variables['lat'][...]
					z[...]	= 	Results[source_run][member_run]['bin_ext_%s'%ext_type]
					z.missing_value = 1e+36
					z.stardard_name = variable_run+" binarys for %s extremes based on %dth percentile"%(ext_type,per)
					z.units         = "0,1"
					x.units         = nc_ano[source_run][member_run].variables['lon'].units
					x.missing_value = 1e+36
					x.setncattr       ("standard_name",nc_ano[source_run][member_run].variables['lon'].standard_name)
					y.units         = nc_ano[source_run][member_run].variables['lat'].units
					y.missing_value = 1e+36
					y.setncattr       ("standard_name",nc_ano[source_run][member_run].variables['lat'].standard_name)
					t.units         = nc_ano[source_run][member_run].variables['time'].units
					t.setncattr       ("calendar", nc_ano[source_run][member_run].variables['time'].calendar)
					t.setncattr       ("standard_name", nc_ano[source_run][member_run].variables['time'].standard_name)
					t.missing_value = 1e+36



# TCE: Calculations:
# ------------------
lags_TCE = np.asarray([0,1,2,3,4], dtype = int)
def Binary_Mat_TCE_Win (bin_ar, win_start_year=2000,lags = lags_TCE, land_frac=  data_lf [source_run]):
	"""
	Aim:
	----
	To save the binary matrix of the Time Continuous Extremes(TCEs) so that the location and duration of the extremes can be identified.
	
	Returns:
	--------
	bin_TCE_01s: are the binary values of extreme values in a TCE only at qualified locations with gaps ( actual as value 0) [hightlight extreme values]
	bin_TCE_1s : are the binary values of extreme values in a TCE only at qualified locations with gaps ( 0 replaced with value 1) [selecting full TCE with only 1s]
	bin_TCE_len : are the len of TCE extreme events, the length  of TCE is captured at the trigger locations
	shape : These matrix are of shape (5,300,192,288) i.e. lags(0-4 months), time(300 months or 25 years {2000-24}), lat(192) and lon(288).
	"""
	from functions import create_seq_mat
	for i,date in enumerate(start_dates):
		if date.year in [win_start_year]:
			start_yr_idx = i
	data = bin_ar[start_yr_idx*win_len: (start_yr_idx+1)*win_len]
	del bin_ar
	
	bin_TCE_1s  = np.ma.zeros((len(lags), data.shape[0],data.shape[1],data.shape[2]))
	bin_TCE_01s = np.ma.zeros((len(lags), data.shape[0],data.shape[1],data.shape[2]))
	bin_TCE_len = np.ma.zeros((len(lags), data.shape[0],data.shape[1],data.shape[2]))

	for lag in lags:
		for lat_i in range( data.shape[1] ):
			for lon_i in range( data.shape[2] ):
				if land_frac[...][lat_i,lon_i] != 0:
					#print lag, lat_i, lon_i
					try:
						tmp = patch_with_gaps_and_eventsize (data[:,lat_i,lon_i], max_gap =2, min_cont_event_size=3, lag=lag)
						for idx, trig in enumerate (tmp[1]):
							bin_TCE_01s [lag, trig:trig+len(tmp[0][idx]), lat_i, lon_i] = tmp[0][idx]
							bin_TCE_1s  [lag, trig:trig+len(tmp[0][idx]), lat_i, lon_i] = np.ones(tmp[0][idx].shape)
							bin_TCE_len [lag, trig, lat_i, lon_i]                       = np.sum(np.ones(tmp[0][idx].shape))
					except:
						bin_TCE_01s[lag, :, lat_i, lon_i]  = np.ma.masked_all(data.shape[0])
						bin_TCE_1s [lag, :, lat_i, lon_i]  = np.ma.masked_all(data.shape[0])
						bin_TCE_len[lag, :, lat_i, lon_i]  = np.ma.masked_all(data.shape[0])
				else:
					bin_TCE_01s[lag, :, lat_i, lon_i]  = np.ma.masked_all(data.shape[0])
					bin_TCE_1s [lag, :, lat_i, lon_i]  = np.ma.masked_all(data.shape[0])
					bin_TCE_len[lag, :, lat_i, lon_i]  = np.ma.masked_all(data.shape[0])
	return bin_TCE_01s, bin_TCE_1s, bin_TCE_len


all_win_start_years = np.arange(1850,2100,25)

# To do TCE analysis for all windows
win_start_years = np.arange(1850,2100,25)

# To check only for win starting at 2000
#win_start_years = [2000] # Testing with the year 2000-24 dataset first

save_TCE_binary = 'n'

if save_TCE_binary in ['y','yy','Y','yes']:
	"""
	To save the binary matrix of the Time Continuous Extremes(TCEs) so that the location and duration of the extremes can be identified.

	If you want to save the binary matrix of extremes as nc files
	this was done so that this coulld be used as input the attribution analysis
	"""
	for start_yr in win_start_years:
		win_idx = np.where( all_win_start_years == start_yr)[0][0]
		for source_run in  source_selected:
			for member_run in common_members [source_run]:
				Binary_Data_TCE = {} # Dictionary to save negative and positive Binary TCEs 
				Binary_Data_TCE ['neg'] = {}
				Binary_Data_TCE ['pos'] = {}

				bin_neg	= Results[source_run][member_run]['bin_ext_neg']
				bin_pos	= Results[source_run][member_run]['bin_ext_pos']
				# Starting with Negative TCEs first
				# ---------------------------------
				
				Binary_Data_TCE ['neg']['bin_TCE_01s'], Binary_Data_TCE ['neg']['bin_TCE_1s'], Binary_Data_TCE ['neg']['bin_TCE_len'] = Binary_Mat_TCE_Win (bin_ar = bin_neg, win_start_year = start_yr,lags = lags_TCE, land_frac=  data_lf [source_run])
				Binary_Data_TCE ['pos']['bin_TCE_01s'], Binary_Data_TCE ['pos']['bin_TCE_1s'], Binary_Data_TCE ['pos']['bin_TCE_len'] = Binary_Mat_TCE_Win (bin_ar = bin_pos, win_start_year = start_yr,lags = lags_TCE, land_frac=  data_lf [source_run])
				
				path_TCE = cori_scratch + 'add_cmip6_data/%s/%s/%s/%s_TCE/'%(source_run,exp,member_run,variable_run)
				# Check if the directory 'path_TCE' already exists? If not, then create one:
				if os.path.isdir(path_TCE) == False:
					os.makedirs(path_TCE)	
				
				for ext_type in ['neg','pos']:
					print("Saving the 01 TCE for %s,%s,%d,%s"%(source_run,member_run,start_yr,ext_type))
					with nc4.Dataset( path_TCE + 'bin_TCE_01s_'+ext_type+'_%d.nc'%start_yr, mode = 'w') as dset:
						dset        .createDimension( "lag",size = lags_TCE.size)
						dset        .createDimension( "time",size = win_len)
						dset        .createDimension( "lat" ,size = nc_ano[source_run][member_run].variables['lat'].size)
						dset        .createDimension( "lon" ,size = nc_ano[source_run][member_run].variables['lon'].size)
						w   =   dset.createVariable(varname = "lag"  ,datatype = float, dimensions = ("lag") , fill_value = 1e+36)
						t   =   dset.createVariable(varname = "time" ,datatype = float, dimensions = ("time"), fill_value = 1e+36)
						x   =   dset.createVariable(varname = "lon"  ,datatype = float, dimensions = ("lon") , fill_value = 1e+36)
						y   =   dset.createVariable(varname = "lat"  ,datatype = float, dimensions = ("lat") , fill_value = 1e+36)
						z   =   dset.createVariable(varname = variable_run +'_TCE_01s'  ,datatype = float, dimensions = ("lag","time","lat","lon"),fill_value = 1e+36) #varible = gpp_bin_ext
						w.axis  =   "T"
						t.axis  =   "T"
						x.axis  =   "X"
						y.axis  =   "Y"
						w[...]  =   lags_TCE
						t[...]  =   nc_ano[source_run][member_run].variables['time'] [...][win_idx * win_len : (win_idx+1)*win_len]
						x[...]  =   nc_ano[source_run][member_run].variables['lon'][...]
						y[...]  =   nc_ano[source_run][member_run].variables['lat'][...]
						z[...]	= 	Binary_Data_TCE [ext_type]['bin_TCE_01s']
						z.missing_value = 1e+36
						z.stardard_name = variable_run+" binary TCE (01s) matrix for 25 years starting at the  year %d"%start_yr
						z.units         = "0,1"
						x.units         =   nc_ano[source_run][member_run].variables['lon'].units
						x.missing_value =   1e+36
						x.setncattr         ("standard_name",nc_ano[source_run][member_run].variables['lon'].standard_name)
						y.units         =   nc_ano[source_run][member_run].variables['lat'].units
						y.missing_value =   1e+36
						y.setncattr         ("standard_name",nc_ano[source_run][member_run].variables['lat'].standard_name)
						t.units         =   nc_ano[source_run][member_run].variables['time'].units
						t.setncattr         ("calendar", nc_ano[source_run][member_run].variables['time'].calendar)
						t.setncattr         ("standard_name", nc_ano[source_run][member_run].variables['time'].standard_name)
						t.missing_value =   1e+36
						w.units         =   "month"
						w.setncattr         ("standard_name","lags in months")
						w.missing_value =   1e+36


					print("Saving the 1s TCE for %s,%s,%d,%s"%(source_run,member_run,start_yr,ext_type))
					with nc4.Dataset( path_TCE + 'bin_TCE_1s_'+ext_type+'_%d.nc'%start_yr, mode = 'w') as dset:
						dset        .createDimension( "lag",size = lags_TCE.size)
						dset        .createDimension( "time",size = win_len)
						dset        .createDimension( "lat" ,size = nc_ano[source_run][member_run].variables['lat'].size)
						dset        .createDimension( "lon" ,size = nc_ano[source_run][member_run].variables['lon'].size)
						w   =   dset.createVariable(varname = "lag"  ,datatype = float, dimensions = ("lag") , fill_value = 1e+36)
						t   =   dset.createVariable(varname = "time" ,datatype = float, dimensions = ("time"), fill_value = 1e+36)
						x   =   dset.createVariable(varname = "lon"  ,datatype = float, dimensions = ("lon") , fill_value = 1e+36)
						y   =   dset.createVariable(varname = "lat"  ,datatype = float, dimensions = ("lat") , fill_value = 1e+36)
						z   =   dset.createVariable(varname = variable_run+'_TCE_1s'  ,datatype = float, dimensions = ("lag","time","lat","lon"),fill_value = 1e+36) #varible = gpp_bin_ext
						w.axis  =   "T"
						t.axis  =   "T"
						x.axis  =   "X"
						y.axis  =   "Y"
						w[...]  =   lags_TCE
						t[...]  =   nc_ano[source_run][member_run].variables['time'] [...][win_idx * win_len : (win_idx+1)*win_len]
						x[...]  =   nc_ano[source_run][member_run].variables['lon'][...]
						y[...]  =   nc_ano[source_run][member_run].variables['lat'][...]
						z[...]	= 	Binary_Data_TCE [ext_type]['bin_TCE_1s']
						z.missing_value = 1e+36
						z.stardard_name = variable_run +" binary TCE (1s) matrix for 25 years starting at the  year %d"%start_yr
						z.units         = "0,1"
						x.units         =   nc_ano[source_run][member_run].variables['lon'].units
						x.missing_value =   1e+36
						x.setncattr         ("standard_name",nc_ano[source_run][member_run].variables['lon'].standard_name)
						y.units         =   nc_ano[source_run][member_run].variables['lat'].units
						y.missing_value =   1e+36
						y.setncattr         ("standard_name",nc_ano[source_run][member_run].variables['lat'].standard_name)
						t.units         =   nc_ano[source_run][member_run].variables['time'].units
						t.setncattr         ("calendar", nc_ano[source_run][member_run].variables['time'].calendar)
						t.setncattr         ("standard_name", nc_ano[source_run][member_run].variables['time'].standard_name)
						t.missing_value =   1e+36
						w.units         =   "month"
						w.setncattr         ("standard_name","lags in months")
						w.missing_value =   1e+36



					
				





# Calculation of TS of gain or loss of carbon uptake
# --------------------------------------------------

def Global_TS_of_Extremes(bin_ar, ano_gC, area = 0, lf = 0):
	"""
	Returns the global TS of :
	1. total carbon loss/gain associated neg/pos extremes   
	2. total freq of extremes
	3. total area affected by extremes

	Parameters:
	-----------
	bin_ar : the binary array of extremes (pos/neg)
	ano_gC : the array which will use the mask or binary arrays to calc the carbon loss/gain
	
	Universal:
	----------
	2-d area array (nlat, nlon), dates_win (# wins, win_size)
	
	Returns:
	--------
	1d array of length # wins x win_size for all : ext_gC_ts, ext_freq_ts, ext_area_ts
	"""
	print (" Calculating Extremes ... " )
	ext_ar      = bin_ar * ano_gC    # extremes array
	if (area == 0) and (lf == 0) :
		print ("The area under extreme will not be calculated... \nGrid area input and land fraction is not provided ... \nThe returned area is 0 (zeros)")
	ext_area_ar = bin_ar * area[...] * lf[...] # area array of extremes
	ext_gC_ts   = []
	ext_freq_ts = []
	ext_area_ts = []
	for i in range(dates_win.flatten().size):
		ext_gC_ts   . append(np.ma.sum(ext_ar[i]))
		ext_freq_ts . append(np.ma.sum(bin_ar[i]))
		ext_area_ts . append(np.ma.sum(ext_area_ar[i]))
	return np.ma.array(ext_gC_ts), np.ma.array(ext_freq_ts),np.ma.array(ext_area_ts)

# Calculating the slopes of GPP extremes
# --------------------------------------
def Slope_Intercept_Pv_Trend_Increase ( time, ts, until_idx1=2100, until_idx2=None):
	"""
	Returns the slope, intercept, r value , p value and trend line points for time period 1850-2100 (as '_21') and 2101-2300 ('_23')

	Parameters:
	-----------
	One dimentional time series of len 5400 from 1850 through 2299

	Returns:
	--------
	single values for slope, intercept, r value , p value, increase percentage**
	1d array for same legnth as 'ts' for 'trend'

	** it return the percent increase of trend line relavtive to the year 1850 (mean trend line value),..
	"""
	until_idx1	= int (until_idx1)
	if until_idx2 !=  None:
		until_idx2	= int (until_idx2)

	# calculation of the magnitudes of global gpp loss and trend from 1850- until idx-1
	slope_1, intercept_1,rv_1,pv_1,std_e1 	= stats.linregress(time[...][:until_idx1],ts[:until_idx1])
	trend_1       							= slope_1*time[...][:until_idx1]+intercept_1
	increase_1    							= (trend_1[-1]-trend_1[0])*100/trend_1[0]

	# calculation of the magnitudes of global gpp loss and trend from index-1 to until-idx2
	if until_idx2 !=  None:
		slope_2, intercept_23,rv_23,pv_23,std_e23  = stats.linregress(time[...][until_idx1:until_idx2],ts[until_idx1:until_idx22])
		trend_2       								= slope_2*time[...][until_idx1:until_idx2]+intercept_23
		increase_2    								= (trend_2[-1]-trend_2[0])*100/trend_2[0]
		increase_2_r1850							= (trend_2[-1]-trend_1[0])*100/trend_1[0]
		return slope_1,intercept_1,pv_1,trend_1,increase_1,slope_2,intercept_2,pv_2,trend_2,increase_2,increase_2_r1850
	else:
		return slope_1,intercept_1,pv_1,trend_1,increase_1

# Saving the results of TS carbon loss/gain 
for source_run in source_selected:
	for member_run in common_members [source_run]:
		Results[source_run][member_run]['ts_global_gC'] 	= {} 
		Results[source_run][member_run]['ts_global_area']	= {}
		Results[source_run][member_run]['ts_global_freq']	= {}
		Results[source_run][member_run]['ts_global_gC']['neg_ext']	= {}
		Results[source_run][member_run]['ts_global_gC']['pos_ext']	= {}
		Results[source_run][member_run]['ts_global_area']['neg_ext']= {}
		Results[source_run][member_run]['ts_global_area']['pos_ext']= {}
		Results[source_run][member_run]['ts_global_freq']['neg_ext']= {}
		Results[source_run][member_run]['ts_global_freq']['pos_ext']= {}

for source_run in source_selected:
	print ("Calculating the global TS of Extremes for %s"%source_run)
	for member_run in common_members [source_run]:
		# Negative Extremes:
		# ------------------
		ts_ext , ts_freq, ts_area	= Global_TS_of_Extremes(bin_ar 	= Results[source_run][member_run]['bin_ext_neg'],
															ano_gC 	= nc_ano[source_run][member_run].variables[variable_run][...],
															area	= data_area [source_run], 
															lf		= data_lf [source_run])
		Results[source_run][member_run]['ts_global_gC'  ]['neg_ext']['ts'] = ts_ext
		Results[source_run][member_run]['ts_global_area']['neg_ext']['ts'] = ts_area
		Results[source_run][member_run]['ts_global_freq']['neg_ext']['ts'] = ts_freq
		del ts_ext , ts_freq, ts_area
		# Positive Extremes:
		# -----------------
		ts_ext , ts_freq, ts_area	= Global_TS_of_Extremes(bin_ar 	= Results[source_run][member_run]['bin_ext_pos'],
															ano_gC 	= nc_ano[source_run][member_run].variables[variable_run][...],
															area	= data_area [source_run], 
															lf		= data_lf [source_run])
		Results[source_run][member_run]['ts_global_gC'  ]['pos_ext']['ts'] = ts_ext
		Results[source_run][member_run]['ts_global_area']['pos_ext']['ts'] = ts_area
		Results[source_run][member_run]['ts_global_freq']['pos_ext']['ts'] = ts_freq
		del ts_ext , ts_freq, ts_area
		# -----------------

for source_run in source_selected:
	for member_run in common_members [source_run]:
		# Negative Extremes gC:
		# ---------------------
		slope,intercept,pv,trend,increase = Slope_Intercept_Pv_Trend_Increase (
											time = nc_ano[source_run][member_run].variables['time'],
											ts	 = Results[source_run][member_run]['ts_global_gC']['neg_ext']['ts'],
											until_idx1 = idx_yr_2099)
		Results[source_run][member_run]['ts_global_gC']['neg_ext']['s21' 	 ] = slope
		Results[source_run][member_run]['ts_global_gC']['neg_ext']['pv21'	 ] = pv
		Results[source_run][member_run]['ts_global_gC']['neg_ext']['trend_21'] = trend
		Results[source_run][member_run]['ts_global_gC']['neg_ext']['inc_21'  ] = increase
		del slope,intercept,pv,trend,increase
		# Positive Extremes gC:
		# ---------------------
		slope,intercept,pv,trend,increase = Slope_Intercept_Pv_Trend_Increase (
											time = nc_ano[source_run][member_run].variables['time'],
											ts	 = Results[source_run][member_run]['ts_global_gC']['pos_ext']['ts'],
											until_idx1 = idx_yr_2099)
		Results[source_run][member_run]['ts_global_gC']['pos_ext']['s21' 	 ] = slope
		Results[source_run][member_run]['ts_global_gC']['pos_ext']['pv21'	 ] = pv
		Results[source_run][member_run]['ts_global_gC']['pos_ext']['trend_21'] = trend
		Results[source_run][member_run]['ts_global_gC']['pos_ext']['inc_21'  ] = increase
		del slope,intercept,pv,trend,increase
		# -----------------------------------
		# -----------------------------------

		# Negative Extremes freq:
		# -----------------------
		slope,intercept,pv,trend,increase = Slope_Intercept_Pv_Trend_Increase (
											time = nc_ano[source_run][member_run].variables['time'],
											ts	 = Results[source_run][member_run]['ts_global_freq']['neg_ext']['ts'],
											until_idx1 = idx_yr_2099)
		Results[source_run][member_run]['ts_global_freq']['neg_ext']['s21' 	 ] 	= slope
		Results[source_run][member_run]['ts_global_freq']['neg_ext']['pv21'	 ] 	= pv
		Results[source_run][member_run]['ts_global_freq']['neg_ext']['trend_21']= trend
		Results[source_run][member_run]['ts_global_freq']['neg_ext']['inc_21'  ]= increase
		del slope,intercept,pv,trend,increase
		# Positive Extremes freq:
		# -----------------------
		slope,intercept,pv,trend,increase = Slope_Intercept_Pv_Trend_Increase (
											time = nc_ano[source_run][member_run].variables['time'],
											ts	 = Results[source_run][member_run]['ts_global_freq']['pos_ext']['ts'],
											until_idx1 = idx_yr_2099)
		Results[source_run][member_run]['ts_global_freq']['pos_ext']['s21' 	 ] 	= slope
		Results[source_run][member_run]['ts_global_freq']['pos_ext']['pv21'	 ]	= pv
		Results[source_run][member_run]['ts_global_freq']['pos_ext']['trend_21']= trend
		Results[source_run][member_run]['ts_global_freq']['pos_ext']['inc_21'  ]= increase
		del slope,intercept,pv,trend,increase
		# -----------------------------------
		# -----------------------------------

		# Negative Extremes area:
		# -----------------------
		slope,intercept,pv,trend,increase = Slope_Intercept_Pv_Trend_Increase (
											time = nc_ano[source_run][member_run].variables['time'],
											ts	 = Results[source_run][member_run]['ts_global_area']['neg_ext']['ts'],
											until_idx1 = idx_yr_2099)
		Results[source_run][member_run]['ts_global_area']['neg_ext']['s21' 	 ] 	= slope
		Results[source_run][member_run]['ts_global_area']['neg_ext']['pv21'	 ] 	= pv
		Results[source_run][member_run]['ts_global_area']['neg_ext']['trend_21']= trend
		Results[source_run][member_run]['ts_global_area']['neg_ext']['inc_21'  ]= increase
		del slope,intercept,pv,trend,increase
		# Positive Extremes area:
		# -----------------------
		slope,intercept,pv,trend,increase = Slope_Intercept_Pv_Trend_Increase (
											time = nc_ano[source_run][member_run].variables['time'],
											ts	 = Results[source_run][member_run]['ts_global_area']['pos_ext']['ts'],
											until_idx1 = idx_yr_2099)
		Results[source_run][member_run]['ts_global_area']['pos_ext']['s21' 	 ] 	= slope
		Results[source_run][member_run]['ts_global_area']['pos_ext']['pv21'	 ]	= pv
		Results[source_run][member_run]['ts_global_area']['pos_ext']['trend_21']= trend
		Results[source_run][member_run]['ts_global_area']['pos_ext']['inc_21'  ]= increase
		del slope,intercept,pv,trend,increase
		# -----------------------------------
	
def Sum_and_Diff_of_Fluxes_perWin(ano_gC, bin_ar = None, data_type = 'ext', diff_ref_yr = 1850):
	"""
	returns a 2-d array sum of fluxes and difference of the sum of fluxes with reference to the ref yr

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

	 
# ----------------------------------------------------------
# Preparing the storage 
# ----------------------------------------------------------
for source_run in source_selected:
	for member_run in common_members [source_run]:
		# Negative Extremes:
		sum_neg_ext	, diff_neg_ext = Sum_and_Diff_of_Fluxes_perWin ( bin_ar = Results[source_run][member_run]['bin_ext_neg'],
													 				 ano_gC = nc_ano[source_run][member_run].variables[variable_run][...],
																	 data_type = 'ext',
													 				 diff_ref_yr = 1850)
		Results[source_run][member_run]['sum_neg_ext']	= sum_neg_ext
		Results[source_run][member_run]['diff_neg_ext']	= diff_neg_ext
		# Positive extremes:
		sum_pos_ext	, diff_pos_ext = Sum_and_Diff_of_Fluxes_perWin ( bin_ar = Results[source_run][member_run]['bin_ext_pos'],
													 				 ano_gC = nc_ano[source_run][member_run].variables[variable_run][...],
																	 data_type = 'ext',
													 				 diff_ref_yr = 1850)
		Results[source_run][member_run]['sum_pos_ext']	= sum_pos_ext
		Results[source_run][member_run]['diff_pos_ext']	= diff_pos_ext

		del sum_neg_ext , diff_neg_ext, sum_pos_ext , diff_pos_ext

		#Negative Flux/Ori
#sum_neg_ori	, diff_neg_ori = Sum_and_Diff_of_Fluxes_perWin ( bin_ar = None,
#													 				 ano_gC = nc_ano[source_run][member_run].variables[variable_run][...],
#																	 data_type = 'ori',
#													 				 diff_ref_yr = 1850)

#		Results[source_run][member_run]['sum_neg_ori']	= sum_neg_ori
#		Results[source_run][member_run]['diff_neg_ori']	= diff_neg_ori

		

			                                                 	
#		Results[source_run][member_run]['sum_pos_ext'] 	= {} 
#		Results[source_run][member_run]['diff_neg_ext'] = {}
#		Results[source_run][member_run]['diff_pos_ext'] = {}


# Regional analysis
# -----------------
import regionmask
# Selection the member_run manually
member_run = common_members[source_run] [0]

lon	= nc_ano[source_run][member_run].variables ['lon']
lat	= nc_ano[source_run][member_run].variables ['lat']

# for the plotting
lon_bounds	= nc_ano[source_run][member_run].variables [lon.bounds]
lat_bounds	= nc_ano[source_run][member_run].variables [lat.bounds]
lon_edges	= np.hstack (( lon_bounds[:,0], lon_bounds[-1,-1]))
lat_edges	= np.hstack (( lat_bounds[:,0], lat_bounds[-1,-1]))

# Creating mask of the regions based on the resolution of the model
mask = regionmask.defined_regions.srex.mask(lon[...], lat[...]).values

# important information:
srex_abr		= regionmask.defined_regions.srex.abbrevs
srex_names		= regionmask.defined_regions.srex.names
srex_nums		= regionmask.defined_regions.srex.numbers 
srex_centroids	= regionmask.defined_regions.srex.centroids 
srex_polygons	= regionmask.defined_regions.srex.polygons

mask_ma 		= np.ma.masked_invalid(mask)

import	matplotlib.pyplot	as plt
import os
"""
Basemaps not working anymore
===========================
#1- Hack to fix missing PROJ4 env var
import os
import conda

conda_file_dir = conda.__file__
conda_dir = conda_file_dir.split('lib')[0]
proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
os.environ["PROJ_LIB"] = proj_lib
#-1 Hack end 

import	matplotlib.pyplot	as plt
from	mpl_toolkits.basemap	import Basemap         
"""

""" 
Regional Plots
--------------
#fig = plt.figure()
#ax = plt.subplot(111, projection=ccrs.PlateCarree())
fig,ax = plt.subplots(tight_layout = True, figsize = (9,5), dpi = 400)
bmap    = Basemap(  projection  =   'eck4',
					lon_0       =   0.,
					resolution  =   'c')
LON,LAT	= np.meshgrid(lon_edges,lat_edges)
ax = bmap.pcolormesh(LON,LAT, mask_ma, cmap ='viridis')
bmap    .drawparallels(np.arange(-90., 90., 30.),fontsize=14, linewidth = .2)
bmap    .drawmeridians(np.arange(0., 360., 60.),fontsize=14, linewidth = .2)
bmap    .drawcoastlines(linewidth = .25,color='lightgrey')
plt.colorbar(ax, orientation='horizontal', pad=0.04)
fig.savefig (web_path + "SREX_regions.pdf")

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

fig = plt.figure(figsize = (9,5))
ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
mask_ma = np.ma.masked_invalid(mask)
h = ax.pcolormesh(lon_edges[...], lat_edges[...], mask_ma, transform = proj_trans)#,  cmap='viridis')
ax.coastlines()
plt.colorbar(h, orientation='horizontal', pad=0.04)

# Plot the abs at the centroids
for idx, abr in enumerate(srex_abr):
	plt.text (	srex_centroids[idx][0], srex_centroids[idx][-1], srex_abr[idx],
				horizontalalignment='center',
				transform = proj_trans)
	ax.add_geometries([srex_polygons[idx]], crs = proj_trans, facecolor='none',  edgecolor='red', alpha=0.8)


fig.savefig (web_path + "SREX_regions_cpy.pdf")
plt.close(fig)
"""


# =================================================================================================
# =================================================================================================
			##		#			##		     ########		
			# #		#		  ##  ##			##
			##		#		  #	   #			##
			#		#		  ##  ##			##
			#		#####		##              ##
# =================================================================================================
# =================================================================================================

# Creating a lis to Unique colors for multiple models:
# ---------------------------------------------------
NUM_COLORS = len(source_selected)
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
sns.reset_orig()  # get default matplotlib styles back
clrs = sns.color_palette('husl', n_colors=NUM_COLORS)

# Creating the ticks for x axis (every 25 years):
# ----------------------------------------------
tmp_idx = np.arange(0, 3013, 300) #for x ticks
tmp_idx[-1]=tmp_idx[-1]-1
dates_ticks = []
years_ticks	= []
for i in tmp_idx:
	a = dates_win.flatten()[i]
	dates_ticks.append(a)
	years_ticks.append(a.year)

# Creating the x-axis years (Monthly)
# -----------------------------------
x_years	= [d.year for d in dates_win.flatten()]

# Caption (optional): This dictionary could be used to save the captions of the figures
# -------------------------------------------------------------------------------------
Captions = {}

# PLOTING THE THRESHOLD FOR QUALIFICATION OF EXTREME EVENTS: fig[1-9]
# ===================================================================

if th_type == 'ind':

	fig1,ax2  = plt.subplots(tight_layout = True, figsize = (9,5), dpi = 400)

	ymin	= 400
	ymax 	= 8000
	for s_idx, source_run in enumerate(source_selected):
		for m_idx, member_run in enumerate(common_members [source_run]):
#		ax2.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_neg'])/10**9,
#				  'r', label = "Th$-$ %s"%source_run, alpha = .7)
			
#		ax2.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_neg'])/10**9,
#				  clrs[s_idx], ls='--', label = "Th$-$ %s"%source_run, alpha = .7)

			ax2.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_neg'])/10**9,
					'r', ls='--', label = "Th$-$ %s"%source_run, alpha = .3)
			ax2.set_ylabel("Negative Extremes (GgC)", {'color': 'r'},fontsize =14)
			ax2.set_xlabel("Time", fontsize = 14)
			ax2.set_ylim([ymin,ymax])
#ax2.set_yticks(np.arange(int(np.floor(ymin/100)*100),int(np.ceil(ymax/100)*100),25))
#ax2.set_yticklabels(-np.arange(int(np.floor(ymin/100)*100),int(np.ceil(ymax/100)*100),25))
#ax2.tick_params(axis='y', colors='red')
#		ax2.set_xticks(dates_ticks)
			ax2.grid(which='major', linestyle=':', linewidth='0.3', color='gray')
			ax1=ax2.twinx()
#		ax1.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_pos'])/10**9,
#				  'g', label = "Th+ %s"%source_run, alpha = .7)
			ax1.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_pos'])/10**9,
					  'g', label = "Th+ %s"%source_run, alpha = .3)
			ax1.set_ylabel("Positive Extremes (GgC)", {'color': 'g'},fontsize =14)
			ax1.set_ylim([ymin,ymax])
#ax1.set_yticks(np.arange(int(np.floor(ymin/100)*100),int(np.ceil(ymax/100)*100),25))
#ax1.tick_params(axis='y', colors='green')
#		ax1.set_xticks(dates_ticks)
#		ax1.grid(which='major', linestyle=':', linewidth='0.3', color='gray')
			lines, labels = ax1.get_legend_handles_labels()
			lines2, labels2 = ax2.get_legend_handles_labels()
			labels, ids = np.unique(labels, return_index=True)
			labels2, ids2 = np.unique(labels2, return_index=True)
			lines = [lines[i] for i in ids]
			lines2 = [lines2[i] for i in ids2]
#		ax2.legend(lines + lines2, labels + labels2, loc= 'best',fontsize =12)   
#continue	
	fig1.savefig(web_path + 'Threshold/ts_threshold_all_scenario_%s_per_%s.pdf'%(variable_run,int(per)))
	plt.close(fig1)
	del fig1

# Threshold per model for the 'th_type' == 'ind' and per = 1.0
# -------------------------------------------------------------

	for source_run in source_selected:
		fig2,ax2 = plt.subplots(tight_layout = True, figsize = (9,5), dpi = 400)
		pd.plotting.deregister_matplotlib_converters()
		if source_run == 'CESM2'		: ymin	= 400	; ymax	= 700
		if source_run == 'CanESM5'		: ymin	= 2000	; ymax	= 8000
		if source_run == 'IPSL-CM6A-LR'	: ymin	= 1700	; ymax	= 2900
		if source_run == 'BCC-CSM2-MR'	: ymin	= 400	; ymax	= 1000
		if source_run == 'CNRM-ESM2-1'	: ymin	= 1000	; ymax	= 1500
		if source_run == 'CNRM-CM6-1'	: ymin	= 1000	; ymax	= 1800

		for m_idx, member_run in enumerate(common_members [source_run]):

			L1= ax2.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_neg'])/10**9,
							  'r', label = "Th$-$ %s"%member_run, linewidth = 0.3, alpha = .7)
			L1[0].set_linestyle(LINE_STYLES[m_idx%NUM_STYLES])
			ax2.set_ylabel("Negative Extremes (GgC)", {'color': 'r'},fontsize =14)
			ax2.set_xlabel("Time", fontsize = 14)
#ax2.set_xlim([dates_ticks[0],dates_ticks[-1]])
#ax2.set_yticks(np.arange(int(np.floor(ymin/100)*100),int(np.ceil(ymax/100)*100),25))
#ax2.set_yticklabels(-np.arange(int(np.floor(ymin/100)*100),int(np.ceil(ymax/100)*100),25))
#ax2.tick_params(axis='y', colors='red')
			ax2.grid(which='major', linestyle='--', linewidth='0.3', color='gray')
			ax1=ax2.twinx()
		for m_idx, member_run in enumerate(common_members [source_run]):
			L2=	ax1.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_pos'])/10**9,
							 'g', label = "Th+ %s"%member_run, linewidth = 0.3, alpha = .7)
			L2[0].set_linestyle(LINE_STYLES[m_idx%NUM_STYLES])
			ax1.set_ylabel("Positive Extremes (GgC)", {'color': 'g'},fontsize =14)
#ax1.set_yticklabels([])
#ax1.set_yticks(np.arange(int(np.floor(ymin/100)*100),int(np.ceil(ymax/100)*100),25))
#ax1.tick_params(axis='y', colors='green')
#		ax1.grid(which='major', linestyle='--', linewidth='0.3', color='gray')

		ax2.set_ylabel("Negative Extremes (GgC)", {'color': 'r'},fontsize =14)
		ax2.set_xlabel("Time", fontsize = 14)
		ax1.set_ylabel("Positive Extremes (GgC)", {'color': 'g'},fontsize =14)

		ax2.set_ylim([ymin,ymax])
		ax1.set_ylim([ymin,ymax])
		ax1.set_xticks(dates_ticks)
		ax1.set_xticklabels(years_ticks)
		lines, labels = ax1.get_legend_handles_labels()
		lines2, labels2 = ax2.get_legend_handles_labels()
		ax2.legend(lines + lines2, labels + labels2, loc=0,fontsize =8)   
		fig2.savefig(web_path + 'Threshold/ts_threshold_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
		# Saving the plots
		path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Global_Extremes/non-TCE/Threshold/"%(source_run,member_run, variable_run)
		if os.path.isdir(path_save) == False:
		    os.makedirs(path_save)
		fig2.savefig(path_save + 'ts_threshold_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
		plt.close(fig2)

	del fig2,ax2

# Plotting thresholds when 'th_type' == 'common':
# -----------------------------------------------
if th_type == 'common':
	fig3  = plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
	plt.title("TS of Thresholds for CMIP6 models for percentile = %d"%int(per))
	pd.plotting.deregister_matplotlib_converters()
	for s_idx, source_run in enumerate(source_selected):
		for m_idx, member_run in enumerate(common_members [source_run]):
			plt.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_neg'])/10**9,
					color=clrs[s_idx], ls='-', label = "$q$ %s"%source_run, alpha = .8, linewidth = .7)
			plt.ylabel("Thresholds (GgC)", {'color': 'k'},fontsize =14)
			plt.xlabel("Time", fontsize = 14)
			plt.grid(which='major', linestyle=':', linewidth='0.3', color='gray')
			plt.legend()
			break #Plotting only the first ensemble member
	fig3.savefig(web_path + 'Threshold/ts_thresholdc_all_models_%s_per_%s.pdf'%(variable_run,int(per)))
	plt.close(fig3)
	del fig3

# Threshold per model for the 'th_type' == 'common' and per = 5.0
# ---------------------------------------------------------------
	for s_idx, source_run in enumerate(source_selected):
		fig4 = plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
		plt.title("TS of %d percentile Thresholds of %s for the model %s"%(per, variable_run.upper(), source_run))
		pd.plotting.deregister_matplotlib_converters()
		if variable_run == 'gpp':
			if source_run == 'CESM2'		: ymin	=  250	; ymax	=  400
			if source_run == 'CanESM5'		: ymin	= 1500	; ymax	= 4500
			if source_run == 'IPSL-CM6A-LR'	: ymin	= 1200	; ymax	= 2100
			if source_run == 'BCC-CSM2-MR'	: ymin	=  300	; ymax	=  600
			if source_run == 'CNRM-ESM2-1'	: ymin	=  700	; ymax	=  900
			if source_run == 'CNRM-CM6-1'	: ymin	=  600	; ymax	= 1100

		if variable_run == 'nbp':
			if source_run == 'CESM2'		: ymin	=  130	; ymax	= 230 
	
		if variable_run == 'ra':
			if source_run == 'CESM2'		: ymin	=  180	; ymax	= 240 

		if variable_run == 'rh':
			if source_run == 'CESM2'		: ymin	=  100	; ymax	= 170

		for m_idx, member_run in enumerate(common_members [source_run]):
			plt.plot( dates_win.flatten(), abs(Results[source_run][member_run]['ts_th_neg'])/10**9,
					color=clrs[s_idx], ls='-', label = "$q$ %s"%source_run, alpha = 1, linewidth = 1)
			break #Plotting only the first ensemble member
		plt.ylim ((ymin,ymax))
		plt.ylabel("Thresholds (GgC)", {'color': 'k'},fontsize =14)
		plt.xlabel("Time", fontsize = 14)
		plt.grid(which='major', linestyle=':', linewidth='0.4', color='gray')
		plt.legend()
		fig4.savefig(web_path + 'Threshold/ts_thresholdc_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
		# Saving the plots
		path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Global_Extremes/non-TCE/Threshold/"%(source_run,member_run, variable_run)
		if os.path.isdir(path_save) == False:
		    os.makedirs(path_save)
		fig4.savefig(path_save + 'ts_thresholdc_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
		plt.close(fig4)
		del fig4





# PLOTING THE GLOBAL TIMESERIES OF THE EXTREME EVENTS : fig[11-19]
# ======================================================================================
for s_idx, source_run in enumerate(source_selected):
	fig11 	= 	plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
	plt.style.use("classic")
	plt.title	("TS global %s extremes for %s when percentile is %d"%(variable_run.upper(), source_run, per))
	pd.plotting.deregister_matplotlib_converters()
	if variable_run == 'gpp':
		if source_run == 'CESM2'		: ymin	= -1.2	; ymax	=  1.2  
		if source_run == 'CanESM5'		: ymin	= -1.5	; ymax	=  1.5
		if source_run == 'IPSL-CM6A-LR'	: ymin	= -0.5	; ymax	=  0.5
		if source_run == 'BCC-CSM2-MR'	: ymin	= -1.6	; ymax	=  1.6
		if source_run == 'CNRM-ESM2-1'	: ymin	= -0.8	; ymax	=  0.8
		if source_run == 'CNRM-CM6-1'	: ymin	= -1.7	; ymax	=  1.7
	if variable_run == 'nbp':
		if source_run == 'CESM2'		: ymin	= -.7	; ymax	=  .7  
	if variable_run == 'ra':
		if source_run == 'CESM2'		: ymin	= -.7	; ymax	=  .7
	if variable_run == 'rh':
		if source_run == 'CESM2'		: ymin	= -.4	; ymax	=  .4

	for m_idx, member_run in enumerate(common_members [source_run]):
		plt.plot(	dates_win.flatten(),	Results[source_run][member_run]['ts_global_gC']['neg_ext']['ts'] / 10**15,
					'r', label = "Negative Extremes" , 	linewidth = 0.5, alpha=0.7 )
		plt.plot(	dates_win.flatten(), 	Results[source_run][member_run]['ts_global_gC']['pos_ext']['ts'] / 10**15,
					'g', label = "Positive Extremes" , 	linewidth = 0.5, alpha=0.7 )
		plt.plot(   dates_win.flatten() [:idx_yr_2099],	Results[source_run][member_run]['ts_global_gC']['neg_ext']['trend_21'] /10**15,
					'k--', label = "Neg Trend 21", 		linewidth = 0.5, alpha=0.9 )
		plt.plot(   dates_win.flatten() [:idx_yr_2099],	Results[source_run][member_run]['ts_global_gC']['pos_ext']['trend_21'] /10**15,
					'k--', label = "Pos Trend 21", 		linewidth = 0.5, alpha=0.9 )
		break #Plotting only the first ensemble member
	plt.ylim ((ymin,ymax)) #| waiting for the first set of graphs to remove this comment

	plt.xlabel(	'Time', fontsize = 14)
	plt.xticks(ticks = dates_ticks, labels = years_ticks, fontsize = 12)
	plt.ylabel( "Intensity of Extremes (PgC/mon)", fontsize = 14)
	plt.grid(	which='major', linestyle=':', linewidth='0.3', color='gray')
	plt.text(	dates_win.flatten()[900],	ymin+0.2,"Slope = %d %s"%(int(Results[source_run][member_run]['ts_global_gC']['neg_ext']['s21']/10**6), 
				'MgC/month'), size =14, color = 'r' )
	plt.text(	dates_win.flatten()[900],	ymax-0.2,"Slope = %d %s"%(int(Results[source_run][member_run]['ts_global_gC']['pos_ext']['s21']/10**6), 
				'MgC/month'), size =14, color = 'g' )
	fig11.savefig(web_path + 'Intensity/ts_global_carbon_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	# Saving the plots
	path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Global_Extremes/non-TCE/Intensity/"%(source_run,member_run, variable_run)
	if os.path.isdir(path_save) == False:
	    os.makedirs(path_save)
	fig11.savefig(path_save + 'ts_global_carbon_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	fig11.savefig(path_save + 'ts_global_carbon_%s_source_%s_per_%s.png'%(source_run,variable_run,int(per)))
	plt.close(fig11)
	del fig11

# Rolling mean of annual losses and gains
# ---------------------------------------
def RM_Nyearly_4m_Mon(ts, rm_years = 5):
	"""
		The rolling mean is calculated to the right end value
		The first 4 years will not be reported in the output of 5 year rolling mean
	"""
	ts 	= np.array(ts)
	yr 	= np.array([np.sum(ts[i:i+12]) for i in range(ts.size//12)])
	yr_rm = pd.Series(yr).rolling(rm_years).mean()
	return yr_rm[rm_years-1:]

# Ploting 5 Year Rolling Mean figures
# -----------------------------------
for s_idx, source_run in enumerate(source_selected):
	fig12 	= 	plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
	plt.title	("5yr RM of TS annual global %s for %s when percentile is %d"%(variable_run.upper(), source_run, per))
	pd.plotting.deregister_matplotlib_converters()

	for m_idx, member_run in enumerate(common_members [source_run]):

		print (source_run,member_run)
		plt.plot(np.arange(1854,2100),	RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_gC']['neg_ext']['ts'] / 10**15),
					'r', label = "Negative Extremes" , 	linewidth = 0.5, alpha=0.7 )
		plt.plot(np.arange(1854,2100), 	RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_gC']['pos_ext']['ts'] / 10**15),
					'g', label = "Positive Extremes" , 	linewidth = 0.5, alpha=0.7 )
		break #Plotting only the first ensemble member

	plt.xlabel(	'Time', fontsize = 14)
	plt.ylabel( "Intensity of Extremes (PgC/mon)", fontsize = 14)
	plt.grid(	which='major', linestyle=':', linewidth='0.3', color='gray')
	fig12.savefig(web_path + 'Intensity/ts_rm5yr_global_carbon_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	# Saving the plots
	path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Global_Extremes/non-TCE/Intensity/"%(source_run,member_run, variable_run)
	if os.path.isdir(path_save) == False:
	    os.makedirs(path_save)
	fig12.savefig(path_save + 'ts_rm5yr_global_carbon_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	plt.close(fig12)
	del fig12

# Frequency of extremes:
# ======================
# Ploting 5 Year Rolling Mean figures:
# ------------------------------------	
for s_idx, source_run in enumerate(source_selected):
	fig14 	= 	plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
	plt.title	("5yr RM of annual TS of global frequency of %s extremes\nfor %s when percentile is %d"%(variable_run.upper(),source_run, per))
	pd.plotting.deregister_matplotlib_converters()
	for m_idx, member_run in enumerate(common_members [source_run]):

		print (source_run,member_run)
		plt.plot(np.arange(1854,2100),	RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_freq']['neg_ext']['ts'] ),
					'r', label = "Negative Extremes" , 	linewidth = 0.5, alpha=0.7 )
		plt.plot(np.arange(1854,2100), 	RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_freq']['pos_ext']['ts'] ),
					'g', label = "Positive Extremes" , 	linewidth = 0.5, alpha=0.7 )

		break #Plotting only the first ensemble member

	plt.xlabel(	'Time', fontsize = 14)
	plt.ylabel( "Frequency of Extremes (count/yr)", fontsize = 14)
	plt.grid(	which='major', linestyle=':', linewidth='0.3', color='gray')
	fig14.savefig(web_path + 'Freq/ts_rm5yr_global_freq_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	# Saving the plots
	path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Global_Extremes/non-TCE/Freq/"%(source_run,member_run, variable_run)
	if os.path.isdir(path_save) == False:
	    os.makedirs(path_save)
	fig14.savefig(path_save + 'ts_rm5yr_global_freq_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	plt.close(fig14)
	del fig14

Captions['fig14']	= "	5 year moving average of the frequency (counts) under positive and negative extremes.\
						All ensemble members have same values"

# Ploting 5 Year Rolling Mean figures (normalized) - pending:
# -------------------------------------------------

# Function to normalize positive and negative freq
def Norm_Two_TS(ts1, ts2):
	ts 			= np.concatenate((ts1,ts2)) 
	norm_ts		= norm(ts)
	norm_ts1	= norm_ts[:len(ts1)]
	norm_ts2	= norm_ts[len(ts1):]
	return norm_ts, norm_ts1, norm_ts2

# TEST
p = np.array([ 8, 6, 7, 8, 6, 5, 4, 6])
n = np.array([ 5, 6, 6, 4, 5, 7, 8, 6])

_,norm_p, norm_n = Norm_Two_TS(p,n)
norm_np = norm_n/norm_p
norm_pn = norm_p/norm_n

mask_np = np.ma.masked_greater(norm_np,1)
mask_pn = np.ma.masked_greater(norm_pn,1)



fig = plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
#plt.plot(  mask_np)
#plt.plot( -mask_pn)
#plt.plot(_)
#plt.plot(norm_np)
plt.plot(p/n)
fig.savefig(web_path + 'ratio_test.pdf')

# Dict to capture the ts of ratios pos to neg extremes of models
ts_ratio_freq	= {}

for s_idx, source_run in enumerate(source_selected):
	fig15 	= 	plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
	plt.title	("5yr Ratio of RM of TS annual global frequency (n/p) for %s when percentile is %d"%(source_run, per))
	pd.plotting.deregister_matplotlib_converters()

	for m_idx, member_run in enumerate(common_members [source_run]):

		print (source_run,member_run)
		ts_ratio	= np.divide ( RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_freq']['neg_ext']['ts'],10) ,
					  			  RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_freq']['pos_ext']['ts'],10) )
		ts_ratio_freq[source_run] = ts_ratio	

		plt.plot 	( np.arange(1859,2100), 	ts_ratio,
					 'k', label = "Pos/Neg Extremes" , 	linewidth = 0.5, alpha=0.7 )

		break #Plotting only the first ensemble member

	plt.xlabel(	'Time', fontsize = 14)
#plt.xticks(ticks = dates_ticks, labels = years_ticks, fontsize = 12)
	plt.ylabel( "Frequency of Extremes (count/yr)", fontsize = 14)
	plt.grid(	which='major', linestyle=':', linewidth='0.3', color='gray')
	fig15.savefig(web_path + 'Freq/ts_ratio_rm5yr_global_freq_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	# Saving the plots
	path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Global_Extremes/non-TCE/Freq/"%(source_run,member_run, variable_run)
	if os.path.isdir(path_save) == False:
	    os.makedirs(path_save)
	fig15.savefig(path_save + 'ts_ratio_rm5yr_global_freq_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	plt.close(fig15)
	del fig15
Captions['fig15']	= "	Shows the ratio the frequency of negative to positive extremes. Before taking the ratio a moving\
						average of 10 years was taken. "


# Area Affected by extremes:
# ==========================

# Ploting 5 Year Rolling Mean figures:
# ------------------------------------	
for s_idx, source_run in enumerate(source_selected):
	fig16 	= 	plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
	plt.title	("5yr RM of annual TS of global area affected by\n %s extremes for %s when percentile is %d"%(variable_run.upper(), source_run, per))
	pd.plotting.deregister_matplotlib_converters()

	for m_idx, member_run in enumerate(common_members [source_run]):

		print (source_run,member_run)
		plt.plot(np.arange(1854,2100),	RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_area']['neg_ext']['ts'] /10**15),
					'r', label = "Negative Extremes" , 	linewidth = 0.5, alpha=0.7 )
		plt.plot(np.arange(1854,2100), 	RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_area']['pos_ext']['ts'] /10**15),
					'g', label = "Positive Extremes" , 	linewidth = 0.5, alpha=0.7 )

		break #Plotting only the first ensemble member

	plt.xlabel(	'Time', fontsize = 14)
#plt.xticks(ticks = dates_ticks, labels = years_ticks, fontsize = 12)
	plt.ylabel( "Area Under Extremes ($10^{15}$ $m^2$)", fontsize = 14)
	plt.grid(	which='major', linestyle=':', linewidth='0.3', color='gray')
	fig16.savefig(web_path + 'Area/ts_rm5yr_global_area_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	# Saving the plots
	path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Global_Extremes/non-TCE/Area/"%(source_run,member_run, variable_run)
	if os.path.isdir(path_save) == False:
	    os.makedirs(path_save)
	fig16.savefig(path_save + 'ts_rm5yr_global_area_%s_source_%s_per_%s.pdf'%(source_run,variable_run,int(per)))
	plt.close(fig16)
	del fig16
Captions['fig16']	= "	5 year moving average of the area under positive and negative extremes.\
						All ensemble members have same values"

# Ploting 10 Year Rolling Mean ratio (Pos/Neg) of area under extremes:
# -------------------------------------------------------------------
# Dict to capture the ts of ratios pos to neg extremes of models
ts_ratio_area	= {}

rm_ratio_yr = 10
for s_idx, source_run in enumerate(source_selected):
	fig17 	= 	plt.figure(tight_layout = True, figsize = (9,5), dpi = 400)
	plt.title	("Ratio of RM (%d years) of annual TS of global area under\n%s extremes (n/p) for %s when percentile is %d"%(rm_ratio_yr, variable_run.upper(), source_run, per))
	pd.plotting.deregister_matplotlib_converters()

	for m_idx, member_run in enumerate(common_members [source_run]):

		print (source_run,member_run)
		ts_ratio	= np.divide ( RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_area']['neg_ext']['ts'],rm_ratio_yr) ,
					  			  RM_Nyearly_4m_Mon(Results[source_run][member_run]['ts_global_area']['pos_ext']['ts'],rm_ratio_yr) )
		ts_ratio_area[source_run] = ts_ratio	

		plt.plot 	( np.arange(1849+rm_ratio_yr,2100), 	ts_ratio,
					 'k', label = "Pos/Neg Extremes" , 	linewidth = 0.5, alpha=0.7 )

		break #Plotting only the first ensemble member

	plt.xlabel(	'Time', fontsize = 14)
#plt.xticks(ticks = dates_ticks, labels = years_ticks, fontsize = 12)
	plt.ylabel( "Ratio of area under Extremes (n/p)", fontsize = 14)
	plt.grid(	which='major', linestyle=':', linewidth='0.3', color='gray')
	fig17.savefig(web_path + 'Area/ts_ratio_rm%dyr_global_freq_%s_source_%s_per_%s.pdf'%(rm_ratio_yr,source_run,variable_run,int(per)))
	# Saving the plots
	path_save   = cori_scratch + "add_cmip6_data/%s/ssp585/%s/%s/Global_Extremes/non-TCE/Area/"%(source_run,member_run, variable_run)
	if os.path.isdir(path_save) == False:
	    os.makedirs(path_save)
	fig17.savefig	(path_save + 'ts_ratio_rm%dyr_global_freq_%s_source_%s_per_%s.pdf'%(rm_ratio_yr,source_run,variable_run,int(per)))
	plt.close(fig17)
	del fig17
Captions['fig17']	= "	Shows the ratio of area under negative to positive extremes. Before taking the ratio a moving\
						average of %d years was taken. "%(rm_ratio_yr)

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
#ymin = -1000
#ymax = 1000
#Units			= nc_data[sce]['var_gC' ].variables[variable].units # Units of gpp
		ts_div_factor	= 10**12
		y_label_text	= 'TgC'
#save_path   = paths['out' ]['comp_lulcc']+'diff_plots/change_%s_sce_%s_'%(variable,sce)

	#if (variable=='gpp' and datatype=='neg_ext' and analysis_type=='independent'): 
	if (variable_run=='gpp' and datatype=='neg_ext'):
		if source_run == 'CESM2'		:  ymax = 100 ; ymin = -ymax
		if source_run == 'BCC-CSM2-MR'	:  ymax = 150 ; ymin = -ymax
		if source_run == 'CNRM-CM6-1'	:  ymax = 200 ; ymin = -ymax
		if source_run == 'CNRM-ESM2-1'	:  ymax = 200 ; ymin = -ymax
		if source_run == 'CanESM5'		:  ymax = 1400; ymin = -ymax
		if source_run == 'IPSL-CM6A-LR' :  ymax = 200 ; ymin = -ymax

		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	if (variable_run=='gpp' and datatype=='pos_ext'): 
		ymin = -14
		ymax = 14
		Units			= nc_data[sce]['ano_gC' ].variables[variable].units # Units of gpp
		ts_div_factor	= 10**12
		y_label_text	= 'TgC'

	for i, data in enumerate (diff_array):
		print ("Plotting %s %s for the win %d"%(datatype,analysis_type,i))
		fig4,ax	= plt.subplots(figsize = (7,2.8),tight_layout=True,dpi=500)
#plt.title ("%s Ano for model %s and win %d"%(variable_run,source_run,i))
		bmap    = Basemap(  projection  =   'eck4',
							lon_0       =   0.,
							resolution  =   'c')
		lat		= nc_ano[source_run][member_run].variables['lat']
		lon		= nc_ano[source_run][member_run].variables['lon']
		LAT,LON = np.meshgrid(lat[...], lon[...],indexing ='ij')
		ax      = bmap.pcolormesh(LON,LAT,np.ma.masked_invalid(data/ts_div_factor),latlon=True,cmap= 'RdGn',vmax= ymax, vmin= ymin)
		cbar    = plt.colorbar(ax)
		cbar    .ax.set_ylabel(y_label_text)
		bmap    .drawparallels(np.arange(-90., 90., 30.),fontsize=14, linewidth = .2)
		bmap    .drawmeridians(np.arange(0., 360., 60.),fontsize=14, linewidth = .2)
		bmap    .drawcoastlines(linewidth = .2)
		plt.title("%s Ano for model %s - %s minus %s" %(variable_run,source_run, text[i], text[diff_yr_idx]))
		fig4.savefig(web_path + 'Spatial_Maps/%s_%s.png'%(savefile_head, format(i,'02')))
		plt.close(fig4)

for s_idx, source_run in enumerate(source_selected):
	for m_idx, member_run in enumerate(common_members [source_run]):
		text = "plot_diff_%s_neg_ext_%s"%(variable_run, source_run)
		Plot_Diff_Plot(diff_array = Results[source_run][member_run]['diff_neg_ext']    , datatype='neg_ext',  analysis_type='independent', savefile_head = text)
		break

"""
Plot_Diff_Plot(diff_array = Results['wo_lulcc']['diff_%s'%variable], datatype='var_ori',  sce = 'wo_lulcc', analysis_type='independent')
# Calling the functions:
Plot_Diff_Plot(diff_array = Results['wo_lulcc']['diff_%s'%variable], datatype='var_ori',  sce = 'wo_lulcc', analysis_type='independent')
Plot_Diff_Plot(diff_array = Results['wo_lulcc']['diff_%s'%variable], datatype='var_ori',  sce = 'wo_lulcc', analysis_type='independent')
Plot_Diff_Plot(diff_array = Results['w_lulcc' ]['diff_%s'%variable], datatype='var_ori',  sce = 'w_lulcc' , analysis_type='independent')

Plot_Diff_Plot(diff_array = Results[source_run][member_run]['diff_neg_ext']    , datatype='neg_ext',  analysis_type='independent')
Plot_Diff_Plot(diff_array = Results['wo_lulcc']['diff_neg_ext']    , datatype='neg_ext',  sce = 'wo_lulcc', analysis_type='independent')
Plot_Diff_Plot(diff_array = Results['w_lulcc' ]['diff_neg_ext']    , datatype='neg_ext',  sce = 'w_lulcc' , analysis_type='independent')
"""

