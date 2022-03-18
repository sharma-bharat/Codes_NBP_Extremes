# python 3
# Bharat Sharma

"""This python file will keep a record of all the functions created for
cmip6 dataset analysis"""

import numpy as np
import datetime as dt
def time_dim_dates(base_date = dt.date(1850,1,1) ,total_timestamps=3012):
	"""this python function create a time or date values for 15th of \
		   	every month from the start date,entered as year,month,day\
			and until the timestamps or the length of the timeseries"""
	from calendar import monthrange
	import datetime as dt
	import numpy as np
	fd          =   base_date+dt.timedelta(14) #first date in the list
	dates_list 	=   [fd]
	for i in range(total_timestamps-1):
		fd  = 	dt.date(fd.year,fd.month,monthrange(fd.year,fd.month)[1])+dt.timedelta(15)
		dates_list     .append(fd)
		dates_array	= 	np.array(dates_list)
	return dates_array


def index_and_dates_slicing(dates_array,start_date,end_date):
	"""This function will generate two np arrays
		1. the indices of the selected dates relative to the original
			input array
		2. the dates array from start_date to end the end date
	   	-----
 		Inputs: 
		-------
			1. dates_array > numpy array with dates format datetime.date(YEAR,MO,DY)
   			2. start_date	> datetime.date(YEAR,MO,DY)
			3. end_date  > datetime.date(YEAR,MO,DY)
			Both start and end date are inclusive"""

	import datetime as dt
	import numpy as np
	idx_dates_list=[]
	for i, j in enumerate(dates_array):
		idx_dates_list	.append((i,j))
	idx_dates_array	=	np.array(idx_dates_list)
	
	idx_array	= idx_dates_array[(idx_dates_array[:,1]>=start_date) & (idx_dates_array[:,1]<=end_date)][:,0]
	dates_array	= idx_dates_array[(idx_dates_array[:,1]>=start_date) & (idx_dates_array[:,1]<=end_date)][:,1]
	return idx_array,dates_array

def geo_idx(dd, dd_array):
	"""
	Search for nearest decimal degree in an array of decimal degrees and return the index.
    np.argmin returns the indices of minium value along an axis.
    so subtract dd from all values in dd_array, take absolute value and find index of minium.

	Inputs:
	______
		dd 		: the value whose index you want to find (lat : 30N = -30.)
		dd_array: the list of values for your search	 (lats: -90 ... 90)

	Outputs:
	_______
		geo_idx : The index of the value closest to the one present in the pool of values 
	"""
	import numpy as np
	geo_idx = (np.abs(dd_array - dd)).argmin()
	return geo_idx

def mpi_local_and_global_index(begin_idx, local_n):
	""" This function will return a list of tuples with a local index and the global index of the given value of begin index and local chunk size for mpi. The reason why we need this is because the we can run a for loop with two indicies if they are provided in a tuple form.\n
	Input:
	------------
	begin_index : 	of the lat/lon running on the local processor when using mpi
	local_n		:	the chunk size of the array we want to run on that processor

	Output:	
	-----------
	result		:	a list of tuple (local_idx,global_idx)
	"""
	local_idx=	0
	result	=	[]
	for i in range(local_n):
		result.append((local_idx,begin_idx))
		begin_idx	+=	1
		local_idx	+=	1
	return result

def ts_lagged(ts,lag = 0, maxlag=3):
	"""This function will shift the timeseries or 1d array based on the lag and maxlags in the analysis.
		The main reason is that if you want to make code general to accept lags as input, its important that the structure/size of the ts are same.
		Note: lag should not be more than the maxlag"""
	import numpy as np
	ts = np.array(ts)
	if 	 lag <  maxlag:  return ts[maxlag-lag:(None if lag==0 else -lag)]
	elif lag == maxlag : return ts[:-lag]
	else: 				 return "lag <= maxlag"


def percentile_colorbar(input_array,in_percentile=99,other=True):
	"""This function will take a list/nd array as first input agrument, the
  	second input argument is the percentile of the value interested
	in (default = 99). The function will mask all the nans and find
	the percentile only on the non_nans. The output will give you the 
	largest nth percentile and smallest nth"""
	import numpy as np
	input_array = np.ma.masked_invalid((np.array(input_array)).flatten())
	non_nans_l	=	input_array[~np.isnan(input_array)]
	if in_percentile   <   1:
		in_percentile  =   in_percentile*100
	max_value   =	np.percentile(non_nans_l,in_percentile)
	if other == False:
		return max_value
	else:
		min_value   =	np.percentile(non_nans_l,100-in_percentile)
		return max_value,min_value

def adjaceny_matrix(struct_id):
	if struct_id        ==  0:
		#small extext short duration
		#struct_cat          =   structs[struct_id]
		struct_mat          =   np.zeros((3,3,3),dtype=int)
		struct_mat[1,1,1]   =   1
	elif struct_id      ==  1:
		#small extext long duration 
		#struct_cat          =   structs[struct_id]
		struct_mat          =   np.zeros((3,3,3),dtype=int)
		struct_mat[:,1,1]   =   1
	elif struct_id      ==  2:
		#large extext short duration 
		#       struct_cat          =   structs[struct_id]
		struct_mat          =   np.zeros((3,3,3),dtype=int)
		struct_mat[1,:,:]   =   1
	elif struct_id      ==  3:
		#large extext long duration or 26 neighbours
		#       struct_cat          =   structs[struct_id]
		struct_mat          =   np.ones((3,3,3),dtype=int)
	elif struct_id      ==  4:
		#18 neighbours
		struct_mat          =   np.zeros((3,3,3),dtype = int)
		struct_mat[1,:,:]   =   np.ones((3,3),dtype = int)
		struct_mat[0,:,1]   =   int(1)
		struct_mat[2,:,1]   =   int(1)
		struct_mat[0,1,:]   =   int(1)
		struct_mat[2,1,:]   =   int(1)
	else:
		# 6 neighbors                                                                                                      
		struct_mat          =   np.zeros((3,3,3),dtype = int)
		struct_mat[1,:,1]   =   int(1)  
		struct_mat[1,1,:]   =   int(1)  
		struct_mat[0,1,1]   =   int(1)
		struct_mat[2,1,1]   =   int(1)
		"""
		Standard types:
		https://www.mathworks.com/help/images/ref/bwconncomp.html
		http://what-when-how.com/computer-graphics-and-geometric-modeling/raster-algorithms-basic-computer-graphics-part-1/
		"""
	return struct_mat

#Create a Matrix of subsequent whole numbers of the same shape as the global grid
def create_seq_mat(nlat =192, nlon = 288):
	"""
		This function will create a 2d matrix from 0 to nlat*nlon
		rows = nlat
		cols = nlon

	"""
	import numpy as np
	mat	= np.zeros((nlat,nlon))
	for i in range(nlat):
		mat[i,:]	= np.arange(i*nlon, (i+1)*nlon)
	return np.asarray(mat, dtype =int)

def cumsum_lagged (ar,lag=1):
	""" This code will cumsum the given array and the previous lagged values and return one array of the same length as input ar
		
		Inputs:
		_______
			ar : 1-d array
			lag: num of lagged values you want to cumsum
		Output:
		______
			cumsumed array  1d same length as ar

	"""
	import numpy as np

	if lag == 0:
		cum_ar = ar
	else:
		cum_ar = np.ma.masked_all((ar.size,lag+1))
		cum_ar[:,0] = ar
		for l in range(lag):
			cum_ar[l+1:,l+1] = ar[:-(l+1)]
	return cum_ar.sum(axis=1)
		
def cumsum_lagged (ar,lag=1, ignore_t0 = False):
	""" This code will cumsum the given array and the previous lagged values and return one array of the same length as input ar (mainly to check the affect of drivers on gpp)
		
		Inputs:
		_______
			ar : 1-d array
			lag: num of lagged values you want to cumsum
			ignore_t0 : True/False
				since the current climate conditions (esp. for PME) where anti correlated with neg gpp ext for lag = 0
				it was decided to check for the possible cases where we can ignore the current condition value for lag = 0 in cumlag effects
		Output:
		______
			cumsumed array  1d same length as ar

	"""
	import numpy as np
	ar	= np.array(ar)
	if (lag == 0 and ignore_t0 == False):
		cum_ar = ar
		return cum_ar

	elif (lag>0 and ignore_t0 == False):
		cum_ar = np.ma.masked_all((ar.size,lag+1))
		cum_ar[:,0] = ar
		for l in range(lag):
			cum_ar[l+1:,l+1] = ar[:-(l+1)]
		return cum_ar.sum(axis=1)
	else:
		if lag ==1:
			cum_ar = np.ma.masked_all((ar.size))
			cum_ar [lag:] = ar[:-lag]
			return cum_ar
		else:
			cum_ar = np.ma.masked_all((ar.size,lag))
			cum_ar[1:,0] = ar[:-1]
			for l in range(1,lag):
				cum_ar[l+1:,l] = ar[:-(l+1)]
			return cum_ar.sum(axis=1)

def cum_av_lagged (ar,lag=1, ignore_t0 = True):
	""" This code will cumsum and then mean/average the given array and the previous lagged values and return one array of the same length as input ar
		
		Inputs:
		-------
			ar : 1-d array
			lag: num of lagged values you want to cum_mean (cum_sum and then average)
			ignore_t0 : True/False
				since the current climate conditions (esp. for PME) where anti correlated with neg gpp ext for lag = 0
				it was decided to check for the possible cases where we can ignore the current condition value for lag = 0 in cumlag effects
		Output:
		-------
			cumsumed array  1d same length as ar
	"""
	import numpy as np
	ar	= np.array(ar)
	if (lag == 0 and ignore_t0 == False):
		cum_ar = ar
		result =  cum_ar

	elif (lag>0 and ignore_t0 == False):
		cum_ar = np.ma.masked_all((ar.size,lag+1))
		cum_ar[:,0] = ar
		for l in range(lag):
			cum_ar[l+1:,l+1] = ar[:-(l+1)]
		result =  cum_ar.sum(axis=1)
	else:
		if lag ==1:
			cum_ar = np.ma.masked_all((ar.size))
			cum_ar [lag:] = ar[:-lag]
			result = cum_ar
		else:
			cum_ar = np.ma.masked_all((ar.size,lag))
			cum_ar[1:,0] = ar[:-1]
			for l in range(1,lag):
				cum_ar[l+1:,l] = ar[:-(l+1)]
			result= cum_ar.sum(axis=1)
	if ignore_t0 ==  True:
		return result/lag
	else:
		return result/(lag+1)
	

def label_count_locations(bin_ar, min_event_size = 3, lag= None):
	"""
		this function will us the ndimage package to find the linear continuous events
		and return is in the form of a dictionary
		also, it will return the the arguments of the begining point of these events
						
			this information can be used to find the triger of extreme events
	
	Inputs:
	-------
	ar				:	1-d array binary array of extremes
	min_event_size	: 	the minimun size that will filter the extreme events and report the first args

	Outputs:
	-------
	dict 			: 	Dictionary with label keys and subsequent counts and locations of extreme events
	ext_arg			: 	The first argument of the extreme events in 'ar'
	"""
	from    scipy	import	ndimage
	import 	numpy	as		np

	dic = {}
	larray,narray	= ndimage.label(bin_ar,structure = np.ones(3))
	locations 		= ndimage.find_objects(larray)
	for idx, l  in enumerate (np.unique(larray, return_counts = True)[0]):
		if (l>0 and np.unique(larray, return_counts = True)[1][idx] >= min_event_size) :
			dic[l] 			= {}
			dic[l]['counts'] 	= np.unique(larray, return_counts = True)[1][idx]
			dic[l]['loc'] 		= locations[idx-1]
	
	args = np.arange(len(bin_ar))
	ext_arg = []
	for k in dic.keys():
		ext_arg.append(args[dic[k]['loc'][0]][0])
			
	ext_arg = np.array(ext_arg)
	ext_arg = ext_arg[ext_arg>=lag]
	return dic, ext_arg


	
def patch_with_gaps(bin_ar, max_gap =3, lag = None):
	"""
	this function will use the ndimage package to find the linear continuous events
		and return it in the form of a dictionary
		also, it will return the the arguments of the begining point of these events
			this information can be used to find the triger of extreme events

		This function will make events continuous with gaps
			e.g. max_gaps = 2
			> [1,1,0,0,0,1,1,0,0,1,1,0,1]  with have 2 extremes
	
	Inputs:
	-------
	ar			:	1-d array binary array of extremes
	max_gape	: 	the maximum gap size that will filter the extreme events and report the first args

	Outputs:
	-------
	dict 		: 	Dictionary with label keys and subsequent counts and locations of extreme events
	ext_arg		: 	The first argument of the extreme events in 'ar'

	"""
	from    scipy   import  ndimage
	import  numpy   as      np
	
	dic = {}
	larray,narray   = ndimage.label(bin_ar,structure = np.ones(3))
	locations       = ndimage.find_objects(larray)
	for idx, l  in enumerate (np.unique(larray, return_counts = True)[0]):
		if l>0:
			dic[l]          = {}
			dic[l]['counts']    = np.unique(larray, return_counts = True)[1][idx]
			dic[l]['loc']       = locations[idx-1]
	
	args = np.arange(len(bin_ar))

	start_args = []
	for k in dic.keys():
		start_args.append(args[dic[k]['loc'][0]][0])
	
	end_args = []
	for k in dic.keys():
		end_args.append(args[dic[k]['loc'][0]][-1])
	
	gaps = np.array(start_args[1:]) - np.array(end_args[:-1])-1

	gaps_mask = ~(gaps <= max_gap)
	
	new_start_args 		= np.zeros((gaps_mask.sum() +1))
	new_start_args[0]	= start_args[0]
	new_start_args[1:]	= np.array(start_args[1:])[gaps_mask]
	new_start_args  	= np.asarray(new_start_args, dtype = int)

	new_start_args		= new_start_args [new_start_args>=lag]

	return dic, new_start_args

def patch_with_gaps_and_eventsize(bin_ar, max_gap =2, min_cont_event_size =3, lag = None):
	"""
	this function will use the ndimage package to find the 1-D continuous events
		and return it in the form of a dictionary
		also, it will return the the arguments of the begining point of these events
			this information can be used to find the triger of extreme events

		This function will make events continuous with gaps
			e.g. max_gaps = 2
			> [1,1,1,0,0,0,1,1,0,0,1,1,1,0,1]  with have 2 extremes
	
	Inputs:
	-------
	bin_ar		:	1-d array binary array of extremes
	max_gap		: 	the maximum gap size that will filter the extreme events and report the first args

	Outputs:
	-------
	dict 		: 	Dictionary with label keys and subsequent counts and locations of extreme events
	ext_arg		: 	The first argument of the extreme events in 'ar'

	"""
	from    scipy   import  ndimage
	import  numpy   as      np
	bin_ar 			= np.asarray(bin_ar, dtype = int)
	bin_ar_0s 		= np.zeros(bin_ar.shape)
	bin_ar_0s[lag:] = bin_ar[lag:]
	del bin_ar
	bin_ar			= bin_ar_0s

	dic = {}
	larray,narray   = ndimage.label(bin_ar,structure = np.ones(3)) # this command labels every continuous event uniquely and also return total number of continuous events
	locations       = ndimage.find_objects(larray) # this command gives the location of the extremes for every label

	for idx, l  in enumerate (np.unique(larray, return_counts = True)[0]): # returns the labels names starting with 1 and total cells with that label 
		if l>0:
			dic[l]          = {}
			dic[l]['counts']    = np.unique(larray, return_counts = True)[1][idx]  	# for every label saving the counts
			dic[l]['loc']       = locations[idx-1]									# for every label saving the location
	
	args = np.arange(len(bin_ar))	# the arguments or index numbers of 'bin_ar'

	start_args = []
	for k in dic.keys():
		start_args.append(args[dic[k]['loc'][0]][0])    # list start args/idx (wrt bin_ar) of the events
	
	end_args = []
	for k in dic.keys():	
		end_args.append(args[dic[k]['loc'][0]][-1])     # list of end args/idx (wrt bin_ar) of the events
	
	gaps = np.array(start_args[1:]) - np.array(end_args[:-1])-1	# ar: gaps/discontinuity(in months) between subsequent events

	gaps_mask = ~(gaps <= max_gap)    # this is the mask of the gaps where the gaps are more than 'max_gap' i.e. gaps>max_gap are True ...
									  # by doing so the events with a larger gap are separate and all other are continuous
	
	new_start_args 		= np.zeros((gaps_mask.sum() +1))  # total events will be one more than number of qualified continuoous event
	new_start_args[0]	= start_args[0] # first arg will be the same as ori start_args
	new_start_args[1:]	= np.array(start_args[1:])[gaps_mask] # all others will follow the new discrete first arg
	new_start_args  	= np.asarray(new_start_args, dtype = int) 

	new_end_args 		= np.zeros((gaps_mask.sum() +1)) # the same goes for the end_args
	new_end_args[-1]	= end_args[-1]
	new_end_args[:-1]	= np.array(end_args[:-1])[gaps_mask]
	new_end_args  		= np.asarray(new_end_args, dtype = int)
	
	new_mask			= new_start_args>=lag   #incase the lags are considered you have to ingore the first few agrs = to the len of lag

	new_start_args		= new_start_args 	[new_mask]  #with lag
	new_end_args		= new_end_args 		[new_mask]  #with lag
	
	new_events 			= {}
	for idx in range(len(new_start_args)):
		new_events[idx] = bin_ar[new_start_args[idx]:new_end_args[idx]+1] # gives you the list of new events after checking for gaps more than max_gap only
	
	new_events 			= {}
	i=0
	triggers 			= []
	for idx in range(len(new_start_args)):
		bin_event 		= bin_ar[new_start_args[idx]:new_end_args[idx]+1] # checking on all qualified previous continuous events with gap
		larray,narray   = ndimage.label(bin_event,structure = np.ones(3)) # generating the labels and the total arrays for the new selected continuous events with gap
		ev_size			= np.unique(larray, return_counts = True)[1]      # event size of the different extreme events without gaps within the selected continous events with gaps
		if (ev_size>= min_cont_event_size).sum() >=1: # looking for any events which is atleast 3-months continuous
			new_events[i] = bin_event 
			i=i+1
			triggers.append(new_start_args[idx])
	triggers = np.asarray(triggers, dtype = int)
	#'new_events' are the qualified continuous events with gaps and at-least one 3 month continuous event
	#'triggers' are the first args of new events
	return new_events, triggers

#to nomalize a timeseries in order to compare everything in a way they are meant to be!
def norm(arr):
	if np.max(arr) == np.min(arr) : return np.array([0]*len(arr))
	return np.array([(x-np.min(arr))/(np.max(arr)-np.min(arr)) for x in arr])

def Unit_Conversions(From ='kg m-2 s-1', To='mm day-1'):
	"""
	To assist with the unit conversion "From" to "To" units

	Returns:
	--------
	the multiplication factor and the new units ("To")	

	"""
	From	= str (From)
	To		= str (To)
	if From == To:
		unit_con_factor = 1
		unit_con_name	= To
		
	elif (From in ['kgm-2s-1', 'kg m-2 s-1']) and (To in ['mmday-1','mm day-1', 'mm/day']):
		unit_con_factor = 86400
		unit_con_name	= To

	elif (To in ['kgm-2s-1', 'kg m-2 s-1']) and (From in ['mmday-1','mm day-1', 'mm/day']):
		unit_con_factor	= 86400**(-1)
		unit_con_name	= To
	
	elif (From in ['kgm-2', 'kg m-2']) and (To in ['mm']):
		unit_con_factor = 1 
		unit_con_name	= To

	elif (To in ['kgm-2', 'kg m-2']) and (From in ['mm']):
		unit_con_factor	= 1
		unit_con_name	= To
	
	elif (From in ['K']) and (To in ['C']):
		unit_con_factor = -273.15
		unit_con_name	= To

	elif (To in ['C']) and (From in ['K']):
		unit_con_factor	= 273.15
		unit_con_name	= To

	return unit_con_factor, unit_con_name

# To make a new masked series when encounded with an error due to ma.core.MaskedConstant error
def MaskedConstant_Resolve (ar):
	"""
		To make a new masked series when encounded with an error due to ma.core.MaskedConstant error

		Input:
		------
		ar: any array or list or series

		Returns:
		-------
		a masked array with error replaced by np.nan and made invalid with masking
	"""
	from numpy.ma import masked 
	values = []
	for val in ar:
		if val is masked:
			val = np.nan 
		values.append(val)
	return np.ma.masked_invalid(values)

# To make a new masked series when encounded with an error due to numpy.ma.core.MaskedArray
def MaskedArray_Resolve (ar):
	"""
		To make a new masked series when encounded with an error due to ma.core.MaskedConstant error

		Input:
		------
		ar: any array or list or series

		Returns:
		-------
		a masked array with error replaced by np.nan and made invalid with masking
	"""
	values = []
	for val in ar:
		try:
			val = float(val)
		except:
			val = np.nan
		values.append(val)
	return np.ma.masked_invalid(values)

# Register RdGn colormap                                                          
import colorsys as cs
import matplotlib.pyplot as plt
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

