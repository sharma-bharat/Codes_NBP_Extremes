# Bharat Sharma
# python 3.7
# Read file names and sort data and file paths

import pandas 	as pd
import glob
import numpy 	as np
import netCDF4	as nc4
import os

web_path	= '/global/homes/b/bharat/results/web/'
in_path		= '/global/homes/b/bharat/results/data_processing/'
files_list 	= glob.glob(in_path+'ls_*.csv')

# Extracting Experiment Vars and Filenames
# --------------------------------------
# Creating the data frame for every experiment and variable 
dict_exp_vars = {} # columns =  ['activity_id','institution_id','source_id','experiment_id','member_id','table_id','variable_id','grid_label','version','filenames']
# dict_exp_vars ['experiment']['variable'] < strucutre
experiments = []
variables	= []
experiment_vars_fname = np.asarray(np.zeros((len(files_list),3)), dtype = 'str')
for idx in range(len(files_list)):
	exp_tmp		= files_list[idx].split('/')[-1].split('_')[1] 
	var_tmp		= files_list[idx].split('/')[-1].split('_')[-1].split('.')[0]
	file_tmp	= files_list[idx].split('/')[-1]
	experiment_vars_fname[idx,0] = exp_tmp		# experiment
	experiment_vars_fname[idx,1] = var_tmp	  	# variables
	experiment_vars_fname[idx,2] =file_tmp		# filename
	print (exp_tmp,var_tmp)
	df_var						 = pd.read_csv(in_path + file_tmp, header =None)
	if exp_tmp not in experiments :
		experiments		.append(exp_tmp) 					# list of experiments
	
	if var_tmp not in variables:
		variables		.append(var_tmp) 	# list of variables 
 	
	if exp_tmp not in  dict_exp_vars : 				 	# building dictionary
		dict_exp_vars[exp_tmp] = {}
		dict_exp_vars[exp_tmp] [var_tmp]  = pd.DataFrame(df_var.iloc[:,0].str.split('/').tolist()
				, columns = ['activity_id','institution_id','source_id','experiment_id','member_id','table_id','variable_id','grid_label','version','filenames']) 

	else:
		dict_exp_vars[exp_tmp] [var_tmp]  = pd.DataFrame(df_var.iloc[:,0].str.split('/').tolist()
				, columns = ['activity_id','institution_id','source_id','experiment_id','member_id','table_id','variable_id','grid_label','version','filenames'])  

# making only one dataframe with all the information
# --------------------------------------------------
df_column_names	= ['activity_id','institution_id','source_id','experiment_id','member_id','table_id','variable_id','grid_label','version','filenames']
df_data 		= pd.DataFrame(columns=df_column_names)     
for exp in experiments:
	for var in variables:
		df_data = df_data.append(dict_exp_vars[exp][var], ignore_index=True)

# df_data is the complete dataframe of all the data that i need for this project
# ------------------------------------------------------------------------------
# Saving the data frame:
# ----------------------
# Removing the files that are about to be saved, sometimes the previous file is not being updated		
try:
	os. remove (in_path 	+ 'df_data_selected.csv')
	os. remove (web_path 	+ 'df_data_selected.csv')
	os. remove (in_path 	+ 'df_data_selected.xlsx')
	os. remove (web_path 	+ 'df_data_selected.xlsx')
except:
	print ("The files does not exist: \n%s\n%s"%(in_path + 'df_data_selected.csv',web_path + 'df_data_selected.csv'))

df_data.to_csv	(in_path 	+ 'df_data_selected.csv',	index=False)
df_data.to_csv	(web_path 	+ 'df_data_selected.csv',	index=False)
df_data.to_excel(in_path 	+ 'df_data_selected.xlsx',	index=False)
df_data.to_excel(web_path 	+ 'df_data_selected.xlsx',	index=False)

print ("The shape of the dataframe %s"%str(df_data.shape))
# Storing the unique enteries in arrays
# ------------------------------------

# Source_id or Model Names:
# -------------------------
source_ids		= np.unique(df_data['source_id'])

# Variable names:
# --------------
variable_ids	= np.unique(df_data['variable_id'])

# Ensemble of models/source ids for every experiment and variable:
# ---------------------------------------------------------------
dict_ensemble	= {}
for model in source_ids:
	dict_ensemble [model] = {}
	for exp in experiments:
		dict_ensemble [model][exp] = {}
		for var in variable_ids:
			filters = (df_data['source_id'] == model) & (df_data['experiment_id'] == exp) & (df_data ['variable_id'] ==var)
			dict_ensemble [model][exp][var] = np.array(df_data[filters]['member_id'])
			
# Dictionary hierarcy :  1. source_id, 2. experiment_id , 3. variable_id




# df_data is the complete dataframe of all the data that i need for this project
# ------------------------------------------------------------------------------

# ploting GPP from all the model and their ensembles
# --------------------------------------------------
#filters	= (

# Extracting the variables from models of different experiments:
# --------------------------------------------------------------

dict_data = {}
dict_data ['source'] = {}
dict_data ['source']['experiment'] = {}
dict_data ['source']['experiment']['member'] = {}
dict_data ['source']['experiment']['member']['variable'] = {}
dict_data ['source']['experiment']['member']['variable']['version'] = {}
dict_data ['source']['experiment']['member']['variable']['version']['filename'] = {}

source_list = ['ScenarioMIP', 'CMIP']
#food for thought
#filters = (temp['experiment_id']=='ssp585') & (temp['institution_id'] == 'MIROC')
		
""" Summary of what data we have:
"""
columns =  ['institution_id','source_id','experiment_id','variable_id','grid_label']
All = ['activity_id','institution_id','source_id','experiment_id','member_id','table_id','variable_id','grid_label','version','filenames']
columns_drop = ['activity_id','member_id','table_id','version','filenames']

# Creating an empty dataframe for summarizing the data sources
# ------------------------------------------------------------

df_brief = df_data.head()
df_brief = df_brief.drop(columns=columns_drop) 
df_brief = df_brief.drop([0,1,2,3,4])      
df_brief = pd.DataFrame(df_brief) 

df_brief2 = df_brief.drop(columns=['grid_label'])     

"""
for ins in np.unique(df_data['institution_id']):
	for source in np.unique(df_data['source_id']):
		for exp in np.unique(df_data['experiment_id']):
			for var in np.unique(df_data['variable_id']):
				df_brief2 = df_brief2.append(pd.DataFrame([[ins,source,exp,var]],columns =  ['institution_id', 'source_id', 'experiment_id', 'variable_id']), ignore_index=True)
				for gr in np.unique(df_data['grid_label']):
					df_brief = df_brief.append(pd.DataFrame([[ins,source,exp,var,gr]],columns =  ['institution_id', 'source_id', 'experiment_id', 'variable_id', 'grid_label']), ignore_index=True)
"""


i = 0
while i < len(df_data)-1:
	ins_1 		= df_data.iloc[i+1].institution_id
	source_1 	= df_data.iloc[i+1].source_id
	exp_1 		= df_data.iloc[i+1].experiment_id
	var_1		= df_data.iloc[i+1].variable_id
	gn_1 		= df_data.iloc[i+1].grid_label


	ins 	= df_data.iloc[i].institution_id
	source 	= df_data.iloc[i].source_id
	exp 	= df_data.iloc[i].experiment_id
	var 	= df_data.iloc[i].variable_id
	gn 		= df_data.iloc[i].grid_label
#print (i,source)
	i = i+1
	print (i,"|", ins, ins_1,"|",source,source_1,"|",exp,exp_1,"|",var,var_1 ,"|" ,gn,gn_1)
	if (ins != ins_1) or (source !=source_1) or (exp!=exp_1) or (var!=var_1) or (gn!=gn_1):
		df_brief = df_brief.append(pd.DataFrame([[ins,source,exp,var,gn]],columns =  ['institution_id', 'source_id', 'experiment_id', 'variable_id', 'grid_label']), ignore_index=True)
		print (i,source)

# Removing the files that are about to be saved, sometimes the previous file is not being updated		
try:
	os. remove (in_path + 'df_df_data_summary_variables_grids.csv')
	os. remove (web_path + 'df_data_summary_variables_grids.csv')
except:
	print ("The files does not exist: \n%s\n%s"%(in_path + 'df_data_summary_variables_grids.csv',web_path + 'df_data_summary_variables_grids.csv'))

df_brief.to_csv(in_path + 'df_data_summary_variables_grids.csv',index=False)
df_brief.to_csv(web_path + 'df_data_summary_variables_grids.csv',index=False)
