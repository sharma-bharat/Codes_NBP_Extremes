# Bharat Sharma
# This file will create a .csv file with all the possile locations for a particular experiment id where that variables stored
# This code should be run at level /global/homes/b/bharat/cmip6_data/CMIP6

#cd /global/homes/b/bharat/cmip6_data/CMIP6/
cd /global/cfs/cdirs/m3522/cmip6/CMIP6/
ls
for var in gpp;
#for var in mrsos;
do
	# Removing the old files with the same name
	rm /global/homes/b/bharat/results/data_processing/ls_ssp585_$var.csv
	rm /global/homes/b/bharat/results/data_processing/ls_historical_$var.csv
	# Now creating the new files
	ls */*/*/ssp585/*/*mon/$var/*/*/*.nc >> /global/homes/b/bharat/results/data_processing/ls_ssp585_$var.csv
	ls */*/*/historical/*/*mon/$var/*/*/*.nc >> /global/homes/b/bharat/results/data_processing/ls_historical_$var.csv
done

"""
for var in ra rh;
#for var in mrsos;
do
	# Removing the old files with the same name
	rm /global/homes/b/bharat/results/data_processing/ls_ssp585_$var.csv
	rm /global/homes/b/bharat/results/data_processing/ls_historical_$var.csv
	# Now creating the new files
	ls */*/*/ssp585/*/*mon/$var/*/*/*.nc >> /global/homes/b/bharat/results/data_processing/ls_ssp585_$var.csv
	ls */*/*/historical/*/*mon/$var/*/*/*.nc >> /global/homes/b/bharat/results/data_processing/ls_historical_$var.csv
done
"""
"""
for var in gpp nbp npp nep mrso pr fFireAll tas tasmax tasmin;
#for var in mrsos;
do
	# Removing the old files with the same name
	rm /global/homes/b/bharat/results/data_processing/ls_ssp585_$var.csv
	rm /global/homes/b/bharat/results/data_processing/ls_historical_$var.csv
	# Now creating the new files
	ls */*/*/ssp585/*/*mon/$var/*/*/*.nc >> /global/homes/b/bharat/results/data_processing/ls_ssp585_$var.csv
	ls */*/*/historical/*/*mon/$var/*/*/*.nc >> /global/homes/b/bharat/results/data_processing/ls_historical_$var.csv
done
"""
"""
for var in areacella sftlf;
do
	# Removing the old files with the same name
	rm /global/homes/b/bharat/results/data_processing/ls_ssp585_$var.csv
	rm /global/homes/b/bharat/results/data_processing/ls_historical_$var.csv
	# Now creating the new files
	ls */*/*/ssp585/*/*/$var/*/*/*.nc >> /global/homes/b/bharat/results/data_processing/ls_ssp585_$var.csv
	ls */*/*/historical/*/*/$var/*/*/*.nc >> /global/homes/b/bharat/results/data_processing/ls_historical_$var.csv
done
"""


