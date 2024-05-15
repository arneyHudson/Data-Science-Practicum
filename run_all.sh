#!/bin/bash

# Define the list of eps values for DBSCAN
eps_values=(0.1 0.2 0.3 0.4 0.5 1.0)

# Define the list of min_samples_dbscan values for DBSCAN
min_samples_values=(3 7 10)

# Define the list of threshold values for BIRCH
threshold_values=(0.3 0.4 0.5 0.6 0.7)

# Define the list of branching_factor values for BIRCH
branching_factor_values=(40 45 50 55 60)

# Loop through each combination of eps and min_samples_dbscan values and run feature_script.py with DBSCAN
for eps in "${eps_values[@]}"
do
    for min_samples_dbscan in "${min_samples_values[@]}"
    do
        python3 mod/feature_script.py dbscan --eps $eps --min_samples_dbscan $min_samples_dbscan &
    done
done

# Loop through each combination of threshold and branching_factor values and run feature_script.py with BIRCH
for threshold in "${threshold_values[@]}"
do
    for branching_factor in "${branching_factor_values[@]}"
    do
        python3 mod/feature_script.py birch --threshold $threshold --branching_factor $branching_factor &
    done
done

# Wait for all background processes to finish
wait
