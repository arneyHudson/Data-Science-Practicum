import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, Birch, OPTICS
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from dataset import DataSet
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
import pickle
from itertools import combinations
import argparse
import ast
import time
from tqdm import tqdm
import os


#################################################################################################################
# Data Pre processing for Model
#################################################################################################################
comb_completed = []

selected_features = [
    'Credit Limit', 
    'Salesperson', 
    'Quote Speed',
    #'Order Date'
    #'Customer Last Activated Date',
    #'Customer Date Opened',
    #'Prospect Date Opened',
    'Material Density',
    'MRR Serial Count',
    'NC Serial Count',
    'Total Shipped Quantity',
    'Total Order Price',
    'Total Order Weight',
    'Latitude', 
    'Longitude'
]

#Uncomment to quickly test the script
#selected_features = ['Credit Limit', 'Salesperson', 'Quote Speed', 'Material Density', 'MRR Serial Count']
#selected_features = ['Quote Speed', 'MRR Serial Count']

def evaluate_combination(subset_data):
    # subset_data of form (data,algorithm,hyperparams)
    algorithm = subset_data[1]
    hyperparams = subset_data[2]
    features = subset_data[3]
    # print(f"\nFeatures subset: {features_subset}")

    subset_data = subset_data[0]

    duplicate_rows = subset_data.duplicated()
    num_duplicate_rows = duplicate_rows.sum()

    if num_duplicate_rows > 2400:
        #print("Not worth investigating this feature set, most columns/features have the same value.")
        return {
            'labels': pickle.dumps('None'),
            'hyperparameters': 'None',
            'silhouette_score': np.nan
        }
    else:
        # change this based on clustering
        if(algorithm == 'dbscan'):
            labels = dbscan_clustering(subset_data, hyperparams[0], hyperparams[1])
        elif(algorithm == 'kmeans'):
            labels = kmeans_clustering(subset_data, hyperparams[0])
        elif(algorithm == 'birch'):
            labels = birch_clustering(subset_data, hyperparams[0], hyperparams[1])
        elif(algorithm == 'optics'):
            labels = optics_clustering(subset_data, hyperparams[0], hyperparams[1], hyperparams[2])
        else:
            raise Exception("Algorithm requested is not in list of available options!")
            
        
        silhouette = silhouette_evaluation(subset_data, labels)
        comb_completed.append(1)
        #print(len(comb_completed))
        results = {
            'labels': pickle.dumps(labels),
            'silhouette_score': silhouette,
            'hyperparameters': None,
            'features': list(map(lambda f: selected_features[f], list(features)))
        }
        if(algorithm == 'dbscan'):
            results['hyperparameters'] = {'eps': hyperparams[0], 'min_samples':hyperparams[1]}
        elif(algorithm == 'kmeans'):
            results['hyperparameters'] = {'n_clusters': hyperparams[0]}
        elif(algorithm == 'birch'):
            results['hyperparameters'] = {'threshold': hyperparams[0], 'branching_factor': hyperparams[1]}
        elif(algorithm == 'optics'):
            results['hyperparameters'] = {'min_samples': hyperparams[0], 'xi': hyperparams[1], 'min_cluster_size': hyperparams[2]}
        return results

def load_dataset():
    """
    Load dataset from file.
    """
    print("\nLoading Data")
    final_df = DataSet().final_customer_df

    columns_to_categorical = ['Customer Status', 'Customer Type', 'Default Currency Indicator',
                          'Customer Category', 'Pricing Category', 'Region Code', 'Country Code',
                          'Account Manager', 'Transit Days from Ringmasters', 
                          'Transit Days from Clinton', 'Transit Days from Spring Grove',
                          'Transit Days from Franklin Park', 'Salesperson', 'City']

    for column in final_df.columns:
        if not pd.api.types.is_integer_dtype(final_df[column]) and not pd.api.types.is_float_dtype(final_df[column]):
            final_df[column] = final_df[column].astype('category')

    # Convert columns to categorical data type
    final_df[columns_to_categorical] = final_df[columns_to_categorical].astype('category')

    columns_to_drop = [col for col in final_df.columns if 'date' in col.lower() or 'name' in col.lower() or 'address' in col.lower()]
    final_df = final_df.drop(columns=columns_to_drop)

    final_df = final_df[selected_features]
    print("Data Loaded\n")
    return final_df

def scale_data(data):
    """
    Preprocess the data: fill missing values, scale features.
    """
    print("Preprocessing Data")
    #non_categorical_cols = data.select_dtypes(exclude=['category']).columns
    #data[non_categorical_cols] = data[non_categorical_cols].fillna(0)
    
    # One-hot encode categorical columns
    #categorical_cols = data.select_dtypes(include=['category']).columns
    #data = pd.get_dummies(data, columns=categorical_cols)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    print("Preprocessing Finished\n")

    data = remove_constant_columns(scaled_data)
    return data

def remove_constant_columns(data):
    """
    Remove columns with constant values across all rows.
    """
    data = pd.DataFrame(data)
    constant_columns = data.columns[data.nunique() == 1]
    data = data.drop(columns=constant_columns)
    return data

def append_to_csv(results, filename):
    """
    Append the results to an existing CSV file or create a new one if it doesn't exist.
    """
    if os.path.isfile(filename):
        results.to_csv(filename, mode='a', header=False, index=False)
    else:
        results.to_csv(filename, index=False)

#################################################################################################################
# Clustering Algorithms
#################################################################################################################


def dbscan_clustering(data, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(data)

def kmeans_clustering(data, n_clusters):
    """
    Perform k-means clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters)
    return kmeans.fit_predict(data)

def birch_clustering(data, threshold, branching_factor):
    #adapted from sai_cluster.py
    birch = Birch(threshold=threshold, branching_factor=branching_factor)
    return birch.fit_predict(data)

def optics_clustering(data, min_samples, xi=0.05, min_cluster_size=0.1):
    #adapted from sai_cluster.py
    optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size)
    return optics.fit_predict(data)

def random_forests(data, target):
    random_forest = RandomForestClassifier()
    random_forest.fit(data, target)
    predictions = random_forest.predict(data)
    return predictions

def silhouette_evaluation(data, labels):
    """
    Evaluate clustering using silhouette score.
    """
    # Filter out noise points for DBSCAN
    non_noise_mask = labels != -1
    non_noise_data = data[non_noise_mask]
    non_noise_labels = labels[non_noise_mask]

    if len(np.unique(non_noise_labels)) > 1 and len(np.unique(non_noise_labels)) <= len(non_noise_labels) - 1:
        return silhouette_score(non_noise_data, non_noise_labels)
        #return 0.0
    else:
        return 0.0


#################################################################################################################
# Model Creation
#################################################################################################################

def load_and_preprocess_data():
    """
    Load dataset and preprocess data.
    """
    dataset = load_dataset()
    preprocessed_data = scale_data(dataset)
    return preprocessed_data

def explore_feature_subsets(preprocessed_data_df, algorithm, algorithm_params, filename):
    """
    Explore different feature subsets for clustering.
    """

    features = list(preprocessed_data_df.columns)

    feature_combinations = []
    for i in range(2, len(features)+1):
        feature_combinations += list(combinations(features, i))
    #print(feature_combinations)
    
    subset_combinations = list(map(lambda c: preprocessed_data_df.loc[:, c], feature_combinations))
    subset_combo_tuples = []
    for i in range(len(subset_combinations)):
        subset_combo_tuples.append((subset_combinations[i],algorithm,algorithm_params,feature_combinations[i]))
    entries = list(tqdm(map(evaluate_combination, subset_combo_tuples), total=len(subset_combo_tuples)))

    print("end")
    results = pd.DataFrame(entries)
    results = results.sort_values('silhouette_score', ascending=False)

    # Filters to only save the models with S.S. over __
    results = results[(results['silhouette_score'] > 0.8)]
    results.reset_index(inplace=True, drop=True)
    count = 0
    filtered_results = pd.DataFrame(columns=results.columns)
    for i in range(len(results)):          
        labels = pickle.loads(results['labels'].loc[i])
        df = pd.DataFrame(columns=['labels'])
        df['labels'] = labels
        
        largest_cluster_percentage = df['labels'].value_counts(normalize=True).max() * 100
        
        if largest_cluster_percentage < 85:
            filtered_results = filtered_results.append(results.loc[i], ignore_index=True)

    # Save the filtered results to a CSV file
    if(filename == None):
        filename = "/home/arneyh/scotforgeproject1/new_results_csv/models_to_analyze.csv"
    append_to_csv(filtered_results, filename)
    #filtered_results.to_csv(path_for_results + "results_"+algorithm+str(algorithm_params)+".csv", index=False)

class LoadModels():

    def __init__(self, filename):
        self.models = pd.read_csv(filename)

    def get_labels(self, entry):
        return pickle.loads(ast.literal_eval(entry))

class LoadModels():

    def __init__(self, filename):
        self.models = pd.read_csv(filename)

    def get_labels(self, entry):
        return pickle.loads(ast.literal_eval(entry))


#################################################################################################################
# Main Running
#################################################################################################################

def main():
    parser = argparse.ArgumentParser(description="Create and save all possible clusters for a given clustering algorithm.\n Data is taken from the dataset.py file.")
    parser.add_argument('algorithm', choices=['dbscan','kmeans','birch','optics'], help='The algorithm used for the clustering.')
    parser.add_argument('--noecho', action='store_true', help='When this flag is called, the program will not echo the user\'s choice')
    parser.add_argument('--eps', type=float, nargs='?', default=0.5, help='(FLOAT) The eps hyperparameter for DBSCAN. Default 0.5')
    parser.add_argument('--min_samples_dbscan', type=int, nargs='?', default=5, help='(INT) The min_samples hyperparameter for DBSCAN. Default 5')
    parser.add_argument('--k_cluster','-k', type=int, nargs='?', default=5, help='(INT) The number of clusters for KMEANS to produce. Deafult 5')
    parser.add_argument('--threshold', type=float, nargs='?', default=0.5, help='(FLOAT) The threshold hyperparameter for BIRCH. Default 0.5')
    parser.add_argument('--branching_factor', type=int, nargs='?', default=50, help='(INT) The branching factor hyperparameter for BIRCH. Default 50')
    parser.add_argument('--min_samples_optics', type=int, nargs='?', default=10, help='(INT) The min_samples hyperparameter for OPTICS. Default 10')
    parser.add_argument('--xi', type=float, nargs='?', default=0.05, help='(FLOAT) The xi hyperparameter for OPTICS. Default 0.05')
    parser.add_argument('--min_cluster_size', type=float, nargs='?', default=0.1, help='(FLOAT) The min_cluster_size hyperparameter for OPTICS. Default 0.1')
    parser.add_argument('--dataset-path', type=str, help='File path to an alternate dataset contained in a .CSV that can be loaded into a Pandas dataframe. Feature names are expected on the first row.')
    parser.add_argument('--nopreprocess', action='store_true', help='When this flag is called, if an alternate dataset is loaded, it will not be preprocessed.')
    parser.add_argument('--filename_out', type=str, help='The filename of the exported CSV, do not include the .csv')
    parsed = parser.parse_args()
    algorithm = parsed.algorithm
    print(parsed.dataset_path)
    # Assign params depending on the algorithm selected 
    params = []
    if(algorithm=='kmeans'):
        params.append(parsed.k_cluster)
    elif(algorithm=='dbscan'):
        params.append(parsed.eps)
        params.append(parsed.min_samples_dbscan)
    elif(algorithm=='birch'):
        params.append(parsed.threshold)
        params.append(parsed.branching_factor)
    elif(algorithm=='optics'):
        params.append(parsed.min_samples_optics)
        params.append(parsed.xi)
        params.append(parsed.min_cluster_size)
        
    if(not parsed.noecho):
        print("Checking all feature combinations with "+algorithm+" with hyperparameters:\n\r\t",end='')
        if(algorithm=='kmeans'):
            print("k="+str(params[0]))
        elif(algorithm=='dbscan'):
            print("eps="+str(params[0])+" min_samples="+str(params[1]))
        elif(algorithm=='birch'):
            print("threshold="+str(params[0])+" branching_factor="+str(params[1]))
        elif(algorithm=='optics'):
            print("min_samples="+str(params[0])+" xi="+str(params[1])+" min_cluster_size="+str(params[2]))
    
    if(parsed.dataset_path==None):
        preprocessed_data = load_and_preprocess_data()
    else:
        print("Loading data from "+parsed.dataset_path)
        preprocessed_data = pd.read_csv(parsed.dataset_path)[selected_features]
        print("Data loaded from "+parsed.dataset_path+" successfully.")
        if(not parsed.nopreprocess):
            preprocessed_data = scale_data(preprocessed_data)
    start_time = time.time()
    explore_feature_subsets(preprocessed_data, algorithm, params, parsed.filename_out)
    end_time = time.time()
    execution_time = end_time - start_time
    print("Execution time:", execution_time, "seconds")
    #explore_feature_subsets(preprocessed_data, 'dbscan', [0.5])

if __name__ == "__main__":
    main()
