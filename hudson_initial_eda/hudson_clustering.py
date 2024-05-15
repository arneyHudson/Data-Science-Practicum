import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
from sklearn.cluster import Birch
import skfuzzy as fuzz
from tqdm import tqdm
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score
from matplotlib.patches import Ellipse
import mplcursors
import plotly.express as px
import plotly.graph_objects as go
from sklearn.covariance import empirical_covariance
from sklearn.preprocessing import StandardScaler
from mod.dataset import DataSet
from imblearn.over_sampling import RandomOverSampler

class ClusteringAnalysis:
    def __init__(self, data):
        self.data = data

    def fuzzy_clustering(self, dataset, num_clusters, fuzziness=2, max_iterations=100, tolerance=1e-5):
        # Transpose the dataset so that each row represents a data point
        dataset = dataset.T

        # Initialize the centroids randomly
        centroids = np.random.rand(num_clusters, dataset.shape[1])

        # Apply fuzzy c-means clustering
        centroids, membership, _, _, _, _, _ = fuzz.cluster.cmeans(
            dataset, num_clusters, fuzziness, error=tolerance, maxiter=max_iterations)

        return centroids, membership

    def calculate_silhouette_score(self, data, num_clusters):
        centroids, membership = self.fuzzy_clustering(data, num_clusters)
        labels = np.argmax(membership, axis=0)
        if len(np.unique(labels)) < 2:
            return -1
        score = silhouette_score(data, labels)
        return score

    def compute_covariance(self, selected_data, membership, centroids, cluster_index):
        cluster_points = selected_data[membership.argmax(axis=0) == cluster_index]
        covariance_matrix = np.cov(cluster_points.T)
        return covariance_matrix

    def plot_clusters(self, data, num_clusters, centroids, membership, feature_names):
        fig, ax = plt.subplots(figsize=(20, 10))

        for i in range(num_clusters):
            cluster_color = plt.cm.tab10(i)
            ax.scatter(data[membership.argmax(axis=0) == i, 0], data[membership.argmax(axis=0) == i, 1],
                       label=f'Cluster {i+1}', alpha=0.5, color=cluster_color)

        for i in range(num_clusters):
            cluster_color = plt.cm.tab10(i)
            ax.scatter(centroids[i, 0], centroids[i, 1], marker='X', color=cluster_color,
                       edgecolors='black', linewidth=1, s=100)

            covariance_matrix = self.compute_covariance(data, membership, centroids, i)

            ellipse = Ellipse((centroids[i, 0], centroids[i, 1]),
                              2*np.sqrt(5.991*covariance_matrix[0, 0]),
                              2*np.sqrt(5.991*covariance_matrix[1, 1]),
                              np.degrees(np.arccos(covariance_matrix[0, 1] /
                                                    np.sqrt(covariance_matrix[0, 0]*covariance_matrix[1, 1]))),
                              edgecolor=cluster_color, linestyle='--', fill=True, alpha=0.1, lw=2, facecolor=cluster_color)
            ax.add_patch(ellipse)

        ax.set_title(f'Fuzzy Clustering Results: {feature_names[0]} vs {feature_names[1]}')
        ax.set_xlabel(feature_names[0])
        ax.set_ylabel(feature_names[1])
        ax.legend()
        plt.grid(True)
        plt.show()

class Main:
    def __init__(self):
        self.final_df = DataSet().final_customer_df
        self.us_df = None
        self.non_us_df = None

    def preprocess_data(self):
        pd.set_option('display.max_columns', None)
        file_path = '~/scotforgeproject1/csv_datasets/'
        df = pd.read_csv(file_path + "Customer_Location.csv")

        df = df[df['Customer ID'].isin(self.final_df['Customer ID'])]
        df.reset_index(drop=True, inplace=True)

        df = pd.merge(df, self.final_df[['Customer ID', 'Subsegment Code']], on='Customer ID', how='left')

        df['Coordinate'] = df.apply(lambda row: (row['Longitude'], row['Latitude']), axis=1)

        region_encoded_df = pd.get_dummies(df['Region'], prefix='Region')
        df = pd.concat([df, region_encoded_df], axis=1)
        df.drop(columns=['Region'], inplace=True)

        self.us_df = df[df['Country'] == 'US']
        self.non_us_df = df[df['Country'] != 'US']
        self.us_df.reset_index(drop=True, inplace=True)
        self.non_us_df.reset_index(drop=True, inplace=True)

    def analyze(self):
        self.preprocess_data()

        # Perform clustering analysis for US customers
        us_cluster_analysis = ClusteringAnalysis(self.us_df[['Latitude', 'Longitude']].values)
        us_silhouette_scores = []
        num_clusters_range = range(2, 11)

        # Calculate silhouette scores for different numbers of clusters
        for num_clusters in tqdm(num_clusters_range, desc="Analyzing US Clusters"):
            silhouette_score = us_cluster_analysis.calculate_silhouette_score(self.us_df[['Latitude', 'Longitude']].values, num_clusters)
            us_silhouette_scores.append(silhouette_score)

        optimal_num_clusters_us = num_clusters_range[np.argmax(us_silhouette_scores)]
        print(f"Optimal number of clusters for US customers: {optimal_num_clusters_us}")

        # Perform clustering with the optimal number of clusters for US customers
        us_centroids, us_membership = us_cluster_analysis.fuzzy_clustering(self.us_df[['Latitude', 'Longitude']].values, optimal_num_clusters_us)
        us_feature_names = ['Latitude', 'Longitude']

        # Plot US clusters
        us_cluster_analysis.plot_clusters(self.us_df[['Latitude', 'Longitude']].values, optimal_num_clusters_us, us_centroids, us_membership, us_feature_names)

        # Perform clustering analysis for non-US customers
        non_us_cluster_analysis = ClusteringAnalysis(self.non_us_df[['Latitude', 'Longitude']].values)
        non_us_silhouette_scores = []

        # Calculate silhouette scores for different numbers of clusters
        for num_clusters in tqdm(num_clusters_range, desc="Analyzing Non-US Clusters"):
            silhouette_score = non_us_cluster_analysis.calculate_silhouette_score(self.non_us_df[['Latitude', 'Longitude']].values, num_clusters)
            non_us_silhouette_scores.append(silhouette_score)

        optimal_num_clusters_non_us = num_clusters_range[np.argmax(non_us_silhouette_scores)]
        print(f"Optimal number of clusters for Non-US customers: {optimal_num_clusters_non_us}")

        # Perform clustering with the optimal number of clusters for non-US customers
        non_us_centroids, non_us_membership = non_us_cluster_analysis.fuzzy_clustering(self.non_us_df[['Latitude', 'Longitude']].values, optimal_num_clusters_non_us)
        non_us_feature_names = ['Latitude', 'Longitude']

        # Plot non-US clusters
        non_us_cluster_analysis.plot_clusters(self.non_us_df[['Latitude', 'Longitude']].values, optimal_num_clusters_non_us, non_us_centroids, non_us_membership, non_us_feature_names)

if __name__ == "__main__":
    analysis = Main()
    analysis.analyze()

