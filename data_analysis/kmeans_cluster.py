import streamlit as st
import sys
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Add the project root directory to the system path
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_access.postgres_handler import PostgresHandler

class KMeans_Cluster:
    # Connect to df 
    def __init__(self):
        self.model = None
        self.scaler = None
        self.pca = None
        self.tiers = {}
        self.tiers_weighted = {}
        
    def connect(self): 
        # Initialize the PostgresHandler
        handler = PostgresHandler()
        handler.connect()

        # Define columns to fetch
        columns = [
            'concord_id', 'data_type', 'name', 'metric', 'queue_depth', 'num_jobs', 'blocksize','unit', 'min_measure', 
            'mean_measure', 'median_measure', 'max_measure', 'stddev_measure', 'capacity_gib', 
            'device_type', 'model', 'operating_pci_speed_gts', 'operating_pci_width', 
        ]

        df = handler.get_data("training_data", columns, limit=None, encode=False)

        # Disconnect from the database
        handler.disconnect()
        
        # Return the data frame for clustering
        return df


    # Filter the df based on type, returns filterd data frame based on data type and metric
    def filter_df(self, df, data_type, metric): 
        df_filtered = df[df['data_type'] == data_type].reset_index(drop=True)
        df_filtered = df[df['metric'] == metric].reset_index(drop=True)
        
        return df_filtered

    # Remove outliers from DF
    def remove_outliers(self, df_filtered, input_columns):
        # features to be used for clutering
        
        # Normalize the data
        self.scaler = StandardScaler()
        df_scaled = self.scaler.fit_transform(df_filtered[input_columns])

        # Step 2: Outlier Detection and Removal using IQR
        Q1 = np.percentile(df_scaled, 25, axis=0)
        Q3 = np.percentile(df_scaled, 75, axis=0)
        IQR = Q3 - Q1

        # Define outlier thresholds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify and remove outliers
        outlier_indices = []
        for i in range(df_scaled.shape[1]):
            outlier_list_col = df_filtered.index[(df_scaled[:, i] < lower_bound[i]) | (df_scaled[:, i] > upper_bound[i])].tolist()
            outlier_indices.extend(outlier_list_col)

        outlier_indices = list(set(outlier_indices))  # Remove duplicates
        df_no_outliers  = df_filtered.drop(index=outlier_indices).reset_index(drop=True)
        np_array_no_outliers = np.delete(df_scaled, outlier_indices, axis=0)
        
        return np_array_no_outliers, df_no_outliers

    # PCA Alogrithm
    def PCA_Alg(self, np_array): 
        self.pca = PCA(n_components=0.95, random_state=42)
        np_pca_no_outliers = self.pca.fit_transform(np_array)
        print("Variance explained by PCA components:", self.pca.explained_variance_ratio_)

        return np_pca_no_outliers
    
    def add_effective_queue_depth(self, df):
        if 'num_jobs' not in df.columns or 'queue_depth' not in df.columns:
            raise ValueError("DataFrame must contain 'num_jobs' and 'queue_depth' columns.")
        
        df = df.copy()
        df['effective_queue_depth'] = df['num_jobs'] * df['queue_depth']
        print(df)
        
        print("Added 'effective_queue_depth' to the DataFrame.")
        return df


    # Kmeans Clustering 
    def KMean_Clustering(self, np_array, df, num_clusters): 
        
        kmeans = KMeans(
            n_clusters=num_clusters,
            init='k-means++',
            n_init=1000,
            max_iter=5000,
            random_state=42
        )
        df['Cluster'] = kmeans.fit_predict(np_array)
        
        self.model = kmeans
        
        return df
    

    def run_alg(self, df, type, metric, num_clusters=4, visual=True):  
        print("PROCESSING DATA TYPE : ", type)
        print("PROCESSING METRIC TYPE :", metric)
        
        # Features to be included in clustering
        input_columns = [
            # 'mean_measure',
            'blocksize', 
            # 'num_jobs', 
            # 'queue_depth', 
            'effective_queue_depth',  # New Feature
            'operating_pci_speed_gts',  
            'operating_pci_width'
        ] 
        
        # Step 1: Compute Effective Queue Depth
        df = self.add_effective_queue_depth(df)
        
        # Step 2: Filter the data frame based on data type and metric
        df_filtered = self.filter_df(df, type, metric) 
        
        # Step 3: Remove outliers 
        np_array, df_no_outliers = self.remove_outliers(df_filtered, input_columns)
        
        # Step 4: Run through PCA 
        np_pca_no_outliers = self.PCA_Alg(np_array)
        
        # Step 5: Run through KMeans 
        df = self.KMean_Clustering(np_pca_no_outliers, df_no_outliers, num_clusters)
        
        # Assign Tier to clusters based on Mean Measure
        # Calculate Tie of 'mean_measure' for each cluster
        cluster_mean_measure = df.groupby('Cluster')['mean_measure'].mean().reset_index()
        
        # Sort clusters based on 'mean_measure' in descending order
        cluster_mean_measure_sorted = cluster_mean_measure.sort_values(by='mean_measure', ascending=False).reset_index(drop=True)
    
        # Assign ranks to clusters
        if metric == 'latency': 
            cluster_mean_measure_sorted['Tier'] = cluster_mean_measure_sorted['mean_measure'].rank(method='dense', ascending=False).astype(int)
        
        else: 
            cluster_mean_measure_sorted['Tier'] = cluster_mean_measure_sorted['mean_measure'].rank(method='dense', ascending=True).astype(int)

        # Store Cluster and the tier into a dictionary
        self.tiers = cluster_mean_measure_sorted.set_index('Cluster')['Tier'].to_dict()
        
        # Map the tiers back to the original DataFrame
        df['Tier'] = df['Cluster'].map(self.tiers)
        
        # Calculate weighted mean measure for each row
        df['weighted_mean_measure'] = df['mean_measure'] / df['effective_queue_depth']

        # Handle division by zero or NaN values in effective queue depth
        df['weighted_mean_measure'] = df['weighted_mean_measure'].replace([float('inf'), -float('inf')], float('nan')).fillna(0)

        # Group by Cluster to calculate the mean of weighted mean measure
        cluster_weighted_mean = df.groupby('Cluster')['weighted_mean_measure'].mean().reset_index()

        # Rename the column for clarity
        cluster_weighted_mean.rename(columns={'weighted_mean_measure': 'cluster_weighted_mean_measure'}, inplace=True)

        # Merge the cluster-weighted mean back into the sorted DataFrame
        cluster_mean_measure_sorted = cluster_mean_measure_sorted.merge(cluster_weighted_mean, on='Cluster', how='left')

        # Assign tiers based on the cluster-weighted mean measure
        if metric == 'latency':
            # Smaller weighted mean measure is better for latency
            cluster_mean_measure_sorted['Tier_weighted'] = cluster_mean_measure_sorted['cluster_weighted_mean_measure'].rank(method='dense', ascending=False).astype(int)
        else:
            # Higher weighted mean measure is better for metrics like bandwidth or IOPS
            cluster_mean_measure_sorted['Tier_weighted'] = cluster_mean_measure_sorted['cluster_weighted_mean_measure'].rank(method='dense', ascending=True).astype(int)

        # Store Cluster and the tier into a dictionary
        self.tiers_weighted = cluster_mean_measure_sorted.set_index('Cluster')['Tier_weighted'].to_dict()
        
        # Map the tiers back to the original DataFrame
        df['Tier_weighted'] = df['Cluster'].map(self.tiers_weighted)
        
        # Step 6: If visual turned on, print visuals
        if visual:  
            df.to_csv('cluster_statistics_full.csv', index=False)
            print("=== Comprehensive Cluster Statistics exported to 'cluster_statistics_full.csv' ===")
                        
            # Calculate Mean of 'mean_measure' for each cluster
            cluster_mean_measure = df.groupby('Cluster')['mean_measure'].mean().reset_index()
            
            # Sort clusters based on 'mean_measure' in descending order
            cluster_mean_measure_sorted = cluster_mean_measure.sort_values(by='mean_measure', ascending=False).reset_index(drop=True)
            
            # Assign ranks to clusters
            cluster_mean_measure_sorted['Tier'] = cluster_mean_measure_sorted['mean_measure'].rank(method='dense', ascending=False).astype(int)
            
            print("\n=== Cluster Ranking Based on 'mean_measure' ===")
            print(cluster_mean_measure_sorted)
            
            # Analyze Cluster Characteristics
            cluster_numerical_stats = df_no_outliers.groupby('Cluster')[input_columns].mean().reset_index()
            # cluster_numerical_stats = df_no_outliers.groupby('Cluster')[input_columns].median().reset_index()

            # Merge the ranking with numerical stats for comprehensive insight
            cluster_stats_ranked = pd.merge(cluster_mean_measure_sorted, cluster_numerical_stats, on='Cluster')
            
            print("\n=== Cluster Numerical Statistics with Ranking ===")
            print(cluster_stats_ranked)
            cluster_stats_ranked.to_csv('cluster_statistics_w_Ranking.csv', index=False)
            
            # Count the number of drives in each cluster
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Cluster', 'Number_of_Drives']
            
            print("\n=== Number of Drives in Each Cluster ===")
            print(cluster_counts)
            
        return df


# Run Main Function (FOR TESTING)
if __name__ == "__main__":

    # Create new KMeans Class 
    km= KMeans_Cluster()
    
    #  Get Df
    df = km.connect()
    
    # run the entire algorithm and return clustered df
    df = km.run_alg(df, 'Random Read', 'iops', True)
    
    print(df)
    
    
    
    
    

