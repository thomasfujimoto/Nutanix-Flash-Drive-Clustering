
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_analysis.kmeans_cluster import KMeans_Cluster # Get clustering functions
from data_access.postgres_handler import PostgresHandler

class KMeans_Predict:
    def __init__(self):
        self.km_cluster = None
        self.input_columns = [
        'blocksize', 'num_jobs', 'queue_depth',
            'operating_pci_speed_gts', 'operating_pci_width'
        ]
        self.predict_columns = [
        'blocksize', 'effective_queue_depth',
            'operating_pci_speed_gts', 'operating_pci_width'
        ]
        self.data_type = None 
        self.metric_type = None
        
    
    # Connect to dataset and get 60% for training dataset and 40% for validation
    def get_prediction_data(self, training=False): 
        # Initialize the PostgresHandler
        handler = PostgresHandler()
        handler.connect()

        # Define columns to fetch
        columns = [
            'concord_id', 'data_type', 'name', 'metric', 'queue_depth', 'num_jobs', 'blocksize','unit', 'min_measure', 
            'mean_measure', 'median_measure', 'max_measure', 'stddev_measure', 'capacity_gib', 
            'device_type', 'model', 'operating_pci_speed_gts', 'operating_pci_width', 
        ]

       
        df_training = handler.get_data("training_data", columns, limit=None, encode=False)
        df_prediction = handler.get_data("validation_data", columns, limit=None, encode=False)
            
        # Disconnect from the database
        handler.disconnect()
        
        # Number of rows    
        print("Number of rows in Traing Data Set:", df_training.shape[0])
        print("Number of rows in Prediction Data Set:", df_prediction.shape[0])
        
        return df_training, df_prediction
    
    # Cleans up validation data to only include appropriate data based on data_type and metric
    def clean_up(self, df, dt, metric):
        filtered_df = df[(df['data_type'] == dt) & (df['metric'] == metric)]
        
        print(f"Filtered DataFrame contains {filtered_df.shape[0]} rows after applying filters for data_type='{dt}' and metric='{metric}'.")
        
        return filtered_df


    def add_effective_queue_depth(self, df):
        if 'num_jobs' not in df.columns or 'queue_depth' not in df.columns:
            raise ValueError("DataFrame must contain 'num_jobs' and 'queue_depth' columns.")
        
        df = df.copy()
        df['effective_queue_depth'] = df['num_jobs'] * df['queue_depth']
        print(df)
        
        print("Added 'effective_queue_depth' to the DataFrame.")
        return df

    # Predict the cluster for a random drive
    def km_predict(self, km, new_data):

        # Check if input columns are in new_data
        for col in self.input_columns:
            if col not in new_data.columns:
                raise ValueError(f"Column '{col}' is missing from the input data.")

        # Compute 'effective_queue_depth' and update new_data
        new_data = self.add_effective_queue_depth(new_data)

        # Filter the necessary columns
        new_data_filtered = new_data[self.predict_columns]
        print("Predicting Cluster Based on the following Data", new_data_filtered.to_string())

        # Apply the same scaling
        new_data_scaled = km.scaler.transform(new_data_filtered)
        print("Data After being scaled", new_data_scaled)
        
        # Apply PCA transformation
        new_data_pca = km.pca.transform(new_data_scaled)
        print("Data After PCA Transformation:\n", new_data_pca)

        # Predict cluster labels
        cluster_labels = km.model.predict(new_data_pca)

        return cluster_labels

        
    def get_valid_input(self, prompt, min_value, max_value):
        while True:
            user_input = input(prompt).strip()

            if user_input.isdigit() and min_value <= int(user_input) <= max_value:
                return int(user_input)
            else:
                print(f"Invalid input. Please enter a number between {min_value} and {max_value}.\n")

    def input_type(self):
        # Get data type from user
        print("Please choose a data type:")
        print("0 - Random Read")
        print("1 - Random Write")
        print("2 - Sequential Write")
        print("3 - Sequential Read")
        dt = self.get_valid_input("Enter a number between 0 and 3: ", 0, 3)

        # Get metric from user
        print("Please choose a metric:")
        print("0 - Bandwidth (bw)")
        print("1 - IOPS")
        print("2 - Latency")
        metric = self.get_valid_input("Enter a number between 0 and 2: ", 0, 2)
        
        # Save user input 
        data_types = ['Random Read', 'Random Write', 'Sequential Read', 'Sequential Write']
        metric_types = ['bw', 'iops','latency']
        
        self.data_type = data_types[dt]
        self.metric_type =  metric_types[metric]
        
        # Ask if user wants to use existing or new data
        print("Would you like to use an existing data or input new data?")
        print("0 - Existing Data")
        print("1 - New Data")
        new_data = self.get_valid_input("Enter a number 0 or 1: ", 0, 1)
        
        if new_data == 0: 
            return False 
        
        else: 
            return True
        
    def input_value(self):
        # Get block size from user
        blocksize = self.get_valid_input("Enter a block size between 4096 and 1048576: ", 4096, 1048576)

        # Get other parameters
        num_jobs = self.get_valid_input("Enter the number of jobs between 1 and 64: ", 1, 64)
        queue_depth = self.get_valid_input("Enter queue depth between 1 and 128: ", 1, 128)
        operating_pci_speed_gts = self.get_valid_input("Enter operating PCI speed (GTS) between 1 and 16: ", 1, 16)
        operating_pci_width = self.get_valid_input("Enter operating PCI width between 1 and 16: ", 1, 16)

        # Create DataFrame entry
        data_entry = {
            'data_type': self.data_type,
            'metric': self.metric_type,
            'blocksize': blocksize,
            'num_jobs': num_jobs,
            'queue_depth': queue_depth,
            'operating_pci_speed_gts': operating_pci_speed_gts,
            'operating_pci_width': operating_pci_width
        }
        
        # Create a DataFrame with a single row containing the user's input
        df_entry = pd.DataFrame([data_entry])

        # Print the DataFrame
        print("\nCreated DataFrame Entry:")
        print(df_entry)

        return df_entry

    def prd_existing_data(self, km, df_clean): 
        # Get current drive models 
        unique_models = df_clean['model'].unique()
        while True:
            print("Please Copy and paste your desired model from the following list:")
            print(unique_models)
            user_input = input("Paste Model here: ")
            
            if user_input not in unique_models: 
                print("Error Model not found, Please Enter a valid Model")
                
            else:
                break
        
        # Get one random data entry with the chosen model 
        model_data = df_clean[df_clean['model'] == user_input].sample(n=1)
        
        # print out model data
        print("Model Data")
        print(model_data)
        
        # Predict with given model 
        print("Model is predicted to be in cluster:", self.km_predict(km, model_data))
    
# Run Main Function (FOR TESTING)
if __name__ == "__main__":
    # Create a new KMeans Class 
    km = KMeans_Cluster()
    prd = KMeans_Predict()
    
    # Store the new instance of KMeans in prd 
    prd.km_cluster = km
    
    #  Get training and validation data 
    df_training, df_prediction= prd.get_prediction_data()
    
    
    while True:
        # Get input from user for data_type, metric and wether or not to use existing or new data
        new_data = prd.input_type()
        print(prd.data_type)
        print(prd.metric_type)
        
        # Run Kmeans on training data based on user input for data type and metric
        km.run_alg(df_training, prd.data_type, prd.metric_type, visual=True)
        
        if new_data: 
            df_prd = prd.input_value()
            # Predict based on input data 
            prediction = prd.km_predict(km, df_prd)
            print(prd.km_predict(km, df_prd))
            print("Drive is in Tier : ", km.tiers[int(prediction[0])])
            
        
        else: 
            # Clean up the prediction data
            df_clean = prd.clean_up(df_prediction, prd.data_type, prd.metric_type)
            prd.prd_existing_data(km, df_clean)
            
            
        
        
    