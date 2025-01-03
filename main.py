import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

from data_access.postgres_handler import PostgresHandler
from data_analysis.kmeans_cluster import KMeans_Cluster
from data_analysis.kmeans_predict import KMeans_Predict


# ===============================
# Gets Environment Variables 
# ===============================

from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)


# ===============================
# K Means Clustering Set UP
# ===============================

# Set up Streamlit page
st.title("KMeans Tiered Analysis")

@st.cache_data
def get_data(_km):
    """Fetch data from the database without hashing the KMeans_Cluster object."""
    df = _km.connect()
    return df

km = KMeans_Cluster()

# Fetch data
df = get_data(km)
if df is None or df.empty:
    st.error("Failed to retrieve data. Please check the connection and data source.")
    st.stop()


@st.cache_data
def process_data(_km, df, data_type, metric, n_clusters, visual):
    """Run KMeans algorithm on the data without hashing the KMeans_Cluster object."""
    return _km.run_alg(df, data_type, metric, n_clusters, visual)

@st.cache_data
def process_data_prd(df, data_type, metric, n_clusters, visual):
    """
    Run KMeans algorithm on the data and cache the DataFrame.
    Access the KMeans_Cluster instance from st.session_state internally.
    """
    km = st.session_state.km  # Accessing the KMeans_Cluster instance
    return km.run_alg(df, data_type, metric, n_clusters, visual)

st.markdown("""---""")


if st.checkbox("📜 Show Instructions"):

    # ===============================
    # Tiering Visualization Instructions
    # ===============================
    st.markdown("""
    ## 📊 **Tiering Visualization Guide**  

    Follow these steps to visualize and analyze your KMeans results:

    1. **🔢 Select Number of Clusters**  
    Choose a number between **4 - 8** using the input box.

    2. **📂 Choose Data Type**  
    Select your desired data type from the dropdown menu.

    3. **📈 Select Metric**  
    Pick a metric to analyze from the available options.

    4. **🔍 Choose Feature to Display**  
    Select one of the following features to visualize:
    - **🔢 Number of Jobs (`num_jobs`)**
    - **📦 Blocksize (`blocksize`)**
    - **🔄 Queue Depth (`queue_depth`)**
    - **📈 Effective Queue Depth (`effective_queue_depth`)**
    - **🔧 Operating PCI Width (`operating_pci_width`)**
    - **⚡ Operating PCI Speed GTS (`operating_pci_speed_gts`)**
    - **📏 Mean Measure (`mean_measure`)**

    5. **📈 View Bar Graph**  
    The bar graph will display the average value of the selected feature for each tier.

    6. **🕵️‍♂️ Analyze Tier Performance**  
    Compare and interpret the performance across different tiers based on the visualization.
    ---
                """)


# ===============================
# K Means Clustering Selection 
# ===============================
st.subheader("Tiering:")
n_clusters = 4
clusters_str = st.text_input("Number of Clusters:", str(n_clusters))
n_clusters = int(clusters_str)

# Choose Data Type
data_types = ["Random Read", "Random Write", "Sequential Write", "Sequential Read"]
data_type_selected = st.selectbox("Select Data Type", data_types)

# Choose Metrics
metrics_options = {"Random Read": ['iops', 'latency'], "Random Write": ['iops', 'latency'],
                   "Sequential Write": ['bw', 'latency'], "Sequential Read": ['bw', 'latency']}
metric_selected = st.selectbox("Select Metric Type", metrics_options[data_type_selected])

# df = process_data(km, df, data_type_selected, metric_selected, n_clusters, True)
df = process_data(km, df, data_type_selected, metric_selected, n_clusters, True)

if st.checkbox("Show Raw Data"):
    st.write(df.head())


# ===============================
# Tiering Visulization 
# ===============================

# dropdown to select y-axis for bar graph
y_axis_options = ['num_jobs', 'blocksize', 'queue_depth', 'effective_queue_depth', 
                  'operating_pci_width', 'operating_pci_speed_gts', 'mean_measure']
selection = st.selectbox("Select Y-Axis for Visualization (Bar Graph)", y_axis_options)

# ==== Table Visulization =====
st.subheader("Summary Table: Metrics by Tier")

# Compute mean for all y_axis_options grouped by Tier
table_data = df.groupby('Tier')[y_axis_options].mean().reset_index()

# Rename columns for better readability
table_data.columns = [
    'Tier' if col == 'Tier' else col.replace('_', ' ').capitalize() 
    for col in table_data.columns
]

# Round numerical values for clarity
table_data = table_data.round(2)

# Display the table using Streamlit
st.dataframe(table_data, use_container_width=True)


# ==== BAR GRAPH ====
st.subheader("Bar Graph: Tier Comparison")

# compute metrics for bar graph
bar_data = df.groupby('Tier')[selection].mean().reset_index()

# create bar graph
fig_bar = px.bar(
    bar_data,
    x='Tier',
    y=selection,
    text_auto=True,
    labels={'Tier': 'Tier', selection: selection.capitalize()},
    title=f"Bar Graph: {selection.capitalize()} by Tier (Higher Tiers = Better Performance w/ Given Metric)",
    color='Tier',  # Optional: add color for better distinction
    height=500
)

fig_bar.update_layout(
    xaxis_title="Tier",
    yaxis_title=selection.capitalize(),
    template="plotly_white"
)

st.plotly_chart(fig_bar)


# ==== Scatter Plot Visulization ====
st.subheader("Scatter Plot")

# Custom Scatter Plot with NVMe info in hover
fig_custom = px.scatter(
    df,
    x=selection,
    y='mean_measure',
    color=df['Tier'].astype(str),
    labels={'x': selection, 'y': 'Mean Measure'},
    title=f"KMeans Tiers for {data_type_selected} ({selection} vs Mean Measure)",
    hover_data={
        'name': True,
        'model': True,
        'capacity_gib': True,
        'operating_pci_speed_gts': True,
        'Tier': True
    }
)
st.plotly_chart(fig_custom)


# ===============================
# Predection Set up 
# ===============================
# ===============================
# Prediction Instructions
# ===============================

st.title("KMeans Prediction")


st.markdown("""---""")


if st.checkbox("📜 Show Prediction Guide", key="predict"):
        
    st.markdown("""
    ## 🔮 **Prediction Guide**

    Follow these steps to predict the tier of a drive:

    1. **📂 Select Data Type**  
    Choose the type of data you want to analyze from the dropdown menu.

    2. **📊 Select Metric**  
    Pick a metric related to the selected data type to focus your analysis.

    3. **🔍 Select Data Source**  
    Decide where your prediction data will come from:
    - **🔄 Use Existing Data**  
        Copy and paste the desired drive information into the input box provided.
    - **➕ Use New Data**  
        Fill out the input fields below with the relevant drive parameters.

    4. **📝 Enter Drive Details**  
    *If you selected **➕ Use New Data**, provide the following information:*
    - **📏 Block Size:** Enter a value between **4096 and 1048576**.
    - **🔢 Number of Jobs:** Enter a number between **1 and 64**.
    - **⚡ Operating PCI Speed (GTS):** Enter a value between **1 and 16**.
    - **🔧 Operating PCI Width:** Enter a value between **1 and 16**.
    - **🔄 Queue Depth:** Enter a value between **1 and 128**.

    5. **🔍 Make Prediction**  
    - **🔘 If Using New Data:** Click the **Predict** button after filling out the input fields.
    - **🔘 If Using Existing Data:** Ensure you've pasted the drive information and then click the **Predict** button.

    6. **📈 View Results**  
    After prediction, view the tier classification and detailed statistics to understand your drive's performance.

    ---

    **💡 Tips:**
    - Ensure all input values are within the specified ranges for accurate predictions.
    - Use the **Show Raw Data** checkbox to verify your input data before making predictions.
    - Experiment with different metrics and data types to gain comprehensive insights into drive performances.
    
    ---
    """)


prd_km = KMeans_Cluster()
prd = KMeans_Predict()

# Cache the results of get_prediction_data
@st.cache_data
def load_prediction_data():
    return prd.get_prediction_data()

# Store the new instance of KMeans in prd 
df_training, df_prediction = load_prediction_data()

# Choose Data Type
prd_data_type_selected = st.selectbox("Select Data Type", data_types, key="prd_data")

# Choose Metrics
prd_metric_selected = st.selectbox("Select Metric Type", metrics_options[prd_data_type_selected], key="prd_metric")

#Decide to use existing data or new data

# Run Kmeans on training data based on user input for data type and metric
# Initialize KMeans_Cluster and store it in session_state if not already present
if 'km' not in st.session_state:
    st.session_state.km = KMeans_Cluster()

# Access the KMeans_Cluster instance from session_state
prd.km_cluster = st.session_state.km
    
prd_df = process_data_prd(df_training, prd_data_type_selected, prd_metric_selected, n_clusters, True)
# prd_df = prd_km.run_alg(df_training, prd_data_type_selected, prd_metric_selected, n_clusters, True)

if st.checkbox("Show Raw Data", key="prd_df"):
    st.write(prd_df.head())

# Switch between "Use new data" and "Use existing data"
data_source = st.radio(
    "Select Data Source:",
    ("Use existing data", "Use new data")
)

# ===============================
# Use New Data to Predict Tier
# ===============================
if data_source == "Use new data":
    # Get drive statistics from user 
    prd_blocksize = st.number_input(
        "Enter a block size between 4096 and 1048576:",
        min_value=4096,
        max_value=1048576,
        step=4096,
        value=4096,  # default value
    )
    prd_num_jobs = st.number_input(
        "Enter the number of jobs between 1 and 64:",
        min_value=1,
        max_value=64,
        step=1,
        value=1  # Default value
    )
    prd_queue_depth = st.number_input(
        "Enter queue depth between 1 and 128:",
        min_value=1,
        max_value=128,
        step=1,
        value=1  # Default value
    )
    prd_operating_pci_speed_gts = st.number_input(
        "Enter operating PCI speed (GTS) between 1 and 16:",
        min_value=1,
        max_value=16,
        step=1,
        value=1  # Default value
    )
    prd_operating_pci_width = st.number_input(
            "Enter operating PCI width between 1 and 16:",
            min_value=1,
            max_value=16,
            step=1,
            value=1  # Default value
        )
    
    # Create DataFrame entry
    data_entry = {
            'data_type': prd_data_type_selected,
            'metric': prd_metric_selected,
            'blocksize': prd_blocksize,
            'num_jobs': prd_num_jobs,
            'queue_depth': prd_queue_depth,
            'operating_pci_speed_gts': prd_operating_pci_speed_gts,
            'operating_pci_width': prd_operating_pci_width
    }
     
    # Create a DataFrame with a single row containing the user's input
    df_prd = pd.DataFrame([data_entry])
    
    # Predict the user inputted drive
    if st.button("Predict"):
        prediction = prd.km_predict(prd.km_cluster, df_prd)  # Run prediction
        # Display Prediction Result with Columns
        
        st.markdown("""
        ### **📝 Note:**
        - **🔼 Higher Tier** = Better Performing Drive
        - **🔽 Lower Tier** = Lower Performing Drive  
          *(Given the specified Metric)*
        """)
        
        # Get the number of tiers 
        num_tiers = len(prd.km_cluster.tiers)

        # Prediction's tier (Replace with your prediction logic)
        tier = prd.km_cluster.tiers[prediction[0]]
        
        # Set colors based on the mode
        background_color = "white" 
        font_color = "#fc8558"
        highlight_color = "#1E90FF"             
        
        # Create the horizontal bar visualization
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor(background_color)  # Set figure background
        ax.set_facecolor(highlight_color)          # Set axes face color to light blue
        
        fig.patch.set_alpha(0)  # Make figure background transparent
        ax.set_facecolor('none')  # Make axes background transparent

        # Divide the bar into equal sections
        bar_sections = np.linspace(0, 1, num_tiers + 1)

        for i in range(num_tiers):
            color = "#fc8558" if i == tier - 1 else "lightgray"
            ax.barh(0, width=bar_sections[i+1] - bar_sections[i], 
                    left=bar_sections[i], color=color, edgecolor="black", height=0.5)

        # Add tier labels
        for i in range(num_tiers):
            ax.text(
                (bar_sections[i] + bar_sections[i+1]) / 2, 
                0, 
                i + 1, 
                ha="center", 
                va="center", 
                fontsize=12, 
                color="black",
                bbox=dict(facecolor="#fc8558" if i == tier - 1 else "lightgray", edgecolor="black", boxstyle="round,pad=0.3")
            )

        # Add "Lowest" and "Highest" labels
        ax.text(-0.05, 0, "Lowest", ha="right", va="center", fontsize=12, color=font_color, fontweight="bold")
        ax.text(1.05, 0, "Highest", ha="left", va="center", fontsize=12, color=font_color, fontweight="bold")

        # Remove axes for a cleaner look
        ax.set_axis_off()

        # Render the plot
        st.pyplot(fig)
        
        st.write("## Cluster Statistics")

        # Define the columns for which you want to compute statistics
        stats_columns = [
            'mean_measure','num_jobs', 
            'blocksize', 'queue_depth', 'effective_queue_depth',
            'operating_pci_width', 'operating_pci_speed_gts'
        ]
        
        # Filter DataFrame for the predicted cluster
        filtered_df = df[df['Cluster'] == prediction[0]][stats_columns].mean().reset_index()

        # Rename columns for better readability
        filtered_df.columns = ['Statistic', 'Value']

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            st.write("The drive was in cluster " + str(prediction[0]) + " the following are the statistics for this cluster")
            st.table(filtered_df)
        else:
            st.write("No data available for the selected cluster.")
 
 
# ===============================
# Use Existing Data to Predict Tier
# ===============================           
elif data_source == "Use existing data":
    df_clean = prd.clean_up(df_prediction, prd_data_type_selected, prd_metric_selected)
    unique_models = df_clean['model'].unique()
    st.write("Choose a model from validation dataset:")
    st.dataframe(unique_models)
    user_input = st.text_input("Paste Valid Model here:")
    
    # Predict the existing drive
    if st.button("# Predict") and user_input in  unique_models:
        model_data = df_clean[df_clean['model'] == user_input].sample(n=1)
        prediction = prd.km_predict(prd.km_cluster, model_data)  # Run prediction
        
        # Get the number of tiers 
        num_tiers = len(prd.km_cluster.tiers)

        # Prediction's tier (Replace with your prediction logic)
        tier = prd.km_cluster.tiers[prediction[0]]
        
        st.markdown("""
        ## Tier Prediction Results: 
        ### **📝 Note:**
        - **🔼 Higher Tier** = Better Performing Drive
        - **🔽 Lower Tier** = Lower Performing Drive  
          *(Given the specified Metric)*
        """)        
        
        # Set colors based on the mode
        background_color = "white" 
        font_color = "#fc8558"
        highlight_color = "#1E90FF"             
        
        # Create the horizontal bar visualization
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.patch.set_facecolor(background_color)  # Set figure background
        ax.set_facecolor(highlight_color)          # Set axes face color to light blue
        
        fig.patch.set_alpha(0)  # Make figure background transparent
        ax.set_facecolor('none')  # Make axes background transparent

        # Divide the bar into equal sections
        bar_sections = np.linspace(0, 1, num_tiers + 1)

        for i in range(num_tiers):
            color = "#fc8558" if i == tier - 1 else "lightgray"
            ax.barh(0, width=bar_sections[i+1] - bar_sections[i], 
                    left=bar_sections[i], color=color, edgecolor="black", height=0.5)

        # Add tier labels
        for i in range(num_tiers):
            ax.text(
                (bar_sections[i] + bar_sections[i+1]) / 2, 
                0, 
                i + 1, 
                ha="center", 
                va="center", 
                fontsize=12, 
                color="black",
                bbox=dict(facecolor="#fc8558" if i == tier - 1 else "lightgray", edgecolor="black", boxstyle="round,pad=0.3")
            )

        # Add "Lowest" and "Highest" labels
        ax.text(-0.05, 0, "Lowest", ha="right", va="center", fontsize=12, color=font_color, fontweight="bold")
        ax.text(1.05, 0, "Highest", ha="left", va="center", fontsize=12, color=font_color, fontweight="bold")

        # Remove axes for a cleaner look
        ax.set_axis_off()

        # Render the plot
        st.pyplot(fig)
          

        # Define the columns for which you want to compute statistics
        stats_columns = [
            'mean_measure','num_jobs', 
            'blocksize', 'queue_depth', 'effective_queue_depth',
            'operating_pci_width', 'operating_pci_speed_gts'
        ]
        
        # Filter DataFrame for the predicted cluster
        filtered_df = df[df['Cluster'] == prediction[0]][stats_columns].mean().reset_index()

        # Rename columns for better readability
        filtered_df.columns = ['Statistic', 'Value']

        # Check if filtered_df is not empty
        if not filtered_df.empty:
            st.subheader("Tier Statistics")
            st.markdown("**The drive was in Cluster:** " + f"**Tier {tier}**")
            st.table(filtered_df)
        else:
            st.write("No data available for the selected cluster.")