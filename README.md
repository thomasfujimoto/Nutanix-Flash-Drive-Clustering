# Nutanix-Flash-Drive-Performance-Tiering

## Disclaimer 
Per the request of Nutanix, the test data used for clustering has been removed, and the SQL server previously used to run this project has been decommissioned. Efforts are currently underway to develop a workaround, including setting up a new SQL server populated with synthetic test data generated using machine learning techniques. 


## Descripion 
This project utilizes the KMeans unsupervised learning algorithm to cluster flash drive test data into 4–8 distinct clusters. The KMeans model is capable of predicting and assigning new, unseen data points to one of the pre-established clusters, enhancing its utility for ongoing analysis.Additionally, a Streamlit application was developed to visualize the clustering and predicition results interactively

## File Structure 
**`/data_access `**: Used to establish a connection to the SQL server, where data used to be stored (again a work around is currently being developed to avoid using restricted data)

**`/data_analysis `** : Stores the code to cluster data using unsupervised learning techniques and to predict unseen data into one of the generated clusters 

**`/docs `** : results of the tests performed in the notebooks, used to determine which unsupervised learning technique to use

**`/notebooks `** : jupiter notebooks used to test various different algorithms with different data sets note: a lot of this code is messy as it was used to strengthen our knowledge of Machine Learning 

**`/reports `** : outlines strategies and techniques used to determine clusters 

**`main.py `** : used to run the clustering and prediciton algorithm in /data_analysis

## Contact 
Please email fujimotothomas@outlook.com with any questions you may have!


## Credits 
Thank you to the Nutanix team for giving us this opportuinity and to teammates, Mattiwos Belachew, Pavitra Sammandam, Ezekiel Norman
and Vara Madem
