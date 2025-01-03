# Nutanix-Flash-Drive-Performance-Tiering

### First time cloning repo:
create a python venv and enter it

Mac/Linux:
```
source .venv/bin/activate
pip install pre-commit
pre-commit install
pip install -r requirements.tx
streamlit run main.py
```

Windows: 
```
.venv\Scripts\activate
```

## Data Table Info:

### Table: ssd_clean_data

This is a cleaned data containing all other tables combined and filters out incorrect tests.
```
The columns are:
concord_id, data_type, metric, queue_depth, num_jobs, blocksize, unit, min_measure, mean_measure, median_measure, max_measure, stddev_measure, device_type, family, vendor, model, firmware, capacity_GiB, operating_pci_speed_GTs, operating_pci_width, linkrate_Gbs, name, reference, created


Categorical columns:
data_type, metric, queue_depth, num_jobs, blocksize
operating_pci_speed_GTs, operating_pci_width, linkrate_Gbs

Numerical columns:
min_measure, mean_measure, median_measure, max_measure, stddev_measure


Categorical values of:
data_type: Random Read, Random Write, Sequential Read, Sequential Write
metric: bw, iops, latency
queue_depth: 1, 8, 16, 32, 64, 128
num_jobs: 1, 2, 4, 8, ...
blocksize: a list of fixed integers

Column 'unit' describes the unit of the numerical columns.
```

### ssd_perf_data_20241016 (Table name: ssd_perf_data)
```
Table ssd_perf_data_20241016.csv

Headers:
"concord_id","data_type","metric","queue_depth","num_jobs","blocksize","unit","min_measure","mean_measure","median_measure","max_measure","stddev_measure"
"concord_id","type","family","vendor","model","firmware","capacity_GiB","operating_pci_speed_GTs","operating_pci_width","linkrate_Gbs"
"concord_id","name","reference","created"
Data sample:
"00211262-d6fd-4b61-85f0-6104fbd5eb44",Sequential Read,bw,1,1,65536,MiBps,528.5,585.3880806367432,587.75,603.0,11.911148144790515

Categorical columns:
data_type, metric, queue_depth, num_jobs, blocksize

Numerical columns
min_measure, mean_measure, median_measure, max_measure, stddev_measure

Categorical values of:
data_type: Random Read, Random Write, Sequential Read, Sequential Write
metric: bw, iops, latency
queue_depth: 1, 8, 16, 32, 64, 128
num_jobs: 1, 2, 4, 8, ...
blocksize: a list of fixed integers

Column 'unit' describes the unit of the numerical columns.
```
### ssd_devices_info_2024101 (Table name: ssd_devices_info)
```
Table ssd_devices_info_20241016.csv

Headers:
"concord_id","type","family","vendor","model","firmware","capacity_GiB","operating_pci_speed_GTs","operating_pci_width","linkrate_Gbs"
"concord_id","name","reference","created"
Data sample:
"3a2fda28-3501-4e60-aad8-559cb930b71b",NVMe,,Samsung,VK001920KYDPU,HPK4,1788.4964218139648,16.0,4,

Categorical columns:
operating_pci_speed_GTs, operating_pci_width, linkrate_Gbs
```
### ssd_devices_tests_20241016 (Table name: ssd_devices_tests)
```
Table ssd_devices_tests_20241016.csv

Headers:
"concord_id","name","reference","created"

Data sample:
"00211262-d6fd-4b61-85f0-6104fbd5eb44",TEST_NVME_SPDK_PERF,aad632695e2d4c32aebe25154cdbd57d,2024-10-11 01:13:16.212
```
