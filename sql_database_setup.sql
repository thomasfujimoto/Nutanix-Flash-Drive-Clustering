-- Loads Data from CSV

DROP TABLE IF EXISTS ssd_perf_data;
DROP TABLE IF EXISTS ssd_devices_info;
DROP TABLE IF EXISTS ssd_devices_tests;

CREATE TABLE ssd_perf_data (
        concord_id UUID,  -- No PRIMARY KEY constraint
        data_type VARCHAR(50),
        metric VARCHAR(50),
        queue_depth INT,
        num_jobs INT,
        blocksize INT,
        unit VARCHAR(50),  
        min_measure FLOAT,  
        mean_measure FLOAT,
        median_measure FLOAT,
        max_measure FLOAT,
        stddev_measure FLOAT
 );
 CREATE TABLE ssd_devices_info (
        concord_id UUID,  -- No PRIMARY KEY constraint
        type VARCHAR(50),
        family VARCHAR(50),
        vendor VARCHAR(50),
        model VARCHAR(50),
        firmware VARCHAR(50),
        capacity_GiB FLOAT,  
        operating_pci_speed_GTs FLOAT,  
        operating_pci_width FLOAT,
        linkrate_Gbs FLOAT 
 );
 
CREATE TABLE ssd_devices_tests (
        concord_id UUID,  -- No PRIMARY KEY constraint
        name VARCHAR(50),
        reference VARCHAR(50),
        created VARCHAR(50)

);
   
COPY ssd_devices_tests(concord_id, name, reference, created)
FROM '/path/to/ssd_devices_tests_20241016.csv'
DELIMITER ','
CSV HEADER;

COPY ssd_perf_data(concord_id, data_type, metric, queue_depth, num_jobs, blocksize, unit, 
                   min_measure, mean_measure, median_measure, max_measure, stddev_measure)
FROM '/path/to/ssd_perf_data_20241016.csv'
DELIMITER ','
CSV HEADER;

COPY ssd_devices_info(concord_id, type, family, vendor, model, firmware, 
                      capacity_GiB, operating_pci_speed_GTs, operating_pci_width, linkrate_Gbs)
FROM '/path/to/ssd_devices_info_20241016.csv'
DELIMITER ','
CSV HEADER;


--- Finish Loading CSV


--- Build cleans data from CSV to ssd_clean_data

-- Step 1: Drop and recreate the `ssd_clean_data` table with distinct dataset
DROP TABLE IF EXISTS ssd_clean_data;

CREATE TABLE ssd_clean_data AS
SELECT DISTINCT
    p.concord_id,
    p.data_type,
    p.metric,
    p.queue_depth,
    p.num_jobs,
    p.blocksize,
    p.unit,
    p.min_measure,
    p.mean_measure,
    p.median_measure,
    p.max_measure,
    p.stddev_measure,
    d.type AS device_type,
    d.family,
    d.vendor,
    d.model,
    d.firmware,
    d.capacity_GiB,
    d.operating_pci_speed_GTs,
    d.operating_pci_width,
    d.linkrate_Gbs,
    t.name,
    t.reference,
    t.created,
    RANDOM() AS random_value  -- Assign a random value to each row
FROM 
    ssd_perf_data p
JOIN 
    ssd_devices_info d ON p.concord_id = d.concord_id
JOIN 
    ssd_devices_tests t ON p.concord_id = t.concord_id
WHERE 
    t.name = 'TEST_NVME_SPDK_PERF'
    AND d.type = 'NVMe'
    AND d.operating_pci_speed_GTs IS NOT NULL
    AND d.operating_pci_width IS NOT NULL;

-- Step 2: Split the data into training and validation sets based on `random_value`
DROP TABLE IF EXISTS training_data, validation_data;

-- 80% of the data for training
CREATE TABLE training_data AS
SELECT *
FROM ssd_clean_data
WHERE random_value <= 0.8;

-- 20% of the data for validation
CREATE TABLE validation_data AS
SELECT *
FROM ssd_clean_data
WHERE random_value > 0.8;

-- Step 3: Verify the counts
SELECT 'Training Count' AS dataset, COUNT(*) FROM training_data
UNION ALL
SELECT 'Validation Count' AS dataset, COUNT(*) FROM validation_data;

select * FROM ssd_clean_data; --Only 147,00 data sets
