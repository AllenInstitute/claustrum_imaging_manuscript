# local data folder

Raw data files are stored here but not synced with github in order to keep the repo from growing too large. Data must be downloaded here manually in order for the figures in this repo to generate.

To access the raw data files:
* Use the following link : https://www.dropbox.com/sh/bkyygkhp2e8anz9/AADB2Kr3eOQCoNEsDGKHZ8Pza?dl=1
* Save the zip file to a location on your local drive (File will be ~2.5 GB)
* Extract all files to this folder (../claustrum_imaging_manuscript/data)

Folders contain:
* raw `traces.csv` files - the raw output of the PCA/ICA Inscopix processing pipeline, one column per cell.
* `filtered_traces.csv` - traces after detrending with a 50 second median filter and z-scoring
* `CXXX.npy` - npy arrays holding the ROI mask for every cell
