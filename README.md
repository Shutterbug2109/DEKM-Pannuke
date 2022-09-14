# DEKM-Pannuke

create new folder log_history
## Single-cell Phenotypic Deep Clustering in Digital Pathology Images

For DEKM

To run the code for single dataset with choice of parameters
python main.py [-h] [--ds_name DS_NAME] [--n_clusters N_CLUSTERS] [--pretrain_epochs PRETRAIN_EPOCHS] [--hidden_units HIDDEN_UNITS] [--environment ENVIRONMENT]

DS_NAME -

PANNUKE_DILATED (for dataset with dilated images)
PANNUKE (for dataset with original squared crops)
PANNUKE_ONLYCELLS (for dataset with onlycell images)


Notebooks
All sequence to run :

RAAN_Data_Preprocessing_NB1
RAAN_Image_Feature_generation_and_PANNUKE_DATASET_NB2
RAAN_DILATED_PANNUKE_DATASET_NB3
RAAN_ONLYCELL_PANNUKE_DATASET_NB4
RAAN_Baseline_KMeans_DBSCAN_NB5
