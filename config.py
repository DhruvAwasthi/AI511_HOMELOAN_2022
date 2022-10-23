"""
Module contains all the configuration details for the entire project
"""
import os

# set directory paths
DATA_DIR = "data/"
LOG_DIR = "log/"
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
DUMP_DIR = os.path.join(DATA_DIR, "dumps")
FIGURES_DIR = os.path.join(DATA_DIR, "figures")

paths = dict()
paths["train_data_path"] = os.path.join(DATASET_DIR, "train_data.csv")
paths["test_data_path"] = os.path.join(DATASET_DIR, "test_data.csv")

# set configuration for preprocessing
preprocess_configuration = dict()
preprocess_configuration["drop_columns_below_std"] = 0.02
preprocess_configuration["dimension_for_hashing"] = 16
preprocess_configuration["train_hash_encoder"] = True

# set configuration for handling outliers
preprocess_configuration["outliers"] = dict()
preprocess_configuration["outliers"]["percentile_range"] = (10, 90)
preprocess_configuration["outliers"]["scaled_factor"] = 1.5

# set configuration for machine learning model
model_configuration = dict()
model_configuration["random_state"] = 42
model_configuration["test_size"] = 0.2
model_configuration["num_stratified_shuffle_splits"] = 5
model_configuration["max_iter"] = 3000
