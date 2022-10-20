"""
Module contains all the configuration details for the entire project
"""
import os

# set directory paths
DATA_DIR = "data/"
LOG_DIR = "log/"

paths = dict()
paths["train_data_path"] = os.path.join(DATA_DIR, "train_data.csv")
paths["test_data_path"] = os.path.join(DATA_DIR, "test_data.csv")

# set configuration for preprocessing
preprocess_configuration = dict()
preprocess_configuration["drop_columns_below_std"] = 0.02

# set configuration for machine learning model
model_configuration = dict()
model_configuration["random_state"] = 42
model_configuration["test_size"] = 0.2
