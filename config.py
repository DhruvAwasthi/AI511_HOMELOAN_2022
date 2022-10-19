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

model_configuration = dict()
