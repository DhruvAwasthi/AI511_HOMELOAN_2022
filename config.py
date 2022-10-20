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
preprocess_configuration["unnecessary_columns"] = [
        "FLAG_EMP_PHONE",  # office phone number is not important
        "FLAG_WORK_PHONE",  # home phone number is not important
        "WEEKDAY_APPR_PROCESS_START",  # does not matter on what day the loan is applied for
        "HOUR_APPR_PROCESS_START",  # does not matter during what hour the loan is applied for
        "REG_REGION_NOT_LIVE_REGION",  # permanent address and contact address (region) are different addresses, and does not matter if they match or not
        "REG_REGION_NOT_WORK_REGION",  # permanent address and work address (region) are different addresses, and does not matter if they match or not
        "LIVE_REGION_NOT_WORK_REGION",  # contact address and work address (region) are different addresses, and does not matter if they match or not
        "REG_CITY_NOT_LIVE_CITY",  # permanent address and contact address (region) are different addresses, and does not matter if they match or not
        "REG_CITY_NOT_WORK_CITY",  # permanent address and work address (region) are different addresses, and does not matter if they match or not
        "LIVE_CITY_NOT_WORK_CITY",  # contact address and work address (region) are different addresses, and does not matter if they match or not,
        "DAYS_LAST_PHONE_CHANGE",  # phone change information does not reveal something important as one can change phone due to multiple things,
        "OBS_30_CNT_SOCIAL_CIRCLE",  # surroundings is biased and does not reveal anything about the person's character
        "DEF_30_CNT_SOCIAL_CIRCLE",  # surroundings is biased and does not reveal anything about the person's character
        "OBS_60_CNT_SOCIAL_CIRCLE",  # surroundings is biased and does not reveal anything about the person's character
        "DEF_60_CNT_SOCIAL_CIRCLE",  # surroundings is biased and does not reveal anything about the person's character
    ]

# set configuration for machine learning model
model_configuration = dict()
model_configuration["random_state"] = 42
model_configuration["test_size"] = 0.2
