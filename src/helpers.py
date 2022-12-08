"""
Module contains helper functions for the development of project
"""
import logging
import os
import pickle as pkl
from datetime import datetime
from typing import Any, NoReturn, Union

import numpy as np
import pandas as pd
from category_encoders.hashing import HashingEncoder
from pandas.core.frame import DataFrame
from pandas.core.series import Series

from config import DUMP_DIR

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_path: str,
) -> DataFrame:
    """
    Loads the dataset from CSV file into pandas Dataframe.

    Args:
        dataset_path: str
            Path of the dataset in CSV format to load.

    Returns:
        DataFrame:
            The dataset stored as CSV format into pandas DataFrame.
    """
    df = pd.read_csv(dataset_path)
    return df


def create_dirs(
    log_dir_path: str,
    dump_dir_path: str,
    figures_dir_path: str,
) -> NoReturn:
    """
    Creates log directory and dump directory to store logs.
    If the directories already exists, then skip.

    Args:
        log_dir_path: str
            Path of the log directory that will store all the logs.
        dump_dir_path: str
            Path of the dump directory that will store all the dumps.
        figures_dir_path: str
            Path of the figures directory that will store all the figures and
            plots.

    Returns:
    """
    # create log directory
    if not os.path.exists(log_dir_path):
        print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {log_dir_path} "
                    f"does not exist, so creating one")
        os.makedirs(log_dir_path)
        print(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                    f"successfully created {log_dir_path}")
    else:
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {log_dir_path} "
            f"exists")

    # create dump directory
    if not os.path.exists(dump_dir_path):
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {dump_dir_path} "
            f"does not exist, so creating one")
        os.makedirs(dump_dir_path)
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
            f"created {dump_dir_path}")
    else:
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {dump_dir_path} "
            f"exists")

    # create figures directory
    if not os.path.exists(figures_dir_path):
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {figures_dir_path} "
            f"does not exist, so creating one")
        os.makedirs(figures_dir_path)
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
            f"created {figures_dir_path}")
    else:
        print(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {figures_dir_path} "
            f"exists")

    return


def calculate_iqr_range(
        data: Series,
        scaled_factor: float = 1.5,
        percentile_range: tuple = (25, 75),
) -> tuple:
    """
    Calculates the IQR range, lower bound, and upper bound of the data to
    detect outliers.

    Args:
        data: Series
            Data for which IQR range, lower bound, and upper bound needs to be
            calculated
        scaled_factor: float
            Defaults to 1.5.

            Set this high to impose more stricter outlier detection i.e.,
            more outliers will be considered as regular data points.
            Lower this value to impose less stricter outlier detection i.e.,
            more data points will be considered as outliers.
        percentile_range: tuple
            Defaults to (25, 75).

            Denotes the percentile range needed to calculate the inter quartile
            range.

    Returns:
    """
    p1, p3 = np.percentile(data.dropna(), percentile_range)
    iqr_range = p3 - p1
    lower_bound = p1 - (scaled_factor * iqr_range)
    upper_bound = p3 + (scaled_factor * iqr_range)
    return iqr_range, lower_bound, upper_bound


def pickle_dump_object(
    object_to_dump: Any,
    file_name: str,
) -> NoReturn:
    """
    Saves the python object to disk for later use. Any Python object can be
    dumped to save.
    Args:
        object_to_dump: Any
            Any python object that needs to be saved. It will be saved as a
            pickle dump file.
        file_name: str
            File name of the dump file where given python object will be saved.

    Returns:
    """
    try:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} creating "
                    f"pickle dump file {file_name}")
        file_path = os.path.join(DUMP_DIR, file_name)
        dump_file = open(file_path, "wb")
        pkl.dump(object_to_dump, dump_file)
        dump_file.close()
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
                    f"created pickle dump file {file_name}")
    except Exception as e:
        logger.error(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to  "
                    f"create pickle dump file {str(e)}")
        logger.error(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
                    f"caused - {str(e)}")
    return


def pickle_load_object(
    file_name: str
) -> Any:
    """
    Loads a saved pickle dump file of a Python object.

    Args:
        file_name: str
            File name of the dump file from where to load the Python object.

    Returns:
        Any:
            The loaded Python object structure.
    """
    try:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} loading "
                    f"pickle dump file {file_name}")
        file_path = os.path.join(DUMP_DIR, file_name)
        dump_file = open(file_path, "rb")
        loaded_object = pkl.load(dump_file)
        dump_file.close()
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
            f"loaded pickle dump file {file_name}")
    except Exception as e:
        logger.error(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to  "
            f"load pickle dump file {str(e)}")
        logger.error(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
                    f"caused - {str(e)}")
        return None
    return loaded_object


def feature_hashing_encoder(
    df: DataFrame,
    columns: list,
    dimensions_to_use: int,
) -> HashingEncoder:
    """
    Trains a hashing encoder to encode categorical columns.

    Args:
        df: DataFrame
            Pandas DataFrame containing the entire dataset, to be used for
            training the hashing encoder.
        columns: list
            List of columns to encode using hashing encoder.
        dimensions_to_use: int
            Number of bits to use for hash encoding of given columns/

    Returns:
        HashingEncoder:
            Trained hashing encoder that can be used to transform data.
    """
    hashing_encoder = None
    try:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} training "
                    f"hash encoder for feature encoding")
        predictors = df.iloc[:, :-1]
        labels = df.iloc[:, -1]
        hashing_encoder = HashingEncoder(
            cols=columns, n_components=dimensions_to_use).fit(predictors, labels)
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} hash "
                    f"encoder for feature hashing trained successfully")
    except Exception as e:
        logger.error(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to "
                    f"train hashing encoder for feature hashing")
        logger.error(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error"
                    f"caused - {str(e)}")
    return hashing_encoder


def transform_data_using_hashing_encoder(
    df: DataFrame,
    hashing_encoder: HashingEncoder,
) -> DataFrame:
    """
    Encodes the categorical columns using hashing encoder.

    Args:
        df: DataFrame
            Pandas DataFrame containing the data to encode.
        hashing_encoder: HashingEncoder
            HashingEncoder that will be used to encode the categorical data.

    Returns:
        DataFrame
            Pandas DataFrame with encoded categorical columns using hashing
            encoder.
    """
    try:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} encoding "
                    f"categorical data using hashing encoder")
        label_column = df.columns[-1]
        predictors = df.iloc[:, :-1]
        labels = df.iloc[:, -1]
        df = hashing_encoder.transform(predictors)
        df[label_column] = labels
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
                    f"encoded categorical data using hashing encoder")
    except Exception as e:
        logger.error(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to encode"
            f"categorical data using hashing encoder")
        logger.error(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error"
                    f"caused - {str(e)}")
    return df


def get_datatype_mapping_for_reduction():
    """
    Returns the datatype mapping for dataset size reduction.

    Returns:
        dict
            Dictionary containing column name and reduced datatype as key-value
            pairs.
    """
    datatype_mapping = {
        "SK_ID_CURR": "unsigned",
        "CNT_CHILDREN": "unsigned",
        "AMT_INCOME_TOTAL": "float",
        "AMT_CREDIT": "float",
        "AMT_ANNUITY": "float",
        "AMT_GOODS_PRICE": "float",
        "REGION_POPULATION_RELATIVE": "float",
        "DAYS_BIRTH": "signed",
        "DAYS_EMPLOYED": "signed",
        "DAYS_REGISTRATION": "signed",
        "DAYS_ID_PUBLISH": "signed",
        "OWN_CAR_AGE": "unsigned",
        "FLAG_MOBIL": "unsigned",
        "FLAG_EMP_PHONE": "unsigned",
        "FLAG_WORK_PHONE": "unsigned",
        "FLAG_CONT_MOBILE": "unsigned",
        "FLAG_PHONE": "unsigned",
        "FLAG_EMAIL": "unsigned",
        "CNT_FAM_MEMBERS": "unsigned",
        "REGION_RATING_CLIENT": "unsigned",
        "REGION_RATING_CLIENT_W_CITY": "unsigned",
        "HOUR_APPR_PROCESS_START": "unsigned",
        "REG_REGION_NOT_LIVE_REGION": "unsigned",
        "REG_REGION_NOT_WORK_REGION": "unsigned",
        "LIVE_REGION_NOT_WORK_REGION": "unsigned",
        "REG_CITY_NOT_LIVE_CITY": "unsigned",
        "REG_CITY_NOT_WORK_CITY": "unsigned",
        "LIVE_CITY_NOT_WORK_CITY": "unsigned",
        "EXT_SOURCE_1": "float",
        "EXT_SOURCE_2": "float",
        "EXT_SOURCE_3": "float",
        "APARTMENTS_AVG": "float",
        "BASEMENTAREA_AVG": "float",
        "YEARS_BEGINEXPLUATATION_AVG": "float",
        "YEARS_BUILD_AVG": "float",
        "COMMONAREA_AVG": "float",
        "ELEVATORS_AVG": "float",
        "ENTRANCES_AVG": "float",
        "FLOORSMAX_AVG": "float",
        "FLOORSMIN_AVG": "float",
        "LANDAREA_AVG": "float",
        "LIVINGAPARTMENTS_AVG": "float",
        "LIVINGAREA_AVG": "float",
        "NONLIVINGAPARTMENTS_AVG": "float",
        "NONLIVINGAREA_AVG": "float",
        "APARTMENTS_MODE": "float",
        "BASEMENTAREA_MODE": "float",
        "YEARS_BEGINEXPLUATATION_MODE": "float",
        "YEARS_BUILD_MODE": "float",
        "COMMONAREA_MODE": "float",
        "ELEVATORS_MODE": "float",
        "ENTRANCES_MODE": "float",
        "FLOORSMAX_MODE": "float",
        "FLOORSMIN_MODE": "float",
        "LANDAREA_MODE": "float",
        "LIVINGAPARTMENTS_MODE": "float",
        "LIVINGAREA_MODE": "float",
        "NONLIVINGAPARTMENTS_MODE": "float",
        "NONLIVINGAREA_MODE": "float",
        "APARTMENTS_MEDI": "float",
        "BASEMENTAREA_MEDI": "float",
        "YEARS_BEGINEXPLUATATION_MEDI": "float",
        "YEARS_BUILD_MEDI": "float",
        "COMMONAREA_MEDI": "float",
        "ELEVATORS_MEDI": "float",
        "ENTRANCES_MEDI": "float",
        "FLOORSMAX_MEDI": "float",
        "FLOORSMIN_MEDI": "float",
        "LANDAREA_MEDI": "float",
        "LIVINGAPARTMENTS_MEDI": "float",
        "LIVINGAREA_MEDI": "float",
        "NONLIVINGAPARTMENTS_MEDI": "float",
        "NONLIVINGAREA_MEDI": "float",
        "TOTALAREA_MODE": "float",
        "OBS_30_CNT_SOCIAL_CIRCLE": "unsigned",
        "DEF_30_CNT_SOCIAL_CIRCLE": "unsigned",
        "OBS_60_CNT_SOCIAL_CIRCLE": "unsigned",
        "DEF_60_CNT_SOCIAL_CIRCLE": "unsigned",
        "DAYS_LAST_PHONE_CHANGE": "signed",
        "FLAG_DOCUMENT_2": "unsigned",
        "FLAG_DOCUMENT_3": "unsigned",
        "FLAG_DOCUMENT_4": "unsigned",
        "FLAG_DOCUMENT_5": "unsigned",
        "FLAG_DOCUMENT_6": "unsigned",
        "FLAG_DOCUMENT_7": "unsigned",
        "FLAG_DOCUMENT_8": "unsigned",
        "FLAG_DOCUMENT_9": "unsigned",
        "FLAG_DOCUMENT_10": "unsigned",
        "FLAG_DOCUMENT_11": "unsigned",
        "FLAG_DOCUMENT_12": "unsigned",
        "FLAG_DOCUMENT_13": "unsigned",
        "FLAG_DOCUMENT_14": "unsigned",
        "FLAG_DOCUMENT_15": "unsigned",
        "FLAG_DOCUMENT_16": "unsigned",
        "FLAG_DOCUMENT_17": "unsigned",
        "FLAG_DOCUMENT_18": "unsigned",
        "FLAG_DOCUMENT_19": "unsigned",
        "FLAG_DOCUMENT_20": "unsigned",
        "FLAG_DOCUMENT_21": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_HOUR": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_DAY": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_WEEK": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_MON": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_QRT": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_YEAR": "unsigned",
    }
    return datatype_mapping


def col_info(df, column_name):
    iqr_range, lower_bound, upper_bound = calculate_iqr_range(
        df[column_name],
        scaled_factor=1.7,
        percentile_range=(25, 75)
    )
    index_of_outliers = df[
        (df[column_name] > upper_bound) | (
                    df[column_name] < lower_bound)].index
    more_than_mean = list()
    less_than_mean = list()
    mean_of_column = df[column_name].mean()
    for index, value in df[column_name][index_of_outliers].iteritems():
        if value > mean_of_column:
            if value not in more_than_mean:
                more_than_mean.append(value)
        elif value < mean_of_column:
            if value not in less_than_mean:
                less_than_mean.append(value)

    less_than_mean.sort()
    more_than_mean.sort()

    unique_values = np.sort(df[column_name].unique())
    return index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound
