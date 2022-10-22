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
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to  "
                    f"create pickle dump file {str(e)}")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
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
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to  "
            f"load pickle dump file {str(e)}")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
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
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to "
                    f"train hashing encoder for feature hashing")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error"
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
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to encode"
            f"categorical data using hashing encoder")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error"
                    f"caused - {str(e)}")
    return df
