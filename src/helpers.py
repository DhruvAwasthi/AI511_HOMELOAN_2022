"""
Module contains helper functions for the development of project
"""
import logging
import os
from datetime import datetime
from typing import NoReturn

import pandas as pd
from pandas.core.frame import DataFrame

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


def create_log_dir(
    log_dir_path: str,
) -> NoReturn:
    """
    Creates log directory to store logs.
    If the log directory already exists, then does nothing.

    Args:
        log_dir_path: str
            Path of the log directory that needs to be creates.

    Returns:
    """
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    return
