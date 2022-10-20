"""
Module contains helper functions for the development of project
"""
import logging
import os
from datetime import datetime
from typing import NoReturn, Union

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from pandas.core.series import Series

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
