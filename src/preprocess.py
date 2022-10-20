"""
Module contains tools for preprocessing of data
"""
import logging
from datetime import datetime

from pandas.core.frame import DataFrame

logger = logging.getLogger(__name__)


def remove_duplicate_rows(
        df: DataFrame,
) -> DataFrame:
    """
    Removes duplicate rows from the dataset.

    Args:
        df: DataFrame
            Pandas DataFrame from which duplicate rows need to be removed.

    Returns:
        DataFrame:
            The same DataFrame but with duplicate rows removed (if present).
    """
    num_of_duplicate_rows = len(df.index[df.duplicated()])
    if num_of_duplicate_rows == 0:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} no dupli"
                    f"cate rows found")
    else:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                    f"{str(num_of_duplicate_rows)} duplicate rows found")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} removing "
                    f"duplicate rows")
        df.drop(axis="rows", labels=df.index[df.duplicated()], inplace=True)
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                    f"{str(num_of_duplicate_rows)} duplicate rows removed "
                    f"successfully")
    return df


def drop_columns_with_low_std(
        df: DataFrame,
        std_value: float = 0.02
) -> DataFrame:
    """
    Drops columns that have very low standard deviation.

    Standard deviation represents the data spread across the features. Low
    standard deviation means data is not much spread, and is uniform in most of
    the sample. Hence, they do not contribute much to the learning.
    So it's better to drop them.

    Args:
        df: DataFrame
            Pandas dataframe from which columns with low standard deviation
            values need to be dropped.
        std_value: float
            Standard deviation value below which columns will be dropped.

    Returns:
        DataFrame:
            Pandas DataFrame after dropping column with low standard deviation
            value.
    """
    columns_with_std_lt_std_value = df.std()[df.std() < std_value].index.values
    if columns_with_std_lt_std_value > 0:
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {len(columns_with_std_lt_std_value)}"
            f"column(s) have standard deviation value less than {std_value}. "
            f"Hence, dropping the columns: {', '.join(columns_with_std_lt_std_value)}")
    df = df.drop(columns_with_std_lt_std_value, axis=1)
    return df


def preprocess_data(
        df: DataFrame,
        preprocessing_configuration: dict,
        is_train_data: bool = False,
        is_test_data: bool = True,
) -> DataFrame:
    """
    Do all the preprocessing of data before feeding it to the learning
    algorithms. It can preprocess both - the training data, and the test data.

    Args:
        df: DataFrame
            Pandas DataFrame to be preprocesses.
        preprocessing_configuration: dict
            Dictionary containing the configuration settings for preprocessing
            steps.
        is_train_data: bool
            Set this to True if data that needs to be preprocessed is training
            data.
        is_test_data: bool
            Set this to True if data that needs to be preprocessed is testing
            data.

    Returns:
        DataFrame:
            Pandas dataframe containing the dataset after doing all the
            preprocessing steps.
    """
    if is_train_data:
        # remove duplicate rows
        df = remove_duplicate_rows(df)

        # drop unique ID column as it doesn't contribute to the learning
        df = df.drop(["SK_ID_CURR"], axis=1)

        # drop columns with low standard deviation values
        df = drop_columns_with_low_std(df,
                                       preprocessing_configuration[
                                           "drop_columns_below_std"]
                                       )

    elif is_test_data:
        pass

    return df
