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
    try:
        columns_with_std_lt_std_value = df.std()[df.std() < std_value].index.values
        if columns_with_std_lt_std_value > 0:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {len(columns_with_std_lt_std_value)}"
                f"column(s) have standard deviation value less than {std_value}. "
                f"Hence, dropping the columns: {', '.join(columns_with_std_lt_std_value)}")
        df = df.drop(columns_with_std_lt_std_value, axis=1, errors="ignore")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
            f"dropped columns with low standard deviation value")
    except Exception as e:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} couldn't"
                    f"delete columns with low standard deviation value")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error caused: "
            f"{str(e)}")
    return df


def drop_unnecessary_columns(
        df: DataFrame,
) -> DataFrame:
    """
    Removes the columns that does not contribute to the learning.

    There are many columns that either contain redundant values from other
    columns or doesn't contain relevant information that might help in learning
    of the machine learning model. Moreover, they add noise to the training
    data and harm the learning of the model.
    Hence, it is better to drop such columns.

    Args:
        df: DataFrame
        Pandas DataFrame from which necessary columns need to be dropped.

    Returns:
        DataFrame:
        Pandas DataFrame after removing unnecessary columns.

    """
    unnecessary_columns_to_drop = [
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

    try:
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} Dropping unnecessary "
            f"columns: {', '.join(unnecessary_columns_to_drop)}")
        df = df.drop(unnecessary_columns_to_drop, axis=1, errors="ignore")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
            f"dropped unnecessary columns")
    except Exception as e:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} couldn't"
                    f"delete unnecessary columns")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error caused: "
            f"{str(e)}")
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
        df = df.drop(["SK_ID_CURR"], axis=1, errors="ignore")

        # drop columns with low standard deviation values
        df = drop_columns_with_low_std(df,
                                       preprocessing_configuration[
                                           "drop_columns_below_std"]
                                       )

    elif is_test_data:
        pass

    return df
