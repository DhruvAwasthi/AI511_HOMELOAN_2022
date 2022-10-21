"""
Module contains tools for preprocessing of data
"""
import logging
from datetime import datetime

from pandas.core.frame import DataFrame

from src.helpers import calculate_iqr_range

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
        if len(columns_with_std_lt_std_value) > 0:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {len(columns_with_std_lt_std_value)}"
                f" column(s) have standard deviation value less than {std_value}. "
                f"Hence, dropping the columns: {', '.join(columns_with_std_lt_std_value)}")
        df = df.drop(columns_with_std_lt_std_value, axis=1, errors="ignore")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
            f"dropped columns with low standard deviation value")
    except Exception as e:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} could not"
                    f"delete columns with low standard deviation value")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error caused: "
            f"{str(e)}")
    return df


def drop_unnecessary_columns(
        df: DataFrame,
        unnecessary_columns: list,
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
        unnecessary_columns: list
            A list containing all the unnecessary columns that needs to be
            dropped.

    Returns:
        DataFrame:
            Pandas DataFrame after removing unnecessary columns.

    """

    try:
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} dropping {len(unnecessary_columns)} "
            f"unnecessary columns: {', '.join(unnecessary_columns)}")
        df = df.drop(unnecessary_columns, axis=1, errors="ignore")
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


def encode_categorical_columns(
    df: DataFrame,
) -> DataFrame:
    """
    Encode categorical columns for building the model.

    Args:
        df: DataFrame
            Pandas DataFrame in which categorical columns needs to be encoded.

    Returns:
        DataFrame:
            Pandas DataFrame after encoding all the categorical columns
    """
    return df


def deal_missing_value_for_categorical_columns(
    df: DataFrame,
) -> DataFrame:
    """
    Deals with missing values for categorical columns.

    Args:
        df: DataFrame
            Pandas DataFrame in which missing values needs to be dealt in
            categorical columns.

    Returns:
        DataFrame:
            Pandas DataFrame after dealing with missing values in all
            categorical columns.
    """
    return df


def deal_missing_value_for_numerical_columns(
    df: DataFrame,
) -> DataFrame:
    """
    Deals with missing values for numerical columns.

    Args:
        df: DataFrame
            Pandas DataFrame in which missing values needs to be dealt in
            numerical columns.

    Returns:
        DataFrame:
            Pandas DataFrame after dealing with missing values in all numerical
            columns.
    """
    return df


def handle_outliers(
    df: DataFrame,
    outliers_handling_configuration: dict,
) -> DataFrame:
    """
    Deals with outliers present in the columns.

    Args:
        df: DataFrame
            Pandas DataFrame in which outliers needs to be handled.
        outliers_handling_configuration: dict
            It contains key-value pairs defining how to handle outliers, and
            other important factors.

    Returns:
        DataFrame:
            Pandas DataFrame with outliers handled.
    """
    for column in list(df.select_dtypes(exclude=["object"]).columns):
        try:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} handling "
                f"outlier(s) for column {column}")
            iqr_range, lower_bound, upper_bound = calculate_iqr_range(
                df[column],
                scaled_factor=outliers_handling_configuration["scaled_factor"],
                percentile_range=outliers_handling_configuration["percentile_range"],
            )

            index_of_outliers = df[
                (df[column] > upper_bound) | (df[column] < lower_bound)].index
            median_of_column = df[column].dropna().median()
            df.loc[index_of_outliers, column] = median_of_column
        except Exception as e:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to "
                f"handle outliers for column {column}")
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error caused -"
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

        # drop columns with low standard deviation values
        df = drop_columns_with_low_std(df,
                                       preprocessing_configuration[
                                           "drop_columns_below_std"])

        # drop unnecessary columns
        df = drop_unnecessary_columns(df, preprocessing_configuration[
                                        "unnecessary_columns"])

        # handle outliers
        df = handle_outliers(df, preprocessing_configuration["outliers"])

        # deal with missing values for categorical columns
        df = deal_missing_value_for_categorical_columns(df)

        # deal with missing values for numerical columns
        df = deal_missing_value_for_numerical_columns(df)

        # encode categorical columns
        df = encode_categorical_columns(df)

    elif is_test_data:
        pass

    return df
