"""
Module contains tools for preprocessing of data
"""
import logging
from datetime import datetime

from pandas.core.frame import DataFrame

from src.helpers import calculate_iqr_range, pickle_dump_object

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
        columns_with_std_lt_std_value = df.std()[
            df.std() < std_value].index.values
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


def get_replacing_criteria_for_categorical_features(
        df: DataFrame
) -> dict:
    """
    Returns the replacing criteria for missing values of categorical features
    i.e., with what value should missing values in categorical features should
    be replaced.

    Args:
        df: DatFrame
            Pandas DataFrame containing the training data, needed to compute
            the replacing criteria for missing values

    Returns:
        dict:
            A dictionary containing column-name and replacing value, as key-
            value pair for each of the categorical feature in training datsaet.
    """
    # There are 6 categorical features in train data that contain missing
    # values - NAME_TYPE_SUITE, OCCUPATION_TYPE, FONDKAPREMONT_MODE,
    # HOUSETYPE_MODE, WALLSMATERIAL_MODE, EMERGENCYSTATE_MODE
    replace_cat_with = dict()
    replace_cat_with["NAME_TYPE_SUITE"] = None
    replace_cat_with["OCCUPATION_TYPE"] = None
    replace_cat_with["FONDKAPREMONT_MODE"] = None
    replace_cat_with["HOUSETYPE_MODE"] = None
    replace_cat_with["WALLSMATERIAL_MODE"] = None
    replace_cat_with["EMERGENCYSTATE_MODE"] = None

    # Replace missing values in NAME_TYPE_SUITE with most common class as there
    # are only 770 missing values as compared to 1,84,506 data points and in
    # which 1,49,059 data points belong to NAME_TYPE_SUITE. So it's safe to
    # assume that most of the data points belong to this category.
    replace_cat_with["NAME_TYPE_SUITE"] = \
    df["NAME_TYPE_SUITE"].value_counts().index[0]

    # In feature column EMERGENCYSTATE_MODE, there are only two categories -
    # Yes (wih 1,443 data points), and No (with 95,727 data points). And the
    # number of values missing is really large (87,336). Here, we can create a
    # new category 'Missing' to replace null values.
    replace_cat_with["EMERGENCYSTATE_MODE"] = "Missing"

    # For now, replace missing values in all other categorical features columns
    # with another cateogir 'Missing', and later, we will try to improve it.
    replace_cat_with["OCCUPATION_TYPE"] = "Missing"
    replace_cat_with["FONDKAPREMONT_MODE"] = "Missing"
    replace_cat_with["HOUSETYPE_MODE"] = "Missing"
    replace_cat_with["WALLSMATERIAL_MODE"] = "Missing"
    return replace_cat_with


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
    try:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} creating "
                    f"missing values replacing criteria for categorical "
                    f"features")
        replacing_criteria_for_cat_features = \
            get_replacing_criteria_for_categorical_features(df)

        # save the replacing criteria to be used for testing data
        pickle_dump_object(replacing_criteria_for_cat_features,
                           "replacing_criteria_for_cat_features.pkl")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
            f"created missing values replacing criteria for categorical "
            f"features")
    except Exception as e:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to"
                    f"create missing replacing criteria for categorical columns")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
                    f"caused - {str(e)}")
        return df

    for column_name, replace_with_value in replacing_criteria_for_cat_features.items():
        try:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} replacing missing"
                f"value in {column_name} with {replace_with_value}")
            df[column_name].fillna(replace_with_value, inplace=True)
        except Exception as e:
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to "
                        f"replace missing values of {column_name}")
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
                        f"caused - {str(e)}")
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
                percentile_range=outliers_handling_configuration[
                    "percentile_range"],
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
