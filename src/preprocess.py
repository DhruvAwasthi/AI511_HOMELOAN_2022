"""
Module contains tools for preprocessing of data
"""
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

from src.helpers import calculate_iqr_range, feature_hashing_encoder, \
    pickle_dump_object, transform_data_using_hashing_encoder, \
    pickle_load_object

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
        pickle_dump_object(list(columns_with_std_lt_std_value),
                           "columns_with_std_lt_std_value.pkl")
        if len(columns_with_std_lt_std_value) > 0:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} {len(columns_with_std_lt_std_value)}"
                f" column(s) have standard deviation value less than {std_value}. "
                f"Hence, dropping the columns: {', '.join(columns_with_std_lt_std_value)}")
            df = df.drop(columns_with_std_lt_std_value, axis=1,
                         errors="ignore")
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
                f"dropped columns with low standard deviation value")
        else:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} there "
                f"are no columns with standard deviation value lower than defined")
    except Exception as e:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} could not"
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
    unnecessary_columns = [
        "SK_ID_CURR",  # this is just a unique identifier for each row
        # "FLAG_EMP_PHONE",  # office phone number is not important
        # "FLAG_WORK_PHONE",  # home phone number is not important
        # "WEEKDAY_APPR_PROCESS_START",
        # # does not matter on what day the loan is applied for
        # "HOUR_APPR_PROCESS_START",
        # # does not matter during what hour the loan is applied for
        # "REG_REGION_NOT_LIVE_REGION",
        # # permanent address and contact address (region) are different addresses, and does not matter if they match or not
        # "REG_REGION_NOT_WORK_REGION",
        # # permanent address and work address (region) are different addresses, and does not matter if they match or not
        # "LIVE_REGION_NOT_WORK_REGION",
        # # contact address and work address (region) are different addresses, and does not matter if they match or not
        # "REG_CITY_NOT_LIVE_CITY",
        # # permanent address and contact address (region) are different addresses, and does not matter if they match or not
        # "REG_CITY_NOT_WORK_CITY",
        # # permanent address and work address (region) are different addresses, and does not matter if they match or not
        # "LIVE_CITY_NOT_WORK_CITY",
        # # contact address and work address (region) are different addresses, and does not matter if they match or not,
        # "DAYS_LAST_PHONE_CHANGE",
        # # phone change information does not reveal something important as one can change phone due to multiple things,
        # "OBS_30_CNT_SOCIAL_CIRCLE",
        # # surroundings is biased and does not reveal anything about the person's character
        # "DEF_30_CNT_SOCIAL_CIRCLE",
        # # surroundings is biased and does not reveal anything about the person's character
        # "OBS_60_CNT_SOCIAL_CIRCLE",
        # # surroundings is biased and does not reveal anything about the person's character
        # "DEF_60_CNT_SOCIAL_CIRCLE",
        # # surroundings is biased and does not reveal anything about the person's character
    ]

    pickle_dump_object(unnecessary_columns, "unnecessary_columns.pkl")

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
        train_one_hot_encoder: bool = False,
):
    """
    Encode categorical columns using one hot encoding.

    Args:
        df: DataFrame
            Pandas DataFrame in which categorical columns needs to be encoded.
        train_one_hot_encoder: bool
            Set this to True to train the one_hot encoder.
            Set this to False to use the existing one_hot encoder.

    Returns:
        tuple:
            If data passed is train data, then it returns the transformed
            predictors and labels.
            If data passed is test data, then it returns the transformed
            predictors.
    """
    # encode columns with two unique values
    df["NAME_CONTRACT_TYPE"].replace({"Cash loans": 1, "Revolving loans": 0}, inplace=True)
    df["FLAG_OWN_CAR"].replace({"Y": 1, "N": 0}, inplace=True)
    df["FLAG_OWN_REALTY"].replace({"Y": 1, "N": 0}, inplace=True)

    # columns to encode using hashing encoder
    columns_using_one_hot_encoder = [
        "OCCUPATION_TYPE",
        "ORGANIZATION_TYPE",
        "CODE_GENDER",
        "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "FONDKAPREMONT_MODE",
        "HOUSETYPE_MODE",
        "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE",
        "WEEKDAY_APPR_PROCESS_START",
    ]

    if train_one_hot_encoder:
        predictors = df.iloc[:, :-1]
        labels = df.iloc[:, -1]
        # train one hot encoder
        transformer = make_column_transformer(
                        (OneHotEncoder(), columns_using_one_hot_encoder),
                        remainder='passthrough')
        predictors_transformed = transformer.fit_transform(predictors)

        # save one hot encoder
        pickle_dump_object(transformer, "cat_encoder.pkl")
        return predictors_transformed, labels
    else:
        # load one hot encoder
        transformer = pickle_load_object("cat_encoder.pkl")
        predictors = df.iloc[:, :]
        predictors_transformed = transformer.transform(predictors)
        return predictors_transformed, "_"


def get_replacing_criteria_for_categorical_features(
        df: DataFrame,
) -> dict:
    """
    Returns the replacing criteria for missing values of categorical features
    i.e., with what value should missing values in categorical features should
    be replaced.

    Args:
        df: DatFrame
            Pandas DataFrame containing the training data, needed to compute
            the replacing criteria for missing values of categorical features.

    Returns:
        dict:
            A dictionary containing column-name and replacing value, as key-
            value pair for each of the categorical feature in training dataset.
    """
    replace_cat_with = dict()

    categorical_columns = list(df.select_dtypes(include=["object"]).columns)
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} there are  "
                f"{len(categorical_columns)} categorical features in dataset")

    categorical_columns_with_missing_values = list()
    for column in categorical_columns:
        missing_values = df[column].isna().sum()
        if missing_values > 0:
            categorical_columns_with_missing_values.append(column)
            replace_cat_with[column] = "Missing"  # default criteria

    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                f"{len(categorical_columns_with_missing_values)} categorical "
                f"column(s) contains missing values - "
                f"{', '.join(categorical_columns_with_missing_values)}")

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
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} replacing "
                f"missing values in {column_name} with {replace_with_value}")
            df[column_name].fillna(replace_with_value, inplace=True)
        except Exception as e:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to "
                f"replace missing values of {column_name}")
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
                f"caused - {str(e)}")
    return df


def get_replacing_criteria_for_numerical_features(
        df: DataFrame
) -> dict:
    """
    Returns the replacing criteria for missing values of numerical features
    i.e., with what value should missing values in categorical features should
    be replaced.

    Args:
        df: DatFrame
            Pandas DataFrame containing the training data, needed to compute
            the replacing criteria for missing values of numerical features.

    Returns:
        dict:
            A dictionary containing column-name and replacing value, as key-
            value pair for each of the numerical feature in training dataset.
    """
    replace_num_with = dict()

    numerical_columns = list(df.select_dtypes(exclude=["object"]).columns)
    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} there are  "
                f"{len(numerical_columns)} categorical features in dataset")

    numerical_columns_with_missing_values = list()
    for column in numerical_columns:
        missing_values = df[column].isna().sum()
        if missing_values > 0:
            numerical_columns_with_missing_values.append(column)
            replace_num_with[column] = df[column].median()  # default criteria

    logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                f"{len(numerical_columns_with_missing_values)} categorical "
                f"column(s) contains missing values - "
                f"{', '.join(numerical_columns_with_missing_values)}")

    # It is a categorical columns that contains integer count, hence missing
    # values can be replaced with the mode of the feature
    replace_num_with["CNT_FAM_MEMBERS"] = \
        df["CNT_FAM_MEMBERS"].value_counts().index[0]
    return replace_num_with


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
    try:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} creating "
                    f"missing values replacing criteria for numerical "
                    f"features")
        replacing_criteria_for_num_features = \
            get_replacing_criteria_for_numerical_features(df)

        # save the replacing criteria to be used for testing data
        pickle_dump_object(replacing_criteria_for_num_features,
                           "replacing_criteria_for_num_features.pkl")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} successfully "
            f"created missing values replacing criteria for numerical "
            f"features")
    except Exception as e:
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to"
                    f"create missing replacing criteria for numerical columns")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
                    f"caused - {str(e)}")
        return df

    for column_name, replace_with_value in replacing_criteria_for_num_features.items():
        try:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} replacing "
                f"missing values in {column_name} with {replace_with_value}")
            df[column_name].fillna(replace_with_value, inplace=True)
        except Exception as e:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to "
                f"replace missing values of {column_name}")
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error "
                f"caused - {str(e)}")
    return df


def get_handle_outlier_criteria(
    df: DataFrame,
    outliers_handling_configuration: dict,
    is_train_data: bool = True,
) -> dict:
    """
    Returns the handling criteria for outliers in all the columns.

    Args:
        df: DataFrame
            Pandas dataframe containing the data.
        outliers_handling_configuration: dict
            It contains key-value pairs defining how to handle outliers, and
            other important factors.
        is_train_data: bool, defaults to True
            Set this to True if the dataframe passed is train data.
            Set this to False if the dataframe passed is test data.

    Returns: dict
        Dictionary containing the column name and list of iqr, lower bound, and
        upper bound values necessary to detect outliers as key-value pairs
        respectively.
    """
    handle_outlier_criteria = dict()
    if is_train_data:
        for column in list(df.select_dtypes(exclude=["object"]).columns):
            if column == "TARGET":
                continue
            iqr_range, lower_bound, upper_bound = calculate_iqr_range(
                df[column],
                scaled_factor=outliers_handling_configuration["scaled_factor"],
                percentile_range=outliers_handling_configuration[
                    "percentile_range"],
            )
            handle_outlier_criteria[column] = [iqr_range, lower_bound,
                                               upper_bound]
        # save the criteria
        pickle_dump_object(handle_outlier_criteria, "handle_outlier.pkl")

    else:
        # load outlier handling criteria
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                    f"loading outlier handling criteria")
        handle_outlier_criteria = pickle_load_object("handle_outlier.pkl")
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                    f"successfully loaded outlier handling criteria")

    return handle_outlier_criteria


def handle_outliers(
    df: DataFrame,
    outliers_handling_configuration: dict,
    is_train_data: bool = True,
) -> DataFrame:
    """
    Deals with outliers present in the columns.

    Args:
        df: DataFrame
            Pandas DataFrame in which outliers needs to be handled.
        outliers_handling_configuration: dict
            It contains key-value pairs defining how to handle outliers, and
            other important factors.
        is_train_data: bool, defaults to True
            Set this to True if the dataframe passed is train data.
            Set this to False if the dataframe passed is test data.

    Returns:
        DataFrame:
            Pandas DataFrame with outliers handled.
    """
    handle_outlier_criteria = get_handle_outlier_criteria(
        df,
        outliers_handling_configuration,
        is_train_data=is_train_data,
    )

    for column in list(df.select_dtypes(exclude=["object"]).columns):
        if column == "TARGET":
            continue
        elif column == "SK_ID_CURR":
            continue
        try:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} handling "
                f"outliers for column {column}")
            iqr_range, lower_bound, upper_bound = handle_outlier_criteria[column]
            index_of_outliers = df[
                (df[column] > upper_bound) | (df[column] < lower_bound)].index
            # df = df.drop(index=index_of_outliers)
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


def scale_numeric_features(
    df: DataFrame,
) -> DataFrame:
    """
    Scales the numeric features using min-max normalization.
    Args:
        df: DataFrame
            Pandas DataFrame containing the dataset.

    Returns:
        DataFrame:
            Pandas DataFrame with scaled numeric features.
    """
    numerical_columns = list(df.select_dtypes(exclude=["object"]).columns)
    categorical_columns = list(df.select_dtypes(include=["object"]).columns)
    target_column = df[["TARGET"]]
    to_scale = df[numerical_columns]
    to_scale = to_scale.drop(["TARGET"], axis=1)

    # create and save scaler
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaler.fit(to_scale)
    pickle_dump_object(scaler, "scaler.pkl")

    # tranform data
    scaled = scaler.transform(to_scale)
    scaled_df = pd.DataFrame(scaled, columns=to_scale.columns)
    df = pd.concat([df[categorical_columns], scaled_df, target_column], axis=1)
    return df


def preprocess_data(
        df: DataFrame,
        preprocessing_configuration: dict,
        is_train_data: bool = False,
        is_test_data: bool = False,
):
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
        tuple:
            If data passed is train data, then it returns the transformed
            predictors and labels.
            If data passed is test data, then it returns the transformed
            predictors.
    """
    if is_train_data:
        # remove duplicate rows
        df = remove_duplicate_rows(df)

        # drop columns with low standard deviation values
        df = drop_columns_with_low_std(df,
                                       preprocessing_configuration[
                                           "drop_columns_below_std"])

        # drop unnecessary columns
        df = drop_unnecessary_columns(df)

        # handle outliers
        df = handle_outliers(df, preprocessing_configuration["outliers"],
                             is_train_data=True)

        # deal with missing values for categorical columns
        df = deal_missing_value_for_categorical_columns(df)

        # deal with missing values for numerical columns
        df = deal_missing_value_for_numerical_columns(df)

        # scale numeric features
        df = scale_numeric_features(df)

        # encode categorical columns
        predictors, labels = encode_categorical_columns(
            df,
            True)
        return predictors, labels

    elif is_test_data:

        # drop columns with low standard deviation values
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} dropping "
            f"columns with lwo standard deviation value ")
        columns_with_std_lt_std_value = pickle_load_object(
                    "columns_with_std_lt_std_value.pkl")
        df = df.drop(columns_with_std_lt_std_value, axis=1,
                     errors="ignore")

        # drop unnecessary columns
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} dropping "
            f"unnecessary columns")
        unnecessary_columns = pickle_load_object("unnecessary_columns.pkl")
        unnecessary_columns.remove("SK_ID_CURR")
        df = df.drop(unnecessary_columns, axis=1,
                     errors="ignore")

        if preprocessing_configuration["outliers"]["at_test_time"]:
            # handle outliers
            df = handle_outliers(df, preprocessing_configuration["outliers"],
                                 is_train_data=False)

        # deal with missing values for categorical columns
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} handling missing "
            f"values of categorical columns")
        replacing_criteria_for_cat_features = pickle_load_object(
            "replacing_criteria_for_cat_features.pkl")
        for column_name, replace_with_value in replacing_criteria_for_cat_features.items():
            df[column_name].fillna(replace_with_value, inplace=True)

        # deal with missing values for numerical columns
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} handling missing "
            f"values of numerical columns")
        replacing_criteria_for_num_features = pickle_load_object(
            "replacing_criteria_for_num_features.pkl")
        for column_name, replace_with_value in replacing_criteria_for_num_features.items():
            df[column_name].fillna(replace_with_value, inplace=True)

        # scale numeric features
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} scaling numeric "
            f"features")
        scaler = pickle_load_object("scaler.pkl")
        unique_identifier_column = df[["SK_ID_CURR"]]
        numerical_columns = list(df.select_dtypes(exclude=["object"]).columns)
        categorical_columns = list(
            df.select_dtypes(include=["object"]).columns)
        to_scale = df[numerical_columns]
        to_scale = to_scale.drop(["SK_ID_CURR"], axis=1)
        scaled = scaler.transform(to_scale)
        scaled_df = pd.DataFrame(scaled, columns=to_scale.columns)
        df = pd.concat([unique_identifier_column, df[categorical_columns],
                        scaled_df], axis=1)

        # encode categorical columns
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} encoding "
            f"categorical features")
        predictors, labels = encode_categorical_columns(
            df,
            False)
        return predictors, labels
