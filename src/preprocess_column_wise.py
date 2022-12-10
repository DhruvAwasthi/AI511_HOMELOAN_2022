import logging

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from src.helpers import col_info

logger = logging.getLogger(__name__)


def column_wise(
        train_df: DataFrame,
        test_df: DataFrame,
):

    # -------------------------------------------------------------------------
    column_name = "CNT_CHILDREN"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    # replace all values with cnt_children >=3 with 3
    train_df.loc[train_df[column_name] >= 3, column_name] = 3

    # test data
    test_df.loc[test_df[column_name] >= 3, column_name] = 3
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_INCOME_TOTAL"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    # there are just four samples with income >9000000; let's just drop them
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(
        train_df, column_name)
    to_delete_indices = train_df[(train_df[column_name] >= 9000000)].index
    train_df.drop(index=to_delete_indices, inplace=True)
    index_of_outliers = index_of_outliers.drop(to_delete_indices)
    # trim outlier values with the extreme value that is not an outlier
    lowest_outlier_value = more_than_mean[0]
    index_lowest_outlier_value = \
        np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(
        f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value - 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]

    # test data
    index_of_outliers = test_df[test_df[column_name] > unique_values[
        index_lowest_outlier_value - 1]].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_CREDIT"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(
        train_df, column_name)
    # trim outlier values with the extreme value that is not an outlier
    lowest_outlier_value = more_than_mean[0]
    index_lowest_outlier_value = \
        np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(
        f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value - 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]

    # test data
    index_of_outliers = test_df[test_df[column_name] > unique_values[
        index_lowest_outlier_value - 1]].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_ANNUITY"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(
        train_df, column_name)
    # there is one sample with amt_annuity >= 258025.5; let's just drop it
    to_delete_indices = train_df[(train_df[column_name] >= 258025.5)].index
    train_df.drop(index=to_delete_indices, inplace=True)
    index_of_outliers = index_of_outliers.drop(to_delete_indices)
    # trim outlier values with the extreme value that is not an outlier
    lowest_outlier_value = more_than_mean[0]
    index_lowest_outlier_value = \
        np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(
        f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value - 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    # there are 6 missing values; let's just replace them with mean calculated without outliers
    train_df[column_name].fillna(column_mean_without_outliers, inplace=True)

    # test data
    test_df[column_name].fillna(column_mean_without_outliers, inplace=True)
    index_of_outliers = test_df[test_df[column_name] > unique_values[
        index_lowest_outlier_value - 1]].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_GOODS_PRICE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(
        train_df, column_name)
    # there are 9 sample with amt_goods_price >= 3375000.0; let's just drop them
    to_delete_indices = train_df[(train_df[column_name] >= 3375000.0)].index
    train_df.drop(index=to_delete_indices, inplace=True)
    index_of_outliers = index_of_outliers.drop(to_delete_indices)
    # trim outlier values with the extreme value that is not an outlier
    lowest_outlier_value = more_than_mean[0]
    index_lowest_outlier_value = \
        np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(
        f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value - 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    # there are 167 missing values; let's just replace them with mean calculated without outliers
    train_df[column_name].fillna(column_mean_without_outliers, inplace=True)

    # test data
    index_of_outliers = test_df[test_df[column_name] > unique_values[
        index_lowest_outlier_value - 1]].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "REGION_POPULATION_RELATIVE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(
        train_df, column_name)
    # trim outlier values with the extreme value that is not an outlier
    # fact - all the outliers here contain the same value
    lowest_outlier_value = more_than_mean[0]
    index_lowest_outlier_value = \
        np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(
        f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value - 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]

    # test data
    index_of_outliers = test_df[test_df[column_name] > unique_values[index_lowest_outlier_value - 1]].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "DAYS_BIRTH"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "DAYS_EMPLOYED"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(
        train_df, column_name)
    # there are 6 samples with days_employed <= 17139; let's just drop them
    to_delete_indices = train_df[(train_df[column_name] <= -17139)].index
    train_df.drop(index=to_delete_indices, inplace=True)
    index_of_outliers = index_of_outliers.drop(to_delete_indices)
    # trim outlier values with the extreme value that is not an outlier
    indices_of_more_than_mean = train_df[train_df[column_name] == more_than_mean[0]].index
    train_df.loc[train_df[column_name] >= 0, column_name] = 0
    index_of_outliers = index_of_outliers.drop(indices_of_more_than_mean)
    lowest_outlier_value = less_than_mean[-1]
    index_lowest_outlier_value = \
    np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value + 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value + 1]

    # test data
    index_of_outliers = test_df[test_df[column_name] > 0].index
    test_df.loc[index_of_outliers, column_name] = 0
    index_of_outliers = test_df[test_df[column_name] < unique_values[index_lowest_outlier_value + 1]].index
    test_df.loc[index_of_outliers, column_name] = unique_values[index_lowest_outlier_value + 1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "DAYS_REGISTRATION"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(
        train_df, column_name)
    # there are 8 samples with days_registration <= 21146; let's just drop them
    to_delete_indices = train_df[(train_df[column_name] <= -21146)].index
    train_df.drop(index=to_delete_indices, inplace=True)
    index_of_outliers = index_of_outliers.drop(to_delete_indices)
    lowest_outlier_value = less_than_mean[-1]
    index_lowest_outlier_value = \
        np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(
        f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value + 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value + 1]

    # test data
    index_of_outliers = test_df[test_df[column_name] < unique_values[
        index_lowest_outlier_value + 1]].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value + 1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "DAYS_ID_PUBLISH"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "DAYS_LAST_PHONE_CHANGE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(
        train_df, column_name)
    # there is just 1 missing value; let's just replace it with mean calculated without outliers
    train_df[column_name].fillna(column_mean_without_outliers, inplace=True)
    # there are 7 samples with days_last_phone_change <= 4115.0; let's just drop them
    to_delete_indices = train_df[(train_df[column_name] <= -4115.0)].index
    train_df.drop(index=to_delete_indices, inplace=True)
    index_of_outliers = index_of_outliers.drop(to_delete_indices)
    lowest_outlier_value = less_than_mean[-1]
    index_lowest_outlier_value = \
        np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(
        f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value + 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value + 1]

    # test data
    index_of_outliers = test_df[test_df[column_name] < unique_values[
        index_lowest_outlier_value + 1]].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value + 1]
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "OWN_CAR_AGE"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"Since number of missing values is huge i.e., 121614, hence dropping column")
    # train data
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_MOBIL"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_EMP_PHONE"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_WORK_PHONE"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_CONT_MOBILE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_PHONE"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_EMAIL"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_2"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_3"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_4"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_5"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_6"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_7"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_8"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_9"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_10"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_11"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_12"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since all values are same, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_13"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_14"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_15"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_16"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_17"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_18"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_19"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_20"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_DOCUMENT_21"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "EXT_SOURCE_1"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since number of missing values is huge i.e., 104055, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "EXT_SOURCE_2"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since number of missing values relatively low i.e., 413, hence replacing them with median as more than 50% values "
                f"are inside 0.56 and median is close to this value, while mean is 0.52.")
    median_value = train_df[column_name].median()
    train_df.fillna(median_value, inplace=True)

    # test data
    test_df.fillna(median_value, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "EXT_SOURCE_3"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 36648, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "CNT_FAM_MEMBERS"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "REGION_RATING_CLIENT"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "REGION_RATING_CLIENT_W_CITY"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "HOUR_APPR_PROCESS_START"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    # replace all values with  >=9 with 3
    train_df[column_name].replace([21, 22, 23, 0, 1, 2], 21, inplace=True)

    # test data
    test_df[column_name].replace([21, 22, 23, 0, 1, 2], 21, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "REG_REGION_NOT_LIVE_REGION"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "REG_REGION_NOT_WORK_REGION"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LIVE_REGION_NOT_WORK_REGION"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "REG_CITY_NOT_LIVE_CITY"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "REG_CITY_NOT_WORK_CITY"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LIVE_CITY_NOT_WORK_CITY"
    logger.info(f"Preprocessing column: {column_name}")
    logger.info(f"No steps required!")
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "APARTMENTS_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 935588, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "BASEMENTAREA_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 107956, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "YEARS_BEGINEXPLUATATION_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 89883, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "YEARS_BUILD_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 122733, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "COMMONAREA_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 128945, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "ELEVATORS_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 98302, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "ENTRANCES_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 92790, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLOORSMAX_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 91709, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLOORSMIN_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 125218, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LANDAREA_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 109525, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LIVINGAPARTMENTS_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 126188, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LIVINGAREA_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 92517, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NONLIVINGAPARTMENTS_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 128121, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NONLIVINGAREA_AVG"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 101769, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "APARTMENTS_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 93558, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "BASEMENTAREA_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 107956, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "YEARS_BEGINEXPLUATATION_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 89883, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "YEARS_BUILD_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 122733, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "COMMONAREA_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 128945, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "ELEVATORS_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 98302, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "ENTRANCES_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 92790, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLOORSMAX_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 91709, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLOORSMIN_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 125218, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LANDAREA_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 109525, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LIVINGAPARTMENTS_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 126188, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LIVINGAREA_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 92517, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NONLIVINGAPARTMENTS_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 128121, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NONLIVINGAREA_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 101769, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "APARTMENTS_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 93558, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "BASEMENTAREA_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 107956, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "YEARS_BEGINEXPLUATATION_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 89883, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "YEARS_BUILD_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 122733, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "COMMONAREA_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 128945, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "ELEVATORS_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 98302, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "ENTRANCES_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 92790, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLOORSMAX_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 91709, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLOORSMIN_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 125218, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LANDAREA_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 109525, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LIVINGAPARTMENTS_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 126188, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "LIVINGAREA_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 92517, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NONLIVINGAPARTMENTS_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 128121, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NONLIVINGAREA_MEDI"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 101769, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "TOTALAREA_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 88931, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "OBS_30_CNT_SOCIAL_CIRCLE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "DEF_30_CNT_SOCIAL_CIRCLE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    # replace all values with def_30_cnt_social_circle >=3 with 3
    train_df.loc[train_df[column_name] > 2, column_name] = 2
    # replace missing values with mode of column
    mode = train_df[column_name].mode()[0]
    train_df[column_name].fillna(mode, inplace=True)

    # test data
    test_df.loc[test_df[column_name] > 2, column_name] = 2
    test_df[column_name].fillna(mode, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "OBS_60_CNT_SOCIAL_CIRCLE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "DEF_60_CNT_SOCIAL_CIRCLE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    # replace all values with def_30_cnt_social_circle >=3 with 3
    train_df.loc[train_df[column_name] > 2, column_name] = 2
    # replace missing values with mode of column
    mode = train_df[column_name].mode()[0]
    train_df[column_name].fillna(mode, inplace=True)

    # test data
    test_df.loc[test_df[column_name] > 2, column_name] = 2
    test_df[column_name].fillna(mode, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_REQ_CREDIT_BUREAU_HOUR"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_REQ_CREDIT_BUREAU_DAY"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_REQ_CREDIT_BUREAU_WEEK"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_REQ_CREDIT_BUREAU_MON"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_REQ_CREDIT_BUREAU_QRT"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "AMT_REQ_CREDIT_BUREAU_YEAR"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NAME_CONTRACT_TYPE"
    logger.info(f"Preprocessing column: {column_name}")
    mapping_dict = {
        "Cash loans": 0,
        "Revolving loans": 1,
    }
    # train data
    train_df[column_name].replace(mapping_dict, inplace=True)

    # test data
    test_df[column_name].replace(mapping_dict, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "CODE_GENDER"
    logger.info(f"Preprocessing column: {column_name}")
    mapping_dict = {
        "F": 0,
        "M": 1,
        "XNA": 2
    }
    # train data
    train_df[column_name].replace(mapping_dict, inplace=True)

    # test data
    test_df[column_name].replace(mapping_dict, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_OWN_CAR"
    logger.info(f"Preprocessing column: {column_name}")
    mapping_dict = {
        "N": 0,
        "Y": 1,
    }
    # train data
    train_df[column_name].replace(mapping_dict, inplace=True)

    # test data
    test_df[column_name].replace(mapping_dict, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FLAG_OWN_REALTY"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NAME_TYPE_SUITE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NAME_INCOME_TYPE"
    logger.info(f"Preprocessing column: {column_name}")
    mapping_dict = {
        "Working": 2,
        "Commercial associate": 2,
        "Pensioner": 1,
        "State servant": 3,
        "Unemployed": 0,
        "Businessman": 3
    }

    # train data
    train_df[column_name].replace(["Student"], "Unemployed", inplace=True)
    train_df[column_name].replace(["Maternity leave"], "Pensioner", inplace=True)
    train_df[column_name].replace(mapping_dict, inplace=True)

    # test data
    test_df[column_name].replace(["Student"], "Unemployed", inplace=True)
    test_df[column_name].replace(["Maternity leave"], "Pensioner", inplace=True)
    test_df[column_name].replace(mapping_dict, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NAME_EDUCATION_TYPE"
    logger.info(f"Preprocessing column: {column_name}")
    mapping_dict = {
        "Secondary / secondary special": 3,
        "Higher education": 2,
        "Incomplete higher": 1,
        "Lower secondary": 0,
        "Academic degree": 4,
    }

    # train data
    train_df[column_name].replace(mapping_dict, inplace=True)

    # test data
    test_df[column_name].replace(mapping_dict, inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NAME_FAMILY_STATUS"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "NAME_HOUSING_TYPE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    unique_values = train_df[column_name].unique()
    encoder = OneHotEncoder()
    encoded_df = pd.DataFrame(encoder.fit_transform(train_df[[column_name]]).toarray())
    encoded_df.columns = [column_name + "_" + str(idx) for idx in range(len(unique_values))]
    train_df.drop(columns=[column_name], axis=1, inplace=True)
    train_df.index = encoded_df.index
    train_df = train_df.join(encoded_df)

    # test data
    encoded_df = pd.DataFrame(encoder.transform(test_df[[column_name]]).toarray())
    encoded_df.columns = [column_name + "_" + str(idx) for idx in range(len(unique_values))]
    test_df.drop(columns=[column_name], axis=1, inplace=True)
    test_df.index = encoded_df.index
    test_df = test_df.join(encoded_df)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "OCCUPATION_TYPE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since number of missing values is huge i.e., 57854, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "WEEKDAY_APPR_PROCESS_START"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(f"Since Since correlation value is too low, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "ORGANIZATION_TYPE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    train_df[column_name].replace(
        ["Business Entity Type 1", "Business Entity Type 2",
         "Business Entity Type 3"], "Business", inplace=True)
    train_df[column_name].replace(
        ["Transport: type 1", "Transport: type 2", "Transport: type 3",
         "Transport: type 4"], "Transport", inplace=True)
    train_df[column_name].replace(
        ["Trade: type 1", "Trade: type 2", "Trade: type 3", "Trade: type 4",
         "Trade: type 5", "Trade: type 6", "Trade: type 7"], "Trade",
        inplace=True)
    train_df[column_name].replace(["Kindergarten", "School", "University"],
                            "Education", inplace=True)
    train_df[column_name].replace(["Restaurant", "Hotel"], "Dining", inplace=True)
    train_df[column_name].replace(
        ["Industry: type 1", "Industry: type 2", "Industry: type 3",
         "Industry: type 4", "Industry: type 5", "Industry: type 6",
         "Industry: type 7", "Industry: type 8", "Industry: type 9",
         "Industry: type 10", "Industry: type 11", "Industry: type 12",
         "Industry: type 13"], "Industry", inplace=True)
    train_df[column_name].replace(
        ["Insurance", "Legal Services", "Services", "Postal"], "Services",
        inplace=True)
    train_df[column_name].replace(["Mobile", "Telecom"], "Mobile", inplace=True)
    train_df[column_name].replace(["Housing", "Cleaning"], "Housing", inplace=True)
    train_df[column_name].replace(["Realtor", "Construction"], "Realtor",
                            inplace=True)
    train_df[column_name].replace(["Culture", "Religion"], "Culture", inplace=True)
    train_df[column_name].replace(
        ["Military", "Police", "Security Ministries", "Emergency", "Security"],
        "Security", inplace=True)

    unique_values = train_df[column_name].unique()
    encoder = OneHotEncoder()
    encoded_df = pd.DataFrame(
        encoder.fit_transform(train_df[[column_name]]).toarray())
    encoded_df.columns = [column_name + "_" + str(idx) for idx in
                          range(len(unique_values))]
    train_df.drop(columns=[column_name], axis=1, inplace=True)
    train_df.index = encoded_df.index
    train_df = train_df.join(encoded_df)

    # test data
    test_df[column_name].replace(
        ["Business Entity Type 1", "Business Entity Type 2",
         "Business Entity Type 3"], "Business", inplace=True)
    test_df[column_name].replace(
        ["Transport: type 1", "Transport: type 2", "Transport: type 3",
         "Transport: type 4"], "Transport", inplace=True)
    test_df[column_name].replace(
        ["Trade: type 1", "Trade: type 2", "Trade: type 3", "Trade: type 4",
         "Trade: type 5", "Trade: type 6", "Trade: type 7"], "Trade",
        inplace=True)
    test_df[column_name].replace(["Kindergarten", "School", "University"],
                            "Education", inplace=True)
    test_df[column_name].replace(["Restaurant", "Hotel"], "Dining", inplace=True)
    test_df[column_name].replace(
        ["Industry: type 1", "Industry: type 2", "Industry: type 3",
         "Industry: type 4", "Industry: type 5", "Industry: type 6",
         "Industry: type 7", "Industry: type 8", "Industry: type 9",
         "Industry: type 10", "Industry: type 11", "Industry: type 12",
         "Industry: type 13"], "Industry", inplace=True)
    test_df[column_name].replace(
        ["Insurance", "Legal Services", "Services", "Postal"], "Services",
        inplace=True)
    test_df[column_name].replace(["Mobile", "Telecom"], "Mobile", inplace=True)
    test_df[column_name].replace(["Housing", "Cleaning"], "Housing", inplace=True)
    test_df[column_name].replace(["Realtor", "Construction"], "Realtor",
                            inplace=True)
    test_df[column_name].replace(["Culture", "Religion"], "Culture", inplace=True)
    test_df[column_name].replace(
        ["Military", "Police", "Security Ministries", "Emergency", "Security"],
        "Security", inplace=True)
    test_df[column_name].value_counts()

    encoded_df = pd.DataFrame(
        encoder.transform(test_df[[column_name]]).toarray())
    encoded_df.columns = [column_name + "_" + str(idx) for idx in
                          range(len(unique_values))]
    test_df.drop(columns=[column_name], axis=1, inplace=True)
    test_df.index = encoded_df.index
    test_df = test_df.join(encoded_df)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "FONDKAPREMONT_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 126230, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "HOUSETYPE_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 92465, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "WALLSMATERIAL_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 93704, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    column_name = "EMERGENCYSTATE_MODE"
    logger.info(f"Preprocessing column: {column_name}")
    # train data
    logger.info(
        f"Since number of missing values is huge i.e., 87319, hence dropping column")
    train_df.drop(columns=[column_name], inplace=True)

    # test data
    test_df.drop(columns=[column_name], inplace=True)
    # -------------------------------------------------------------------------

    # train_data
    train_new_df = train_df.copy()
    train_target_column = train_new_df[["TARGET"]]
    train_to_scale = train_new_df.drop(columns=["SK_ID_CURR", "TARGET"], axis=1)

    scaler = StandardScaler()
    scaler.fit(train_to_scale)
    train_scaled = scaler.transform(train_to_scale)
    # train_scaled_df = pd.DataFrame(train_scaled, columns=train_to_scale.columns, index=train_target_column.index)
    train_scaled_df = pd.DataFrame(train_scaled,
                                   columns=train_to_scale.columns,
                                   index=train_target_column.index)
    train_scaled_complete_df = pd.concat([train_scaled_df, train_target_column], axis=1)

    # test data
    test_new_df = test_df.copy()
    test_unique_column = test_new_df[["SK_ID_CURR"]]
    test_new_df_dropped = test_new_df.drop(columns=["SK_ID_CURR"], axis=1)
    test_scaled = scaler.transform(test_new_df_dropped)
    # test_scaled_df = pd.DataFrame(test_scaled, columns=test_new_df_dropped.columns, index=test_unique_column.index)
    test_scaled_df = pd.DataFrame(test_scaled,
                                  columns=test_new_df_dropped.columns,
                                  index=test_unique_column.index)
    test_scaled_complete_df = pd.concat([test_scaled_df, test_unique_column], axis=1)

    return train_scaled_complete_df, test_scaled_complete_df
