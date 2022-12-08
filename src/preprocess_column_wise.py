import logging

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame

from src.helpers import col_info


logger = logging.getLogger(__name__)


def column_wise(
    train_df: DataFrame,
    test_df: DataFrame,
) -> tuple[DataFrame, DataFrame]:
    # CNT_CHILDREN
    # train data
    # replace all values with cnt_children >=3 with 3
    column_name = "CNT_CHILDREN"
    logger.info(f"Preprocessing column: {column_name}")
    train_df[column_name].replace([3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 19], 3,
                               inplace=True)

    # test data
    test_df[column_name].replace([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], 3,
                                     inplace=True)

    # AMT_INCOME_TOTAL
    # train data
    # there are just four samples with income >9000000; let's just drop them
    column_name = "AMT_INCOME_TOTAL"
    logger.info(f"Preprocessing column: {column_name}")
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(train_df, column_name)
    to_delete_indices = train_df[(train_df[column_name] >= 9000000)].index
    train_df.drop(index=to_delete_indices, inplace=True)
    index_of_outliers = index_of_outliers.drop(to_delete_indices)
    # trim outlier values with the extreme value that is not an outlier
    lowest_outlier_value = more_than_mean[0]
    index_lowest_outlier_value = \
    np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value - 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[index_lowest_outlier_value - 1]

    # test data
    index_of_outliers = test_df[
        (test_df[column_name] > upper_bound) | (
                test_df[column_name] < lower_bound)].index
    test_df.loc[index_of_outliers, column_name] = unique_values[index_lowest_outlier_value - 1]


    # AMT_CREDIT
    # train data
    column_name = "AMT_CREDIT"
    logger.info(f"Preprocessing column: {column_name}")
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound, column_mean_without_outliers = col_info(train_df, column_name)
    lowest_outlier_value = more_than_mean[0]
    index_lowest_outlier_value = \
        np.where(unique_values == lowest_outlier_value)[0][0]
    logger.info(f"Lowest outlier value in unique values: {lowest_outlier_value}")
    logger.info(
        f"Index of lowest outlier value in unique values: {index_lowest_outlier_value}")
    logger.info(
        f"Outliers will be trimmed to: {unique_values[index_lowest_outlier_value - 1]}")
    train_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]

    # test data
    index_of_outliers = test_df[
        (test_df[column_name] > upper_bound) | (
                test_df[column_name] < lower_bound)].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]


    # AMT_ANNUITY
    # train data
    column_name = "AMT_ANNUITY"
    logger.info(f"Preprocessing column: {column_name}")
    index_of_outliers, less_than_mean, more_than_mean, unique_values, iqr_range, lower_bound, upper_bound,column_mean_without_outliers = col_info(train_df, column_name)
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
    index_of_outliers = test_df[
        (test_df[column_name] > upper_bound) | (
                test_df[column_name] < lower_bound)].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    test_df[column_name].fillna(column_mean_without_outliers, inplace=True)


    # AMT_GOODS_PRICE
    # train data
    column_name = "AMT_GOODS_PRICE"
    logger.info(f"Preprocessing column: {column_name}")
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
    index_of_outliers = test_df[
        (test_df[column_name] > upper_bound) | (
                test_df[column_name] < lower_bound)].index
    test_df.loc[index_of_outliers, column_name] = unique_values[
        index_lowest_outlier_value - 1]
    test_df[column_name].fillna(column_mean_without_outliers, inplace=True)


    # REGION_POPULATION_RELATIVE
    # train data
    column_name = "REGION_POPULATION_RELATIVE"

    return train_df, test_df
