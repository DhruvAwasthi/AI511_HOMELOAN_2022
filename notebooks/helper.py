import os
from datetime import datetime

import numpy as np
import pandas as pd


def get_datatype_mapping_for_reduction():
    """
    Returns the datatype mapping for dataset size reduction.

    Returns:
        dict
            Dictionary containing column name and reduced datatype as key-value
            pairs.
    """
    datatype_mapping = {
        "SK_ID_CURR": "unsigned",
        "CNT_CHILDREN": "unsigned",
        "AMT_INCOME_TOTAL": "float",
        "AMT_CREDIT": "float",
        "AMT_ANNUITY": "float",
        "AMT_GOODS_PRICE": "float",
        "REGION_POPULATION_RELATIVE": "float",
        "DAYS_BIRTH": "signed",
        "DAYS_EMPLOYED": "signed",
        "DAYS_REGISTRATION": "signed",
        "DAYS_ID_PUBLISH": "signed",
        "OWN_CAR_AGE": "unsigned",
        "FLAG_MOBIL": "unsigned",
        "FLAG_EMP_PHONE": "unsigned",
        "FLAG_WORK_PHONE": "unsigned",
        "FLAG_CONT_MOBILE": "unsigned",
        "FLAG_PHONE": "unsigned",
        "FLAG_EMAIL": "unsigned",
        "CNT_FAM_MEMBERS": "unsigned",
        "REGION_RATING_CLIENT": "unsigned",
        "REGION_RATING_CLIENT_W_CITY": "unsigned",
        "HOUR_APPR_PROCESS_START": "unsigned",
        "REG_REGION_NOT_LIVE_REGION": "unsigned",
        "REG_REGION_NOT_WORK_REGION": "unsigned",
        "LIVE_REGION_NOT_WORK_REGION": "unsigned",
        "REG_CITY_NOT_LIVE_CITY": "unsigned",
        "REG_CITY_NOT_WORK_CITY": "unsigned",
        "LIVE_CITY_NOT_WORK_CITY": "unsigned",
        "EXT_SOURCE_1": "float",
        "EXT_SOURCE_2": "float",
        "EXT_SOURCE_3": "float",
        "APARTMENTS_AVG": "float",
        "BASEMENTAREA_AVG": "float",
        "YEARS_BEGINEXPLUATATION_AVG": "float",
        "YEARS_BUILD_AVG": "float",
        "COMMONAREA_AVG": "float",
        "ELEVATORS_AVG": "float",
        "ENTRANCES_AVG": "float",
        "FLOORSMAX_AVG": "float",
        "FLOORSMIN_AVG": "float",
        "LANDAREA_AVG": "float",
        "LIVINGAPARTMENTS_AVG": "float",
        "LIVINGAREA_AVG": "float",
        "NONLIVINGAPARTMENTS_AVG": "float",
        "NONLIVINGAREA_AVG": "float",
        "APARTMENTS_MODE": "float",
        "BASEMENTAREA_MODE": "float",
        "YEARS_BEGINEXPLUATATION_MODE": "float",
        "YEARS_BUILD_MODE": "float",
        "COMMONAREA_MODE": "float",
        "ELEVATORS_MODE": "float",
        "ENTRANCES_MODE": "float",
        "FLOORSMAX_MODE": "float",
        "FLOORSMIN_MODE": "float",
        "LANDAREA_MODE": "float",
        "LIVINGAPARTMENTS_MODE": "float",
        "LIVINGAREA_MODE": "float",
        "NONLIVINGAPARTMENTS_MODE": "float",
        "NONLIVINGAREA_MODE": "float",
        "APARTMENTS_MEDI": "float",
        "BASEMENTAREA_MEDI": "float",
        "YEARS_BEGINEXPLUATATION_MEDI": "float",
        "YEARS_BUILD_MEDI": "float",
        "COMMONAREA_MEDI": "float",
        "ELEVATORS_MEDI": "float",
        "ENTRANCES_MEDI": "float",
        "FLOORSMAX_MEDI": "float",
        "FLOORSMIN_MEDI": "float",
        "LANDAREA_MEDI": "float",
        "LIVINGAPARTMENTS_MEDI": "float",
        "LIVINGAREA_MEDI": "float",
        "NONLIVINGAPARTMENTS_MEDI": "float",
        "NONLIVINGAREA_MEDI": "float",
        "TOTALAREA_MODE": "float",
        "OBS_30_CNT_SOCIAL_CIRCLE": "unsigned",
        "DEF_30_CNT_SOCIAL_CIRCLE": "unsigned",
        "OBS_60_CNT_SOCIAL_CIRCLE": "unsigned",
        "DEF_60_CNT_SOCIAL_CIRCLE": "unsigned",
        "DAYS_LAST_PHONE_CHANGE": "signed",
        "FLAG_DOCUMENT_2": "unsigned",
        "FLAG_DOCUMENT_3": "unsigned",
        "FLAG_DOCUMENT_4": "unsigned",
        "FLAG_DOCUMENT_5": "unsigned",
        "FLAG_DOCUMENT_6": "unsigned",
        "FLAG_DOCUMENT_7": "unsigned",
        "FLAG_DOCUMENT_8": "unsigned",
        "FLAG_DOCUMENT_9": "unsigned",
        "FLAG_DOCUMENT_10": "unsigned",
        "FLAG_DOCUMENT_11": "unsigned",
        "FLAG_DOCUMENT_12": "unsigned",
        "FLAG_DOCUMENT_13": "unsigned",
        "FLAG_DOCUMENT_14": "unsigned",
        "FLAG_DOCUMENT_15": "unsigned",
        "FLAG_DOCUMENT_16": "unsigned",
        "FLAG_DOCUMENT_17": "unsigned",
        "FLAG_DOCUMENT_18": "unsigned",
        "FLAG_DOCUMENT_19": "unsigned",
        "FLAG_DOCUMENT_20": "unsigned",
        "FLAG_DOCUMENT_21": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_HOUR": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_DAY": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_WEEK": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_MON": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_QRT": "unsigned",
        "AMT_REQ_CREDIT_BUREAU_YEAR": "unsigned",
    }
    return datatype_mapping


def reduce_dataset_size(
    df,
):
    """
    Reduced the dataset size by changing the datatype of each column.

    Args:
        df: DataFrame
            Pandas DataFrame whose size needs to be reduced.

    Returns:
        DataFrame:
            Pandas DataFrane with reduced size.
    """
    datatype_mapping = get_datatype_mapping_for_reduction()
    print(
        f"Reducing dataset size")
    for column_name, column_data_type in datatype_mapping.items():
        try:
            df[column_name] = pd.to_numeric(df[column_name],
                                            downcast=column_data_type)
        except Exception as e:
            print(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to "
                f"change datatype of {column_name}")
            print(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error caused -"
                f"{str(e)}")
    return df


def calculate_iqr_range(
        data,
        scaled_factor,
        percentile_range,
):
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
