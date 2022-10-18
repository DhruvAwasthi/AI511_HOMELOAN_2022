import logging
from datetime import datetime

from pandas.core.frame import DataFrame

logger = logging.getLogger(__name__)


def remove_duplicate_rows(
    df: DataFrame,
) -> DataFrame:
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
