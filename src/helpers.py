import logging
import os
from datetime import datetime
from typing import NoReturn

import pandas as pd
from pandas.core.frame import DataFrame

logger = logging.getLogger(__name__)


def load_dataset(
    dataset_path: str,
) -> DataFrame:
    df = pd.read_csv(dataset_path)
    return df


def create_log_dir(
    log_dir_path: str,
) -> NoReturn:
    if not os.path.exists(log_dir_path):
        os.makedirs(log_dir_path)
    return
