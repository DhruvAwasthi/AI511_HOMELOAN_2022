import logging
from datetime import datetime
from typing import NoReturn

from pandas.core.frame import DataFrame

from src.helpers import load_dataset

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config_info):
        self.config_info = config_info

    def preprocess_data(
        self,
        df: DataFrame,
    ):
        return

    def build_model(self):
        return

    def train_model(
        self,
        train_df: DataFrame,
    ):
        return

    def test_model(
        self,
        test_df: DataFrame,
    ):
        return

    def train_and_test_model(
        self,
        train_df: DataFrame,
        test_df: DataFrame,
    ):
        return

    def run(
        self,
        train: bool = False,
        test: bool = False,
        train_and_test: bool = False,
    ) -> NoReturn:

        if train:
            try:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"loading train dataset...")
                train_df = load_dataset(self.config_info.paths["train_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"train dataset loaded successfully")
            except Exception as e:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"failed to load train dataset")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"error caused - {str(e)}")
                return

            self.train_model(train_df)

        elif test:
            try:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"loading test dataset...")
                test_df = load_dataset(self.config_info.paths["test_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"test dataset loaded successfully")
            except Exception as e:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"failed to load train dataset")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"error caused - {str(e)}")
                return

            self.test_model(test_df)

        elif train_and_test:
            try:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"loading train dataset...")
                train_df = load_dataset(self.config_info.paths["train_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"train dataset loaded successfully")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"loading test dataset...")
                test_df = load_dataset(self.config_info.paths["test_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"test dataset loaded successfully")
            except Exception as e:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"failed to load datasets")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"error caused - {str(e)}")
                return

            self.train_and_test_model(train_df, test_df)
        return
