"""
Module contains tools to run build the entire pipeline of project
"""
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
        """
        Run the entire project as a pipeline.

        To run, we can give options either to run the entire pipeline, or to
        run only a part of it i.e. training only, testing only, or training and
        testing both.

        Args:
            train: bool
                Set to true if only training needs to be done.
            test: bool
                Set to true if only testing needs to be done.
            train_and_test:
                Set to true if both training and testing needs to be done.

        Returns:
        """
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

            try:
                self.train_model(train_df)
            except Exception as e:
                print(f"failed to train model")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"failed to train model")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"error caused - {str(e)}")

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

            try:
                self.test_model(test_df)
            except Exception as e:
                print(f"failed to test model")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"failed to test model")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"error caused - {str(e)}")

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

            try:
                self.train_and_test_model(train_df, test_df)
            except Exception as e:
                print(f"failed to train and test model")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"failed to train and test model")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"error caused - {str(e)}")
        return
