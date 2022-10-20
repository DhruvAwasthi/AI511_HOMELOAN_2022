"""
Module contains tools to run build the entire pipeline of project
"""
import logging
from datetime import datetime
from typing import NoReturn

from pandas.core.frame import DataFrame
from sklearn.model_selection import train_test_split

from src.helpers import load_dataset
from src.preprocess import preprocess_data

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config_info):
        self.config_info = config_info
        self.preprocessing_configuration = config_info.preprocessing_configuration
        self.model_configuration = config_info.model_configuration

    def build_model(self):
        return

    def train_model(
            self,
            train_df: DataFrame,
    ) -> NoReturn:
        """
        Trains the machine learning model.

        It takes the data in pandas DataFrame format, preprocesses it, builds a
        model on top it, trains it, and saves it.

        Args:
            train_df: DataFrame
                Pandas dataframe containing the labels and predictors that will
                be used for training the machine learning model.

        Returns:

        """
        # preprocess the dataset
        preprocessed_df = preprocess_data(train_df,
                                          self.preprocessing_configuration,
                                          is_train_data=True
                                          )

        # separate the predictors and the labels
        predictors = preprocessed_df.drop("TARGET", axis=1)
        labels = preprocessed_df["TARGET"].copy()

        # split the data into train set and validation set
        X_train, X_val, y_train, y_val = train_test_split(
            predictors,
            labels,
            test_size=self.config_info.model_configuration["test_size"],
            random_state=self.config_info.model_configuration["random_state"],
        )

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
        Runs the entire project as a pipeline.

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
                train_df = load_dataset(
                    self.config_info.paths["train_data_path"])
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
                test_df = load_dataset(
                    self.config_info.paths["test_data_path"])
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
                train_df = load_dataset(
                    self.config_info.paths["train_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"train dataset loaded successfully")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                            f"loading test dataset...")
                test_df = load_dataset(
                    self.config_info.paths["test_data_path"])
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
