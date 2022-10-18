import logging
from datetime import datetime
from typing import NoReturn

from src.helpers import load_dataset

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config_info):
        self.config_info = config_info

    def preprocess_data(self):
        return

    def build_model(self):
        return

    def train_model(self):
        return

    def test_model(self):
        return

    def train_and_test_model(self):
        return

    def run(
        self,
        train=False,
        test=False,
        train_and_test=False,
    ) -> NoReturn:

        if train:
            try:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} loading train dataset...")
                train_df = load_dataset(self.config_info.paths["train_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} train dataset loaded successfully")
            except Exception as e:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to load train dataset")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error caused - {str(e)}")

        elif test:
            try:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} loading test dataset...")
                test_df = load_dataset(self.config_info.paths["test_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} test dataset loaded successfully")
            except Exception as e:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to load train dataset")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error caused - {str(e)}")

        elif train_and_test:
            try:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} loading train dataset...")
                train_df = load_dataset(self.config_info.paths["train_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} train dataset loaded successfully")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} loading test dataset...")
                test_df = load_dataset(self.config_info.paths["test_data_path"])
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} test dataset loaded successfully")
            except Exception as e:
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to load datasets")
                logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error caused - {str(e)}")
        return
