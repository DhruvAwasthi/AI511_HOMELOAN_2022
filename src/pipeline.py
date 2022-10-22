"""
Module contains tools to run build the entire pipeline of project
"""
import logging
from datetime import datetime
from typing import NoReturn

from pandas.core.frame import DataFrame
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

from src.helpers import load_dataset, pickle_dump_object
from src.preprocess import preprocess_data

logger = logging.getLogger(__name__)


class Pipeline:
    def __init__(self, config_info):
        self.config_info = config_info
        self.preprocessing_configuration = config_info.preprocess_configuration
        self.model_configuration = config_info.model_configuration

    def build_model(
            self,
    ) -> LogisticRegression:
        """
        Returns a logistic regression model
        Returns:
            LogisticRegression
                A logistic regression model
        """
        clf = None
        try:
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                    f"building the logistic regression model")
            clf = LogisticRegression(
                random_state=self.model_configuration["random_state"],
                max_iter=self.model_configuration["max_iter"],
            )
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                        f"successfully built logistic regression model")
        except Exception as e:
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} failed to "
                f"built logistic regression model")
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} error"
                        f"caused - {str(e)}")
        return clf

    def scale_data(
            self,
    ):
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
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                    f"preprocessing data")
        preprocessed_df = preprocess_data(train_df,
                                          self.preprocessing_configuration,
                                          is_train_data=True
                                          )
        logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                    f"preprocessing done")
        # separate the predictors and the labels
        predictors = preprocessed_df.drop("TARGET", axis=1)
        labels = preprocessed_df["TARGET"].copy()

        # build the model
        clf = self.build_model()

        stratified_splits = StratifiedShuffleSplit(
            n_splits=self.config_info.model_configuration["num_stratified_shuffle_splits"],
            test_size=self.config_info.model_configuration["test_size"],
            random_state=self.config_info.model_configuration["random_state"],
        )


        # split the data using stratified fold for training
        iter = 0
        for train_indices, val_indices in stratified_splits.split(predictors, labels):
            iter += 1
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} training "
                f"model - iteration {iter}")
            X_train, X_val = predictors.loc[train_indices], predictors.loc[val_indices]
            y_train, y_val = labels.loc[train_indices], labels.loc[val_indices]
            clf.fit(X_train.iloc[:, :-1].values, y_train.values)
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} complete "
                f"training model - iteration {iter}")
            y_predict = clf.predict(X_val.iloc[:, :-1].values)
            logger.info(
                f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} calculating "
                f"classification scores")
            accuracy = accuracy_score(y_val.values, y_predict)
            f1 = f1_score(y_val.values, y_predict)
            f1_macro = f1_score(y_val.values, y_predict, average="macro")
            precision = precision_score(y_val.values, y_predict)
            recall = recall_score(y_val.values, y_predict)
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                        f"accuracy score is: {accuracy}")
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                        f"f1 score score is: {f1}")
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                        f"f1 score (macro) score is: {f1_macro}")
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                        f"precision score is: {precision}")
            logger.info(f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} "
                        f"recall score is: {recall}")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} training done")

        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} saving model")
        pickle_dump_object(clf, "model.sav")
        logger.info(
            f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')} model saved")
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
                            f"loading train dataset")
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
