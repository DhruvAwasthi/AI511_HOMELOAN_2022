"""
Main module that contains tools to run the project
"""
import argparse
import logging
import os
from datetime import datetime

import config
from src.pipeline import Pipeline
from src.helpers import create_dirs

# create directory structure
create_dirs(config.LOG_DIR, config.DUMP_DIR, config.FIGURES_DIR)

# build command line interface for easy usage
parser = argparse.ArgumentParser(
    description="Home Loan Default Risk Prediction",
    epilog="Thank you for using Home Loan Default Risk Prediction pipeline!"
)
parser.add_argument(
    "--train_model",
    default=False,
    type=bool,
    help="set this to true to build and train the model"
)
parser.add_argument(
    "--test_model",
    default=False,
    type=bool,
    help="set this to true to test the model"
)
parser.add_argument(
    "--train_and_test_model",
    default=False,
    type=bool,
    help="set this to true to train and test the model at once")

args = parser.parse_args()

# run pipeline to build and train the model only
if args.train_model:
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(config.LOG_DIR,
                              datetime.now().strftime(
                                  "%Y-%m-%d_%H-%M-%S") + "_train_model.log")
    )
    train_pipeline = Pipeline(config).run(train=True)

# run pipeline to test the model only by generating predictions for test data
elif args.test_model:
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(config.LOG_DIR,
                              datetime.now().strftime(
                                  "%Y-%m-%d_%H-%M-%S") + "_test_model.log")
    )
    test_pipeline = Pipeline(config).run(test=True)

# run pipeline to train and test the model at once
elif args.train_and_test_model:
    logging.basicConfig(
        level=logging.INFO,
        filename=os.path.join(config.LOG_DIR,
                              datetime.now().strftime("%Y-%m-%d_%H-%M-%S") +
                              "_train_and_test_model.log")
    )
    train_and_test_pipeline = Pipeline(config).run(train_and_test=True)

else:
    print(f"No argument supplied.")
    print(f"Run python main.py -h to see a list of available options.")
