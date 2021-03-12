# -*- coding: utf-8 -*-

"""
This script is used to train and export ML model according to config
Usage:
    python3 ./src/train.py
"""
import logging

# from cloudpickle import dump
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sklearn
import sklearn.cluster as cluster
import xgboost as xgb
from rich import pretty, print
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# metrics
from sklearn.model_selection import train_test_split

# load custom packages/functions
from utility import (
    BaseMetrics,
    SimpleAccuracy,
    SimpleMetrics,
    load_data,
    parse_config,
    precision,
    recall,
    roc_auc,
    set_logger,
)

pretty.install()


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def train(config_file):
    """
    Main function that trains & persists model based on training set
    Args:
        config_file [str]: path to config file
    Returns:
        None
    """
    ##################
    # configure logger
    ##################
    logger = set_logger("./log/train.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    processed_train = Path(config["train"]["processed_train"])
    ensemble_model = config["train"]["ensemble_model"]
    model_config = config["train"]["model_config"]
    model_eval = config["train"]["model_eval"]
    model_path = Path(config["train"]["model_path"])
    test_size = config["train"]["test_size"]
    random_state = config["train"]["random_state"]

    logger.info(f"config: {config['train']}")

    ##################
    # Load data
    ##################
    logger.info(f"-------------------Load the processed data-------------------")
    # Load engineered dataset from EDA section

    # data = pickle.load(open(processed_data + 'credicard.engineered.pkl','rb'))
    data = pickle.load(open(processed_train, "rb"))

    # data columns will be all other columns except class
    data_cols = list(data.columns[data.columns != "Class"])
    label_cols = ["Class"]

    n_real = np.sum(data.Class == 0)  # 200000
    n_fraud = np.sum(data.Class == 1)  # 473

    X = data[data_cols]
    y = data[label_cols]

    # split into train test sets - 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=1, stratify=y
    )

    logger.info(f"classes 0, 1: {n_real, n_fraud}")
    logger.info(f"train, test: {len(X_train), len(X_test)}")
    logger.info(f"cols: {data_cols + label_cols}")
    logger.info("\n")

    ##################
    # Set & train model
    ##################
    # Load model
    # Limited to sklearn ensemble for the moment
    logger.info(f"-------------------Initiate model-------------------")

    model = xgb

    dtrain = model.DMatrix(X_train, y_train, feature_names=data_cols)
    dtest = model.DMatrix(X_test, y_test, feature_names=data_cols)

    results_dict = {}

    test = model.train(
        model_config,
        dtrain,
        num_boost_round=100,
        verbose_eval=False,
        early_stopping_rounds=20,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=results_dict,
        # feval = recall, maximize=True
        feval=model_eval,
        maximize=True,
    )

    # model = initiate_model(ensemble_model, model_config)

    # Train model
    logger.info(f"Train model using {ensemble_model}, {model_config}")
    y_pred = test.predict(dtest, ntree_limit=test.best_iteration + 1)
    y_true = y_test["Class"].values

    logger.info(f"best iteration: {test.best_iteration}")
    logger.info(f"Recall: {recall( np.round(y_pred), dtest)}")
    logger.info(f"Precision: {precision( y_pred, dtest )}")
    logger.info(f"ROC_AUC: {roc_auc( y_pred, dtest)}")
    logger.info(SimpleMetrics(np.round(y_pred), y_true))
    logger.info("\n")
    ##################
    # Persist model
    ##################

    # logger.info(f"-------------------Persist model-------------------")
    # model_path.parent.mkdir(parents=True, exist_ok=True)
    # with open(model_path, "wb") as f:
    #     dump(test, f)
    # logger.info(f"Persisted model to {model_path}")


def initiate_model(ensemble_model, model_config):
    """
    initiate model using eval, implement with defensive programming
    Args:
        ensemble_model [str]: name of the ensemble model

    Returns:
        [sklearn.model]: initiated model
    """
    if ensemble_model in dir(sklearn.ensemble):
        return eval("sklearn.ensemble." + ensemble_model)(**model_config)
    else:
        raise NameError(f"{ensemble_model} is not in sklearn.ensemble")


if __name__ == "__main__":
    train()
