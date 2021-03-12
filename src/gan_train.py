# -*- coding: utf-8 -*-

"""
This script is used to train and export GAN model according to config
Usage:
    python3 ./src/gan_train.py
"""
import logging
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sklearn.cluster as cluster
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# metrics
from sklearn.model_selection import train_test_split
from ydata_synthetic.preprocessing.regular.credit_fraud import *
from ydata_synthetic.synthesizers.regular import CGAN, WGAN_GP, VanilllaGAN

from utility import load_data, parse_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def gan_train(config_file):
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
    logger = set_logger("./log/gan_train.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    data_name = config["etl"]["data_name"]
    processed_data_path = config["etl"][
        "processed_data_path"
    ]  # this is where we put the cleaned and pre-processed data
    target = config["etl"]["target"]  # the target feature
    model_path = config["gan_train"]["model_path"]
    # gan_model = config["gan_train"]["gan_model"]
    model_name = config["gan_train"]["model_name"]
    gan_args = list(config["gan_train"]["gan_args"].values())
    train_args = list(config["gan_train"]["train_args"].values())
    logger.info(f"config: {config['etl'],config['gan_train']}")

    ##################
    # Load data
    ##################
    logger.info(f"-------------------Load the processed data-------------------")

    data = pickle.load(
        open(processed_data_path + str(data_name) + ".engineered.pkl", "rb")
    )
    # data columns will be all other columns except class
    data_cols = list(data.columns[data.columns != target])
    label_cols = [target]

    sorted_cols = [
        "V4",
        "V14",
        "V12",
        "V8",
        "Amount",
        "V19",
        "V10",
        "V11",
        "Time",
        "V26",
        "V28",
        "V22",
        "V7",
        "V24",
        "V16",
        "V20",
        "V18",
        "V3",
        "V21",
        "V6",
        "V25",
        "V15",
        "V1",
        "V9",
        "V23",
        "V13",
        "V2",
        "V5",
        "V27",
        "V17",
        "Class",
    ]
    data = data[sorted_cols].copy()

    logger.info(f"cols: {data_cols + label_cols}")
    logger.info(f"Processed data shape: {data.shape}")
    logger.info("\n")

    ##################
    # Set & train model
    ##################
    # Load model
    logger.info(f"-------------------Initiate model-------------------")
    # For the purpose of this example we will only synthesize the minority class - and create labels for the conditional gan which requires labels
    train_data = data.loc[data[target] == 1].copy()

    algorithm = cluster.KMeans
    args, kwds = (), {"n_clusters": 2, "random_state": 0}
    labels = algorithm(*args, **kwds).fit_predict(train_data[data_cols])

    fraud_w_classes = train_data.copy()
    fraud_w_classes[target] = labels

    logger.info(
        "Training dataset info: Number of records - {} Number of variables - {}".format(
            train_data.shape[0], train_data.shape[1]
        )
    )
    logger.info(
        f"Unique labels in training data: { pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) )} "
    )

    models_dir = "./cache"

    train_sample = fraud_w_classes.copy().reset_index(drop=True)
    train_sample = pd.get_dummies(
        train_sample, columns=[target], prefix=target, drop_first=True
    )
    label_cols = [i for i in train_sample.columns if target in i]
    data_cols = [i for i in train_sample.columns if i not in label_cols]
    train_sample[data_cols] = (
        train_sample[data_cols] / 10
    )  # scale to random noise size, one less thing to learn
    train_no_label = train_sample[data_cols]

    logger.info(f"train data with no label shape: {train_no_label.shape}")
    logger.info(train_no_label.head())

    logger.info(f"train data with label shape: {train_sample.shape}")
    logger.info(train_sample.head())
    logger.info("\n")

    # Training the WGAN_GP model
    gan_model = WGAN_GP
    synthesizer = gan_model(gan_args, n_critic=2)
    synthesizer.train(train_sample, train_args)

    logger.info(f"target data synthesized: {target}")
    logger.info(f"gan model used: {gan_model}")
    logger.info(f'gan args used: {config["gan_train"]["gan_args"]}')
    logger.info(f'train args used: {config["gan_train"]["train_args"]}')
    logger.info(synthesizer.generator.summary())
    logger.info(synthesizer.critic.summary())

    #     ##################
    #     # Persist model
    #     ##################
    #  Saving the synthesizer to later generate new events
    # synthesizer.save(path = model_path + gan_model + data_name + '.pkl')
    synthesizer.save(path="models/wgangp_credit_fraud.pkl")
    # synthesizer.save(path = "model_path + model_name + '_' + data_name + '.pkl'")
    logger.info(
        f"Persisted model to {model_path + model_name + '_' + data_name + '.pkl'}"
    )


if __name__ == "__main__":
    gan_train()
