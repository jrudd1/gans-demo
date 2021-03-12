# -*- coding: utf-8 -*-

"""
This script is used to ingest raw data from external soures and convert the raw data to processed for GAN training
It is designed to be idempotent [stateless transformation]
Usage:
    python3 ./src/etl.py
"""
import logging
import os
import pickle
import zipfile
from pathlib import Path
from time import sleep

import click
import dotenv
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv

# kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
from rich import pretty, print
from rich.progress import track
from sklearn.model_selection import train_test_split

# preprocessing
from sklearn.preprocessing import LabelEncoder, PowerTransformer

from utility import parse_config, set_logger

pretty.install()


# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()

# load up the entries as environment variables
load_dotenv(dotenv_path)


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def etl(config_file):
    """
    ETL function that load raw data and convert to train and test set
    Args:
        config_file [str]: path to config file
    Returns:
        None
    """
    ##################
    # configure logger
    ##################
    logger = set_logger("./log/etl.log")

    ##################
    # Load config from config file
    ##################
    logger.info(f"Load config from {config_file}")
    config = parse_config(config_file)

    external_data_url = config["etl"]["external_data_url"]  # external url of raw data
    external_data_path = config["etl"][
        "external_data_path"
    ]  # this is where you ingest the kaggle zip files
    raw_data_path = config["etl"][
        "raw_data_path"
    ]  # this is where you unzip the csv files
    processed_data_path = config["etl"][
        "processed_data_path"
    ]  # this is where we put the cleaned and pre-processed data
    data_name = config["etl"]["data_name"]  # what we're naming the data file
    target = config["etl"]["target"]  # the target feature
    logger.info(f"config: {config['etl']}")

    ##################
    # Load SECRET KEYS
    ##################
    KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME")
    KAGGLE_KEY = os.environ.get("KAGGLE_KEY")

    ##################
    # Data extraction
    ##################
    logger.info("-------------------Start data ingestion-------------------")
    # kaggle imports
    api = KaggleApi()
    api.authenticate()

    def kaggle_download_raw(url, in_path):
        api.dataset_download_files(url, path=in_path)

    def unzip(in_path, out_path):
        pathlist = Path(in_path).rglob("*.zip")
        for path in pathlist:  # get the list of files
            with zipfile.ZipFile(path) as zipref:  # treat the file as a zip
                zipref.extractall(out_path)  # extract it in the raw data directory

    def read_files(in_path, index=False):
        pathlist = Path(in_path).rglob("*.csv")
        dataframes = [pd.read_csv(path, index_col=index) for path in pathlist]
        return dataframes

    [kaggle_download_raw(url, external_data_path) for url in external_data_url]
    unzip(external_data_path, raw_data_path)
    logger.info("End data extraction")
    logger.info("\n")
    # ##################
    # # Data transformation
    # ##################
    logger.info("-------------------Start data transformation-------------------")

    # raw_data_file[0] = read_files(raw_data)[0]
    def credit_transformations(data):

        # drop duplicate records
        data = data.drop_duplicates()

        # convert time to time of day in hours
        data["Time"] = (data["Time"].values / 3600) % 24

        # Log transformation to Amount variable
        Amountlog = np.log10(data["Amount"].values + 1)
        data["Amount"] = Amountlog

        # transform data columns to be approx Gaussian
        data_cols = list(data.columns[data.columns != target])

        # data[data_cols] = StandardScaler().fit_transform(data[data_cols])
        data[data_cols] = PowerTransformer(
            method="yeo-johnson", standardize=True, copy=True
        ).fit_transform(data[data_cols])

        pickle.dump(
            data, open(processed_data_path + data_name + ".engineered.pkl", "wb")
        )

        return data

    data = read_files(raw_data_path)[0]

    data = credit_transformations(data)
    data_cols = list(data.columns[data.columns != target])

    logger.info(f"Dataset name: {data_name}")
    logger.info(f"Dataset target: {target}")
    logger.info(f"Dataset columns: {data_cols}")
    logger.info(f"# of data columns: {len(data_cols)}")
    logger.info(data.head())
    logger.info("End data transformation")
    logger.info("\n")


if __name__ == "__main__":
    etl()
