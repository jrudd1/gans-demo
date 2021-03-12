# -*- coding: utf-8 -*-

"""
This script is used to produce synthetic credit fraud data and evaluate results
Usage:
    python3 ./src/gan_evaluate.py
"""
import logging
import pickle
from pathlib import Path

import click
import numpy as np
import pandas as pd
import sklearn.cluster as cluster

# visualization
import sweetviz
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, roc_auc_score

# metrics
from sklearn.model_selection import train_test_split
from table_evaluator import TableEvaluator, load_data
from ydata_synthetic.preprocessing.regular.credit_fraud import *
from ydata_synthetic.synthesizers.regular import CGAN, WGAN_GP, VanilllaGAN

from utility import load_data, parse_config, set_logger


@click.command()
@click.argument("config_file", type=str, default="src/config.yml")
def gan_evaluate(config_file):
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
    logger = set_logger("./log/gan_evaluate.log")

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
    model_name = config["gan_train"]["model_name"]
    random_state = config["gan_evaluate"]["random_state"]
    test_size = config["gan_evaluate"]["test_size"]
    noise_dim = config["gan_evaluate"]["noise_dim"]

    logger.info(f"config: {config['etl'],config['gan_train']}")

    ##################
    # Load model
    ##################
    logger.info(
        f"-------------------Load the synthetic model and data-------------------"
    )

    # Loading the synthesizer
    # sample of synthetic data
    # synthesizer = WGAN_GP.load(path='models/wgangp_credit_fraud.pkl')
    synthesizer = WGAN_GP.load(path=model_path + model_name + "_" + data_name + ".pkl")

    synth_data = synthesizer.sample(1000)
    logger.info(f"Synthetic data shape: {synth_data.shape}")
    logger.info(synth_data.head())

    models = {model_name: [model_name, False, synthesizer.generator]}

    # load processed real data
    logger.info(
        f"-------------------Load the preprocessed real data-------------------"
    )
    # data = pickle.load(open(processed_train,'rb'))
    data = pickle.load(
        open(processed_data_path + str(data_name) + ".engineered.pkl", "rb")
    )
    data_cols = list(data.columns[data.columns != target])

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

    train_data = data.loc[data[target] == 1].copy()

    algorithm = cluster.KMeans
    args, kwds = (), {"n_clusters": 2, "random_state": 0}
    labels = algorithm(*args, **kwds).fit_predict(train_data[data_cols])

    fraud_w_classes = train_data.copy()
    fraud_w_classes[target] = labels

    train_sample = fraud_w_classes.copy().reset_index(drop=True)
    # train_sample = pd.get_dummies(train_sample, columns=[target], prefix=target, drop_first=True)
    label_cols = [i for i in train_sample.columns if target in i]
    data_cols = [i for i in train_sample.columns if i not in label_cols]
    train_sample[data_cols] = (
        train_sample[data_cols] / 10
    )  # scale to random noise size, one less thing to learn
    train_no_label = train_sample[data_cols]

    logger.info(f"cols: {data_cols + label_cols}")
    logger.info(f"Processed data shape: {data.shape}")
    logger.info(
        "Training dataset info: Number of records - {} Number of variables - {}".format(
            train_data.shape[0], train_data.shape[1]
        )
    )
    logger.info(
        f"Unique labels in training data: { pd.DataFrame( [ [np.sum(labels==i)] for i in np.unique(labels) ], columns=['count'], index=np.unique(labels) )} "
    )
    logger.info(f"train data with no label shape: {train_no_label.shape}")
    logger.info(train_no_label.head())

    logger.info(f"train data with label shape: {train_sample.shape}")
    logger.info(train_sample.head())
    logger.info("\n")

    ##################
    # Create synthetic data
    ##################
    # Load model
    logger.info(f"-------------------Create synthetic data -------------------")

    np.random.seed(random_state)
    z = np.random.normal(size=(test_size, noise_dim))
    real = synthesizer.get_data_batch(
        train=train_sample, batch_size=test_size, seed=random_state
    )
    real_samples = pd.DataFrame(real, columns=data_cols + label_cols)
    labels = fraud_w_classes[target]

    base_dir = "cache/"
    model_names = [model_name]
    colors = ["deepskyblue", "blue"]
    markers = ["o", "^"]
    class_labels = ["Class 1", "Class 2"]

    col1, col2 = "V17", "V10"

    # Actual fraud data visualization
    model_steps = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
    rows = len(model_steps)
    columns = 5

    axarr = [[]] * len(model_steps)

    fig = plt.figure(figsize=(14, rows * 3))

    for model_step_ix, model_step in enumerate(model_steps):
        axarr[model_step_ix] = plt.subplot(rows, columns, model_step_ix * columns + 1)

        for group, color, marker, label in zip(
            real_samples.groupby(target), colors, markers, class_labels
        ):
            plt.scatter(
                group[1][[col1]],
                group[1][[col2]],
                label=label,
                marker=marker,
                edgecolors=color,
                facecolors="none",
            )

        plt.title("Actual Fraud Data")
        plt.ylabel(col2)  # Only add y label to left plot
        plt.xlabel(col1)
        xlims, ylims = axarr[model_step_ix].get_xlim(), axarr[model_step_ix].get_ylim()

        if model_step_ix == 0:
            legend = plt.legend()
            legend.get_frame().set_facecolor("white")

        for i, model_name in enumerate(model_names[:]):

            [model_name, with_class, generator_model] = models[model_name]
            # [model_name, generator_model] = models[model_name]

            generator_model.load_weights(
                base_dir + "_generator_model_weights_step_" + str(model_step) + ".h5"
            )

            ax = plt.subplot(rows, columns, model_step_ix * columns + 1 + (i + 1))

            if with_class:
                # labels = x[:,-label_dim:]
                g_z = generator_model.predict([z, labels])
                gen_samples = pd.DataFrame(g_z, columns=data_cols + label_cols)
                for group, color, marker, label in zip(
                    gen_samples.groupby(target), colors, markers, class_labels
                ):
                    plt.scatter(
                        group[1][[col1]],
                        group[1][[col2]],
                        label=label,
                        marker=marker,
                        edgecolors=color,
                        facecolors="none",
                    )
            else:
                g_z = generator_model.predict(z)
                gen_samples = pd.DataFrame(g_z, columns=data_cols + ["label"])
                gen_samples.to_csv(
                    model_path + model_name + "_generated_sample.csv", index=False
                )
                plt.scatter(
                    gen_samples[[col1]],
                    gen_samples[[col2]],
                    label=class_labels[0],
                    marker=markers[0],
                    edgecolors=colors[0],
                    facecolors="none",
                )
            plt.title(model_name)
            plt.xlabel(data_cols[0])
            ax.set_xlim(xlims), ax.set_ylim(ylims)

    plt.suptitle("Comparison of GAN outputs", size=16, fontweight="bold")
    plt.tight_layout(rect=[0.075, 0, 1, 0.95])

    # Adding text labels for traning steps
    vpositions = np.array([i._position.bounds[1] for i in axarr])
    vpositions += (vpositions[0] - vpositions[1]) * 0.35
    for model_step_ix, model_step in enumerate(model_steps):
        fig.text(
            0.05,
            vpositions[model_step_ix],
            "training\nstep\n" + str(model_step),
            ha="center",
            va="center",
            size=12,
        )

    plt.savefig("reports/figures/Comparison_of_GAN_outputs.png")

    logger.info("\n")

    ##################
    # Load synthetic data and create reports
    ##################
    # Load modatadel
    logger.info(f"-------------------Load synthetic data -------------------")

    synth = pd.read_csv(model_path + model_name + "_generated_sample.csv")

    algorithm = cluster.KMeans
    args, kwds = (), {"n_clusters": 2, "random_state": 0}
    labels = algorithm(*args, **kwds).fit_predict(synth[data_cols])
    synth["label"] = labels
    synth = synth.rename(columns={"label": target})

    my_report = sweetviz.compare([train_sample, "Real"], [synth, "Synthetic"])
    my_report.show_html("reports/" + data_name + "_comparison_report.html")

    table_evaluator = TableEvaluator(train_sample, synth)
    # logger.info(print(table_evaluator.visual_evaluation(plot_correlation_comparison)))
    logger.info(table_evaluator.evaluate(target_col=target))
    logger.info(synth.head())


if __name__ == "__main__":
    gan_evaluate()
