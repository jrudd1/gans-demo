import logging
from pathlib import Path

import pandas as pd
import yaml


def parse_config(config_file):

    with open(config_file, "rb") as f:
        config = yaml.safe_load(f)
    return config


def set_logger(log_path):
    """
    Read more about logging: https://www.machinelearningplus.com/python/python-logging-guide/
    Args:
        log_path [str]: eg: "../log/train.log"
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_path, mode="w")
    formatter = logging.Formatter(
        "%(asctime)s : %(levelname)s : %(name)s : %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info(f"Finished logger configuration!")
    return logger


def load_data(processed_data_path, target):
    """
    Load data from specified file path
    ***I know this function is dumb, why we need another function? Just to demo unit test?
    In this case it is easy, but if you have complex pipeline, you will
    want to safeguard the behavior!
    Args:
        processed_data [str]: file path to processed data

    Returns:
        [tuple]: feature matrix and target variable
    """

    data = pd.read_csv(processed_data_path)
    return data.drop(target, axis=1).to_numpy(), data[target], list(data.columns)
    # return data


# https://github.com/dmlc/xgboost/blob/master/demo/guide-python/custom_objective.py


def recall(preds, dtrain):
    labels = dtrain.get_label()
    preds = np.round(preds).astype("Int64")
    return "recall", recall_score(labels, preds, average="binary")


def precision(preds, dtrain):
    labels = dtrain.get_label()
    return "precision", precision_score(labels, np.round(preds))


def roc_auc(preds, dtrain):
    labels = dtrain.get_label()
    return "roc_auc", roc_auc_score(labels, preds)


def BaseMetrics(y_pred, y_true):
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    return TP, TN, FP, FN


def SimpleMetrics(y_pred, y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred, y_true)
    ACC = (TP + TN) / (TP + TN + FP + FN)

    # Reporting
    from IPython.display import display

    print("Confusion Matrix")
    display(
        pd.DataFrame(
            [[TN, FP], [FN, TP]],
            columns=["Pred 0", "Pred 1"],
            index=["True 0", "True 1"],
        )
    )
    print("Accuracy : {}".format(ACC))


def SimpleAccuracy(y_pred, y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred, y_true)
    ACC = (TP + TN) / (TP + TN + FP + FN)
    return ACC
