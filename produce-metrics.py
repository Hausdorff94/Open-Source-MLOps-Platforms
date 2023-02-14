from mlflow import log_metric
from random import choice

metric_names = ["accuracy", "RMSE", "precision", "recall", "F1"]
percentages = [i for i in range(0, 101)]

for i in range(100):
    log_metric(choice(metric_names), choice(percentages))