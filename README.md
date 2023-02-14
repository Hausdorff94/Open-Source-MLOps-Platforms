# Open Source MLOps Platforms
---
## MLflow

### MLflow Tracking

```python
import mlflow

# Log parameters
mlflow.log_param("num_dimensions", 8)
mlflow.log_param("regularization", 0.1)

# Log metricts
mlflow.log_metric("accuracy", 0.1)

# Log artifacts
mlflow.log_artifact("roc.png")
mlflow.log_artifact("model.pkl")
```

### Using MLflow

Run MLflow Tracking Server UI

```bash
mlflow ui --port 5000
```

Registry some metrics, parameters and artifacts

```bash
python3 test-mlflow.py
```

#### Create an experiment

```bash
mlflow experiments create -n "produce-metrics"
```

Use the script `produce-metrics.py` to produce some metrics, parameters and artifacts.

```bash
MLFLOW_EXPERIMENT_ID=460706163268634032 python3 produce-metrics.py
```

#### Parameters, Version, Artifacts and Metrics

Train a regression model with the wine quality dataset. Use the script `train.py` to train and log the parameters, metrics and artifacts.

```bash
mlflow experiments create -n "wine-quality-lr"
```

```bash
MLFLOW_EXPERIMENT_ID=806893229162250730 python3 train.py
``` 

### MLflow Projects

- Create a conda environment

```bash
conda create -n exploration python=3.8
```

- Activate virtual environment

```bash
conda activate exploration
```

- Export the environment to a YAML file

```bash
conda env export --no-builds > conda_env.yaml
```



```yaml
name: My Project
conda_env: conda.yaml
entry_points:
  main:
    parameters:
      data_file: path
      regularization: {type: float, default: 0.1}
    command: "python train.py -r {regularization} {data_file}"
  validate:
    parameters:
      data_file: path
    command: "python validate.py {data_file}"
```

### MLflow Models

```yaml
time_created: 2022-01-05T14:45:55.23
flavors:
  sklearn:
    sklearn_version: 0.19.1
    pickled_model: model.pkl
  python_function:
    loader_module: mlflow.sklearn
    pickled_model: model.pkl
```

---

