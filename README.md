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

### MLflow Projects

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
