from mlflow import log_metric, log_param, log_artifact

if __name__ == "__main__":
    log_param("threshold", 0.5)
    log_param("verbosity", "INFO")
    
    log_metric("accuracy", 0.9342)
    log_metric("RMSE", 2)
    
    log_artifact("hyperparameters.json")