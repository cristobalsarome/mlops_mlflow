import mlflow
from model import ScorePredictionModel

mlflow.set_experiment("score_prediction_features")

mlflow.start_run()

model_checkpoint = "saved_models/model.xgb"

model = ScorePredictionModel(model_checkpoint)


mlflow.log_param("model_checkpoint", model_checkpoint)

mlflow.pyfunc.log_model(
    artifact_path="score_prediction_model",
    python_model=model,
    registered_model_name="score_prediction_features"
)

mlflow.end_run()