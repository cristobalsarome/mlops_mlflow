import mlflow
from model import ScorePredictionModel

mlflow.set_experiment("score_prediction")

mlflow.start_run()

model_checkpoint = "score_prediction_1"

model = ScorePredictionModel(model_checkpoint)


mlflow.log_param("model_checkpoint", model_checkpoint)

mlflow.pyfunc.log_model(
    artifact_path="score_prediction_model",
    python_model=model,
    registered_model_name="score_prediction_model"
)

mlflow.end_run()