import mlflow
import pandas as pd
import xgboost as xgb


class ScorePredictionModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint

    def load_context(self, context):
        self.model = xgb.Booster()
        self.model.load_model(context.artifacts["model_path"])

    def predict(self, context, model_input):
        return self.model.predict(xgb.DMatrix(model_input))
