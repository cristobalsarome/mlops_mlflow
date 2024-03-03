import mlflow
import pandas as pd
import xgboost as xgb


class ScorePredictionModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model_checkpoint):
        self.model_checkpoint = model_checkpoint

    def load_context(self):
        self.model = xgb.Booster()
        self.model.load_model(self.model_checkpoint)

    def predict(self, model_input):
        return self.model.predict(model_input)
