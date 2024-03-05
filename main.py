
import pandas as pd
import xgboost as xgb
from scipy.stats import uniform, randint
from sklearn.model_selection import RandomizedSearchCV
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import os
from sklearn.metrics import mean_squared_error
from mlflow.exceptions import RestException

mlflow.set_tracking_uri(uri="http://localhost:8002")
# Create a new MLflow Experiment
mlflow.set_experiment("Score Prediction Experiment")

def load_vars_from_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith('#'):  # Ignore empty lines and comments
                var, value = line.split('=', 1)
                os.environ[var.strip()] = value.strip()
    
def print_model_version_info(mv):
    print(f"Name: {mv.name}")
    print(f"Version: {mv.version}")
    print(f"Source: {mv.source}")

def load_data():
    load_vars_from_file("/home/cristobal/environments/local")
    db_user = os.environ.get("AIRBYTE_POSTGRES_USER")
    db_password = os.environ.get("AIRBYTE_POSTGRES_PASS")
    database = os.environ.get("AIRBYTE_POSTGRES_DB")
    db_uri = f"postgresql://{db_user}:{db_password}@localhost:5432/{database}"


    conn = create_engine(db_uri)

    df = pd.read_sql(
        """SELECT 
                     user_id, movie_id, rating,
                        user_occupation, user_active_since,
                        movie_name, movie_release_date,
                        genre_unknown, genre_action, genre_adventure, genre_animation, 
                     genre_children, genre_comedy, genre_crime, genre_documentary, 
                     genre_drama, genre_fantasy, genre_film_noir, genre_horror, genre_musical, 
                     genre_mystery, genre_romance, genre_sci_fi, genre_thriller, 
                     genre_war, genre_western, movie_imdb_url
                      
                      FROM target.scores_users_movies""",
        conn,
    )
    conn.dispose()
    return df


def prepare_training_data(df):
    # categorical varables to one hot encoding
    df = pd.get_dummies(df, columns=["user_occupation", "movie_name"], drop_first=True)
    # transform dates to years
    df["user_active_since"] = pd.to_datetime(df["user_active_since"]).dt.year
    df["movie_release_date"] = pd.to_datetime(df["movie_release_date"]).dt.year

    # to_numeric
    df = df.apply(pd.to_numeric, errors="coerce")

    target = "rating"
    X = df[[col for col in df.columns if col != target]]
    y = df[target]
    # Split data into training and test sets
    X_train, X_test, y_train , y_test= train_test_split(X,y, test_size=0.2)

    return X_train, y_train, X_test, y_test




def train_xgboost_with_mlflow(X_train, y_train, X_test, y_test):
    # Define parameter grid for RandomizedSearchCV
    param_grid = {
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 10),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'gamma': uniform(0, 0.5)
    }

    # Create XGBoost regressor
    xgb_reg = xgb.XGBRegressor()

    # Create RandomizedSearchCV object
    random_search = RandomizedSearchCV(estimator=xgb_reg, param_distributions=param_grid, n_iter=1,
                                       scoring='neg_mean_squared_error', cv=5, random_state=42, n_jobs=-1)

    # Train the model
    mlflow.xgboost.autolog()
    with mlflow.start_run() as run:

        # tag runName
        mlflow.set_tag("runName", "score_prediction_model")

        random_search.fit(X_train, y_train)

        # Log parameters
        mlflow.log_params(random_search.best_params_)

        # Log metrics
        mlflow.log_metric("mean_squared_error", -random_search.best_score_)

        # Log model
        mlflow.xgboost.log_model(random_search.best_estimator_, "score_prediction_model")

        # log feature importance
        feature_importance = random_search.best_estimator_.feature_importances_
        feature_importance = pd.DataFrame({'feature': X_train.columns, 'importance': feature_importance})
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')

        # Log the best model
        mlflow.set_tag("model", "XGBoost")
        mlflow.set_tag("runName", "score_prediction_model")

        # get test metrics
        y_pred = random_search.best_estimator_.predict(X_test)
        test_mse = mean_squared_error(y_test, y_pred)
        mlflow.log_metric("test_mean_squared_error", test_mse)

        from mlflow.tracking import MlflowClient
        from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
        # Create source model version
        client = MlflowClient()
        src_name = "Score Prediction Model"

        registered_models = client.search_registered_models()
        model_list = [m.name for m in registered_models]
        if src_name in model_list:
            print(f"Model {src_name} already exists. Updating it...")
        else:
            client.create_registered_model(src_name)
        src_uri = f"runs:/{run.info.run_id}/xgboost-model"
        mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
        print_model_version_info(mv_src)
        print("--")

    return random_search.best_estimator_



def check_and_update_model():
        
    # # Create an MLflow client
    # try:
    #     client = mlflow.tracking.MlflowClient()
    #     src_name = "Score Prediction Model"
    #     model = client.get_registered_model(src_name)
    # except RestException as e:
    #     print("Model not found")
    #     return

    
    retrain_model = False
    if False:
        # Get the latest model version
        latest_model_version = ['run_id'][0]    
        # Get the creation time of the latest model version
        creation_time = mlflow.get_run(run_id=latest_model_version).info.start_time
        #    Calculate the age of the model
        model_age = datetime.now() - creation_time

        if model_age > timedelta(days=30):
            print("Model is older than 1 month")
            retrain_model = True
    else:
        print("No model found")
        retrain_model = True


    # If the model is older than 1 month, retrain it
    if retrain_model:
        print("Retraining model...")
        df = load_data()
        X_train, y_train, X_test, y_test = prepare_training_data(df)
        train_xgboost_with_mlflow(X_train, y_train, X_test, y_test)
    else:
        print("Model is up to date.")

def tiny_example():
    df = load_data()
    X_train, y_train, X_test, y_test = prepare_training_data(df)
    train_xgboost_with_mlflow(X_train, y_train, X_test, y_test)

def main():
    check_and_update_model()

if __name__ == "__main__":
    main()


