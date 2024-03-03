import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

def load_data(db_uri):
    print(db_uri)
    conn =  create_engine(db_uri)
    df = pd.read_sql("""SELECT 
                     user_id, movie_id, rating,
                        user_occupation, user_active_since,
                        movie_name, movie_release_date,
                        genre_unknown, genre_action, genre_adventure, genre_animation, 
                     genre_children, genre_comedy, genre_crime, genre_documentary, 
                     genre_drama, genre_fantasy, genre_film_noir, genre_horror, genre_musical, 
                     genre_mystery, genre_romance, genre_sci_fi, genre_thriller, 
                     genre_war, genre_western, movie_imdb_url
                      
                      FROM target.scores_users_movies""", conn)
    conn.dispose()
    return df
def prepare_data(df):
    # categorical varables to one hot encoding
    df = pd.get_dummies(df, columns=["user_occupation", "movie_name"], drop_first=True)
    # transform dates to years
    df["user_active_since"] = pd.to_datetime(df["user_active_since"]).dt.year
    df["movie_release_date"] = pd.to_datetime(df["movie_release_date"]).dt.year

    # to_numeric
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

def train(df):
    target = "rating"
    X = df[[col for col in df.columns if col != target]]
    y = df[target]
    # Split data into training and test sets
    X_train, X_test, y_train , y_test= train_test_split(X,y, test_size=0.2)
    
    print( X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    # train an XGBoost regressor
    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    # calculate performance metrics
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2_score_model = r2_score(y_test, y_pred)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2_score_model)

    # log model
    mlflow.xgboost.log_model(model, 
                             "model",
                             registered_model_name="score_prediction_model")




    return model

def main(db_uri):
    mlflow.start_run()

    df = load_data(db_uri)
    df = prepare_data(df)
    model = train(df)



    mlflow.end_run()

if __name__ == "__main__":
    import sys
    print(sys.argv[1])
    main(sys.argv[1])
    """
    db_uri = 'postgresql://airbyte:airbyte@localhost:5432/mlops'
    with create_engine(db_uri) as conn:
        df = pd.read_sql('SELECT user_id, movie_id, rating FROM target.scores_users_movies', conn)
    print(df.head())"""