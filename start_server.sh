conda activate mlops
set -o allexport && source ../environments/local && set +o allexport
mlflow server --backend-store-uri postgresql://$POSTGRES_USER:$POSTGRES_PASSWORD@$POSTGRES_HOST/$MLFLOW_POSTGRES_DB --default-artifact-root $MLFLOW_ARTIFACTS_PATH -h 0.0.0.0 -p 8002
