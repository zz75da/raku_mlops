# /opt/airflow/dags/train_dag.py
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="train_model_dag",
    default_args=default_args,
    description="Train text+image model, log to MLflow, push metrics to Prometheus",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:
    # Environment variables are defined once to avoid repetition
    env_vars = (
        "export MLFLOW_TRACKING_URI=http://mlflow:5000 && "
        "export MLFLOW_S3_ENDPOINT_URL=http://minio:9000 && "
        "export AWS_ACCESS_KEY_ID=minio && "
        "export AWS_SECRET_ACCESS_KEY=minio123 && "
        "export TRAIN_CSV_X_PATH=/opt/airflow/data/X_train_update.csv && "
        "export TRAIN_CSV_Y_PATH=/opt/airflow/data/Y_train_CVw08PX.csv && "
        "export TEST_CSV_X_PATH=/opt/airflow/data/X_test_update.csv && "
        "export IMAGE_ROOT=/opt/airflow/data && "        
        "export IMAGE_ROOT_TRAIN=/opt/airflow/data/images/image_train && "
        "export IMAGE_ROOT_TEST=/opt/airflow/data/images/image_test && "        
        "export ARTIFACTS_DIR=/opt/airflow/artifacts && "
        "export MLFLOW_EXPERIMENT=rakuten-text-image && "
        "export PROM_PUSHGATEWAY_URL=http://pushgateway:9091 && "
        "export PCA_BATCH_SIZE=1024 && "
    )

    # --- Step 1: Pre-check for data files & directories ---
    check_data = BashOperator(
        task_id="check_training_data",
        bash_command=(
            "echo '=== Checking training data availability ===' && "
            "ls -lh /opt/airflow/data/ || exit 1 && "
            "[ -f /opt/airflow/data/X_train_update.csv ] || (echo 'Missing X_train_update.csv' && exit 1) && "
            "[ -f /opt/airflow/data/Y_train_CVw08PX.csv ] || (echo 'Missing Y_train_CVw08PX.csv' && exit 1) && "
            "[ -f /opt/airflow/data/X_test_update.csv ] || (echo 'Missing X_test_update.csv' && exit 1) && "
            "[ -d /opt/airflow/data/images/ ] || (echo 'Missing image root directory' && exit 1) && "
            "[ -d /opt/airflow/data/images/image_train/ ] || (echo 'Missing image root directory for training' && exit 1) && "
            "[ -d /opt/airflow/data/images/image_test/ ] || (echo 'Missing image root directory for testing' && exit 1) && "
            "echo 'All training files are present ✅'"
        ),
    )

    # --- Step 2: Preprocess and extract features ---
    preprocess_and_extract = BashOperator(
        task_id="preprocess_and_extract_features",
        bash_command=(
            f"{env_vars}"
            "echo '=== Starting data preprocessing and feature extraction ===' && "
            "python3 /opt/airflow/scripts/train.py --step preprocess" # Assuming your script can handle steps
        ),
    )

    # --- Step 3: Train the model ---
    train_model = BashOperator(
        task_id="train_model",
        bash_command=(
            f"{env_vars}"
            "echo '=== Starting model training ===' && "
            "python3 /opt/airflow/scripts/train.py --step train" # Assuming your script can handle steps
        ),
    )

    # --- Step 4: Evaluate and log artifacts ---
    evaluate_and_log = BashOperator(
        task_id="evaluate_and_log",
        bash_command=(
            f"{env_vars}"
            "echo '=== Evaluating model and logging artifacts to MLflow ===' && "
            "python3 /opt/airflow/scripts/train.py --step evaluate" # Assuming your script can handle steps
        ),
    )

    # --- Step 5: Push final metrics to Prometheus ---
    push_metrics = BashOperator(
        task_id="push_final_metrics",
        bash_command=(
            f"{env_vars}"
            "echo '=== Pushing final metrics to Prometheus ===' && "
            "python3 /opt/airflow/scripts/train.py --step push_metrics" # Assuming your script can handle steps
        ),
    )

    # Define the order of the tasks
    check_data >> preprocess_and_extract >> train_model >> evaluate_and_log >> push_metrics