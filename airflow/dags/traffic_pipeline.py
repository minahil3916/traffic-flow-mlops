
from datetime import datetime

from airflow.sdk import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT_DIR = "/mnt/c/Users/HP/Documents/traffic-flow-mlops"
PYTHON = f"{PROJECT_DIR}/.venv-linux/bin/python"


with DAG(
    dag_id="traffic_flow_ml_pipeline",
    description="Automated traffic flow preprocessing and forecasting",
    start_date=datetime(2026, 11, 21),
    schedule=None,
    catchup=False,
    tags=["traffic", "mlops", "forecasting"],
) as dag:

    split_data = BashOperator(
        task_id="split_dataset",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/data/split_data.py",
    )

    normalize_data = BashOperator(
        task_id="normalize_dataset",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/data/normalize_data.py",
    )

    create_sequences = BashOperator(
        task_id="create_sequences",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/data/create_sequences.py",
    )

    train_random_forest = BashOperator(
        task_id="train_random_forest",
        bash_command=f"cd {PROJECT_DIR} && {PYTHON} src/models/random_forest.py",
    )

    split_data >> normalize_data >> create_sequences >> train_random_forest