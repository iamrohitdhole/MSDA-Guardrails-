from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="llm_enrich_drugbank_dag",
    description="Run LLM enrichment from latest DrugBank silver sample to gold layer in MinIO",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,
    start_date=datetime(2025, 12, 7),
    catchup=False,
    tags=["drugbank", "llm", "gold"],
) as dag:

    debug_paths = BashOperator(
        task_id="debug_paths",
        bash_command=(
            "echo 'PWD:' $(pwd) && "
            "echo 'Listing /opt/airflow:' && ls -R /opt/airflow && "
            "echo 'Listing current dir:' && ls"
        ),
    )
