from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
from airflow.providers.standard.operators.bash import BashOperator
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine, text
from pathlib import Path

# Base directory and filenames
project_dir = Path("/Users/jacob.joseph/Documents/IITJ/Capstone Project/capstone_project")
csv_path = project_dir / "cleaned_data.csv"
bash_script = project_dir / "week3_datainjestion.py"
log_path = project_dir / "week3_datainjestion.log"
bash_command = f'python {bash_script} > {log_path} 2>&1'

def etl():
    try:
        database_url = 'postgresql://jacob:test123@localhost:5432/healthcare_db'
        engine = create_engine(database_url)
        # Truncate existing data safely before loading
        with engine.connect() as conn:
            conn.execute(text("TRUNCATE TABLE diabetes_hospital_data;"))
        # Read and load data in chunks to avoid OOM errors
        chunk_size = 5000 
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            chunk.to_sql('diabetes_hospital_data', engine, if_exists='append', index=False)
        print("ETL job completed: Data loaded successfully.")
    except Exception as e:
        print(f"ETL job failed: {e}")
        raise    

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 10, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'healthcare_data_etl',
    default_args=default_args,
    description='ETL pipeline for healthcare data loading',
    schedule="0 9 * * *",
    catchup=False
)

etl_task = PythonOperator(
    task_id='load_data',
    python_callable=etl,
    dag=dag,
)

data_ingest_task = BashOperator(
    task_id='run_external_python_script',
    bash_command=bash_command,
    dag=dag,
)

# Set dependency so ingestion happens before ETL
data_ingest_task >> etl_task
