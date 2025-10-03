from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sqlalchemy import create_engine

def etl():
    cleaned_df = pd.read_csv('/path/to/cleaned_data.csv')
    database_url = 'postgresql://user:password@localhost:5432/healthcare_db'
    engine = create_engine(database_url)
    cleaned_df.to_sql('diabetes_hospital_data', engine, if_exists='replace', index=False)
    print("ETL job completed: Data loaded successfully.")

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
    schedule='@daily',
)

etl_task = PythonOperator(
    task_id='load_data',
    python_callable=etl,
    dag=dag,
)
