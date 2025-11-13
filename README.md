Capstone Project: Automated Healthcare ETL & Data Pipeline
Overview
This project builds a robust, automated ETL pipeline for healthcare data using Airflow 2.x, Python, Bash, and PostgreSQL.
It demonstrates enterprise-grade data workflow orchestration and is designed with CI/CD best practices.
Features
•	End-to-end data pipeline:
Structured ingestion, cleaning, and loading of healthcare data.
•	Airflow DAG:
Automates ETL tasks via PythonOperator and BashOperator.
•	Safe database updates:
Handles Postgres table/view dependencies cleanly using TRUNCATE and chunked inserts.
•	Project scripts included:
o	Data cleaning (week2_dataclearning.py)
o	ETL automation (week3_automateETL.py)
o	Data ingestion (week3_datainjestion.py)
o	Analysis & dashboard scripts (week5_analysis.py, week6_dashboard.py)
•	CI/CD ready:
Version-controlled DAGs and workflows, with template for GitHub Actions.
Folder Structure
text
capstone_project/
├── dags/                    # Airflow DAGs
│   └── week3_automateETL.py
├── week2_dataclearning.py   # Cleaning script
├── week3_datainjestion.py   # Ingestion/ETL script
├── week5_analysis.py        # Data analysis
├── week6_dashboard.py       # Dashboard code
├── cleaned_data.csv         # Input data
├── requirements.txt         # Python dependencies
├── README.md                # This file!
└── Project Weekly Updates/  # Progress logs & notes
Getting Started
1. Clone the repository
bash
git clone <your-repo-url>
cd capstone_project
2. Set up dependencies
bash
conda create -n myenv python=3.11
conda activate myenv
pip install -r requirements.txt
3. Start Airflow and PostgreSQL
Make sure PostgreSQL is running and your database (healthcare_db) is accessible.
Initialize and start Airflow:
bash
export AIRFLOW_HOME=~/airflow
airflow db init
airflow standalone
4. Place DAGs in the dags/ directory
Move your DAG script(s) (week3_automateETL.py/others) into dags/.
5. Run or Trigger DAGs
Use Airflow UI (http://localhost:8080) to trigger workflows, check status and logs.
6. CI/CD Reference (GitHub Actions)
Example workflow for validating DAGs on every push (see .github/workflows/):
text
name: Airflow DAG CI
on: [push]
jobs:
  airflow_validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install Airflow
        run: pip install apache-airflow
      - name: Validate DAGs
        run: airflow dags list
Troubleshooting
•	For memory errors, use pandas chunked reads in ETL (chunksize=5000)
•	Logs are stored locally in ~/airflow/logs
•	Airflow UI may show remote log server errors—these can be ignored unless remote logging is enabled.
