import pandas as pd
from sqlalchemy import create_engine, text

def load_data_to_db(df, database_url, table_name):
    # Create SQLAlchemy engine
    engine = create_engine(database_url)
    
    with engine.connect() as conn:
        conn.execute(text(f"TRUNCATE TABLE {table_name};"))

    # Write dataframe to SQL table, replace existing data
    df.to_sql(table_name, con=engine, if_exists='append', index=False)
    print(f'Data loaded into table {table_name} successfully.')

if __name__ == "__main__":
    # Example: Load cleaned CSV data (replace 'cleaned_data.csv' with your file)
    cleaned_df = pd.read_csv('cleaned_data.csv')
    
    # Database connection URL format: 
    # postgresql://username:password@host:port/database
    database_url = 'postgresql://jacob:test123@localhost:5432/healthcare_db?gssencmode=disable'

    load_data_to_db(cleaned_df, database_url, 'diabetes_hospital_data')