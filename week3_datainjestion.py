import pandas as pd
from sqlalchemy import create_engine

def load_data_to_db(df, db_url, table_name):
    # Create SQLAlchemy engine
    engine = create_engine(db_url)

    # Write dataframe to SQL table, replace existing data
    df.to_sql(table_name, con=engine, if_exists='replace', index=False)
    print(f'Data loaded into table {table_name} successfully.')

if __name__ == "__main__":
    # Example: Load cleaned CSV data (replace 'cleaned_data.csv' with your file)
    cleaned_df = pd.read_csv('cleaned_data.csv')
    
    # Database connection URL format: 
    # postgresql://username:password@host:port/database
    database_url = 'postgresql://jacob:test123@localhost:5432/healthcare_db?gssencmode=disable'

    load_data_to_db(cleaned_df, database_url, 'diabetes_hospital_data')

