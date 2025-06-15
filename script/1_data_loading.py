# Databricks notebook source
# MAGIC %md
# MAGIC # Data Loading

# COMMAND ----------

# MAGIC %md
# MAGIC Este notebook foi desenvolvido para realizar a preparação dos dados de voos utilizando o PySpark no Databricks. Ele abrange as etapas de importação, limpeza, transformação e preparação de arquivos CSV, seguido pela conversão para o formato Parquet para otimização de leitura e processamento.

# COMMAND ----------

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
spark = SparkSession.builder.appName("Data loading").getOrCreate()

import pandas as pd
import os

def merge_csv_files(directory_path: str, output_path: str, chunk_size: int = 5000) -> None:

    dtype_mapping = {
        'FL_DATE': str,
        'OP_CARRIER': str,
        'OP_CARRIER_FL_NUM': 'float64',
        'ORIGIN': str,
        'DEST': str,
        'CRS_DEP_TIME': 'float64',
        'DEP_TIME': 'float64',
        'DEP_DELAY': 'float64',
        'TAXI_OUT': 'float64',
        'WHEELS_OFF': 'float64',
        'WHEELS_ON': 'float64',
        'TAXI_IN': 'float64',
        'CRS_ARR_TIME': 'float64',
        'ARR_TIME': 'float64',
        'ARR_DELAY': 'float64',
        'CANCELLED': 'float64',
        'CANCELLATION_CODE': str,
        'DIVERTED': 'float64',
        'CRS_ELAPSED_TIME': 'float64',
        'ACTUAL_ELAPSED_TIME': 'float64',
        'AIR_TIME': 'float64',
        'DISTANCE': 'float64',
        'CARRIER_DELAY': 'float64',
        'WEATHER_DELAY': 'float64',
        'NAS_DELAY': 'float64',
        'SECURITY_DELAY': 'float64',
        'LATE_AIRCRAFT_DELAY': 'float64'
    }

    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    if not csv_files:
        raise ValueError(f"No CSV files found in {directory_path}")

    first_file = True
    total_rows_processed = 0

    print("Processing CSV files...")

    for file in csv_files:
        file_path = os.path.join(directory_path, file)

        try:
            total_rows = sum(1 for _ in open(file_path)) - 1  # subtract header row

            chunks = pd.read_csv(
                file_path,
                dtype=dtype_mapping,
                na_values=['', 'NA', 'null'],
                keep_default_na=True,
                chunksize=chunk_size
            )

            for chunk in chunks:
                if 'Unnamed: 27' in chunk.columns:
                    chunk = chunk.drop('Unnamed: 27', axis=1)

                string_columns = ['FL_DATE', 'OP_CARRIER', 'ORIGIN', 'DEST', 'CANCELLATION_CODE']
                for col in string_columns:
                    if col in chunk.columns:
                        chunk[col] = chunk[col].astype(str).replace('nan', '')

                for col, dtype in dtype_mapping.items():
                    if col in chunk.columns:
                        try:
                            chunk[col] = chunk[col].astype(dtype)
                        except Exception as e:
                            print(f"Warning: Could not convert column {col} to {dtype} in {file}: {str(e)}")

                chunk.to_csv(
                    output_path,
                    mode='w' if first_file and total_rows_processed == 0 else 'a',
                    header=first_file and total_rows_processed == 0,
                    index=False
                )

                total_rows_processed += len(chunk)
                first_file = False

        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
            continue

    if total_rows_processed == 0:
        raise ValueError("No valid data was processed")

    print(f"\nProcessing complete! Total rows processed: {total_rows_processed}")
    print(f"Merged data saved to: {output_path}")

# COMMAND ----------

merge_csv_files(directory_path='dbfs:/FileStore/airplanes/datasetsCSV',output_path='dbfs:/FileStore/airplanes/datasetsCSV/merged_flights_data.csv')

# COMMAND ----------

df = pd.read_csv('dbfs:/FileStore/airplanes/datasetsCSV/merged_flights_data.csv')
df.to_parquet('dbfs:/FileStore/airplanes/merged_flights.parquet')