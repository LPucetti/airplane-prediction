# Databricks notebook source
# MAGIC %md
# MAGIC FL_DATE: string -> Divide into year, month, day (converter para dia da semana)
# MAGIC OP_CARRIER: string -> Rename to airline_identifier
# MAGIC ORIGIN: string -> Rename to origin_airport
# MAGIC DEST: string -> Rename to destination_airport
# MAGIC
# MAGIC CRS_DEP_TIME: double -> Rename to scheduled_departure_time
# MAGIC DEP_DELAY: double -> Rename to departure_delay
# MAGIC
# MAGIC TAXI_OUT: double -> Rename to taxi_out_time
# MAGIC TAXI_IN: double -> Rename to taxi_in_time
# MAGIC DISTANCE: double -> Rename to distance
# MAGIC
# MAGIC DELAYED -> o voo atrasou ou não (CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY ou LATE_AIRCRAFT_DELAY tem valor diferente de 0 ou null)
# MAGIC DELAYED_TIME -> quanto tempo atrasou (somar CARRIER_DELAY, WEATHER_DELAY, NAS_DELAY, SECURITY_DELAY e LATE_AIRCRAFT_DELAY. Se for null, 0)
# MAGIC
# MAGIC objetivo:
# MAGIC
# MAGIC CANCELLED: double -> Rename to cancelled and convert to boolean
# MAGIC DIVERTED: double -> Rename to diverted and convert to boolean
# MAGIC ARR_DELAY: double -> Rename to arrival_delay

# COMMAND ----------

# In a Databricks notebook, make sure you have a Spark session:
# Usually Spark session is already available as 'spark'.
# If needed, you can create one as follows:
#from pyspark.sql import SparkSession
#spark = SparkSession.builder.getOrCreate()

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, BooleanType

# 1. Read the Parquet file into a Spark DataFrame
df = spark.read.parquet("dbfs:/FileStore/airplanes/merged_flights.parquet")

# 2. Convert FL_DATE to a date type
df = df.withColumn("FL_DATE", F.to_date(F.col("FL_DATE"), "yyyy-MM-dd"))

# 3. Extract year, month, day_of_week
df = df.withColumn("year", F.year("FL_DATE")) \
       .withColumn("month", F.month("FL_DATE")) \
       .withColumn("day_of_week", F.date_format(F.col("FL_DATE"), "EEEE"))

# 4. Rename columns
df = df.withColumnRenamed("OP_CARRIER", "airline_identifier") \
       .withColumnRenamed("ORIGIN", "origin_airport") \
       .withColumnRenamed("DEST", "destination_airport") \
       .withColumnRenamed("CRS_DEP_TIME", "scheduled_departure_time") \
       .withColumnRenamed("DEP_DELAY", "departure_delay") \
       .withColumnRenamed("TAXI_OUT", "taxi_out_time") \
       .withColumnRenamed("TAXI_IN", "taxi_in_time") \
       .withColumnRenamed("DISTANCE", "distance") \
       .withColumnRenamed("CANCELLED", "cancelled") \
       .withColumnRenamed("DIVERTED", "diverted") \
       .withColumnRenamed("ARR_DELAY", "arrival_delay")

# 5. Convert cancelled and diverted to integer
df = df.withColumn("cancelled", F.col("cancelled").cast(IntegerType())) \
       .withColumn("diverted", F.col("diverted").cast(IntegerType()))

# 6. Fill null values in delay columns with 0
delay_columns = [
    "CARRIER_DELAY", "WEATHER_DELAY", "NAS_DELAY",
    "SECURITY_DELAY", "LATE_AIRCRAFT_DELAY"
]

for col_name in delay_columns:
    df = df.withColumn(col_name, F.when(F.col(col_name).isNull(), 0).otherwise(F.col(col_name)))

# 7. Create delayed column (1 if any delay > 0, else 0)
#    We'll sum them up, and check if the sum is greater than 0.
df = df.withColumn("delayed", 
                   F.when(
                       (F.col("CARRIER_DELAY") > 0) | 
                       (F.col("WEATHER_DELAY") > 0) | 
                       (F.col("NAS_DELAY") > 0) |
                       (F.col("SECURITY_DELAY") > 0) |
                       (F.col("LATE_AIRCRAFT_DELAY") > 0),
                       1
                   ).otherwise(0)
                  )

# 8. Create delayed_time column by summing the delay columns
df = df.withColumn("delayed_time",
                   F.col("CARRIER_DELAY") + 
                   F.col("WEATHER_DELAY") + 
                   F.col("NAS_DELAY") + 
                   F.col("SECURITY_DELAY") + 
                   F.col("LATE_AIRCRAFT_DELAY")
                  )

# 9. Convert numeric columns to numeric types
numeric_columns = [
    "scheduled_departure_time", "departure_delay", "taxi_out_time", 
    "taxi_in_time", "distance", "arrival_delay", "delayed_time", 
    "cancelled", "diverted", "delayed"
]

for col_name in numeric_columns:
    df = df.withColumn(col_name, F.col(col_name).cast(DoubleType()))

# 10. Select final columns to keep
columns_to_keep = [
    "year", "month", "day_of_week",
    "airline_identifier", "origin_airport", "destination_airport",
    "scheduled_departure_time", "departure_delay",
    "taxi_out_time", "taxi_in_time", "distance",
    "delayed", "delayed_time",
    "cancelled", "diverted", "arrival_delay"
]

df_final = df.select(*columns_to_keep)

# 11. Write the cleaned data to Parquet
df_final.write.mode("overwrite").parquet("dbfs:/FileStore/airplanes/cleaned_merged_flights.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC Este notebook realiza o processamento e limpeza de dados de voos armazenados no formato Parquet, usando PySpark no Databricks. O objetivo é preparar os dados para análises e modelagem preditiva.
# MAGIC
# MAGIC O arquivo Parquet consolidado é carregado em um DataFrame do PySpark para posterior manipulação.
# MAGIC
# MAGIC A coluna FL_DATE é convertida para o tipo date.
# MAGIC Novas colunas são extraídas: ano, mês, e dia da semana.
# MAGIC
# MAGIC As colunas cancelled e diverted são convertidas para IntegerType (valores 0 e 1).
# MAGIC Colunas numéricas específicas são convertidas para DoubleType para garantir consistência.
# MAGIC
# MAGIC Valores nulos nas colunas de atrasos são substituídos por 0 para evitar problemas de processamento.
# MAGIC
# MAGIC delayed: Indica se houve algum atraso com base nas colunas de atrasos específicas (1 para atrasado, 0 caso contrário).
# MAGIC delayed_time: Soma dos tempos de atraso para capturar a magnitude total dos atrasos.
# MAGIC
# MAGIC Apenas as colunas relevantes para análise ou modelagem são mantidas no DataFrame final.
# MAGIC
# MAGIC Os dados limpos e transformados são gravados no formato Parquet, permitindo acesso eficiente e integração com futuros pipelines de análise e aprendizado de máquina.

# COMMAND ----------

print(df_final.head())
print(df_final.dtypes)
print(df_final.isnull().sum())

# COMMAND ----------

import pandas as pd

df = pd.read_parquet('/dbfs:/FileStore/airplanes/merged_flights.parquet')

df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format='%Y-%m-%d')
df['year'] = df['FL_DATE'].dt.year
df['month'] = df['FL_DATE'].dt.month
df['day'] = df['FL_DATE'].dt.day
df['day_of_week'] = df['FL_DATE'].dt.day_name()

# 3) Rename columns
df = df.rename(columns={
    'OP_CARRIER': 'airline_identifier',
    'ORIGIN': 'origin_airport',
    'DEST': 'destination_airport',
    'CRS_DEP_TIME': 'scheduled_departure_time',
    'DEP_DELAY': 'departure_delay',
    'TAXI_OUT': 'taxi_out_time',
    'TAXI_IN': 'taxi_in_time',
    'DISTANCE': 'distance',
    'CANCELLED': 'cancelled',
    'DIVERTED': 'diverted',
    'ARR_DELAY': 'arrival_delay'
})

# 4) Cast the `cancelled` column to integer (Change #3)
df['cancelled'] = df['cancelled'].astype(int)
df['diverted'] = df['diverted'].astype(int)

# 5) Handle delay columns
delay_columns = [
    'CARRIER_DELAY', 'WEATHER_DELAY', 'NAS_DELAY',
    'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY'
]
df[delay_columns] = df[delay_columns].fillna(0)

# 6) Create a `delayed` boolean column (True if any delay > 0)
df['delayed'] = df[delay_columns].gt(0).any(axis=1)

# 7) Sum the delay columns to get `delayed_time`
df['delayed_time'] = df[delay_columns].sum(axis=1)

# 8) Convert numeric columns to numeric types
numeric_columns = [
    'scheduled_departure_time', 'departure_delay',
    'taxi_out_time', 'taxi_in_time', 'distance',
    'arrival_delay', 'delayed_time'
]
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# 9) Filter out rows with null `scheduled_departure_time` (Change #1)
df_cleaned = df[df['scheduled_departure_time'].notnull()].copy()

# 10) Replace null values with 0 in the specified columns (Change #2)
columns_to_replace = ["arrival_delay", "taxi_in_time", "taxi_out_time", "departure_delay"]
for column in columns_to_replace:
    df_cleaned[column] = df_cleaned[column].fillna(0)

# 11) Select final columns
columns_to_keep = [
    'year', 'month', 'day_of_week',
    'airline_identifier', 'origin_airport', 'destination_airport',
    'scheduled_departure_time', 'departure_delay',
    'taxi_out_time', 'taxi_in_time', 'distance',
    'delayed', 'delayed_time',
    'cancelled', 'diverted', 'arrival_delay'
]
df_final = df_cleaned[columns_to_keep]

# 12) Write to Parquet
df_final.write.mode("overwrite").parquet("dbfs:/FileStore/airplanes/cleaned_merged_flights_v3.parquet")
