# Databricks notebook source
# MAGIC %md
# MAGIC # Generate avro schema

# COMMAND ----------

# MAGIC %md
# MAGIC Este projeto combina PySpark, Apache Kafka e Avro para criar um pipeline completo que processa dados de voos em tempo real, realiza previsões de cancelamento e desvio utilizando modelos previamente treinados com Gradient Boosted Trees (GBT), e publica os resultados das previsões de volta no Kafka, em formato Avro, para consumo por outras aplicações.
# MAGIC
# MAGIC Ao Final do código uma descrição completa do  trabalho.

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, LongType, FloatType, DoubleType,
    BooleanType, ArrayType, MapType, TimestampType, DateType, BinaryType, DecimalType
)
import json

spark = SparkSession.builder.getOrCreate()
parquet_file_path = "dbfs:/FileStore/airplanes/cleaned_merged_flights_v3.parquet"

df = spark.read.parquet(parquet_file_path)
df.printSchema()

def spark_type_to_avro(field):

    field_type = field.dataType
    nullable = field.nullable
    field_name = field.name

    if isinstance(field_type, StringType):
        avro_type = "string"
    elif isinstance(field_type, IntegerType):
        avro_type = "int"
    elif isinstance(field_type, LongType):
        avro_type = "long"
    elif isinstance(field_type, FloatType):
        avro_type = "float"
    elif isinstance(field_type, DoubleType):
        avro_type = "double"
    elif isinstance(field_type, BooleanType):
        avro_type = "boolean"
    elif isinstance(field_type, BinaryType):
        avro_type = "bytes"
    elif isinstance(field_type, TimestampType):
        avro_type = {
            "type": "long",
            "logicalType": "timestamp-millis"
        }
    elif isinstance(field_type, DateType):
        avro_type = {
            "type": "int",
            "logicalType": "date"
        }
    elif isinstance(field_type, DecimalType):
        avro_type = {
            "type": "bytes",
            "logicalType": "decimal",
            "precision": field_type.precision,
            "scale": field_type.scale
        }
    elif isinstance(field_type, ArrayType):
        items = spark_type_to_avro(StructField("item", field_type.elementType, True))
        avro_type = {
            "type": "array",
            "items": items["type"] if isinstance(items["type"], dict) else items["type"]
        }
    elif isinstance(field_type, MapType):
        values = spark_type_to_avro(StructField("value", field_type.valueType, True))
        avro_type = {
            "type": "map",
            "values": values["type"] if isinstance(values["type"], dict) else values["type"]
        }
    elif isinstance(field_type, StructType):
        avro_type = {
            "type": "record",
            "name": field_name.capitalize(),
            "fields": [spark_type_to_avro(f) for f in field_type.fields]
        }
    else:
        raise NotImplementedError(f"Type {field_type} is not supported.")

    if nullable:
        return {
            "name": field_name,
            "type": ["null", avro_type]
        }
    else:
        return {
            "name": field_name,
            "type": avro_type
        }

def generate_avro_schema_from_spark(df, record_name="RootRecord"):
    spark_schema = df.schema
    avro_fields = [spark_type_to_avro(field) for field in spark_schema.fields]

    avro_schema = {
        "type": "record",
        "name": record_name,
        "fields": avro_fields
    }

    return avro_schema

avro_schema = generate_avro_schema_from_spark(df)
avro_schema_json = json.dumps(avro_schema, indent=2)
print(avro_schema_json)
avro_schema_path = "dbfs:/FileStore/airplanes/flights_schema.avsc"
dbutils.fs.put(avro_schema_path, avro_schema_json, overwrite=True)
print(f"Avro schema has been saved to {avro_schema_path}")


# COMMAND ----------

from pyspark.sql.types import (StructType, StructField, IntegerType, StringType,
                               DoubleType, LongType)
from pyspark.sql import Row
import random

schema = StructType([
    StructField("year", IntegerType(), True),
    StructField("month", IntegerType(), True),
    StructField("day_of_week", StringType(), True),
    StructField("airline_identifier", StringType(), True),
    StructField("origin_airport", StringType(), True),
    StructField("destination_airport", StringType(), True),
    StructField("scheduled_departure_time", DoubleType(), True),
    StructField("departure_delay", DoubleType(), True),
    StructField("taxi_out_time", DoubleType(), True),
    StructField("taxi_in_time", DoubleType(), True),
    StructField("distance", DoubleType(), True),
    StructField("delayed", LongType(), True),
    StructField("delayed_time", DoubleType(), True),
    StructField("cancelled", LongType(), True),
    StructField("diverted", LongType(), True),
    StructField("arrival_delay", DoubleType(), True)
])

days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
airlines = ["AA", "DL", "UA", "SW", "BA", "LH", "AF", "FR", "EK"]
airports = ["JFK", "LAX", "ORD", "DFW", "MIA", "SFO", "SEA", "DEN", "ATL"]

rows = []
for _ in range(15):
    year = random.randint(2009, 2017)
    month = random.randint(1, 12)
    day_of_week = random.choice(days_of_week)
    airline_identifier = random.choice(airlines)
    origin_airport = random.choice(airports)
    destination_airport = random.choice(airports)
    
    while destination_airport == origin_airport:
        destination_airport = random.choice(airports)
    
    scheduled_departure_time = round(random.uniform(0, 23.99), 2)
    departure_delay = round(random.uniform(0, 180), 2)
    taxi_out_time = round(random.uniform(1, 30), 2)
    taxi_in_time = round(random.uniform(1, 30), 2)
    distance = round(random.uniform(100, 4000), 2)
    
    delayed = random.randint(0, 1)
    delayed_time = round(random.uniform(0, 180), 2) if delayed == 1 else 0.0
    
    cancelled = random.randint(0, 1)
    diverted = random.randint(0, 1)
    
    arrival_delay = round(random.uniform(0, 180), 2)
    
    rows.append(Row(
        year=year,
        month=month,
        day_of_week=day_of_week,
        airline_identifier=airline_identifier,
        origin_airport=origin_airport,
        destination_airport=destination_airport,
        scheduled_departure_time=scheduled_departure_time,
        departure_delay=departure_delay,
        taxi_out_time=taxi_out_time,
        taxi_in_time=taxi_in_time,
        distance=distance,
        delayed=delayed,
        delayed_time=delayed_time,
        cancelled=cancelled,
        diverted=diverted,
        arrival_delay=arrival_delay
    ))

df = spark.createDataFrame(rows, schema=schema)

output_path = "dbfs:/FileStore/airplanes/batch_flights.csv"
df.coalesce(1).write.mode("overwrite").option("header", "true").csv(output_path)

print(f"CSV file has been saved to: {output_path}")


# COMMAND ----------

# MAGIC %md
# MAGIC # 1. Kafka Producer

# COMMAND ----------

from pyspark.sql.functions import struct, monotonically_increasing_id
from pyspark.sql.avro.functions import to_avro
from pyspark.sql import SparkSession
import json
from pyspark.sql.functions import col


USERNAME = "bigdata2612"
PASSWORD = "lVAq6PPmpfYpidWpyWglrQXM7kC67J"
BOOTSTRAP_SERVERS = "ctmka04u3np52g1pdhj0.any.eu-central-1.mpx.prd.cloud.redpanda.com:9092"
TOPIC = "flights"

# Read Avro schema content from DBFS
AVRO_SCHEMA = dbutils.fs.head("dbfs:/FileStore/airplanes/flights_schema.avsc")

spark = SparkSession.builder.getOrCreate()

# Read CSV from DBFS
df = spark.read.csv("dbfs:/FileStore/airplanes/batch_flights.csv", header=True, inferSchema=True)

# Add a flight_id column for the Kafka message key
df = df.withColumn("flight_id", monotonically_increasing_id())

df = df.withColumn("delayed", col("delayed").cast("long")) \
       .withColumn("cancelled", col("cancelled").cast("long")) \
       .withColumn("diverted", col("diverted").cast("long"))
       
# Convert all columns to a single Avro-encoded binary column named "value"
df_avro = df.withColumn(
    "value", 
    to_avro(struct([df[c] for c in df.columns if c != "flight_id"]), AVRO_SCHEMA)
)

# Convert flight_id column to string for the key
df_avro = df_avro.withColumn("key", df_avro["flight_id"].cast("string"))

# Now select only the key and value columns to send to Kafka
df_avro.select("key", "value") \
    .write \
    .format("kafka") \
    .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS) \
    .option(
        "kafka.sasl.jaas.config",
        f"kafkashaded.org.apache.kafka.common.security.scram.ScramLoginModule required username='{USERNAME}' password='{PASSWORD}';"
    ) \
    .option("kafka.ssl.endpoint.identification.algorithm", "https") \
    .option("kafka.security.protocol", "SASL_SSL") \
    .option("kafka.sasl.mechanism", "SCRAM-SHA-512") \
    .option("topic", TOPIC) \
    .option("checkpointLocation", "dbfs:/FileStore/airplanes/checkpoints/producer") \
    .save()


# COMMAND ----------

# MAGIC %md
# MAGIC # 2. Kafka Consumer

# COMMAND ----------

# === Imports and Setup ===
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, struct, monotonically_increasing_id
from pyspark.sql.avro.functions import from_avro as from_avro_func
from pyspark.sql.avro.functions import to_avro as to_avro_func
from pyspark.ml import PipelineModel
from pyspark.ml.classification import GBTClassificationModel

# Kafka/Redpanda configs
USERNAME = "bigdata2612"
PASSWORD = "lVAq6PPmpfYpidWpyWglrQXM7kC67J"
BOOTSTRAP_SERVERS = "ctmka04u3np52g1pdhj0.any.eu-central-1.mpx.prd.cloud.redpanda.com:9092"
TOPIC = "flights"
PREDICTIONS_TOPIC = "flight_predictions"

# Avro schema for input
avro_schema = dbutils.fs.head("dbfs:/FileStore/airplanes/flights_schema.avsc")

spark = SparkSession.builder.getOrCreate()

# COMMAND ----------

# === 1. Read from Kafka as a streaming DataFrame ===
streaming_df = (
    spark.readStream
    .format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
    .option("kafka.sasl.jaas.config", f"kafkashaded.org.apache.kafka.common.security.scram.ScramLoginModule required username='{USERNAME}' password='{PASSWORD}';")
    .option("kafka.ssl.endpoint.identification.algorithm", "https")
    .option("kafka.security.protocol", "SASL_SSL")
    .option("kafka.sasl.mechanism", "SCRAM-SHA-512")
    .option("subscribe", TOPIC)
    .option("startingOffsets", "earliest")
    .load()
)

# COMMAND ----------

# 1. Parse Avro from Kafka
parsed_streaming_df = streaming_df.select(
    from_avro_func(streaming_df.value, avro_schema).alias("data")
).select("data.*")

# 2. Load the feature engineering pipeline
feature_model = PipelineModel.load("dbfs:/FileStore/airplanes/feature_pipeline_model")

# 3. Transform parsed streaming data to include a 'features' column
streaming_df_features = feature_model.transform(parsed_streaming_df)

# 4. Load the final pipelines for cancelled and diverted
pipeline_cancelled_model = PipelineModel.load("dbfs:/FileStore/airplanes/model_MLP_cancelled")
pipeline_diverted_model  = PipelineModel.load("dbfs:/FileStore/airplanes/model_GBT_diverted")

# 5. Apply each pipeline to the DataFrame containing 'features'
predictions_stream_cancelled = pipeline_cancelled_model.transform(streaming_df_features).select(
    "origin_airport",
    "destination_airport",
    col("prediction").alias("cancelled_prediction"),
    col("probability").alias("cancelled_probability")
)

predictions_stream_diverted = pipeline_diverted_model.transform(streaming_df_features).select(
    "origin_airport",
    "destination_airport",
    col("prediction").alias("diverted_prediction"),
    col("probability").alias("diverted_probability")
)

# COMMAND ----------

# === 5. Combine the two model outputs into one DataFrame ===
# We'll join on flight_id, origin_airport, and destination_airport
final_predictions_stream = (
    predictions_stream_cancelled
    .join(
        predictions_stream_diverted,
        on=["origin_airport", "destination_airport"],
        how="inner"
    )
)

# COMMAND ----------

# === 6. Write the combined predictions to the console (for debugging/inspection) ===
query_console = (
    final_predictions_stream
    .writeStream
    .outputMode("append")
    .format("console")
    .option("truncate", False)
    .start()
)

# === 7. Write the combined predictions to Kafka in Avro format ===

# We define a new Avro schema that includes both cancelled and diverted results
predictions_avro_schema = """
{
  "type": "record",
  "name": "FlightPredictions",
  "fields": [
    {"name":"origin_airport", "type":"string"},
    {"name":"destination_airport", "type":"string"},
    {"name":"cancelled_prediction", "type":"double"},
    {"name":"cancelled_probability", "type":{"type":"array","items":"double"}},
    {"name":"diverted_prediction", "type":"double"},
    {"name":"diverted_probability", "type":{"type":"array","items":"double"}}
  ]
}
"""

output_df = final_predictions_stream.select(
    "origin_airport",
    "destination_airport",
    "cancelled_prediction",
    "cancelled_probability",
    "diverted_prediction",
    "diverted_probability"
)

output_avro_df = (
    output_df
    .withColumn(
        "value",
        to_avro_func(struct([col(c) for c in output_df.columns]), predictions_avro_schema)
    )
)

query_kafka = (
    output_avro_df
    .select("value")
    .writeStream
    .format("kafka")
    .option("kafka.bootstrap.servers", BOOTSTRAP_SERVERS)
    .option("kafka.sasl.jaas.config", f"kafkashaded.org.apache.kafka.common.security.scram.ScramLoginModule required username='{USERNAME}' password='{PASSWORD}';")
    .option("kafka.ssl.endpoint.identification.algorithm", "https")
    .option("kafka.security.protocol", "SASL_SSL")
    .option("kafka.sasl.mechanism", "SCRAM-SHA-512")
    .option("topic", PREDICTIONS_TOPIC)
    .option("checkpointLocation", "dbfs:/FileStore/airplanes/checkpoints/consumer")
    .start()
)

# === 8. Wait for both streaming queries to terminate (they won’t, unless stopped) ===
query_console.awaitTermination()
query_kafka.awaitTermination()

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Arquitetura Geral
# MAGIC
# MAGIC Os dados de entrada são armazenados como arquivos CSV e enviados ao Kafka. O processamento em tempo real começa quando o Kafka fornece os dados para o Spark Structured Streaming, onde as mensagens são decodificadas do formato Avro para um DataFrame Spark. O Spark aplica os modelos previamente treinados para prever cancelamento e desvio, e os resultados são combinados em um único DataFrame. Por fim, os resultados são exibidos no console para depuração e enviados de volta ao Kafka.
# MAGIC
# MAGIC 2. Componentes do Pipeline
# MAGIC
# MAGIC A criação do schema Avro é baseada no DataFrame Spark para garantir a compatibilidade de tipos, sendo salvo como um arquivo .avsc no Databricks FileStore. Os dados simulados de voos, gerados aleatoriamente, incluem variáveis como year, month, airline_identifier, distance, cancelled, e diverted. Esses dados são convertidos para Avro e enviados ao Kafka como mensagens no tópico flights. O Spark Structured Streaming lê os dados do Kafka no tópico flights, decodifica as mensagens do formato Avro e as converte para um DataFrame Spark. Os modelos GBT para cancelamento e desvio são então carregados e aplicados aos dados de streaming, produzindo previsões que são publicadas no Kafka.
# MAGIC
# MAGIC 3. Tecnologias Utilizadas
# MAGIC
# MAGIC O pipeline utiliza PySpark para ingestão e processamento de dados em tempo real, além de aplicar os modelos GBT para previsões. O Apache Kafka atua como transporte de mensagens em tempo real entre os componentes do pipeline, enquanto o Avro é usado para serialização eficiente e compatível para mensagens Kafka. O Databricks fornece um ambiente robusto para desenvolvimento, execução e armazenamento de modelos e dados.
# MAGIC
# MAGIC 4. Fluxo Detalhado
# MAGIC
# MAGIC Os dados simulados de voos são gerados aleatoriamente com informações como airline_identifier, distance, cancelled, diverted, entre outros. Esses dados são escritos como mensagens Kafka no tópico flights. O Spark lê os dados do Kafka, decodifica o formato Avro e os prepara para os modelos de machine learning. Dois modelos GBT são aplicados: um para prever cancelamento e outro para prever desvio. As previsões incluem a probabilidade de cada classe e são combinadas em um único DataFrame. Os resultados combinados das previsões são codificados em Avro e publicados no tópico flight_predictions no Kafka.
# MAGIC 5. Exemplos de Resultados
# MAGIC
# MAGIC As previsões de cancelamento incluem a classe prevista (0 ou 1) e a probabilidade associada a cada classe, por exemplo, cancelled_prediction: 1 e cancelled_probability: [0.2, 0.8]. As previsões de desvio seguem o mesmo formato, como diverted_prediction: 0 e diverted_probability: [0.95, 0.05]. Uma mensagem Kafka combinada, enviada ao tópico flight_predictions, inclui todas essas informações, como o flight_id, aeroportos de origem e destino, além das previsões e probabilidades.
# MAGIC
# MAGIC 6. Pontos de Destaque
# MAGIC
# MAGIC O pipeline é escalável, capaz de processar fluxos de dados em larga escala, e integra de forma eficiente Kafka e Spark para ingestão e publicação de mensagens. As predições em tempo real são rápidas e precisas, graças aos modelos GBT treinados. O uso do Avro garante consistência e eficiência no transporte de mensagens. Este projeto oferece uma solução robusta para previsões de voos em tempo real e pode ser expandido para incluir mais variáveis ou modelos.