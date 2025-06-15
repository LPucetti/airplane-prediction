# Databricks notebook source
# MAGIC %md
# MAGIC # Data loading

# COMMAND ----------

# MAGIC %md
# MAGIC Devido ao tamanho dos ficheiros para agilidade e eficiencia os mesmos foram convertidos inicialmente de .CSV para .PARQUET via script py de maneira a permitir que o schema fosse identificado de acordo com o que viesse do próprio .csv, dessa maneira as validações dos datasets poderiam ser feitas exclusivamente via Databricks. 

# COMMAND ----------

# DBTITLE 1,Comparativo entre os Schemas de cada Parquet
from pyspark.sql.types import StructType

# Lista de anos e caminhos dos datasets
datasets = {
    "2009": 'dbfs:/FileStore/airplanes/2009.parquet',
    "2010": 'dbfs:/FileStore/airplanes/2010.parquet',
    "2011": 'dbfs:/FileStore/airplanes/2011.parquet',
    "2012": 'dbfs:/FileStore/airplanes/2012.parquet',
    "2013": 'dbfs:/FileStore/airplanes/2013.parquet',
    "2014": 'dbfs:/FileStore/airplanes/2014.parquet',
    "2015": 'dbfs:/FileStore/airplanes/2015.parquet',
    "2016": 'dbfs:/FileStore/airplanes/2016.parquet',
    "2017": 'dbfs:/FileStore/airplanes/2017.parquet',
    "2018": 'dbfs:/FileStore/airplanes/2018.parquet'
}

# Função para obter os tipos das colunas de um schema
def get_column_types(schema: StructType):
    return {field.name: str(field.dataType) for field in schema.fields}

# Criando uma lista para armazenar os dados comparativos
comparative_schema = []

# Iterar pelos datasets e obter o tipo das colunas
for year, path in datasets.items():
    df = spark.read.parquet(path)
    column_types = get_column_types(df.schema)
    column_types["Year"] = year  # Adiciona o ano como uma coluna
    comparative_schema.append(column_types)

# Transformando a lista em um DataFrame
schema_comparison_df = spark.createDataFrame(comparative_schema)

# Organizando as colunas para exibição (colocando "Year" primeiro)
columns_ordered = ["Year"] + [col for col in schema_comparison_df.columns if col != "Year"]
schema_comparison_df = schema_comparison_df.select(*columns_ordered)

# Mostrando a tabela comparativa
schema_comparison_df.show(truncate=False)


# COMMAND ----------

# DBTITLE 1,Esse tem que ver o que esta acontecendo
from pyspark.sql.types import BooleanType, StructField, StructType, LongType, StringType, IntegerType, DateType, DoubleType

schemaPad = StructType([
    StructField ('FL_DATE',StringType(),True),
    StructField ('OP_CARRIER',StringType(),True),
    StructField ('OP_CARRIER_FL_NUM',LongType(),True),
    StructField ('ORIGIN',StringType(),True),
    StructField ('DEST',StringType(),True),
    StructField ('CRS_DEP_TIME',LongType(),True),
    StructField ('DEP_TIME',DoubleType(),True),
    StructField ('DEP_DELAY',DoubleType(),True),
    StructField ('TAXI_OUT',DoubleType(),True),
    StructField ('WHEELS_OFF',DoubleType(),True),
    StructField ('WHEELS_ON',DoubleType(),True),
    StructField ('TAXI_IN',DoubleType(),True),
    StructField ('CRS_ARR_TIME',LongType(),True),
    StructField ('ARR_TIME',DoubleType(),True),
    StructField ('ARR_DELAY',DoubleType(),True),
    StructField ('CANCELLED',DoubleType(),True),
    StructField ('CANCELLATION_CODE',StringType(),True),
    StructField ('DIVERTED',DoubleType(),True),
    StructField ('CRS_ELAPSED_TIME',DoubleType(),True),
    StructField ('ACTUAL_ELAPSED_TIME',DoubleType(),True),
    StructField ('AIR_TIME',DoubleType(),True),
    StructField ('DISTANCE',DoubleType(),True),
    StructField ('CARRIER_DELAY',DoubleType(),True),
    StructField ('WEATHER_DELAY',DoubleType(),True),
    StructField ('NAS_DELAY',DoubleType(),True),
    StructField ('SECURITY_DELAY',DoubleType(),True),
    StructField ('LATE_AIRCRAFT_DELAY',DoubleType(),True),
    StructField ('Unnamed: 27',DoubleType(),True)
])

# COMMAND ----------

# MAGIC %md
# MAGIC Devido aos schemas importados foi possível ver que os ficheiros estavam com formatos que não eram adequados para os datasets, devido ao import automático, logo com o próprio parquet foi feito um schema padrão para todos de modo a ser possível fazer as primeiras análises sobre os datasets.

# COMMAND ----------

# DBTITLE 1,Leitura dos Parquets
df_airplane2009 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2009.parquet')
df_airplane2010 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2010.parquet')
df_airplane2011 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2011.parquet')
df_airplane2012 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2012.parquet')
df_airplane2013 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2013.parquet')
df_airplane2014 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2014.parquet')
df_airplane2015 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2015.parquet')
df_airplane2016 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2016.parquet')
df_airplane2017 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2017.parquet')
df_airplane2018 = spark.read.schema(schemaPad).parquet('dbfs:/FileStore/airplanes/2018.parquet')

ds_dict={
    "2009": df_airplane2009,
    "2010": df_airplane2010,
    "2011": df_airplane2011,
    "2012": df_airplane2012,
    "2013": df_airplane2013,
    "2014": df_airplane2014,
    "2015": df_airplane2015,
    "2016": df_airplane2016,
    "2017": df_airplane2017,
    "2018": df_airplane2018
}

# COMMAND ----------

# DBTITLE 1,Unificação dos Parquets
parquet_path = "dbfs:/FileStore/airplanes/*.parquet"
df = spark.read.schema(schemaPad).parquet(parquet_path)
#df = df.dropDuplicates()  # Remove registros duplicados, se necessário
#df.write.parquet("dbfs:/FileStore/airplanes/unificado.parquet", mode="overwrite")

# COMMAND ----------

# MAGIC %md
# MAGIC O schema esta com algum problema de conversão, tem que ver ainda o que ta acontecendo, mudar talvés o tipo dos atributos ou ver se esta algo mal na conversão

# COMMAND ----------

# MAGIC %md
# MAGIC # Data exploration

# COMMAND ----------

# DBTITLE 1,Relatório Inicial para cada Ficheiro
from pyspark.sql.functions import col, countDistinct, count, mean, min, max, stddev

# Função para explorar um dataset
def explore_dataset(df, year):
    print(f"Explorando o dataset do ano {year}\n")

    #Contagem de registros totais
    print(f"Total de registros: {df.count()}")

    #Valores distintos e nulos para colunas categóricas
    for cat_col in ["FL_DATE", "OP_CARRIER", "ORIGIN", "DEST", "CANCELLATION_CODE"]:
        print(f"\n{cat_col}")
        df.select(countDistinct(cat_col).alias("Valores Distintos")).show()

    #Estatísticas para colunas numéricas
    for num_col in [
        "DEP_DELAY", "WEATHER_DELAY", "CARRIER_DELAY", "ARR_DELAY", "NAS_DELAY", "SECURITY_DELAY",
        "TAXI_OUT", "TAXI_IN", "LATE_AIRCRAFT_DELAY",
        "CRS_ELAPSED_TIME", "ACTUAL_ELAPSED_TIME", "AIR_TIME", "DISTANCE"
    ]:
        print(f"\n{num_col}")
        df.select(
            mean(num_col).alias("Média"),
            min(num_col).alias("Mínimo"),
            max(num_col).alias("Máximo"),
            stddev(num_col).alias("Desvio Padrão")
        ).show()

    #Frequência de cancelamentos e atrasos
    for flag_col in ["CANCELLED", "DIVERTED"]:
        print(f"\nFrequência da Coluna Binária: {flag_col}")
        df.groupBy(flag_col).count().show()

# iteração entre os datasets
for year, df in ds_dict.items():
    explore_dataset(df, year)


# COMMAND ----------

from pyspark.sql.functions import count
import matplotlib.pyplot as plt

for year, df in ds_dict.items():
    # Agrupar por CANCELLED e contar as ocorrências
    cancelled_counts = df.groupBy("CANCELLED").agg(count("*").alias("Count"))
    
    # Converter para Pandas
    cancelled_counts_pd = cancelled_counts.toPandas()
    
    # Criar o gráfico de pizza
    plt.figure(figsize=(6, 6))
    plt.pie(
        cancelled_counts_pd["Count"], 
        labels=cancelled_counts_pd["CANCELLED"].astype(str),  # Convertendo booleano para string
        autopct='%1.1f%%', 
        startangle=90, 
        colors=['lightgreen', 'tomato']
    )
    plt.title(f"Proporção de Voos Cancelados - {year}")
    plt.show()


# COMMAND ----------

from pyspark.sql.functions import count
import matplotlib.pyplot as plt

for year, df in ds_dict.items():
    # Agrupar por diverted e contar as ocorrências
    diverted_counts = df.groupBy("DIVERTED").agg(count("*").alias("Count"))
    
    # Converter para Pandas
    diverted_counts_pd = diverted_counts.toPandas()
    
    # Criar o gráfico de pizza
    plt.figure(figsize=(6, 6))
    plt.pie(
        diverted_counts_pd["Count"], 
        labels=diverted_counts_pd["DIVERTED"].astype(str),  # Convertendo booleano para string
        autopct='%1.1f%%', 
        startangle=90, 
        colors=['lightgreen', 'tomato']
    )
    plt.title(f"Proporção de Voos Cancelados - {year}")
    plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
from pyspark.sql.functions import col, count

code_labels = {
    "A": "Airline/Carrier",
    "B": "Weather",
    "C": "National Air System",
    "D": "Security"
}

for year, df in ds_dict.items():
    # Filtrar valores nulos antes de agrupar
    cancellation_code_counts = (
        df.filter(col("CANCELLATION_CODE").isNotNull())  # Ignorar valores nulos
        .groupBy("CANCELLATION_CODE")
        .agg(count("*").alias("Count"))
    )

    # Mostrar o resultado
    print(f"Contagem dos cancelamentos do ano {year}\n")
    cancellation_code_counts.show()

    # Converter para Pandas
    cancellation_code_counts_pd = cancellation_code_counts.toPandas()

    # Criar o gráfico de barras
    plt.figure(figsize=(10, 6))
    plt.bar(cancellation_code_counts_pd["CANCELLATION_CODE"],  # Valores já filtrados
            cancellation_code_counts_pd["Count"], 
            color='skyblue',
            label= "Cancelamento por código")
    
    plt.legend(
    labels=[f"{code} - {desc}" for code, desc in code_labels.items()],
    loc="upper right",
    title="Motivos de Cancelamento"
    )
    plt.xlabel("Cancellation Code")
    plt.ylabel("Count")
    plt.title(f"Contagem dos motivos de cancelamento do ano {year}")
    plt.xticks(rotation=45)
    plt.legend(loc="upper right")
    plt.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns

numeric_cols = ['DEP_DELAY', 'TAXI_OUT', 'ARR_DELAY', 'CARRIER_DELAY', 'WEATHER_DELAY',
                'NAS_DELAY', 'SECURITY_DELAY', 'LATE_AIRCRAFT_DELAY', 'DISTANCE',
                'AIR_TIME', 'CRS_ELAPSED_TIME', 'ACTUAL_ELAPSED_TIME']

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Conversão para um único ficheiro

# COMMAND ----------

# DBTITLE 1,Adicionar todos os ficheiros a um parquet unificado
parquet_path = "dbfs:/FileStore/airplanes/*.parquet"
df = spark.read.schema(schema).parquet(parquet_path)
#df = df.dropDuplicates()  # Remove registros duplicados, se necessário
#df.write.parquet("dbfs:/FileStore/airplanes/unificado.parquet", mode="overwrite")


# COMMAND ----------

df.count()