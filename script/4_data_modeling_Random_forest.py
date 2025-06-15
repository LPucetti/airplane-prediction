# Databricks notebook source
# MAGIC %md
# MAGIC # Random Forest Models

# COMMAND ----------

# MAGIC %md
# MAGIC Este notebook implementa um pipeline de Random Forest Classifier para prever as probabilidades de um voo ser cancelado, desviado ou atrasado. 
# MAGIC
# MAGIC O modelo foi desenvolvido com avaliação de métricas como AUC, precisão e F1-Score.

# COMMAND ----------

# DBTITLE 1,Import Libraries
# PySpark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType

# ML
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Python 
import numpy as np
#import pandas as pd


# COMMAND ----------

# DBTITLE 1,Data for ML
# df_forest = spark.read.parquet("dbfs:/FileStore/airplanes/cleaned_merged_flights_2.parquet")
df_forest = spark.read.parquet("dbfs:/FileStore/airplanes/cleaned_merged_flights_v3.parquet")

df_forest = df_forest.withColumn("cancelled", F.col("cancelled").cast(IntegerType()))
df_forest = df_forest.withColumn("diverted", F.col("diverted").cast(IntegerType()))
df_forest = df_forest.withColumn("delayed", F.col("delayed").cast(IntegerType()))

numeric_cols = [
    "scheduled_departure_time", "departure_delay", "taxi_out_time",
    "taxi_in_time", "distance", "arrival_delay", "delayed_time"
]
df_forest = df_forest.fillna(0, subset=numeric_cols)

train_data, test_data = df_forest.randomSplit([0.8,0.2], seed = 42)

# COMMAND ----------

# MAGIC %md
# MAGIC Carregamento dos Dados:
# MAGIC - Os dados foram carregados de um arquivo Parquet (cleaned_merged_flights_2.parquet).
# MAGIC
# MAGIC Divisão do Dataset:  
# MAGIC - O dataset foi dividido em 80% para treinamento e 20% para teste.

# COMMAND ----------

# MAGIC %md
# MAGIC ## RF - Cancelled

# COMMAND ----------

# DBTITLE 1,RF - Cancelled
# Índice para colunas categóricas
categorical_cols = ['day_of_week', 'airline_identifier', 'origin_airport', 'destination_airport']
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_index") for col in categorical_cols]

# OneHotEncoder para colunas categóricas indexadas
encoders = [OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec") for col in categorical_cols]

# Seleção de colunas numéricas
numeric_cols = [
    "scheduled_departure_time", "departure_delay", "taxi_out_time",
    "taxi_in_time", "distance", "arrival_delay", "delayed_time"
]

# Vetorização das features
feature_cols = numeric_cols + [f"{col}_vec" for col in categorical_cols]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

target_col = "cancelled"


# COMMAND ----------

# DBTITLE 1,Create RandomForest
# Modelo RF
rf = RandomForestClassifier(featuresCol="features", labelCol=target_col, numTrees=10, maxDepth=5)


# COMMAND ----------

# DBTITLE 1,Pipeline and Training
# Construir o pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# Treinar o modelo
model = pipeline.fit(train_data)


# COMMAND ----------


predictions = model.transform(test_data)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(predictions)
print(f"Acurácia: {accuracy:.3f}")

"Acurácia: 0.984"


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator_auc = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)
print(f"Área Sob a Curva ROC: {auc:.3f}")

"Área Sob a Curva ROC: 1.000"


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_precision = MulticlassClassificationEvaluator(labelCol="cancelled", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="cancelled", predictionCol="prediction", metricName="weightedRecall")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="cancelled", predictionCol="prediction", metricName="f1")

precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

"""Precision: 0.982
Recall: 0.982
F1-Score: 0.982
"""

# COMMAND ----------

model.write().overwrite().save("dbfs:/FileStore/airplanes/random_forest_model")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RF - Diverted

# COMMAND ----------

target_col = "diverted"

rf = RandomForestClassifier(featuresCol="features", labelCol=target_col, numTrees=10, maxDepth=5)


# COMMAND ----------

# Construir o pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# Treinar o modelo
model_diverted = pipeline.fit(train_data)


# COMMAND ----------


predictions = model_diverted.transform(test_data)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Avaliador de Acurácia
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(predictions)
print(f"Acurácia: {accuracy:.3f}")


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Avaliador de AUC (Área Sob a Curva ROC)
evaluator_auc = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)
print(f"Área Sob a Curva ROC: {auc:.3f}")


# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_precision = MulticlassClassificationEvaluator(labelCol="diverted", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="diverted", predictionCol="prediction", metricName="weightedRecall")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="diverted", predictionCol="prediction", metricName="f1")

precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")

"""Precision: 0.982
Recall: 0.982
F1-Score: 0.982
"""

# COMMAND ----------

model_diverted.write().overwrite().save("dbfs:/FileStore/airplanes/random_forest_model_diverted")

# COMMAND ----------

# MAGIC %md
# MAGIC ## RF - Delayed

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

target_col = "delayed"

rf = RandomForestClassifier(featuresCol="features", labelCol=target_col, numTrees=10, maxDepth=5)


# COMMAND ----------

# Construir o pipeline
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

# Treinar o modelo
model_delayed = pipeline.fit(train_data)


# COMMAND ----------

predictions = model_delayed.transform(test_data)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Avaliador de Acurácia
evaluator_accuracy = MulticlassClassificationEvaluator(labelCol=target_col, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator_accuracy.evaluate(predictions)
print(f"Acurácia: {accuracy:.3f}")

"Acurácia: 0.982"


# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Avaliador de AUC (Área Sob a Curva ROC)
evaluator_auc = BinaryClassificationEvaluator(labelCol=target_col, metricName="areaUnderROC")
auc = evaluator_auc.evaluate(predictions)
print(f"Área Sob a Curva ROC: {auc:.3f}")

"Área Sob a Curva ROC: 1.000"

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator_precision = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="weightedPrecision")
evaluator_recall = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="weightedRecall")
evaluator_f1 = MulticlassClassificationEvaluator(labelCol="delayed", predictionCol="prediction", metricName="f1")

precision = evaluator_precision.evaluate(predictions)
recall = evaluator_recall.evaluate(predictions)
f1 = evaluator_f1.evaluate(predictions)

print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score: {f1:.3f}")


"""Precision: 0.982
Recall: 0.982
F1-Score: 0.982
"""

# COMMAND ----------

model_delayed.write().overwrite().save("dbfs:/FileStore/airplanes/random_forest_model_delayed")

# COMMAND ----------

# MAGIC %md
# MAGIC Pipeline de Features
# MAGIC
# MAGIC Indexação e Codificação One-Hot:
# MAGIC
# MAGIC Colunas categóricas como day_of_week, airline_identifier, origin_airport e destination_airport foram indexadas e codificadas.
# MAGIC
# MAGIC Vetorização:
# MAGIC
# MAGIC Colunas numéricas como scheduled_departure_time, departure_delay, distance, e os vetores codificados foram combinados em uma única coluna chamada features.
# MAGIC
# MAGIC 1. Previsão de Cancelamentos
# MAGIC
# MAGIC Hiperparâmetros do Random Forest:
# MAGIC - numTrees: 10 árvores.
# MAGIC - maxDepth: Profundidade máxima de 5.
# MAGIC
# MAGIC Métricas de Desempenho:
# MAGIC - Acurácia: Alta, indicando uma boa taxa de acertos.
# MAGIC - AUC (Área Sob a Curva ROC): Elevada, indicando boa separação entre voos cancelados e não cancelados.
# MAGIC - F1-Score: Alta, refletindo equilíbrio entre precisão e recall.
# MAGIC
# MAGIC 2. Previsão de Desvios
# MAGIC
# MAGIC Hiperparâmetros do Random Forest:
# MAGIC - numTrees: 10 árvores.
# MAGIC - maxDepth: Profundidade máxima de 5.
# MAGIC
# MAGIC Métricas de Desempenho:
# MAGIC -   Acurácia: Elevada, sugerindo boa previsão de desvios.
# MAGIC -   AUC (Área Sob a Curva ROC): Moderada, indicando espaço para melhorar a separação entre as classes.
# MAGIC -   F1-Score: Excelente, confirmando a eficácia do modelo.
# MAGIC
# MAGIC 3. Previsão de Atrasos
# MAGIC
# MAGIC Hiperparâmetros do Random Forest:
# MAGIC - numTrees: 10 árvores.
# MAGIC - maxDepth: Profundidade máxima de 5.
# MAGIC
# MAGIC Métricas de Desempenho:
# MAGIC - Acurácia: Alta, destacando a capacidade do modelo em identificar atrasos.
# MAGIC - AUC (Área Sob a Curva ROC): Boa, sugerindo separação confiável entre voos atrasados e pontuais.
# MAGIC - F1-Score: Satisfatória, indicando uma boa performance geral.

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Resultados Consolidados
# MAGIC
# MAGIC Cancelamentos:
# MAGIC - Acurácia: Excelente.
# MAGIC - AUC: Boa separação entre classes.
# MAGIC - F1-Score: Alta.
# MAGIC
# MAGIC Desvios:
# MAGIC - Acurácia: Alta.
# MAGIC - AUC: Moderada, com espaço para melhorias.
# MAGIC - F1-Score: Muito boa.
# MAGIC
# MAGIC Atrasos:
# MAGIC - Acurácia: Boa.
# MAGIC - AUC: Consistente, indicando um modelo confiável.
# MAGIC - F1-Score: Boa.
# MAGIC
# MAGIC Esses modelos fornecem uma base sólida para prever eventos importantes na operação de voos, sendo úteis para análises preditivas e suporte à tomada de decisão em tempo real.