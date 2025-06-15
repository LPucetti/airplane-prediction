# Databricks notebook source
# MAGIC %md
# MAGIC # Linear Regression Model

# COMMAND ----------

# MAGIC %md
# MAGIC Este notebook utiliza Regressão Logística com o PySpark para prever a probabilidade de um voo ser cancelado ou desviado. Com um pipeline de pré-processamento robusto, as features foram transformadas e utilizadas para treinar dois modelos distintos, um para cada variável de interesse.
# MAGIC 1. Pré-processamento de Dados
# MAGIC
# MAGIC     Carregamento e Amostragem:
# MAGIC         O dataset foi carregado a partir de um arquivo Parquet (cleaned_merged_flights_v3.parquet) e 8% dos dados foram utilizados para treinamento e teste.
# MAGIC     Divisão do Dataset:
# MAGIC         O conjunto de dados foi dividido em 80% para treino e 20% para teste.
# MAGIC     Pipeline de Features:
# MAGIC         Indexação e Codificação One-Hot:
# MAGIC             Colunas categóricas como airline_identifier, origin_airport, destination_airport, e day_of_week foram transformadas em representações numéricas.
# MAGIC         Montagem de Features:
# MAGIC             Colunas como year, month, scheduled_departure_time, atrasos e distância foram combinadas em uma única coluna features.
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

spark = SparkSession.builder.appName("Predict Cancelled and Diverted").getOrCreate()

# Read data
df = spark.read.parquet("dbfs:/FileStore/airplanes/cleaned_merged_flights_v3.parquet")
df = df.sample(fraction=0.08, seed=42)

# Split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# Indexers
airline_indexer = StringIndexer(inputCol="airline_identifier", outputCol="airline_index", handleInvalid="skip")
origin_indexer = StringIndexer(inputCol="origin_airport", outputCol="origin_index", handleInvalid="skip")
dest_indexer   = StringIndexer(inputCol="destination_airport", outputCol="dest_index", handleInvalid="skip")
dow_indexer    = StringIndexer(inputCol="day_of_week", outputCol="dow_index", handleInvalid="skip")

# Encoders
airline_encoder = OneHotEncoder(inputCols=["airline_index"], outputCols=["airline_vec"], handleInvalid="keep")
origin_encoder  = OneHotEncoder(inputCols=["origin_index"],  outputCols=["origin_vec"],  handleInvalid="keep")
dest_encoder    = OneHotEncoder(inputCols=["dest_index"],    outputCols=["dest_vec"],    handleInvalid="keep")
dow_encoder     = OneHotEncoder(inputCols=["dow_index"],     outputCols=["dow_vec"],     handleInvalid="keep")

# VectorAssembler
assembler = VectorAssembler(
    inputCols=[
        "year", "month", "scheduled_departure_time",
        "departure_delay", "taxi_out_time", "taxi_in_time",
        "distance", "delayed_time",
        "airline_vec", "origin_vec", "dest_vec", "dow_vec"
    ],
    outputCol="features"
)

feature_pipeline = Pipeline(stages=[
    airline_indexer, origin_indexer, dest_indexer, dow_indexer,
    airline_encoder, origin_encoder, dest_encoder, dow_encoder,
    assembler
])

# Fit on train set, transform both train & test
feature_model = feature_pipeline.fit(train_df)
train_features = feature_model.transform(train_df)
test_features  = feature_model.transform(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Modelo 1: Previsão de Cancelamentos
# MAGIC
# MAGIC     Configuração do Modelo:
# MAGIC         A Regressão Logística foi configurada para prever a coluna cancelled (0 ou 1).
# MAGIC         Hiperparâmetros principais:
# MAGIC             Número de Iterações (maxIter): 10.
# MAGIC             Regularização (regParam): 0.3.
# MAGIC             ElasticNet (elasticNetParam): 0.8 (80% L1, 20% L2).
# MAGIC     Métricas de Desempenho:
# MAGIC         Acurácia: 98.39%.
# MAGIC         F1-Score: 0.9759.
# MAGIC         Precisão: 0.9681.
# MAGIC         Recall: 0.9839.
# MAGIC     Conclusão:
# MAGIC         O modelo apresentou um desempenho sólido, indicando que as features selecionadas conseguem capturar bem os padrões relacionados a cancelamentos de voos.
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Logistic Regression for 'cancelled'
label_indexer_cancelled = StringIndexer(inputCol="cancelled", outputCol="label_cancelled")
log_reg_cancelled = LogisticRegression(
    featuresCol="features",
    labelCol="label_cancelled",
    maxIter=10,   # Number of iterations
    regParam=0.3, # Regularization parameter
    elasticNetParam=0.8 # ElasticNet mixing parameter (0 = L2, 1 = L1)
)

pipeline_cancelled = Pipeline(stages=[
    label_indexer_cancelled,
    log_reg_cancelled
])

model_cancelled = pipeline_cancelled.fit(train_features)
print("Finished training Logistic Regression model for Cancelled")

predictions_cancelled = model_cancelled.transform(test_features)

# ========== Evaluations for Cancelled ==========
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label_cancelled",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_cancelled",
    predictionCol="prediction",
    metricName="accuracy"
)
acc_cancelled = multi_evaluator.evaluate(predictions_cancelled)

f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_cancelled",
    predictionCol="prediction",
    metricName="f1"
)
f1_cancelled = f1_evaluator.evaluate(predictions_cancelled)

precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_cancelled",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision_cancelled = precision_evaluator.evaluate(predictions_cancelled)

recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_cancelled",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall_cancelled = recall_evaluator.evaluate(predictions_cancelled)

print("=== Cancelled (Logistic Regression) Model ===")
print("Accuracy (cancelled):", acc_cancelled)
print("F1-Score (cancelled):", f1_cancelled)
print("Precision (cancelled):", precision_cancelled)
print("Recall (cancelled):", recall_cancelled)

# Save the Logistic Regression model for 'cancelled'
model_cancelled.write().overwrite().save("dbfs:/FileStore/airplanes/model_LogReg_cancelled")

"""
Finished training Logistic Regression model for Cancelled
=== Cancelled (Logistic Regression) Model ===
Accuracy (cancelled): 0.9839122449186682
F1-Score (cancelled): 0.9759335960352226
Precision (cancelled): 0.9680833057008934
Recall (cancelled): 0.9839122449186682
"""

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Modelo 2: Previsão de Desvios
# MAGIC
# MAGIC     Configuração do Modelo:
# MAGIC         Um modelo separado de Regressão Logística foi treinado para prever a coluna diverted (0 ou 1).
# MAGIC         Hiperparâmetros principais:
# MAGIC             Número de Iterações (maxIter): 10.
# MAGIC             Regularização (regParam): 0.3.
# MAGIC             ElasticNet (elasticNetParam): 0.8 (80% L1, 20% L2).
# MAGIC     Métricas de Desempenho:
# MAGIC         Acurácia: 99.77%.
# MAGIC         F1-Score: 0.9965.
# MAGIC         Precisão: 0.9953.
# MAGIC         Recall: 0.9977.
# MAGIC     Conclusão:
# MAGIC         O modelo apresentou alta precisão e recall, sugerindo que é eficaz para prever desvios, embora o desafio seja menor devido à frequência mais baixa de desvios.
# MAGIC
# MAGIC

# COMMAND ----------

# Logistic Regression for 'diverted'
label_indexer_diverted = StringIndexer(inputCol="diverted", outputCol="label_diverted")
log_reg_diverted = LogisticRegression(
    featuresCol="features",
    labelCol="label_diverted",
    maxIter=10,   # Number of iterations
    regParam=0.3, # Regularization parameter
    elasticNetParam=0.8 # ElasticNet mixing parameter (0 = L2, 1 = L1)
)

pipeline_diverted = Pipeline(stages=[
    label_indexer_diverted,
    log_reg_diverted
])

model_diverted = pipeline_diverted.fit(train_features)
print("Finished training Logistic Regression model for Diverted")

predictions_diverted = model_diverted.transform(test_features)

# ========== Evaluations for Diverted ==========
binary_evaluator_div = BinaryClassificationEvaluator(
    labelCol="label_diverted",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

multi_evaluator_div = MulticlassClassificationEvaluator(
    labelCol="label_diverted",
    predictionCol="prediction",
    metricName="accuracy"
)
acc_diverted = multi_evaluator_div.evaluate(predictions_diverted)

f1_evaluator_div = MulticlassClassificationEvaluator(
    labelCol="label_diverted",
    predictionCol="prediction",
    metricName="f1"
)
f1_diverted = f1_evaluator_div.evaluate(predictions_diverted)

precision_evaluator_div = MulticlassClassificationEvaluator(
    labelCol="label_diverted",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision_diverted = precision_evaluator_div.evaluate(predictions_diverted)

recall_evaluator_div = MulticlassClassificationEvaluator(
    labelCol="label_diverted",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall_diverted = recall_evaluator_div.evaluate(predictions_diverted)

print("=== Diverted (Logistic Regression) Model ===")
print("Accuracy (diverted):", acc_diverted)
print("F1-Score (diverted):", f1_diverted)
print("Precision (diverted):", precision_diverted)
print("Recall (diverted):", recall_diverted)

# Save the Logistic Regression model for 'diverted'
model_diverted.write().overwrite().save("dbfs:/FileStore/airplanes/model_LogReg_diverted")

"""
Finished training Logistic Regression model for Diverted
=== Diverted (Logistic Regression) Model ===
Accuracy (diverted): 0.9976539113316488
F1-Score (diverted): 0.9964822446465265
Precision (diverted): 0.9953133267953375
Recall (diverted): 0.9976539113316488
"""

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Resultados Consolidados
# MAGIC
# MAGIC Os modelos de Regressão Logística forneceram resultados robustos para ambas as tarefas:
# MAGIC
# MAGIC   Cancelamentos:
# MAGIC       Alta F1-Score (0.9759) e precisão (0.9681).
# MAGIC       Adequado para prever voos com maior probabilidade de cancelamento.
# MAGIC   
# MAGIC   Desvios:
# MAGIC       Excelente F1-Score (0.9965) e recall (0.9977).
# MAGIC       Confirma a capacidade de detectar desvios com alto grau de confiança.