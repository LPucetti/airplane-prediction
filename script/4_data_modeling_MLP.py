# Databricks notebook source
# MAGIC %md
# MAGIC # Multilayer Perceptron Model

# COMMAND ----------

# MAGIC %md
# MAGIC Este notebook utiliza Redes Neurais Artificiais (Multilayer Perceptron) para prever a probabilidade de um voo ser cancelado ou desviado. O pipeline implementado combina engenharia de features, treinamento de modelos, e avaliação de desempenho utilizando PySpark no Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC 1. Pré-processamento de Dados
# MAGIC
# MAGIC     Carregamento e Amostragem:
# MAGIC         O dataset foi carregado de um arquivo Parquet (cleaned_merged_flights_v3.parquet) e uma amostra de 8% foi utilizada para treino e teste.
# MAGIC     Divisão do Dataset:
# MAGIC         Os dados foram divididos em 80% para treino e 20% para teste.
# MAGIC     Pipeline de Features:
# MAGIC         Indexação e Codificação One-Hot:
# MAGIC             Colunas categóricas como airline_identifier, origin_airport, destination_airport, e day_of_week foram transformadas em representações numéricas.
# MAGIC         Montagem do Vetor de Features:
# MAGIC             Colunas como year, month, scheduled_departure_time, atrasos e distância foram combinadas na coluna features.

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

first_row_features = train_features.select("features").first().features
print("Feature vector size:", len(first_row_features))

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Modelo 1: Previsão de Cancelamentos
# MAGIC
# MAGIC Configuração do Modelo:
# MAGIC
# MAGIC Um Multilayer Perceptron Classifier foi configurado para prever a coluna cancelled (0 ou 1).
# MAGIC
# MAGIC Arquitetura da Rede Neural:
# MAGIC
# MAGIC Camadas: [784 (input), 20 (hidden), 10 (hidden), 2 (output)].
# MAGIC
# MAGIC Hiperparâmetros:
# MAGIC - Número de Iterações (maxIter): 3.
# MAGIC - Tamanho do Bloco (blockSize): 64.
# MAGIC
# MAGIC Métricas de Desempenho:
# MAGIC - AUC (Área Sob a Curva ROC): 0.4516.
# MAGIC - Acurácia: 98.39%.
# MAGIC - F1-Score: 0.9759.
# MAGIC - Precisão: 0.9681.
# MAGIC - Recall: 0.9839.
# MAGIC   
# MAGIC Conclusão:
# MAGIC
# MAGIC O modelo apresentou alta precisão e recall, apesar do baixo valor de AUC, sugerindo que ele é eficiente na tarefa de classificação binária, mas pode ser sensível à separação dos dados.

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Index 'cancelled' as the label
label_indexer_cancelled = StringIndexer(inputCol="cancelled", outputCol="label_cancelled")

# MLP for 'cancelled'
mlp_cancelled = MultilayerPerceptronClassifier(
    featuresCol="features",
    labelCol="label_cancelled",
    maxIter=3,
    layers=[784, 20, 10, 2],
    blockSize=64,
    seed=123
)

pipeline_cancelled = Pipeline(stages=[
    label_indexer_cancelled,
    mlp_cancelled
])

model_cancelled = pipeline_cancelled.fit(train_features)
print("Finished model training")
predictions_cancelled = model_cancelled.transform(test_features)

# ========== EVALUATIONS ==========

# 1. AUC (Area Under ROC)
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label_cancelled",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc_cancelled = binary_evaluator.evaluate(predictions_cancelled)

# 2. Accuracy
multi_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_cancelled",
    predictionCol="prediction",
    metricName="accuracy"
)
acc_cancelled = multi_evaluator.evaluate(predictions_cancelled)

# 3. F1-Score
f1_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_cancelled",
    predictionCol="prediction",
    metricName="f1"
)
f1_cancelled = f1_evaluator.evaluate(predictions_cancelled)

# 4. Weighted Precision
precision_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_cancelled",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision_cancelled = precision_evaluator.evaluate(predictions_cancelled)

# 5. Weighted Recall
recall_evaluator = MulticlassClassificationEvaluator(
    labelCol="label_cancelled",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall_cancelled = recall_evaluator.evaluate(predictions_cancelled)

# Print results
print("=== Cancelled Model ===")
print("AUC (cancelled):", auc_cancelled)
print("Accuracy (cancelled):", acc_cancelled)
print("F1-Score (cancelled):", f1_cancelled)
print("Precision (cancelled):", precision_cancelled)
print("Recall (cancelled):", recall_cancelled)

model_cancelled.write().overwrite().save("dbfs:/FileStore/airplanes/model_MLP_cancelled")

"""
Finished model training
=== Cancelled Model ===
AUC (cancelled): 0.45160997940950903
Accuracy (cancelled): 0.9839122449186682
F1-Score (cancelled): 0.9759335960352226
Precision (cancelled): 0.9680833057008934
Recall (cancelled): 0.9839122449186682
"""

# COMMAND ----------

# MAGIC %md
# MAGIC 3. Modelo 2: Previsão de Desvios
# MAGIC
# MAGIC Configuração do Modelo:
# MAGIC
# MAGIC Um Multilayer Perceptron Classifier foi configurado para prever a coluna diverted (0 ou 1).
# MAGIC
# MAGIC Arquitetura da Rede Neural:
# MAGIC
# MAGIC Camadas: [784 (input), 20 (hidden), 10 (hidden), 2 (output)].
# MAGIC
# MAGIC Hiperparâmetros:
# MAGIC - Número de Iterações (maxIter): 3.
# MAGIC - Tamanho do Bloco (blockSize): 64.
# MAGIC
# MAGIC Métricas de Desempenho:
# MAGIC - AUC (Área Sob a Curva ROC): 0.5353.
# MAGIC - Acurácia: 99.77%.
# MAGIC - F1-Score: 0.9965.
# MAGIC - Precisão: 0.9953.
# MAGIC - Recall: 0.9977.
# MAGIC
# MAGIC Conclusão:
# MAGIC
# MAGIC O modelo de desvio apresentou desempenho excelente em métricas de acurácia e F1-Score, embora o AUC indique que há espaço para melhorar a separação entre as classes.

# COMMAND ----------

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

# Index 'diverted' as the label
label_indexer_diverted = StringIndexer(inputCol="diverted", outputCol="label_diverted")

# MLP for 'diverted'
mlp_diverted = MultilayerPerceptronClassifier(
    featuresCol="features",
    labelCol="label_diverted",
    maxIter=3,
    layers=[784, 20, 10, 2],
    blockSize=64,
    seed=123
)

pipeline_diverted = Pipeline(stages=[label_indexer_diverted, mlp_diverted])
model_diverted = pipeline_diverted.fit(train_features)
print("Finished model training")
predictions_diverted = model_diverted.transform(test_features)

# ========== EVALUATIONS ==========

# 1. AUC (Area Under ROC)
binary_evaluator_div = BinaryClassificationEvaluator(
    labelCol="label_diverted",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc_diverted = binary_evaluator_div.evaluate(predictions_diverted)

# 2. Accuracy
multi_evaluator_div = MulticlassClassificationEvaluator(
    labelCol="label_diverted",
    predictionCol="prediction",
    metricName="accuracy"
)
acc_diverted = multi_evaluator_div.evaluate(predictions_diverted)

# 3. F1-Score
f1_evaluator_div = MulticlassClassificationEvaluator(
    labelCol="label_diverted",
    predictionCol="prediction",
    metricName="f1"
)
f1_diverted = f1_evaluator_div.evaluate(predictions_diverted)

# 4. Weighted Precision
precision_evaluator_div = MulticlassClassificationEvaluator(
    labelCol="label_diverted",
    predictionCol="prediction",
    metricName="weightedPrecision"
)
precision_diverted = precision_evaluator_div.evaluate(predictions_diverted)

# 5. Weighted Recall
recall_evaluator_div = MulticlassClassificationEvaluator(
    labelCol="label_diverted",
    predictionCol="prediction",
    metricName="weightedRecall"
)
recall_diverted = recall_evaluator_div.evaluate(predictions_diverted)

print("=== Diverted Model ===")
print("AUC (diverted):", auc_diverted)
print("Accuracy (diverted):", acc_diverted)
print("F1-Score (diverted):", f1_diverted)
print("Precision (diverted):", precision_diverted)
print("Recall (diverted):", recall_diverted)

model_diverted.write().overwrite().save("dbfs:/FileStore/airplanes/model_MLP_diverted")

"""
Finished model training
=== Diverted Model ===
AUC (diverted): 0.5352573048254481
Accuracy (diverted): 0.9976539113316488
F1-Score (diverted): 0.9964822446465265
Precision (diverted): 0.9953133267953375
Recall (diverted): 0.9976539113316488
"""