# Databricks notebook source
# MAGIC %md
# MAGIC # Decision Tree Model

# COMMAND ----------

# MAGIC %md
# MAGIC Este notebook implementa um pipeline de machine learning usando Gradient Boosted Trees (GBT) com o objetivo de prever se um voo será cancelado ou desviado. Utilizando o PySpark no Databricks, os modelos foram treinados e avaliados em um subconjunto dos dados de voos.
# MAGIC

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier

# 1. Spark session
spark = SparkSession.builder.appName("Predict Cancelled and Diverted with GBT").getOrCreate()

# 2. Read data
df = spark.read.parquet("dbfs:/FileStore/airplanes/cleaned_merged_flights_v3.parquet")
df = df.sample(fraction=0.08, seed=42)

# 3. Train-Test Split
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC 1. Pré-processamento de Dados
# MAGIC
# MAGIC     Os dados foram carregados a partir de um arquivo Parquet (cleaned_merged_flights_v3.parquet) e uma amostra de 8% foi utilizada para agilizar o treinamento.
# MAGIC     O dataset foi dividido em 80% para treinamento e 20% para teste.
# MAGIC     Um pipeline de pré-processamento foi criado para realizar:
# MAGIC         Indexação e One-Hot Encoding das colunas categóricas:
# MAGIC             airline_identifier, origin_airport, destination_airport, e day_of_week.
# MAGIC         Montagem do vetor de features:
# MAGIC             Colunas como year, month, scheduled_departure_time, atrasos, distância e as variáveis codificadas foram combinadas em uma única coluna chamada features.

# COMMAND ----------

# 4. Feature Engineering
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

# 5. Fit the pipeline on the train set
feature_model = feature_pipeline.fit(train_df)
train_features = feature_model.transform(train_df)
test_features  = feature_model.transform(test_df)

# COMMAND ----------

feature_model.write().overwrite().save("dbfs:/FileStore/airplanes/feature_pipeline_model")

# COMMAND ----------

# MAGIC %md
# MAGIC 2. Modelo 1: Previsão de Cancelamentos
# MAGIC
# MAGIC     Foi criado um modelo GBTClassifier para prever a coluna cancelled (0 ou 1).
# MAGIC     Hiperparâmetros principais:
# MAGIC         Número de iterações (maxIter): 10.
# MAGIC         Semente para reprodução (seed): 123.
# MAGIC     Métricas de Desempenho:
# MAGIC         AUC (Área Sob a Curva ROC): 0.9999.
# MAGIC         Acurácia: 99.96%.
# MAGIC         F1-Score: 0.9996.
# MAGIC         Precisão: 0.9996.
# MAGIC         Recall: 0.9996.
# MAGIC     Conclusão:
# MAGIC         O modelo apresenta desempenho praticamente perfeito para a tarefa de prever cancelamentos, indicando uma separação clara entre os dados de voos cancelados e não cancelados.
# MAGIC
# MAGIC 3. Modelo 2: Previsão de Desvios
# MAGIC
# MAGIC     Foi criado um segundo modelo GBTClassifier para prever a coluna diverted (0 ou 1).
# MAGIC     Hiperparâmetros principais:
# MAGIC         Número de iterações (maxIter): 6.
# MAGIC         Semente para reprodução (seed): 123.
# MAGIC     Métricas de Desempenho:
# MAGIC         AUC (Área Sob a Curva ROC): 0.8079.
# MAGIC         Acurácia: 99.79%.
# MAGIC         F1-Score: 0.9972.
# MAGIC         Precisão: 0.9977.
# MAGIC         Recall: 0.9979.
# MAGIC     Conclusão:
# MAGIC         O modelo é altamente preciso, mas a AUC menor sugere que a tarefa de prever desvios é mais desafiadora, possivelmente devido a uma menor separação nos dados ou desequilíbrio nas classes.

# COMMAND ----------

# 6. Cancelled Prediction (GBTClassifier)
label_indexer_cancelled = StringIndexer(inputCol="cancelled", outputCol="label_cancelled")
gbt_cancelled = GBTClassifier(
    featuresCol="features",
    labelCol="label_cancelled",
    maxIter=10,
    seed=123
)

pipeline_cancelled = Pipeline(stages=[
    label_indexer_cancelled,
    gbt_cancelled
])

model_cancelled = pipeline_cancelled.fit(train_features)
print("Finished training GBT model for Cancelled")

predictions_cancelled = model_cancelled.transform(test_features)

# ========== Evaluations for Cancelled ==========
binary_evaluator = BinaryClassificationEvaluator(
    labelCol="label_cancelled",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc_cancelled = binary_evaluator.evaluate(predictions_cancelled)

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

print("=== Cancelled (GBT) Model ===")
print("AUC (cancelled):", auc_cancelled)
print("Accuracy (cancelled):", acc_cancelled)
print("F1-Score (cancelled):", f1_cancelled)
print("Precision (cancelled):", precision_cancelled)
print("Recall (cancelled):", recall_cancelled)

# Save the GBT model for 'cancelled'
model_cancelled.write().overwrite().save("dbfs:/FileStore/airplanes/model_GBT_cancelled")

"""
Finished training GBT model for Cancelled
=== Cancelled (GBT) Model ===
AUC (cancelled): 0.9999069899599846
Accuracy (cancelled): 0.999589028585345
F1-Score (cancelled): 0.9995892641660367
Precision (cancelled): 0.9995895436151179
Recall (cancelled): 0.999589028585345
"""

# COMMAND ----------

# 7. Diverted Prediction (GBTClassifier)
label_indexer_diverted = StringIndexer(inputCol="diverted", outputCol="label_diverted")
gbt_diverted = GBTClassifier(
    featuresCol="features",
    labelCol="label_diverted",
    maxIter=6,
    seed=123
)

pipeline_diverted = Pipeline(stages=[
    label_indexer_diverted,
    gbt_diverted
])

model_diverted = pipeline_diverted.fit(train_features)
print("Finished training GBT model for Diverted")

predictions_diverted = model_diverted.transform(test_features)

# ========== Evaluations for Diverted ==========
binary_evaluator_div = BinaryClassificationEvaluator(
    labelCol="label_diverted",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
auc_diverted = binary_evaluator_div.evaluate(predictions_diverted)

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

print("=== Diverted (GBT) Model ===")
print("AUC (diverted):", auc_diverted)
print("Accuracy (diverted):", acc_diverted)
print("F1-Score (diverted):", f1_diverted)
print("Precision (diverted):", precision_diverted)
print("Recall (diverted):", recall_diverted)

# Save the GBT model for 'diverted'
model_diverted.write().overwrite().save("dbfs:/FileStore/airplanes/model_GBT_diverted")

"""
Finished training GBT model for Diverted
=== Diverted (GBT) Model ===
AUC (diverted): 0.8078581792895039
Accuracy (diverted): 0.9979238332978173
F1-Score (diverted): 0.9971543281247596
Precision (diverted): 0.9977057917433334
Recall (diverted): 0.9979238332978173
"""

# COMMAND ----------

# MAGIC %md
# MAGIC 4. Resultados Consolidados
# MAGIC
# MAGIC     Ambos os modelos apresentam desempenho robusto, com excelente precisão e recall.
# MAGIC     A tarefa de prever cancelamentos foi resolvida quase perfeitamente, enquanto a previsão de desvios, embora também forte, apresenta mais margem para melhoria.