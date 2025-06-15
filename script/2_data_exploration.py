# Databricks notebook source
# MAGIC %md
# MAGIC ### Environment Configuration

# COMMAND ----------

RUN_LOCALLY = False #Alterar esta variavel para False para correr no Databricks

if RUN_LOCALLY:
    from pyspark.context import SparkContext
    from pyspark.sql.session import SparkSession
    sc = SparkContext('local')
    spark = SparkSession(sc)
    dataset_dir = "datasetsParquet/"
    print("Running locally.")
else:
    dataset_dir = "dbfs:/FileStore/airplanes/"
    print("Running on Databricks.")

# COMMAND ----------

#To terminate the hommie
if sc is not None:
    sc.stop()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Loading

# COMMAND ----------

converted_parquet_path = dataset_dir + 'merged_flights.parquet'
df = spark.read.parquet(converted_parquet_path)
df = df.sample(fraction=0.08, seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Exploration

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Count null values in each column

# COMMAND ----------

from pyspark.sql.functions import col, count, when, isnan
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

print("Starting Data Exploration...")
print("\nTotal number of rows:", df.count())
print("Total number of columns:", len(df.columns))

null_counts = []

for column in df.columns:
    null_count = df.filter(
        col(column).isNull() | isnan(col(column))
    ).count()
    null_counts.append((column, null_count))

null_df = pd.DataFrame(null_counts, columns=['Column', 'Null Count'])
null_df = null_df[null_df['Null Count'] > 0]
null_df = null_df.sort_values('Null Count', ascending=False)

print("\nColumns with Null Values:")
print("========================")
print(null_df.to_string(index=False))

plt.figure(figsize=(15, 6))
plt.bar(null_df['Column'], null_df['Null Count'])
plt.xticks(rotation=45, ha='right')
plt.title('Null Values by Column')
plt.xlabel('Columns')
plt.ylabel('Number of Null Values')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Histograms for numerical columns

# COMMAND ----------

numeric_columns = [field.name for field in df.schema.fields
                  if field.dataType.typeName() in ('integer', 'double', 'float', 'long')]

for col in numeric_columns:

    pdf = df.select(col).toPandas()

    plt.figure(figsize=(10, 6))
    sns.histplot(data=pdf, x=col, bins=30)
    plt.title(f'Distribution of {col}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.close()

    del pdf

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Pie charts for categorical columns

# COMMAND ----------

categorical_columns = [field.name for field in df.schema.fields
                     if field.dataType.typeName() in ('string', 'boolean')]

TOP_N_COUNT_VALUES = 15

for col in categorical_columns:
    if col == "FL_DATE":
        continue

    value_counts = df.groupBy(col).count().orderBy('count', ascending=False)

    top_n = value_counts.limit(TOP_N_COUNT_VALUES).toPandas()
    total_count = df.count()
    top_n_count = top_n['count'].sum()
    print(f"\n{col}: Top {TOP_N_COUNT_VALUES} values represent {(top_n_count/total_count)*100:.2f}% of the data")

    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_n, x=col, y='count')

    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top 15 Most Frequent Values in {col}')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()
    plt.close()


# COMMAND ----------

# MAGIC %md
# MAGIC ##### Basic Statistics for Numerical Columns

# COMMAND ----------

print(df.describe().toPandas().transpose())

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Correlation matrix for numerical columns

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import col, count, when, isnan, coalesce, lit

numeric_cols = [field.name for field in df.schema.fields
                if field.dataType.typeName() in ('integer', 'double', 'float', 'long')]

# Replace nulls with 0 for numeric columns - May not be the best approach for all columns
df_clean = df.select([coalesce(col(c), lit(0)).alias(c) for c in numeric_cols])

vector_col = "correlation_features"
assembler = VectorAssembler(inputCols=numeric_cols, outputCol=vector_col, handleInvalid="skip")
df_vector = assembler.transform(df_clean).select(vector_col)

correlation_matrix = Correlation.corr(df_vector, vector_col).collect()[0][0]
correlation_matrix_pd = pd.DataFrame(correlation_matrix.toArray(), columns=numeric_cols, index=numeric_cols)

# Create heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(correlation_matrix_pd,
            annot=True,
            cmap='coolwarm',
            fmt=".2f",
            linewidths=.5,
            center=0,
            vmin=-1, vmax=1)

plt.title("Correlation Heatmap of Numeric Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

print("\nStrongest Correlations:")
correlations = []
for i in range(len(numeric_cols)):
    for j in range(i+1, len(numeric_cols)):
        correlations.append((
            numeric_cols[i],
            numeric_cols[j],
            abs(correlation_matrix_pd.iloc[i, j])
        ))

correlations.sort(key=lambda x: x[2], reverse=True)

print("\nTop 10 Strongest Correlations:")
for col1, col2, corr in correlations[:10]:
    print(f"{col1} - {col2}: {corr:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Departure Delays by Carrier

# COMMAND ----------

import pyspark.sql.functions as F

sampled_df = df.select('OP_CARRIER', 'DEP_DELAY')
pdf = sampled_df.toPandas()

plt.figure(figsize=(12,6))
sns.boxplot(x='OP_CARRIER', y='DEP_DELAY', data=pdf)
plt.title("Departure Delays by Carrier")
plt.xlabel("Carrier")
plt.ylabel("Departure Delay (minutes)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

summary_stats = sampled_df.groupBy('OP_CARRIER').agg(
    F.avg('DEP_DELAY').alias('avg_delay'),
    F.expr('percentile_approx(DEP_DELAY, 0.5)').alias('median_delay'),
    F.count('*').alias('flight_count')
).orderBy('avg_delay', ascending=False)

summary_stats.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Distance vs. Air Time scatter plot

# COMMAND ----------

sampled_df = df.select('DISTANCE', 'AIR_TIME')
pdf = sampled_df.toPandas()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='DISTANCE', y='AIR_TIME', data=pdf, alpha=0.5)
plt.title("Distance vs. Air Time")
plt.xlabel("Distance (miles)")
plt.ylabel("Air Time (minutes)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Late Aircraft Delay vs. Arrival Delay scatter plot

# COMMAND ----------

sampled_df = df.select('LATE_AIRCRAFT_DELAY', 'ARR_DELAY')
pdf = sampled_df.toPandas()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='LATE_AIRCRAFT_DELAY', y='ARR_DELAY', data=pdf, alpha=0.5)
plt.title("Late Aircraft Delay vs. Arrival Delay")
plt.xlabel("Late Aircraft Delay (minutes)")
plt.ylabel("Arrival Delay (minutes)")
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Monthly Delays bar plot

# COMMAND ----------

from pyspark.sql.functions import month, avg
from pyspark.sql.types import DateType

df_with_month = df.withColumn('FL_DATE', df['FL_DATE'].cast(DateType()))\
                  .withColumn('month', month('FL_DATE'))

monthly_delays = df_with_month.groupBy('month')\
                             .agg(avg('ARR_DELAY').alias('avg_delay'))\
                             .orderBy('month')

pdf_monthly = monthly_delays.toPandas()

plt.figure(figsize=(10, 6))
plt.bar(pdf_monthly['month'], pdf_monthly['avg_delay'])
plt.title('Average Arrival Delay by Month')
plt.xlabel('Month')
plt.ylabel('Average Delay (minutes)')
plt.xticks(range(1, 13))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nMonthly Delay Statistics:")
monthly_delays.show()