import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os

for i in range(2009, 2019):
    # Caminho para o arquivo CSV
    csv_path = r"D:/Users/leono/Documents/2024/Documentos/MestradoCienciadosDados/A2_S1/BigData/Projeto/CSV/" + str(i) + ".csv"
    # Leitura do CSV
    df = pd.read_csv(csv_path)

    # Conversão para tabela Parquet
    table = pa.Table.from_pandas(df)

    # Caminho para salvar o Parquet
    parquet_path = r"D:/Users/leono/Documents/2024/Documentos/MestradoCienciadosDados/A2_S1/BigData/Projeto/PARQUET"
    parquet_file = os.path.join(parquet_path, f"{i}.parquet")

    # Escrita do arquivo Parquet
    pq.write_table(table, parquet_file)
