# ✈️ Airplane Streaming Pipeline – Delay & Cancellation Forecast

Este projeto implementa um pipeline completo de dados para previsão de atrasos e cancelamentos de voos, utilizando dados reais de companhias aéreas dos EUA entre 2009 e 2018. O pipeline abrange desde ingestão, preparação e modelagem até uma camada de simulação de streaming de dados.

---

## 🎯 Objetivo

Criar uma solução de análise preditiva e simulação de dados em tempo real, capaz de:

- Prever atrasos ou cancelamentos de voos com base em dados históricos
- Simular um fluxo de ingestão contínuo (streaming) para testes de escalabilidade
- Avaliar diferentes algoritmos de machine learning em cenários realistas

---

## 🧱 Estrutura do Projeto

airplane-streaming-pipeline/
├── scripts/
│   ├── 1_data_loading.py
│   ├── 2_data_exploration.py
│   ├── 3_data_preparation.py
│   ├── 4_model_linear_regression.py
│   ├── 4_model_random_forest.py
│   ├── 4_model_decision_tree.py
│   ├── 4_model_mlp.py
│   ├── 5_data_streaming.py
│   ├── convert.py
│   └── main_pipeline.py
│
├── docs/
│   └── Project-2.pdf
│
├── README.md
├── requirements.txt
└── .gitignore

🔎 Fonte dos Dados

Os dados utilizados neste projeto foram obtidos do Kaggle:

📂 Dataset:
Airline Delay and Cancellation Data (2009–2018) – por Wendy Yuanyu Mu

⚠️ Os arquivos originais não estão incluídos neste repositório devido ao seu tamanho (~10GB). Recomenda-se o download manual via link acima.
⚙️ Tecnologias Utilizadas

´´´
    Python
    Pandas / NumPy – Manipulação de dados
    Scikit-learn – Modelos de regressão, árvores, Random Forest, MLP
    Matplotlib / Seaborn – Visualizações analíticas
    Simulação de Streaming – Script para ingestão contínua simulada
    Jupyter Notebook / Scripts – Modularização por etapas

🔄 Pipeline de Processamento
    Carga de dados (1_data_loading.py)
    Exploração inicial (2_data_exploration.py)
    Pré-processamento (3_data_preparation.py)
    Modelagem Preditiva com diferentes algoritmos
    Simulação de streaming com dados de voo em tempo real (5_data_streaming.py)

📄 Documentação
    Consulte docs/Project-2.pdf para detalhes metodológicos, decisões técnicas e resultados.
