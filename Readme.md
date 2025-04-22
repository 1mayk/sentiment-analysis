# Análise de Sentimento com VADER

Este projeto utiliza o NLTK e o analisador de sentimentos VADER para classificar mensagens de clientes como **positivo**, **negativo** ou **neutro**.  
A abordagem é baseada em regras, dispensando a necessidade de treinamento de modelo, o que permite uma análise rápida e simples.

## Estrutura do Projeto

    sentiment_analysis/
    │
    ├── 📁 data/
    │   ├── 📁 raw/
    │   │   └── client_conversations.csv  # Dados brutos de conversas
    │   └── 📁 processed/
    │       └── predictions.csv             # Resultados das análises em batch
    │
    ├── 📁 src/
    │   ├── preprocess.py  # Pré-processamento de texto (limpeza, normalização)
    │   └── predict.py     # Script principal para análise com VADER
    │
    ├── 📄 requirements.txt  # Dependências necessárias
    ├── 📄 README.md         # Documentação do projeto
    └── 📄 .gitignore        # Arquivos ignorados pelo Git

## Pré-requisitos

- Python 3.8+
- Pip

## Configuração do Ambiente Virtual

Recomenda-se criar um ambiente virtual para gerenciar as dependências do projeto sem interferir com outras instalações Python em sua máquina.

1. **Crie o ambiente virtual:**

   ```bash
   # No Linux/MacOS:
   python3 -m venv .venv

   # No Windows:
   python -m venv .venv
   ```

2. **Ative o ambiente virtual:**

   ```bash
   # No Linux/MacOS:
   source .venv/bin/activate

   # No Windows:
   .venv\Scripts\activate
   ```

3. **Como saber se o ambiente está ativo?**
- Ao ativar, o nome do ambiente aparecerá no início da sua linha de comando, por exemplo:

         (.venv) user@maquina:~/seu-projeto$

- Para verificar com um comando, você pode rodar:

         which python      # Linux/MacOS
         where python      # Windows

4. **Instale as dependências com o ambiente ativado:**

   ```bash
   pip install -r requirements.txt
   ```

## Instalação

1. Clone o repositório:

   ```bash
   git clone git@github.com:1mayk/sentiment-analysis.git
   cd sentiment_analysis
   ```

2. Instale as dependências (preferencialmente dentro de um ambiente virtual, conforme descrito acima):

   ```bash
   pip install -r requirements.txt
   ```

## Utilização

1. **Analisando uma Única Mensagem**

   Você pode analisar rapidamente uma mensagem usando o script `predict.py`:

   - **Modo Interativo:**  
     Basta executar o script sem argumentos e digitar a mensagem quando solicitado.
     
         python src/predict.py

   - **Via Linha de Comando:**  
     Forneça a mensagem diretamente:
     
         python src/predict.py --message "Eu amo a Labware!"

2. **Analisando Mensagens em Batch (Arquivo CSV)**

   Para processar várias mensagens contidas em um arquivo CSV, utilize o argumento `--csv`:

         python src/predict.py --csv ../data/raw/client_conversations.csv --out ../data/processed/predictions.csv

   As predições serão salvas no arquivo especificado (por padrão, `../data/processed/predictions.csv`).