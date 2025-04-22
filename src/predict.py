import argparse
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from src.preprocess import preprocess_text

# Download do léxico VADER
nltk.download("vader_lexicon")


def analyze_message_vader(message):
    """
    Analisa a mensagem usando o SentimentIntensityAnalyzer (VADER).
    Retorna 'positivo', 'negativo' ou 'neutro' com base na pontuação compound.
    """
    sia = SentimentIntensityAnalyzer()
    scores = sia.polarity_scores(message)
    compound = scores["compound"]

    # Define os limites para classificar a mensagem
    if compound >= 0.05:
        return "positivo"
    elif compound <= -0.05:
        return "negativo"
    else:
        return "neutro"


def predict_single_message(message):
    # Pré-processa a mensagem
    message_clean = preprocess_text(message)
    sentiment = analyze_message_vader(message_clean)
    print("Mensagem:", message)
    print("Sentimento Previsto:", sentiment)


def predict_from_csv(input_file, output_file):
    # Lê os dados do CSV; espera que exista uma coluna "texto"
    data = pd.read_csv(input_file)

    # Pré-processar os textos
    data["texto_limpo"] = data["texto"].apply(preprocess_text)

    # Aplica o VADER para cada mensagem
    data["sentimento_previsto"] = data["texto_limpo"].apply(analyze_message_vader)

    # Salva os resultados no arquivo CSV de saída
    data.to_csv(output_file, index=False)
    print(f"Predições salvas em: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Análise de Sentimento com VADER")

    # Parâmetro para analisar uma mensagem
    parser.add_argument("-m", "--message", type=str, help="Mensagem para análise")

    # Parâmetros para análise em CSV
    parser.add_argument(
        "--csv",
        type=str,
        help='Caminho para o arquivo CSV com as mensagens (deve conter a coluna "texto")',
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../data/processed/predictions.csv",
        help="Arquivo de saída para predições",
    )

    args = parser.parse_args()

    # Se o argumento CSV for fornecido, processa o arquivo
    if args.csv:
        predict_from_csv(args.csv, args.out)
    # Se uma mensagem individual for fornecida, processa a mensagem
    elif args.message:
        predict_single_message(args.message)
    # Caso nenhum argumento seja fornecido, solicita a mensagem interativamente
    else:
        user_input = input("Digite a mensagem para análise de sentimento: ")
        predict_single_message(user_input)


if __name__ == "__main__":
    main()
