#!/usr/bin/env python3
import argparse
import pandas as pd
import nltk
from preprocess import preprocess_text
from sentiment_dict_pt import analyze_sentiment

# Downloads necessários (executar na primeira vez)
nltk.download("punkt")
nltk.download("stopwords")


def analyze_message_pt(message):
    """
    Analisa a mensagem usando o dicionário de português em sentiment_dict_pt.
    Retorna 'positivo', 'negativo' ou 'neutro'.
    """
    processed = preprocess_text(message)
    # Se preprocess_text retornar tupla, pega só o texto processado
    if isinstance(processed, tuple):
        processed_text = processed[0]
    else:
        processed_text = processed

    score, sentiment = analyze_sentiment(processed_text)
    return sentiment


def predict_single_message(message):
    """
    Analisa uma única mensagem e imprime o resultado.
    """
    processed = preprocess_text(message)
    if isinstance(processed, tuple):
        processed_text = processed[0]
    else:
        processed_text = processed

    score, sentiment = analyze_sentiment(processed_text)
    print("Mensagem:", message)
    # print("Texto processado:", processed_text)
    # print("Pontuação:", score)
    print("Sentimento Previsto:", sentiment)


def predict_from_csv(input_file, output_file):
    """
    Lê um CSV com coluna "texto", aplica preprocessamento e análise
    de sentimento, e salva um novo CSV com as colunas:
      - texto_limpo
      - pontuacao
      - sentimento_previsto
    """
    # Lê os dados do CSV
    data = pd.read_csv(input_file)

    # Aplica preprocess_text e extrai string limpa
    processed_series = data["texto"].apply(preprocess_text)
    data["texto_limpo"] = processed_series.apply(
        lambda x: x[0] if isinstance(x, tuple) else x
    )

    # Analisa sentimento em cada texto limpo
    results = data["texto_limpo"].apply(lambda txt: analyze_sentiment(txt))
    data["pontuacao"] = results.apply(lambda tup: tup[0])
    data["sentimento_previsto"] = results.apply(lambda tup: tup[1])

    # Salva os resultados no arquivo CSV de saída
    data.to_csv(output_file, index=False)
    print(f"Predições salvas em: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Análise de Sentimento com dicionário PT"
    )
    parser.add_argument(
        "-m", "--message", type=str, help="Mensagem única para análise de sentimento"
    )
    parser.add_argument(
        "--csv",
        type=str,
        help='Caminho para o arquivo CSV com as mensagens (coluna "texto")',
    )
    parser.add_argument(
        "--out",
        type=str,
        default="../data/processed/predictions.csv",
        help="Arquivo de saída para predições (quando usar --csv)",
    )

    args = parser.parse_args()

    if args.csv:
        predict_from_csv(args.csv, args.out)
    elif args.message:
        predict_single_message(args.message)
    else:
        user_input = input("Digite a mensagem para análise de sentimento: ")
        predict_single_message(user_input)


if __name__ == "__main__":
    main()
