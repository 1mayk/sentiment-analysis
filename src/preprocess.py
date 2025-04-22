import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentiment_dict_pt import analyze_sentiment

# Downloads necessários (executar na primeira vez)
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_text(text):
    """
    Pré-processa o texto:
      - Converte para minúsculas
      - Remove URLs
      - Remove pontuação
      - Tokeniza o texto
      - Remove stopwords
    Em seguida, analisa o sentimento usando o dicionário em sentiment_dict_pt.
    Retorna tupla: (processed_text, score, sentiment_label).
    """
    # Converte para minúsculas
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove pontuações
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenização em português
    tokens = word_tokenize(text, language="portuguese")

    # Remove stopwords
    stop_words = set(stopwords.words("portuguese"))
    tokens = [word for word in tokens if word not in stop_words]

    # Junta de volta em string
    processed = " ".join(tokens)

    # Chama o seu analisador de sentimento
    score, sentiment = analyze_sentiment(processed)

    return processed, score, sentiment
