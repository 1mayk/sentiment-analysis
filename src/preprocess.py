import nltk
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Downloads necessários (executar na primeira vez)
nltk.download("punkt")
nltk.download("stopwords")


def preprocess_text(text):
    """
    Pré-processa o texto:
      - Converte para minúsculas
      - Remove URLs
      - Remove pontuação
      - Tokeniza o texto (opcional para análises mais robustas)
      - (Opcional) Remove stopwords
    Retorna o texto processado.
    """
    # Converte para minúsculas
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    # Remove pontuações
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenização
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('portuguese'))
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)
