# Dicionário de palavras com sentimentos
sentiment_dict = {
    # Palavras positivas
    "bom": 1,
    "ótimo": 2,
    "excelente": 2,
    "maravilhoso": 2,
    "incrível": 2,
    "adorei": 2,
    "gostei": 1,
    "feliz": 1,
    "contente": 1,
    "satisfeito": 1,
    "melhor": 1,
    "perfeito": 2,
    "legal": 1,
    "amo": 2,
    "fantástico": 2,
    "recomendo": 1,
    "positivo": 1,
    "eficiente": 1,
    "eficaz": 1,
    "aprovado": 1,
    "qualidade": 1,
    "satisfação": 1,
    "sucesso": 1,
    "vitorioso": 1,
    "alegre": 1,
    "bonito": 1,
    "agradável": 1,
    "superior": 1,
    "espetacular": 2,
    # Palavras negativas
    "ruim": -1,
    "péssimo": -2,
    "horrível": -2,
    "terrível": -2,
    "decepcionante": -2,
    "odiei": -2,
    "detestei": -2,
    "triste": -1,
    "chateado": -1,
    "insatisfeito": -1,
    "pior": -1,
    "falha": -1,
    "quebrado": -1,
    "odeio": -2,
    "desastre": -2,
    "não recomendo": -2,
    "negativo": -1,
    "ineficiente": -1,
    "ineficaz": -1,
    "reprovado": -1,
    "problema": -1,
    "reclamação": -1,
    "fracasso": -1,
    "derrotado": -1,
    "angustiado": -1,
    "feio": -1,
    "desagradável": -1,
    "inferior": -1,
    "decepção": -2,
    "complicado": -1,
    "defeito": -1,
    "erro": -1,
    "falhou": -1,
    "insatisfação": -1,
}

# Modificadores que podem inverter ou amplificar o sentimento
modifiers = {
    "não": -1,  # inverte o sentimento
    "nunca": -1,  # inverte o sentimento
    "jamais": -1,  # inverte o sentimento
    "muito": 1.5,  # amplifica o sentimento
    "bastante": 1.5,  # amplifica o sentimento
    "extremamente": 2,  # amplifica fortemente o sentimento
    "pouco": 0.5,  # reduz o sentimento
    "quase": 0.8,  # reduz ligeiramente o sentimento
    "super": 2,  # amplifica fortemente
    "mega": 2,  # amplifica fortemente
}


def analyze_sentiment(processed_text):
    """
    Analisa o sentimento de um texto pré-processado em português.

    Args:
        processed_text (str): Texto já pré-processado (tokenizado e sem stopwords)

    Returns:
        tuple: (pontuação, classificação)
    """
    # Tokeniza novamente para processar palavra por palavra
    tokens = processed_text.split()

    score = 0
    modifier = 1  # modificador padrão

    for token in tokens:
        # Verifica se o token atual é um modificador
        if token in modifiers:
            modifier = modifiers[token]
            continue

        # Verifica se o token está no dicionário de sentimentos
        if token in sentiment_dict:
            # Aplica o modificador ao sentimento
            token_score = sentiment_dict[token] * modifier
            score += token_score
            # Reseta o modificador após uso
            modifier = 1

    # Classifica o sentimento com base na pontuação
    if score > 0.5:
        sentiment = "positivo"
    elif score < -0.5:
        sentiment = "negativo"
    else:
        sentiment = "neutro"

    return score, sentiment
