import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import spacy
from nltk import ngrams

nltk.download('stopwords')

### Tecnicas de Pré-processamento
# 1. Stopwords
def remove_stopwords(df):

    stop_pt = set(stopwords.words('portuguese'))
    df['excerpt'] = df['excerpt'].apply(lambda x: ' '.join([word.lower() for word in x.split() if word.lower() not in stop_pt]))
    
    return df['excerpt']

# 2. Normalização
#remover pontuacao e caracteres especiais e acentos e lowercase
#remover jargoes ?

def normalizacao(df):
    df['excerpt'] = df['excerpt'].str.replace('[.,;?:!/°\']', '', regex=True)
    df['excerpt'] = df['excerpt'].str.replace('[ã]', 'a', regex=True)
    df['excerpt'] = df['excerpt'].str.replace('[é]', 'e', regex=True)
    df['excerpt'] = df['excerpt'].str.replace('[í]', 'i', regex=True)
    df['excerpt'] = df['excerpt'].str.replace('[ó]', 'o', regex=True)
    df['excerpt'] = df['excerpt'].str.replace('[ã]', 'a', regex=True)
    df['excerpt'] = df['excerpt'].str.replace('[\\d]+', '', regex=True)

    return df['excerpt']

#ref: Investigating the impact of pre‑processing techniques and pre‑trained word embeddings 
#in detecting Arabic health information on social media

# 3. Lematização
def lematizacao(lista):
    list_lemmas = [[token.lemma_ for token in doc] for doc in lista]

    return list_lemmas

# 4. POS
def pos(lista):
    list_pos = [[token.text for token in doc if token.pos_ == 'NOUN'] for doc in lista]
    list_pos = list(filter(lambda x: len(x) > 0, list_pos))

    return list_pos

# 5. Embeddings

# 6. N-Gramas
def n_grams_pp(df):
    word_list = df['excerpt'].tolist()
    ngram_list = [word_tuple[0].split() for word_tuple in ngrams(word_list, 3)]

    return ngram_list

def documentNgrams(df):
    ngrams_all = []
    word_list = df['excerpt'].tolist()

    for document in word_list:
        tokens = document.split()
        if len(tokens) <= 4:
            continue
        else:
            output = list(ngrams(tokens, 4))
        for ngram in output:
            ngrams_all.append(" ".join(ngram))

    return pd.DataFrame({'ngrams': ngrams_all})

##
## outra ref: Repositorio Fabio Colado

