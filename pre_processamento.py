import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
import spacy
from nltk import ngrams


nltk.download('stopwords')
nlp = spacy.load("pt_core_news_lg")

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
def embeddings(model, tokens):
    embeddings = []
    for token in tokens:
        if token in model.wv:
            embeddings.append(model.wv[token])
    return embeddings
#vetores = [modelo_embeddings[palavra] if palavra in modelo_embeddings else [0.0] * dimensao_do_vetor for palavra in tokens]

# Juntar os vetores de volta em um texto (opcional)
# texto_preprocessado = " ".join(tokens)

# Processar o texto pré-processado com spaCy para NER
#doc = nlp.make_doc(" ".join(tokens))

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
        if len(tokens) <= 3:
            continue
        else:
            output = list(ngrams(tokens, 3))
            for ngram in output:
                ngrams_all.append(" ".join(ngram))

    return ngrams_all
#return pd.DataFrame({'ngrams': ngrams_all})

def documentNgrams3(df):
    ngrams_all = []
    total_entity_counts = {}
    mais_entidades_real = 0
    doc_mais_entidades_real = 0
    menos_entidades_real = float('inf')
    doc_menos_entidades_real = 0

    for idx, document in enumerate(df['excerpt'].tolist()):
        doc = nlp(document)

        entities = {}
        entity_counts = {}

        for entidade in doc.ents:
            if entidade.text in entities:
                entities[entidade.text] += 1
            else:
                entities[entidade.text] = 1

            if entidade.label_ in entity_counts:
                entity_counts[entidade.label_] += 1
            else:
                entity_counts[entidade.label_] = 1

        total_entities_count = sum(entity_counts.values())

        if total_entities_count > mais_entidades_real:
            mais_entidades_real = total_entities_count
            doc_mais_entidades_real = idx

        if total_entities_count < menos_entidades_real:
            menos_entidades_real = total_entities_count
            doc_menos_entidades_real = idx

        for ngram in ngrams(document.split(), 4):
            ngrams_all.append(" ".join(ngram))

        # Update total_entity_counts inside the loop
        for key, value in entity_counts.items():
            if key in total_entity_counts:
                total_entity_counts[key] += value
            else:
                total_entity_counts[key] = value

    return pd.DataFrame({
        'ngrams': ngrams_all,
        'total_entity_counts': total_entity_counts,
        'mais_entidades_real': mais_entidades_real,
        'doc_mais_entidades_real': doc_mais_entidades_real,
        'menos_entidades_real': menos_entidades_real,
        'doc_menos_entidades_real': doc_menos_entidades_real
    })


def documentNgrams2(df, n=4):
    ngrams_all = {f'ngram_{i}': [] for i in range(n)}
    documentos = []
    word_list = df['excerpt'].tolist()

    for idx, document in enumerate(word_list):
        tokens = document.split()
        if len(tokens) <= n:
            continue
        else:
            output = list(ngrams(tokens, n))
        for i, ngram in enumerate(output):
            ngrams_all[f'ngram_{i}'].append(" ".join(ngram))
            documentos.append(idx)

    result_df = pd.DataFrame({'documento_original': documentos})
    result_df.update(pd.DataFrame(ngrams_all))

    return result_df
##
## outra ref: Repositorio Fabio Colado

