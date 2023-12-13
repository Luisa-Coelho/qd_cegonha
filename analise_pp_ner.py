### Classificacao por meio de Reconhecimento de Entidades Nomeadas

import pre_processamento
import zipfile
import pandas as pd
import numpy as np
import spacy

PATH = (".")

with zipfile.ZipFile("./raw_data/saude.csv.zip", 'r') as zip_ref:
    with zip_ref.open('saude.csv') as arquivo_csv:
        df = pd.read_csv(arquivo_csv)

nlp = spacy.load("pt_core_news_lg")

new_df = df[['excerpt', 'source_territory_id', 'source_state_code', 'source_territory_name', 'source_date']]
new_df['ano'] = new_df['source_date'].str.extract(r'(\d{4})')
new_df.loc[:, 'ano'] = pd.to_numeric(new_df['ano'], errors='coerce')
new_df = new_df[(new_df['ano'] >= 2011) & (new_df['ano'] <= 2021)]

#print(new_df.head(8))
def nlp_preprocessamento(df):
    #doc_full = [nlp(excerpt) for excerpt in df]
    doc1 = [nlp(excerpt) for excerpt in df[1:1000]]
    return doc1

##### APLICANDO DIEFRENTES PRÉ-PROCESSAMENTOS NA AMOSTRA ###
list_df_stopwords = nlp_preprocessamento(pre_processamento.remove_stopwords(new_df))
list_df_normalizacao = nlp_preprocessamento(pre_processamento.normalizacao(new_df))
list_df_ngrams = nlp_preprocessamento(pre_processamento.documentNgrams(new_df))
list_df_lemmas = pre_processamento.lematizacao(nlp_preprocessamento(new_df['excerpt'].tolist()))
list_df_pos = pre_processamento.pos(nlp_preprocessamento(new_df['excerpt'].tolist()))
list_df_sem_processamento = nlp_preprocessamento(new_df['excerpt'])

print(f'stopwords {len(list_df_stopwords)}')
print(f'normalizacao {len(list_df_normalizacao)}')
print(f'ngram {len(list_df_ngrams)}')
print(f'lematizacao {len(list_df_lemmas)}')
print(f'pos {len(list_df_pos)}')
print(len(list_df_sem_processamento))
print(len(list(new_df['excerpt'])))

print(f'Tokens em stopwords: {sum(len(doc) for doc in list_df_stopwords)}')
print(f'Tokens em normalizacao: {sum(len(doc) for doc in list_df_normalizacao)}')
print(f'Tokens em n-grama: {sum(len(doc) for doc in list_df_ngrams)}')
print(f'Tokens em lemmas: {sum(len(doc) for doc in list_df_lemmas)}')
print(f'Tokens em POS: {sum(len(doc) for doc in list_df_pos)}')
print(f'Tokens sem pre-processamento: {sum(len(doc) for doc in list_df_sem_processamento)}')

def media_entidades_lemmas(lista_lemmas):
    media_entidades_lemmas = []
    for doc in list_df_lemmas:
        doc_lemmas = nlp(" ".join(doc))
        print([ent.text for ent in doc_lemmas.ents])
        media_entidades_lemmas.append(len(doc_lemmas.ents))
    
    return np.mean(media_entidades_lemmas)

media_entidades_stopwords = np.mean([len(w.ents) for w in list_df_stopwords])
media_entidades_normalizacao = np.mean([len(w.ents) for w in list_df_normalizacao])
media_entidades_ngram = np.mean([len(w.ents) for w in list_df_ngrams])
media_entidades_lematizacao = media_entidades_lemmas(list_df_lemmas)
media_entidades_pos = np.mean([len(w.ents) for w in list_df_pos])

def resultado_media_entidades(**media_entidades):
    print(f'Media entidades Stopwords {media_entidades["stopwords"]}\n')
    print(f'Media entidades Normalizacao {media_entidades["normalizacao"]}\n')
    print(f'Media entidades N-Grama {media_entidades["ngram"]}\n')
    print(f'Media entidades Lematização {media_entidades["lematizacao"]}\n')
    print(f'Media entidades POS {media_entidades["pos"]}\n')
    

def count_docs(lista):
    entidades = {}
    total_entity_counts = {}
    mais_entidades_doc = 0
    menos_entidades_doc = 100
    excerpt = 0
    count = 0

    for doc in nlp_preprocessamento(lista):
        entity_counts = {}
        count = 0

        for entidade in doc.ents:
            count += 1

            if entidade.text in entidades:
                entidades[entidade.text] += 1
            else:
                entidades[entidade.text] = 1

            if entidade.label_ in entity_counts:
                entity_counts[entidade.label_] += 1
            
            else:
                entity_counts[entidade.label_] = 1

        if count > mais_entidades_doc:
            mais_entidades_doc = count

        if count < menos_entidades_doc:
            menos_entidades_doc = count

    for entity_type, count in entity_counts.items():
        if entity_type in total_entity_counts:
            total_entity_counts[entity_type] += count
        else:
            total_entity_counts[entity_type] = count

    print(f"Excerto com maior numero de entidades: {mais_entidades_doc}: {lista[mais_entidades_doc]}\n")
    print(f"Excerto com menor numero de entidades: {menos_entidades_doc}: {lista[menos_entidades_doc]}\n")

    return total_entity_counts, entidades

def frequencia_entidades(total_entity_counts, entidades):
    print("Contagem entidades:\n")
    for entity_type, total_count in total_entity_counts.items():
        print(f"{entity_type}: {total_count}\n")

    print('Entidade que mais aparece entre os documentos - MISC\n')
    for i in sorted(entidades, key = entidades.get, reverse=False):
        print(i, entidades[i])

    for i in sorted(entidades, key=lambda x: (entidades[x], x == 'MISC'), reverse=False):
        print(i, entidades[i])

#total_entidades, entidades_n  = count_docs(list_df_stopwords)
#frequencia_entidades(total_entidades, entidades_n)
#resultado_media_entidades(stopwords = media_entidades_stopwords, normalizacao = media_entidades_normalizacao, ngram = media_entidades_ngram,
# lematizacao=media_entidades_lematizacao, pos = media_entidades_pos)

#### ENTIDADEs QUE MAIS APARECEM - TOTAL
### MEDIA DE ENTIDADES POR DOCUMENTO
### TOTAL de entidades
### ENTIDADES QUE MAIS APARECEM POR DOCUMENTO > ENTRE E INTER. Será que tem ents que repetem no mesmo doc?
### EXCERTO COM MAIS E MENOS ENTIDADES >> explorar excerto