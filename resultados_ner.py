import analise_pp_ner
import pre_processamento
import zipfile
import pandas as pd
import numpy as np
import spacy

from gensim.models import Word2Vec

#model = Word2Vec.load('./corpus_nilc.txt')

PATH = (".")
nlp = spacy.load("pt_core_news_lg")

with zipfile.ZipFile("./raw_data/saude.csv.zip", 'r') as zip_ref:
    with zip_ref.open('saude.csv') as arquivo_csv:
        df = pd.read_csv(arquivo_csv)


new_df = df[['excerpt', 'source_territory_id', 'source_state_code', 'source_territory_name', 'source_date']]
new_df['ano'] = new_df['source_date'].str.extract(r'(\d{4})')
new_df.loc[:, 'ano'] = pd.to_numeric(new_df['ano'], errors='coerce')
new_df = new_df[(new_df['ano'] >= 2011) & (new_df['ano'] <= 2021)]

df_antes_2016 = new_df[(new_df['ano'] <= 2016)]
df_pos_2016 = new_df[(new_df['ano'] > 2016)] 

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
#list_df_embeddings = pre_processamento.embeddings(model, nlp_preprocessamento(new_df['excerpt'].tolist()))
list_df_sem_processamento = nlp_preprocessamento(new_df['excerpt'])

list_stopwords_antes = nlp_preprocessamento(pre_processamento.remove_stopwords(df_antes_2016))
list_normalizacao_antes = nlp_preprocessamento(pre_processamento.normalizacao(df_antes_2016))
list_ngrams_antes = nlp_preprocessamento(pre_processamento.documentNgrams(df_antes_2016))
list_lemmas_antes = pre_processamento.lematizacao(nlp_preprocessamento(df_antes_2016['excerpt'].tolist()))
list_pos_antes = pre_processamento.pos(nlp_preprocessamento(df_antes_2016['excerpt'].tolist()))
#list_embedding_antes = pre_processamento.embeddings(model, nlp_preprocessamento(df_antes_2016['excerpt'].tolist()))
list_sem_processamento_antes = nlp_preprocessamento(df_antes_2016['excerpt'])

list_stopwords_depois = nlp_preprocessamento(pre_processamento.remove_stopwords(df_pos_2016))
list_normalizacao_depois = nlp_preprocessamento(pre_processamento.normalizacao(df_pos_2016))
list_ngrams_depois = nlp_preprocessamento(pre_processamento.documentNgrams(df_pos_2016))
list_lemmas_depois = pre_processamento.lematizacao(nlp_preprocessamento(df_pos_2016['excerpt'].tolist()))
list_pos_depois = pre_processamento.pos(nlp_preprocessamento(df_pos_2016['excerpt'].tolist()))
#list_embedding_depois = pre_processamento.embeddings(model, nlp_preprocessamento(df_pos_2016['excerpt'].tolist()))
list_sem_processamento_depois = nlp_preprocessamento(df_pos_2016['excerpt'])

#print(f'stopwords {len(list_df_stopwords)}')
#print(f'normalizacao {len(list_df_normalizacao)}')
#print(f'ngram {len(list_df_ngrams)}')
#print(f'lematizacao {len(list_df_lemmas)}')
#print(f'pos {len(list_df_pos)}')
#print(len(list_df_sem_processamento))
#print(len(list(new_df['excerpt'])))
#   
#print(f'Tokens em stopwords: {sum(len(doc) for doc in list_df_stopwords)}')
#print(f'Tokens em normalizacao: {sum(len(doc) for doc in list_df_normalizacao)}')
#print(f'Tokens em n-grama: {sum(len(doc) for doc in list_df_ngrams)}')
#print(f'Tokens em lemmas: {sum(len(doc) for doc in list_df_lemmas)}')
#print(f'Tokens em POS: {sum(len(doc) for doc in list_df_pos)}')
#print(f'Tokens sem pre-processamento: {sum(len(doc) for doc in list_df_sem_processamento)}')
#
##media_entidades_stopwords = np.mean([len(w.ents) for w in list_df_stopwords])
##media_entidades_normalizacao = np.mean([len(w.ents) for w in list_df_normalizacao])
##media_entidades_ngram = np.mean([len(w.ents) for w in list_df_ngrams])
##media_entidades_lematizacao = analise_pp_ner.media_entidades_lemmas(list_df_lemmas)
##media_entidades_pos = analise_pp_ner.media_entidades_pos(list_df_pos)

total_entidades, entidades_n, mais_entidades_real, doc_mais_entidades_real, menos_entidades_real, doc_menos_entidades_real, entidades_misc, entidades_org, entidades_per, entidades_loc = analise_pp_ner.count_docs(list_df_ngrams)
analise_pp_ner.frequencia_entidades(total_entidades, entidades_n, entidades_misc, entidades_org, entidades_per, entidades_loc)
#analise_pp_ner.resultado_media_entidades(stopwords = media_entidades_stopwords, normalizacao = media_entidades_normalizacao, ngram = media_entidades_ngram,
# lematizacao=media_entidades_lematizacao, pos = media_entidades_pos)


print(f"Excerto com maior numero de entidades: {mais_entidades_real}: {list_df_stopwords[doc_mais_entidades_real - 1]}\n")
print(f"Excerto com menor numero de entidades: {menos_entidades_real}: {list_df_stopwords[doc_menos_entidades_real - 1]}\n")