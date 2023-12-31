### Classificacao por meio de Reconhecimento de Entidades Nomeadas
import pandas as pd
import numpy as np
import spacy
import pathlib
from collections import Counter
#from gensim.models import KeyedVectors

PATH = (".")

#model = KeyedVectors.load_word2vec_format('./corpus_nilc.txt')

nlp = spacy.load("pt_core_news_lg")

def media_entidades_lemmas(lista_lemmas):
    media_entidades_lemmas = []
    for doc in lista_lemmas:
        doc_lemmas = nlp(" ".join(doc))
        media_entidades_lemmas.append(len(doc_lemmas.ents))
    
    return np.mean(media_entidades_lemmas)

def media_entidades_pos(lista_pos):
    media_entidades_pos = []
    for doc in lista_pos:
        doc_pos = nlp(" ".join(doc))
        media_entidades_pos.append(len(doc_pos.ents))
    
    return np.mean(media_entidades_pos)


def resultado_media_entidades(**media_entidades):
    print(f'Media entidades Stopwords {media_entidades["stopwords"]}\n')
    print(f'Media entidades Normalizacao {media_entidades["normalizacao"]}\n')
    print(f'Media entidades N-Grama {media_entidades["ngram"]}\n')
    print(f'Media entidades Lematização {media_entidades["lematizacao"]}\n')
    print(f'Media entidades POS {media_entidades["pos"]}\n')
    

def count_docs(lista_nlp_preprocessed):
    
    total_entity_counts = {}
    mais_entidades = 0
    menos_entidades = 100
    count = 0
    entidades_contagem = []
    entidades = []
    entidades_misc = []
    entidades_org = []
    entidades_per = []
    entidades_loc = []
    doc_count = 0
    doc_mais_entidades_real = 0
    doc_menos_entidades_real = 0
    doc_mais_entidades = 0
    doc_menos_entidades = 0
    mais_entidades_real = 0
    menos_entidades_real = 200

    for doc in lista_nlp_preprocessed:
        #doc_new = nlp(" ".join(doc))
        doc_new = doc
        
        doc_count +=1
        entities = {}
        entity_counts = {}
        entities_misc = {}
        entities_org = {}
        entities_per = {}
        entities_loc = {}

        for entidade in doc_new.ents:
            count += 1

            if entidade.text in entities:
                entities[entidade.text] += 1
                if entidade.label_ == 'MISC':
                    if entidade.text in entities_misc:
                        entities_misc[entidade.text] += 1
                    else:
                        entities_misc[entidade.text] = 1

                if entidade.label_ == 'ORG':
                    if entidade.text in entities_org:
                        entities_org[entidade.text] += 1
                    else:
                        entities_org[entidade.text] = 1

                if entidade.label_ == 'PER':
                    if entidade.text in entities_per:
                        entities_per[entidade.text] += 1
                    else:
                        entities_per[entidade.text] = 1

                if entidade.label_ == 'LOC':
                    if entidade.text in entities_loc:
                        entities_loc[entidade.text] += 1
                    else:
                        entities_loc[entidade.text] = 1

            else:
                entities[entidade.text] = 1
                if entidade.label_ == 'MISC':
                    if entidade.text in entities_misc:
                        entities_misc[entidade.text] += 1
                    else:
                        entities_misc[entidade.text] = 1

                if entidade.label_ == 'ORG':
                    if entidade.text in entities_org:
                        entities_org[entidade.text] += 1
                    else:
                        entities_org[entidade.text] = 1

                if entidade.label_ == 'PER':
                    if entidade.text in entities_org:
                        entities_per[entidade.text] += 1
                    else:
                        entities_per[entidade.text] = 1

                if entidade.label_ == 'LOC':
                    if entidade.text in entities_org:
                        entities_loc[entidade.text] += 1
                    else:
                        entities_loc[entidade.text] = 1

            if entidade.label_ in entity_counts:
                entity_counts[entidade.label_] += 1
            
            else:
                entity_counts[entidade.label_] = 1

            if count > mais_entidades:
                mais_entidades = count
                doc_mais_entidades = doc_count
#
            if count < menos_entidades:
                menos_entidades = count
                doc_menos_entidades = doc_count

        total_entities_count = sum(entity_counts.values())
        if total_entities_count > mais_entidades_real:
            mais_entidades_real = total_entities_count
            doc_mais_entidades_real = doc_count

        if total_entities_count < menos_entidades_real:
            menos_entidades_real = total_entities_count
            doc_menos_entidades_real = doc_count
    
        entidades.append(entities)
        entidades_contagem.append(entity_counts)
        entidades_misc.append(entities_misc)
        entidades_org.append(entities_org)
        entidades_per.append(entities_per)
        entidades_loc.append(entities_loc)

    for dicionario in entidades_contagem:
        for chave, valor in dicionario.items():
            if chave in total_entity_counts:
                total_entity_counts[chave] += 1
            else:
                total_entity_counts[chave] = 1
    

    return total_entity_counts, entidades, mais_entidades_real, doc_mais_entidades_real, menos_entidades_real, doc_menos_entidades_real, entidades_misc, entidades_org, entidades_per, entidades_loc

def frequencia_entidades(total_entity_counts, entidades, entidades_misc, entidades_org, entidades_per, entidades_loc):
    print("Contagem entidades:\n")
    for entity_type, total_count in total_entity_counts.items():
        print(f"{entity_type}: {total_count}\n")

    print('Entidade que mais aparece entre os documentoS\n')
    contagem_total = Counter
    lista_de_tuplas = [tuple(d.items()) for d in entidades]
    todas_as_tuplas = [item for sublist in lista_de_tuplas for item in sublist]
    contagem = contagem_total(todas_as_tuplas)
    oito_maiores_entidades = contagem.most_common(8)

    for entidade, contagem in oito_maiores_entidades:
      print(f'{entidade}: {contagem} ocorrências')

    print('\n\nENTIDADES MISC')
    contagem_total = Counter
    lista_de_tuplas = [tuple(d.items()) for d in entidades_misc]
    todas_as_tuplas = [item for sublist in lista_de_tuplas for item in sublist]
    contagem = contagem_total(todas_as_tuplas)
    oito_maiores_misc = contagem.most_common(8)

    for entidade, contagem in oito_maiores_misc:
        print(f'{entidade}: {contagem} ocorrências') 
        
    print('\n\nENTIDADES ORG')
    contagem_total = Counter
    lista_de_tuplas = [tuple(d.items()) for d in entidades_org]
    todas_as_tuplas = [item for sublist in lista_de_tuplas for item in sublist]
    contagem = contagem_total(todas_as_tuplas)
    oito_maiores_org = contagem.most_common(8)

    for entidade, contagem in oito_maiores_org:
        print(f'{entidade}: {contagem} ocorrências')  
        
    print('\n\nENTIDADES PER')
    contagem_total = Counter
    lista_de_tuplas = [tuple(d.items()) for d in entidades_per]
    todas_as_tuplas = [item for sublist in lista_de_tuplas for item in sublist]
    contagem = contagem_total(todas_as_tuplas)
    oito_maiores_per = contagem.most_common(8)

    for entidade, contagem in oito_maiores_per:
        print(f'{entidade}: {contagem} ocorrências')

    print('\n Entidades LOC')
    contagem_total = Counter
    lista_de_tuplas = [tuple(d.items()) for d in entidades_loc]
    todas_as_tuplas = [item for sublist in lista_de_tuplas for item in sublist]
    contagem = contagem_total(todas_as_tuplas)
    oito_maiores_loc = contagem.most_common(8)

    for entidade, contagem in oito_maiores_loc:
        print(f'{entidade}: {contagem} ocorrências')
