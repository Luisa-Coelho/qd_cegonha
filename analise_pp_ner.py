### Classificacao por meio de Reconhecimento de Entidades Nomeadas
import spacy
import pre_processamento
import zipfile
import pandas as pd
import numpy as np

PATH = (".")

with zipfile.ZipFile("./raw_data/saude.csv.zip", 'r') as zip_ref:
    with zip_ref.open('saude.csv') as arquivo_csv:
        df = pd.read_csv(arquivo_csv)

nlp = spacy.load("pt_core_news_sm")

df_excerpt, list_df_stopwords = pre_processamento.remove_stopwords(df)

#doc = [nlp(excerpt) for excerpt in list_df_stopwords]
doc1 = [nlp(excerpt) for excerpt in list_df_stopwords[1:200]]
doc2 = [nlp(excerpt) for excerpt in list_df_stopwords[201:400]]
doc3 = [nlp(excerpt) for excerpt in list_df_stopwords[401:600]]
doc4 = [nlp(excerpt) for excerpt in list_df_stopwords[601:800]]
doc5 = [nlp(excerpt) for excerpt in list_df_stopwords[801:1000]]

#print([(w.ents) for w in doc2])

media_entidades = np.mean([len(w.ents) for w in doc2])
print(media_entidades)

total_entity_counts = {}

for doc in doc2:
    entity_counts = {}
    for entidade in doc.ents:
        if entidade.label_ in entity_counts:
            entity_counts[entidade.label_] += 1
        else:
            entity_counts[entidade.label_] = 1

    for entity_type, count in entity_counts.items():
        if entity_type in total_entity_counts:
            total_entity_counts[entity_type] += count
        else:
            total_entity_counts[entity_type] = count

print("Contagem entidades:")
for entity_type, total_count in total_entity_counts.items():
    print(f"{entity_type}: {total_count}")

#### ENTIDADEs QUE MAIS APARECEM - TOTAL
### MEDIA DE ENTIDADES POR DOCUMENTO
### TOTAL de entidades
### ENTIDADES QUE MAIS APARECEM POR DOCUMENTO > ENTRE E INTER. Será que tem ents que repetem no mesmo doc?