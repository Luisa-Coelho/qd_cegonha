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

new_df, df_excerpt, list_df_stopwords = pre_processamento.remove_stopwords(df)

new_df = new_df[['excerpt', 'source_territory_id', 'source_state_code', 'source_territory_name', 'source_date']]
subset_df = new_df[new_df['source_state_code'] == 'GO']
list_go = subset_df['excerpt'].tolist()

#doc = [nlp(excerpt) for excerpt in list_df_stopwords]
print(len(list_go))
doc_full = [nlp(excerpt) for excerpt in list_go]
doc1 = [nlp(excerpt) for excerpt in list_df_stopwords[1:1000]]
doc2 = [nlp(excerpt) for excerpt in list_df_stopwords[1001:2000]]
doc3 = [nlp(excerpt) for excerpt in list_df_stopwords[2001:3000]]
doc4 = [nlp(excerpt) for excerpt in list_df_stopwords[3001:4000]]

#print([(w.ents) for w in doc2])

media_entidades = np.mean([len(w.ents) for w in doc1])
print(media_entidades)
media_entidades = np.mean([len(w.ents) for w in doc_full])
print(media_entidades)
print('\n')

entidades = {}
total_entity_counts = {}
mais_entidades_doc = 0
menos_entidades_doc = 100
excerpt = 0
count = 0

for doc in doc1:
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

print("Contagem entidades:\n")
for entity_type, total_count in total_entity_counts.items():
    print(f"{entity_type}: {total_count}\n")

print('Entidade que mais aparece entre os documentos - MISC\n')
#for i in sorted(entidades, key = entidades.get, reverse=True):
#    print(i, entidades[i])

for i in sorted(entidades, key=lambda x: (entidades[x], x == 'MISC'), reverse=True):
    print(i, entidades[i])

print(f"Excerto com maior numero de entidades: {mais_entidades_doc}: {doc1[mais_entidades_doc]}\n")
print(f"Excerto com menor numero de entidades: {menos_entidades_doc}: {doc1[menos_entidades_doc]}\n")


#### ENTIDADEs QUE MAIS APARECEM - TOTAL
### MEDIA DE ENTIDADES POR DOCUMENTO
### TOTAL de entidades
### ENTIDADES QUE MAIS APARECEM POR DOCUMENTO > ENTRE E INTER. Será que tem ents que repetem no mesmo doc?
### EXCERTO COM MAIS E MENOS ENTIDADES >> explorar excerto