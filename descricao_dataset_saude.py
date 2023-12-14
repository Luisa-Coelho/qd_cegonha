import zipfile
import pandas as pd
import numpy as np
import re

PATH = (".")

with zipfile.ZipFile("./raw_data/saude.csv.zip", 'r') as zip_ref:
    with zip_ref.open('saude.csv') as arquivo_csv:
        df = pd.read_csv(arquivo_csv)

new_df = df[['excerpt', 'source_territory_id', 'source_state_code', 'source_territory_name', 'source_date']]
new_df['ano'] = new_df['source_date'].str.extract(r'(\d{4})')
new_df.loc[:, 'ano'] = pd.to_numeric(new_df['ano'], errors='coerce')
new_df.loc[:, 'source_territory_id'] = new_df['source_territory_id'].astype(str)
new_df = new_df[(new_df['ano'] >= 2011) & (new_df['ano'] <= 2021)]

print(new_df.describe(include='all'))

count_id = new_df['source_territory_name'].value_counts()
new_df['count_id'] = new_df['source_territory_name'].map(count_id)

df_cities = new_df.drop_duplicates(subset=['source_territory_name'])
print("Cidades que mais publicaram")
pd.set_option('display.max_columns', None)
print(df_cities.sort_values(by='count_id', ascending=False))

def extrair_pontuacoes(texto):
    # Usando expressão regular para encontrar pontuações
    pontuacoes = re.findall(r'[.,;!?]', texto)
    return pontuacoes

def extrair_numeros(texto):
    # Usando expressão regular para encontrar números
    numeros = re.findall(r'\b\d+(\.\d+)?\b', texto)
    #numeros_validos = [num for num in numeros if '.' not in num or num.count('.') <= 1]
    numeros_validos = [num for num in numeros if '.' not in num or num.count('.') <= 1]
    numeros_validos = [num for num in numeros_validos if num != '']
    return numeros_validos

# Aplicando a função para extrair pontuações e criando uma nova coluna no DataFrame
new_df['pontuacoes'] = new_df['excerpt'].apply(extrair_pontuacoes)
new_df['media_pontuacoes'] = new_df['pontuacoes'].apply(lambda x: len(x) if x else 0)

new_df['numeros'] = new_df['excerpt'].apply(extrair_numeros)
new_df['soma_numeros'] = new_df['numeros'].apply(lambda x: sum(map(float, x)) if x else 0)

# Calculando a soma e média dos números
print("Soma dos números:", np.sum(new_df['soma_numeros']))
print("Média das pontuações:", np.mean(new_df['media_pontuacoes']))

new_df['num_excertos'] = new_df['excerpt'].apply(lambda x: len(x))

# Somar o número total de excertos para cada ano
soma_por_ano = new_df.groupby('ano')['num_excertos'].sum()

# Calcular a média dividindo pelo total de anos
media_total = soma_por_ano.sum() / len(soma_por_ano)

# Exibir o resultado
print(f'Soma de excertos por ano:\n{soma_por_ano}')
print(f'Média total de excertos: {media_total}')