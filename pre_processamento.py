import pandas as pd
import pathlib
import nltk
from nltk.corpus import stopwords

### Tecnicas de Pré-processamento
# 1. Stopwords
def remove_stopwords(df):
    stop_pt = set(nltk.corpus.stopwords.words('portuguese'))
    df['excerpt'] = df['excerpt'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_pt]))
    
    return df, df['excerpt'], list(df['excerpt'])

# 2. Normalização
#remover pontuacao
#remover jargoes ?
#ref: Investigating the impact of pre‑processing techniques and pre‑trained word embeddings 
#in detecting Arabic health information on social media

# 3. Lowercase

# 4. Lematização

# 5. Tokenização

# 6. POS

# 7. Remover números

# 8. Generalização das palavras

## outra ref: Repositorio Fabio Colado

