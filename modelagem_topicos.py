from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
import pandas as pd
import PyPDF2
import nltk
from nltk.corpus import stopwords
import re
import string

# https://rustup.rs
#https://medium.com/leti-pires/modelagem-de-tópicos-em-python-utilizando-o-modelo-de-alocação-latente-de-dirichlet-lda-3276a469f421
#https://maartengr.github.io/BERTopic/index.html#attributes
#@article{grootendorst2022bertopic,
#  title={BERTopic: Neural topic modeling with a class-based TF-IDF procedure},
#  author={Grootendorst, Maarten},
#  journal={arXiv preprint arXiv:2203.05794},
#  year={2022}
#}
nltk.download('punkt')
nltk.download('stopwords')
 
def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return [text]

def preprocess(docs):
    new_doc = []
    for doc in docs:
        tokens = nltk.word_tokenize(re.sub(f"[{re.escape(string.punctuation)}]", ' ', doc.lower()))
        tokens_hifens = nltk.word_tokenize(re.sub(f"[{re.escape(string.punctuation)}]", '-', doc.lower()))
        hifens_filtered = [w for w in tokens_hifens if (len(w) > 1) and (w not in stopwords_set)]
        filtered_tokens = [w for w in tokens if (len(w) > 1) and (w not in stopwords_set)]
        #new_doc.extend(filtered_tokens)
        new_doc.extend(hifens_filtered)
        ## add feature para não separar palavras com hífen
    return new_doc

pt_stopwords = stopwords.words('portuguese')
new_stopwords = ['anexo', 'parágrafo', 'após', 'meio', 'incluindo', 'quais',
                  'outros', 'âmbito', 'diretrizes', 'rede', 'investimentos', 'investimento']
stopwords_set = set(pt_stopwords + [word.lower() for word in new_stopwords])
doc = preprocess(extract_text('./raw_data/portaria_rede_cegonha.pdf'))

#https://www.sbert.net/docs/pretrained_models.html
topic_model_emb = BERTopic(language='portuguese', embedding_model='multi-qa-MiniLM-L6-cos-v1',
                            vectorizer_model=CountVectorizer(ngram_range=(1, 1)), top_n_words=10)
topics_emb, probs_emb = topic_model_emb.fit_transform(doc)

#https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html
topic_model_tfidf = BERTopic(language='portuguese', ctfidf_model=ClassTfidfTransformer(),
                              vectorizer_model=CountVectorizer(ngram_range=(1, 1)), top_n_words=10)
topics_tfidf, probs_emb = topic_model_tfidf.fit_transform(doc)

#https://maartengr.github.io/BERTopic/getting_started/visualization/visualization.html#visualize-documents
# Modelagem - Embedings
fig = topic_model_emb.visualize_barchart()
fig.write_html("./images/emb1.html")

#Modelagem TFIDF
#topic_model_tfidf.visualize_documents()
fig = topic_model_tfidf.visualize_barchart()
fig.write_html("./images/tfidf1.html")
#topic_model.get_document_info(docs)

# Fine-tune your topic representations
#representation_model = KeyBERTInspired()
#topic_model = BERTopic(representation_model=representation_model)

##documents = df['corpus'].tolist()
##D. Vianna, E. Moura. Organizing Portuguese Legal Documents through Topic Discovery. Proceedings 
# of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2022