import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import word2vec
import os
import re


data = pd.read_csv('/home/pan/Downloads/Top30.csv')

# print(data.info())
# print(data.Query.value_counts())

data = data.reset_index(drop=True)
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;-•    ]')
# STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub('[^\s]*.com[^\s]*', "", text)
    text = re.sub('[^\s]*www.[^\s]*', "", text)
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = text.replace(r'\r','').replace(r'\t','').replace(r'\n','').replace('-','')
    return text

data['Description'] = data['Description'].apply(clean_text)
data['Query'] = data['Query'].apply(clean_text)
data = data.sort_values(by = ['Query'])
data.to_csv("/home/pan/Downloads/cleanedTop30.csv")
last_query = data['Query'][0]
file1 = open("/home/pan/Documents/NLP/structured_data/{}.txt".format(last_query),"w+")
counter = len(data['Description'])
for desc, query in zip(data['Description'],data['Query']):
    if query != last_query:
        file1.close() 
        file1 = open("/home/pan/Documents/NLP/structured_data/{}.txt".format(query),"w+")
    file1.write(query+": "+desc)
    last_query = query
    counter -=1
    print(counter)

file1.close() 





