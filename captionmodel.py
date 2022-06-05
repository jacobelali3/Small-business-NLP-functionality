
#!python -m spacy download en_core_web_lg

import en_core_web_lg

import pandas as pd
import numpy as np
import spacy
import heapq
nlp = spacy.load('en_core_web_lg')

def similarity(x,y):
    x = nlp(x)
    y= nlp(y)
    x=nlp(' '.join([str(t) for t in x if not t.is_stop]))
    y=nlp(' '.join([str(t) for t in y if not t.is_stop]))
    return x.similarity(y)

my_df=pd.read_csv('/content/Dataset1.csv')
my_df=my_df.iloc[:,:10]# 
my_df.dropna(inplace=True)

def caption_model(caption,df):
    hashtags=[]
    if type(caption) !=str:
        return ('invalid input')
    x=[]
    for i in df.iloc[:,0]:
        x.append(similarity(i,caption))
    x=np.array(x)
    rows=heapq.nlargest(3, range(len(x)), x.take)
    y=[]
    z=[]
    for row in rows:
        for i in df.iloc[row][1:]:
            y.append(similarity(i,caption))
            z.append(i)
        
    y=np.array(y)
    final_indices=heapq.nlargest(3, range(len(y)), y.take)
    for i in final_indices:
        hashtags.append(z[i])
    
    return hashtags

x=caption_model('birthday cake for son',my_df)


