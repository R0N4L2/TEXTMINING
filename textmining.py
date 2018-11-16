# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:49:35 2018

@author: rcastillo
"""

import pandas as pd
from bson.objectid import ObjectId
import os,gc,sys
import numpy as np
import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score
from scipy.sparse import hstack,csr_matrix
import dask.array as da
import sparse as ss
import nltk
from sklearn import mixture
import string
from numba import jit, prange
@jit
def text2num(lista,inv=False,inverse=False):
    dic=pd.DataFrame({'D':lista})    
    dic.D=dic.D.str.upper()
    val=nltk.FreqDist(dic.D.str.split('[\W+,\d+]').dropna().sum())
    val=pd.DataFrame.from_dict(val,orient='index').reset_index()
    val['palabra']=val.pop("index")
    val['valor']=val.pop(0)
    if inverse:
        val['valor']=(val['valor']/val['valor'].sum())**((-1)**inv)
        val['valor']=val['valor']/val['valor'].sum()
    else:
        val['valor']=(-1)**inv*val['valor']/val['valor'].sum()+inv
    val['word2num']=val['palabra'].apply(word2num)*val['valor']
    dic['word2num']=0
    v,d=val.shape[0],dic.shape[0]
    if v<d:
        for j in prange(v):
            pos=dic.D.str.split('[\W+,\d+]',expand=True).isin([val.loc[j,'palabra']]).any(axis=1)
            if pos.any():
                dic.word2num[pos]+=val.loc[j,'word2num']
    else:
        for j in prange(d):
            pos=val.palabra.isin(dic.loc[j,'D'].str.split('[\W+,\d+]'))
            if pos.any():
                dic.loc[j,'word2num']=val.word2num[pos].sum()   
    return dic.word2num.values
@jit
def word2num(word):
    num,word=0,str(word)
    if word!='None' and word!='nan' and word.isalpha():
        num=np.inner(pd.Series(list(word)).apply(ord),np.power(256.0,-np.arange((pd.Series(word).str.count('')-1).values)))
    return num
def sentencespacerep(word):
    word=str(word)
    if word!='None' and word!='nan':
        x=pd.Series(pd.Series(word).str.split('[\W+,\d+]').sum())
        z=pd.DataFrame({"word":x.tolist(),"count":x.str.count('').tolist()})
        z.loc[z.query('count<=2').index,'word']=' '
        word1=(z.word+' ').sum().strip().replace('  ',' ')
    return word1
if __name__ == '__main__':
    secret = None
    gc.collect()
    os.system('clear')
    with open('textinputest.csv',encoding="utf8",errors='ignore') as f:
        contents = f.read()
    x=pd.DataFrame(pd.Series(contents.split('\n')).str.split(';').tolist())
    rep1=x[1:-1]
    for i in x.keys():
         rep1[str(x.loc[0,i])]=y.pop(i)
    rep1[["precio","costo"]]=rep1[["precio","costo"]].astype(float)
    rep1.descripcion=rep1.descripcion.astype(str)
    a1=1; a2=1; a3=1; a4=1;b=rep1.shape[0]-1
    rep1['descripcion']=rep1['descripcion'].str.lower()
    rep1['descripcion']=rep1['descripcion'].apply(sentencespacerep)
    rep1['Actividad econmica principal']=rep1['Actividad econmica principal'].str.lower()
    rep1['Actividad econmica principal']=rep1['Actividad econmica principal'].apply(sentencespacerep)
    rep1=rep1.drop_duplicates()
    vectorizer=TfidfVectorizer()
    X=vectorizer.fit_transform(rep1.descripcion.tolist())
    Y=vectorizer.fit_transform(rep1['Actividad econmica principal'].tolist())
    X=hstack([X,Y])
    rep1['word2num1']=text2num.py_func(rep1.descripcion)
    rep1['word2num2']=text2num.py_func(rep1['Actividad econmica principal'])
    Y=csr_matrix(rep1[["precio","costo",'word2num1','word2num2']].values)
    X=hstack([X,Y])
    X=da.from_array(ss.COO.from_scipy_sparse(X),chunks=(100,100))
    db = DBSCAN().fit(X)
    rep1=rep1.join(pd.DataFrame({"labels":db.labels_.tolist()}))
    rep1.loc[rep1.query('labels==labels').index,'labels']=rep1.loc[rep1.query('labels==labels').index,'labels']-rep1.query('labels==labels')['labels'].min()
    uax=rep1['labels'].drop_duplicates()
    tam=0
    k=0
    while np.isnan(uax).any() and k<10**3:
        tam+=uax.dropna().shape[0]
        aux=X[rep1.query('labels!=labels').index,:]
        db = DBSCAN().fit(aux)
        rep1.loc[rep1.query('labels!=labels').index,'labels']=(db.labels_-db.labels_.min()+tam).tolist()
        uax=rep1['labels'].drop_duplicates()
        k+=1
    rep1.to_csv("PRUEBALABELS3.csv",sep="|",index=False)
    kmeans = KMeans(n_clusters=rep1['labels'].drop_duplicates().shape[0],init='k-means++',max_iter=1000,n_init=10).fit(X)
    rep1['labels2'] =kmeans.labels_.tolist()
    rep1.to_csv("PRUEBALABELS4.csv",sep="|",index=False)
    dpgmm = mixture.BayesianGaussianMixture(n_components=rep2['labels'].drop_duplicates().shape[0],weight_concentration_prior_type='dirichlet_distribution',tol=1**-10,mean_precision_prior=1**-10,max_iter=2**32-1,random_state=2**32-1,warm_start=True).fit(X)
    rep1['labels3']=dpgmm.predict(X)
    rep1.to_csv("PRUEBALABELS5.csv",sep="|",index=False)
    
	
	
