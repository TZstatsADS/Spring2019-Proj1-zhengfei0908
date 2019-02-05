#! /usr/bin/env python3
# -*- coding:UTF-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE

def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def Tsne_Plot(data, num_features, num_topics, dim, threshold=0.05):
    '''
    data: list of critical words in documents
    num_features: the max features that tf-idf can extract
    num_topics: the topics NMF will decompose
    threshold: this control how many words displayed on the result
    '''

    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_features)
    tfidf = tfidf_vectorizer.fit_transform(data)
    #tfidf_feature_names = tfidf_vectorizer.get_feature_names()

    nmf = NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    W = nmf.fit_transform(tfidf)
    #H = nmf.components_
    
    idx = np.max(W, axis=1) > threshold  # idx of doc that above the threshold
    W1 = W[idx]
    tsne_model = TSNE(n_components=dim, verbose=0, random_state=0, angle=.99, init='pca')
    tsne_nmf = tsne_model.fit_transform(W1)
    if dim == 2:
        result = {'x': tsne_nmf[:,0], 'y': tsne_nmf[:,1]}
    elif dim == 3:
        result = {'x': tsne_nmf[:,0], 'y': tsne_nmf[:,1], 'z': tsne_nmf[:,2]}

    colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
    ])

    nmf_keys = [] # topic_num for each docunment
    for i in range(len(W)):
        nmf_keys.append(W[i].argmax())
    result['category'] = nmf_keys

    nmf_keys1 = [x for i,x in enumerate(nmf_keys) if idx[i] == True]

    color = colormap[nmf_keys1]
    result['color'] = color
    
    text = ['Topic ' + str(i) + 'Content: ' + str(sent) for i,sent in 
    zip(nmf_keys, [sent for i, sent in enumerate(data) if idx[i] == True])]
    result['text'] = text

    return result