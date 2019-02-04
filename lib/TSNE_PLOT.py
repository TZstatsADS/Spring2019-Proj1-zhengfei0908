#! /usr/bin/env python3
# -*- coding:UTF-8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.plotting import save
from bokeh.models import HoverTool

def display_topics(model, feature_names, no_top_words=10):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))

def Tsne_Plot(data, num_features, num_topics, threshold, title, display_words = 5, print = False):
    '''
    data: list of critical words in documents
    num_features: the max features that tf-idf can extract
    num_topics: the topics NMF will decompose
    threshold: this control how many words displayed on the result
    title: the title of result and the name of HTML file
    display_words: the number of critical words displayed on the result
    print: whether to print the result topics
    '''
    print('Begin TF-IDF transformation')
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=num_features)
    tfidf = tfidf_vectorizer.fit_transform(data)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print('Done\n')

    print('Begin NMF decomposition')
    nmf = NMF(n_components=num_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    W = nmf.fit_transform(tfidf)
    H = nmf.components_
    print('Done\n')
    
    print('Begin t-SNE computing')
    idx = np.amax(W, axis=1) > threshold  # idx of doc that above the threshold
    W = W[idx]
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_nmf = tsne_model.fit_transform(W)
    print('Done\n')

    if display:
        print('The result topics is:')
        display_topics(nmf, tfidf_feature_names, 10)

    colormap = np.array([
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"
    ])
    nmf_keys = []
    for i in range(W.shape[0]):
        nmf_keys.append(W[i].argmax())
    topic_summaries = []
    for i, topic_dist in enumerate(H):
        topic_words = np.array(tfidf_feature_names)[np.argsort(topic_dist)][:-(display_words + 1):-1] # get!
        topic_summaries.append(' '.join(topic_words)) # append!
    num_example = len(W)
    plot_nmf = bp.figure(plot_width=800, plot_height=600,
                     title=title,
                     tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
                     x_axis_type=None, y_axis_type=None, min_border=1)

    content = [sent for i, sent in enumerate(data) if idx[i] == True]
    source = bp.ColumnDataSource(data=dict(x = tsne_nmf[:, 0], y = tsne_nmf[:, 1], 
                                       color = colormap[nmf_keys][:num_example],
                                       content = content[:num_example], keys = nmf_keys[:num_example]))

    plot_nmf.scatter(x='x', y='y', color = 'color', source = source)

    topic_coord = np.empty((W.shape[1], 2)) * np.nan
    for topic_num in nmf_keys:
        if not np.isnan(topic_coord).any():
            break
        topic_coord[topic_num] = tsne_nmf[nmf_keys.index(topic_num)]
    for i in range(W.shape[1]):
        plot_nmf.text(topic_coord[i, 0], topic_coord[i, 1], [topic_summaries[i]])
    hover = plot_nmf.select(dict(type=HoverTool))
    hover.tooltips = {"content": "@content - topic: @keys"}
    save(plot_nmf, '../figs/{}.html'.format(title))
