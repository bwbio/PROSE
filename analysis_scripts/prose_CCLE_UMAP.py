# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:54:13 2021

@author: Bertrand Jern Han Wong
"""
import prose as pgx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
import itertools
import glob
import os
import random
from tqdm import tqdm
import scipy.stats
import gtfparse
import itertools
from pylab import *
import collections
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import quantile_transform
import pickle
import re
import umap

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%%
result = pd.read_csv('ccle/ccle_prose_formatted.tsv.gz',sep='\t')

#%%


reducer = umap.UMAP(n_neighbors=15,
                    min_dist=0.35,
                    random_state=2)

result_scores = result.drop(columns=['cell_line','tissue'])

scaled = StandardScaler().fit_transform(result_scores)
u = reducer.fit_transform(scaled)

umapDf = pd.DataFrame(u)
umapDf.columns = ['UMAP-1','UMAP-2']

tissues = result['tissue']
umapDf['tissue'] = tissues

cmap = cm.get_cmap('seismic', len(tissues.unique()))
lut = dict(zip(tissues.unique(), [cmap(i)[:3] for i in range(cmap.N)]))
row_colors = tissues.map(lut)
umap_palette = sns.color_palette("husl", len(tissues.unique()))

#Complete UMAP plot
fig, ax = plt.subplots(figsize=[30,30])
g=sns.scatterplot(data=umapDf, x='UMAP-1', y='UMAP-2', hue='tissue',
                  alpha=0.9, s=1350, palette=umap_palette)
plt.xlabel('UMAP-1',size=100,labelpad=20)
plt.ylabel('UMAP-2',size=100,labelpad=20)
ax.legend(markerscale=6).remove()
umap_xlim, umap_ylim = ax.get_xlim(), ax.get_ylim()
plt.xticks([]); plt.yticks([])
plt.savefig('plots/CCLE_tissue_combined.png',
            format='png', dpi=300, bbox_inches='tight')
plt.show()


umapDf.to_csv('source_data/Fig 2c (CCLE UMAP).csv',index=False)

#%%

#Individual tissue plots
fig, axes = plt.subplots(nrows = 6, 
                         ncols = int(ceil(tissues.nunique())/6)+1,
                         figsize=[32,48])

for i, ax in enumerate(axes.flat):
    
    if i < tissues.nunique():
        tissue = tissues.unique()[i]
        g0=sns.scatterplot(data=umapDf[umapDf.tissue!=tissue],
                           x='UMAP-1', y='UMAP-2',
                           alpha=0.05, s=80, color='gray',
                           ax=ax)
        g=sns.scatterplot(data=umapDf[umapDf.tissue==tissue],
                          x='UMAP-1', y='UMAP-2',
                          alpha=0.9, s=80, color=umap_palette[i],
                          ax=ax)

        
        g.set_title(tissue+' (n='+str(tissues.value_counts()[tissue])+')',size=30)
        
        ax.set_xlim(umap_xlim)
        ax.set_ylim(umap_ylim)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([]); ax.set_yticks([])
        
    elif i >= tissues.nunique():
        ax.set_visible(False)
    
plt.savefig('plots/CCLE_umap_individual.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show()    
    
    
    