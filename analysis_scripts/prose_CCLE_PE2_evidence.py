# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 05:21:45 2021

@author: bw98j
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

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%% Read formatted matrices
score = pd.read_csv('ccle/ccle_prose_formatted.tsv.gz', sep='\t').drop_duplicates('cell_line')

#%%
pe=glob.glob('databases/nextprot*')

pedict = {}
for i in pe:
    pe = i.split('_')[3]
    df = pd.read_csv(i)
    pro = [i.split('_')[-1] for i in df.values.T[0]]
    pedict[pe] = set(pro)

#%%

score = score.sort_values(by='tissue')


df = score[list(score.columns.intersection(pedict['PE2']))]
data = df.mean().sort_values(ascending=True)


fig, ax = plt.subplots(figsize=[10,9])

g = sns.scatterplot(y=df.mean().sort_values(ascending=True),
                 x=df.mean().sort_values(ascending=True).rank(),
                 )
sns.despine()

plt.ylabel('Mean PROSE score', labelpad=10)
plt.xlabel('rank')

plt.text(s='Protein',x=1.4,y=.95,size=30,ha='right',transform = ax.transAxes, weight='bold')
plt.text(s='Score',x=1.45,y=.95,size=30,ha='left',transform = ax.transAxes,weight='bold')

highscore = df.mean().sort_values(ascending=False)[:10]    
for p,m,i in zip(highscore.index, round(highscore,3),range(len(highscore))):
    print(p,m)    
    plt.text(s=m,x=1.45,y=.85-i*0.08, ha='left',
          size=30,transform = ax.transAxes)

    plt.text(s=p,x=1.4,y=.85-i*0.08, ha='right',
              size=30,transform = ax.transAxes)

plt.savefig('plots/CCLE_PE2_rank.png',
            format='png', dpi=600, bbox_inches='tight') 

data.to_csv('source_data/Fig S4a (PE2 rank plot).csv')

#%%





cmap = sns.diverging_palette(9, 255, as_cmap=True)

g = sns.clustermap(data=df[highscore.index].T, 
                    cmap=cmap,
                    vmin=0,vmax=2,center=0,
                    figsize=[12,10],
                    xticklabels=False,yticklabels=True,
                    dendrogram_ratio=0.1,
                    row_cluster=False,col_cluster=False,
                    cbar_kws={"orientation": "horizontal", 'aspect':50},
                    )

ax = g.ax_heatmap
ax.set_xlabel('cell lines',size=40,labelpad=10)

g.cax.set_position([.45, -0.08, .3, .02])
ax.text(x=0.3,y=-0.2,s='PROSE score',ha='center',size=40, transform = ax.transAxes)

g.savefig('plots/CCLE_PE2_heatmap.png',
            format='png', dpi=600, bbox_inches='tight') 

df[highscore.index].T.to_csv('source_data/Fig S4b (PE2 matrix).csv')
