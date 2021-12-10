# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 00:10:40 2021

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
import pickle
import sklearn
from operator import itemgetter

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%% Get HeLa DDA protein lists

with open('interim_files/HeLa_DDA_sample.pkl', 'rb') as handle:
    testdata = pickle.load(handle)

panel_corr = pd.read_csv('interim_files/klijn_panel_spearmanCorr.tsv', sep='\t',index_col=0)


#%% Generate KDE plots for basic evidence levels

obs = testdata['HeLa R1']['two peptide']
unobs = testdata['HeLa R1']['no evidence']
q = pgx.prose(obs, unobs, panel_corr)
data = q.summary

#%% Generate KDE plots for basic evidence levels


palette = ['#bababa', '#1f77b4', '#FF7F0E'] 

sns.color_palette(itemgetter(7,0,1)(sns.color_palette("tab10", 10)))

fig, axes = plt.subplots(nrows=2,ncols=1,figsize=[15,10])

ax=axes[0]
g = sns.kdeplot(data=data, x='score_norm',hue='y_true',common_norm=False,ax=ax,
                palette=palette,legend=False,lw=6)
ax.set_xlabel('PROSE score')
ax.set_ylabel('')

ax=axes[1]
g = sns.kdeplot(data=data, x='prob',hue='y_true',common_norm=False,ax=ax,
                palette=palette,legend=False,lw=6)
ax.set_xlabel(r'$P_{\rmLR}$ (protein exists)')
ax.set_ylabel('')

plt.subplots_adjust(hspace=0.5)
fig.supylabel('Density',x=0.01)

plt.savefig('plots/HeLaR1_bench_KDE.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show() 

data.drop(columns=['y_pred', 'y_true']).to_csv(
    'source_data/Fig 1e (HeLa R1, Score and LR prob distributions).csv',
    index=False)

#%% Generate similarity plots for different evidence levels

groups = testdata['HeLa R1'].keys()

rename_method = {'percolator':'Percolator',
                 'epifany':'EPIFANY',
                 'fido':'Fido'}

result = pd.DataFrame()
ks_mat = pd.DataFrame()

for group in groups:
    data = q.summary
    members = testdata['HeLa R1'][group]
    subset = data[data.protein.isin(members)]
    subset['group'] = group
    result = result.append(subset)

for i, j in itertools.combinations(groups,2):
    d=scipy.stats.ks_2samp(result[result.group == i].score_norm,
                           result[result.group == j].score_norm)[0]
    
    if i in rename_method.keys():
        i = rename_method[i]
    if j in rename_method.keys():
        j = rename_method[j]       
    
    ks_mat.loc[i,j] = d
    ks_mat.loc[j,i] = d
    ks_mat.loc[i,i] = 0
    ks_mat.loc[j,j] = 0

ks_mat = 1-ks_mat
ks_mat = ks_mat.reindex(['two peptide',
                         'Percolator',
                         'EPIFANY',
                         'Fido',
                         'one peptide',
                         'weak evidence',
                         'ambiguous',
                         'no evidence',
                         ])

ks_mat = ks_mat[ks_mat.index]

ks_mat.to_csv('source_data/Fig 1f (KS matrix for inference approach).csv')

mask = np.triu(np.ones_like(ks_mat, dtype=bool))
np.fill_diagonal(mask, False)


fig, ax = plt.subplots(figsize=[12, 12])

cmap = sns.diverging_palette(9, 255, as_cmap=True)

g=sns.heatmap(ks_mat, mask=mask, cmap=cmap, center=0.7,vmax=1,vmin = 0.4,
              square=True, linewidths=2.5,
              cbar_kws=dict(shrink=.6,ticks=[0,.2,.4,.6,.8,1],
                            label='',
                            pad=0.12),
              )
plt.text(s='Similarity' + '\n' +r' (1 - $D_{\rmKS}$)',
         x=6.3,y=4,rotation=0)

plt.savefig('plots/HeLaR1_bench_KS_subclasses.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show() 
