# -*- coding: utf-8 -*-
"""
See patterns of protein recovery based on PROSE scores
"""

import pandas as pd
import pickle 
import os.path
import numpy as np


# load datasets 
with open('datasets/proteomics/fig2_proteinlists.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

# get sample data as agg_id
samplenames = datasets.keys()
sampledata = pd.DataFrame([i.split('_') for i in samplenames],
                          index = samplenames,
                          columns = ['method', 'sample', 'rep'])

agg_id = pd.read_csv('source_data/figure3_aggregated.csv', index_col=0)

#%%

# import glob

# fs = glob.glob('source_data/figure2/*_1000_1rhk_PROSE_random_k0.csv.gz')

                 
#%% Plot missing recovery

import seaborn as sns
import matplotlib.pyplot as plt

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 1.0)

fig, ax = plt.subplots(figsize=[10,10])

for dda_samp in ['DDA HeLa R1', 'DDA HeLa R2', 'DDA HeLa R3', 'DDA HeLa R4']:
# dda_samp = 'DDA HeLa R1'
    f = f'source_data/figure2/{"_".join(dda_samp.split(" "))}_1000_1rhk_PROSE_random_k0.csv.gz'
    
    dia_ids = agg_id[(agg_id['DIA HeLa R1'] == 1) |
                     (agg_id['DIA HeLa R2'] == 1) |
                     (agg_id['DIA HeLa R3'] == 1) |
                     (agg_id['DIA HeLa R4'] == 1)].index
    
    mismatch_ids = agg_id[(agg_id['DDA THP1 R1'] == 1) |
                          (agg_id['DDA THP1 R2'] == 1) |
                          (agg_id['DDA THP1 R3'] == 1)].index

    dia_specific = dia_ids.difference(mismatch_ids)
    mismatch_specific = mismatch_ids.difference(dia_ids)
    
    missing_dda = agg_id[(agg_id[dda_samp] == 0) &
                         (agg_id.index.isin(dia_specific))].index
    
    mismatch_dda = agg_id[(agg_id[dda_samp] == 0) &
                         (agg_id.index.isin(mismatch_specific))].index

    
    result = pd.read_csv(f, index_col=0)
    
    missing_scores = result[result.index.isin(missing_dda)]\
                     .sort_values(by='score_norm', ascending=False)
                     
                     
    mismatch_scores = result[result.index.isin(mismatch_dda)]\
                     .sort_values(by='score_norm', ascending=False)
                     
    
    sns.kdeplot(data = missing_scores, x='prob', color='orange', cut=1)
    sns.kdeplot(data = mismatch_scores, x='prob', color='cornflowerblue', cut=1)
    
    sns.despine()


plt.xlabel('LR Prob(protein)')
plt.ylabel('HeLa DDA \nunobserved proteins', labelpad=20)

#%% Phenotype specificity of scores
import itertools as it
import glob

n_vals = [2000,1000,500]
methods = ['MaxLink', 'MaxLink (norm.)', 'RWRH', 'PROSE']
samplemethods = ['top_n','random']

for method, samplemethod, n in it.product(methods, samplemethods, n_vals):
    
    fs = glob.glob(f'source_data/figure2/*_{n}_1rhk_{method}_{samplemethod}_k0.csv.gz')
        
    agg = pd.DataFrame()
    
    for f in fs:
        fname_clean =  f.replace('.csv.gz','').replace('top_n', 'topn').split('\\')[-1]
        identifiers = fname_clean.split('_')
        samplename = '_'.join(identifiers[:3])
        result = pd.read_csv(f, index_col=0)
        
        if len(agg.index) <= 1:
            agg = pd.DataFrame(index=result.index)
        
        agg.loc[result.index, samplename] = result['score']
        
    agg.columns = [i.replace('_', ' ') for i in agg.columns]
    
    data = agg.corr(method='spearman')
        
    g = sns.clustermap(data=data,
                       yticklabels=1,xticklabels=0,
                       vmin=0.6, vmax=1,
                       figsize=[16,10],
                       dendrogram_ratio=[.2,0],
                       cmap='rocket')
    
    cax = g.cax
    cax.set_visible(False)
    
    ax = g.ax_heatmap
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    
    samplemethod_label = samplemethod.replace("top_n", f'top {n} proteins')\
                                     .replace("random", f'{n} random proteins')
    
    ax.set_title(f'{method}, {samplemethod_label}', pad=40)
    
    g.savefig(f'plots/suppfig_phenotype_score/{n}_{method}_{samplemethod}.png',
                bbox_inches='tight',
                dpi=600,)
    


#%%

g = sns.clustermap(data=data,
                    yticklabels=1,xticklabels=0,
                    vmin=0.6, vmax=1,
                    figsize=[16,10],
                    dendrogram_ratio=[.2,0],
                    cmap='rocket',
                    cbar_pos=(.0, -.16, .35, .05),
                    cbar_kws=dict(orientation='horizontal'))

cax = g.cax
cax.yaxis.set_ticks_position('left')
cax.tick_params(size=12, labelsize=30)
cax.text(x=-1,y=0.5,s='Rank correlation ($ρ$), score', size=30, transform=cax.transAxes,  va='center')


