# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 01:42:30 2021

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

#%% Read formatted matrices

score = pd.read_csv('ccle/ccle_prose_formatted.tsv.gz', sep='\t')
tmt = pd.read_csv('ccle/ccle_tmt_formatted.tsv.gz', sep='\t')
tpm = pd.read_csv('ccle/ccle_tpm_formatted.tsv.gz', sep='\t')
drive = pd.read_csv('ccle/ccle_drive_formatted.tsv.gz', sep='\t')


#%% TMT-score spearman correlation

result = pd.DataFrame()
cell_lines = score.cell_line.to_list()

common_cols = tmt.columns.intersection(score.columns).to_list()
common_prot = common_cols[2:]
tmt_sub = tmt[common_prot]
score_sub = score[common_prot]

for i, rowi in tqdm(score_sub.iterrows()):
    for j, rowj in tmt_sub.iterrows():
        rho = scipy.stats.spearmanr(rowi,rowj)[0]
        result.loc[i,j] = rho

result.index = cell_lines
result.columns = cell_lines

#%% TMT-score correlation heatmap
 

toi = ['breast',
       'skin',
       'kidney',
       'haematopoietic and lymphoid tissue',
       'central nervous system',
       'endometrium']

mapper = score[['cell_line','tissue']]
coi = mapper[mapper.tissue.isin(toi)].sort_values(by='tissue')
coi = coi.cell_line.drop_duplicates()
coi = coi.to_list()

clusterdf = result.loc[coi,coi]
clusterdf = clusterdf[~clusterdf.index.duplicated(keep='first')]
clusterdf = clusterdf.loc[:,~clusterdf.columns.duplicated()]

clusterdf.to_csv('source_data/Fig 2d (PROSE-PROSE correlation).csv')

tissues = score.tissue
lut = dict(zip(tissues.unique(), sns.color_palette("husl", len(tissues.unique()))))
row_colors = pd.DataFrame(zip(cell_lines, tissues.map(lut)),
                          columns=['cell_line','tissues'],
                          ).drop_duplicates().set_index('cell_line')
row_colors = row_colors.loc[coi]
row_colors.columns=['']

cmap = sns.diverging_palette(9, 255, as_cmap=True)

g = sns.clustermap(clusterdf, 
                   cmap=cmap,
                   vmin=-0.1,vmax=0.5,center=0.2,
                   figsize=[15,15],
                   xticklabels=False,yticklabels=False,
                   row_colors=row_colors,
                   col_colors=row_colors,
                   dendrogram_ratio=0.15,
                    row_cluster=False,col_cluster=False,
                   cbar_kws={"orientation": "horizontal", 'aspect':50},
                   )


ax = g.ax_heatmap
ax.set_xlabel('Protein expression (TMT quant)',size=50,labelpad=30)
ax.set_ylabel('PROSE score',size=50,labelpad=30)
g.cax.set_position([.56, -0.1, .4, .02])
ax.text(x=62,y=168,s=r'$ρ_{\rmscore,TMT}$ ',ha='right',size=50)

g.savefig('plots/CCLE_heatmap_TMT_score.png',
            format='png', dpi=600, bbox_inches='tight') 


#%% TPM-score spearman correlation

result = pd.DataFrame()
cell_lines = score.cell_line.to_list()
tpm_matches = pd.merge(tmt[['cell_line', 'tissue']],tpm, on='cell_line').dropna()


common_cols = tpm_matches.columns.intersection(score.columns).to_list()
common_prot = common_cols[2:]
tpm_sub = tpm_matches[common_prot]
score_sub = score[common_prot]

for i, rowi in tqdm(score_sub.iterrows()):
    for j, rowj in tpm_sub.iterrows():
        rho = scipy.stats.spearmanr(rowi,rowj)[0]
        result.loc[i,j] = rho

result.index = cell_lines
result.columns = tpm_matches.cell_line

#%% TPM-score correlation heatmap

toi = ['breast',
       'skin',
       'kidney',
       'haematopoietic and lymphoid tissue',
       'central nervous system',
       'endometrium']

mapper = tpm_matches[['cell_line','tissue']]
coi = mapper[mapper.tissue.isin(toi)].sort_values(by='tissue')
coi = coi.cell_line.drop_duplicates()
coi = coi.to_list()

clusterdf = result.loc[coi,coi]
clusterdf = clusterdf[~clusterdf.index.duplicated(keep='first')]
clusterdf = clusterdf.loc[:,~clusterdf.columns.duplicated()]

clusterdf.to_csv('source_data/Fig 2f (PROSE-TPM correlation).csv')

tissues = score.tissue
lut = dict(zip(tissues.unique(), sns.color_palette("husl", len(tissues.unique()))))
row_colors = pd.DataFrame(zip(cell_lines, tissues.map(lut)),
                          columns=['cell_line','tissues'],
                          ).drop_duplicates().set_index('cell_line')
row_colors = row_colors.loc[coi]
row_colors.columns=['']

cmap = sns.diverging_palette(9, 255, as_cmap=True)

g = sns.clustermap(clusterdf, 
                   cmap=cmap,
                   vmin=0.2,vmax=0.8,center=0.5,
                   figsize=[15,15],
                   xticklabels=False,yticklabels=False,
                   row_colors=row_colors,
                   col_colors=row_colors,
                   dendrogram_ratio=0.15,
                    row_cluster=False,col_cluster=False,
                   cbar_kws={"orientation": "horizontal", 'aspect':50},
                   )

ax = g.ax_heatmap
ax.set_xlabel('Gene expression (TPM)',size=50,labelpad=30)
ax.set_ylabel('PROSE score',size=50,labelpad=30)
g.cax.set_position([.56, -0.1, .4, .02])
ax.text(x=62,y=168,s=r'$ρ_{\rmscore,TPM}$ ',ha='right',size=50)

g.savefig('plots/CCLE_heatmap_TPM_score.png',
            format='png', dpi=600, bbox_inches='tight') 


#%% score-score spearman correlation

result = pd.DataFrame()
cell_lines = score.cell_line.to_list()

for i, rowi in tqdm(score_sub.iterrows()):
    for j, rowj in score_sub.iterrows():
        r = scipy.stats.pearsonr(rowi,rowj)[0]
        result.loc[i,j] = r

result.index = cell_lines
result.columns = cell_lines


#%% score-score correlation heatmap

toi = ['breast',
       'skin',
       'kidney',
       'haematopoietic and lymphoid tissue',
       'central nervous system',
       'endometrium']

mapper = score[['cell_line','tissue']]
coi = mapper[mapper.tissue.isin(toi)].sort_values(by='tissue')
coi = coi.cell_line.drop_duplicates()
coi = coi.to_list()

clusterdf = result.loc[coi,coi]
clusterdf = clusterdf[~clusterdf.index.duplicated(keep='first')]
clusterdf = clusterdf.loc[:,~clusterdf.columns.duplicated()]

clusterdf.to_csv('source_data/Fig 2e (PROSE-TMT correlation).csv')

tissues = score.tissue
lut = dict(zip(tissues.unique(), sns.color_palette("husl", len(tissues.unique()))))
row_colors = pd.DataFrame(zip(cell_lines, tissues.map(lut)),
                          columns=['cell_line','tissues'],
                          ).drop_duplicates().set_index('cell_line')
row_colors = row_colors.loc[coi]
row_colors.columns=['']

cmap = sns.diverging_palette(9, 255, as_cmap=True)

g = sns.clustermap(clusterdf, 
                   cmap=cmap,
                   vmin=0,vmax=1,center=0.5,
                   figsize=[15,15],
                   xticklabels=False,yticklabels=False,
                   row_colors=row_colors,
                   col_colors=row_colors,
                   dendrogram_ratio=0.15,
                    row_cluster=False,col_cluster=False,
                   cbar_kws={"orientation": "horizontal", 'aspect':50},
                   )

ax = g.ax_heatmap
ax.set_xlabel('PROSE score',size=50,labelpad=30)
ax.set_ylabel('PROSE score',size=50,labelpad=30)
g.cax.set_position([.56, -0.1, .4, .02])
ax.text(x=56,y=168,s=r'$r_{\rmscore, score}$ ',ha='right',size=50)

g.savefig('plots/CCLE_heatmap_score_score.png',
            format='png', dpi=600, bbox_inches='tight') 
