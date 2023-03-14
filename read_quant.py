# -*- coding: utf-8 -*-
"""
Parse MaxQuant and DIANN files
"""

import pandas as pd
import numpy as np

#%% read MaxQuant proteinGroups.txt

path = 'datasets/proteomics/maxquant_all_dda/proteinGroups.txt'
only_unique_map = True

df = pd.read_csv(path, sep='\t')
intensity_cols = [i for i in df.columns if 'iBAQ ' in i]
cols = ['Protein IDs']+intensity_cols
df = df[cols]

if only_unique_map:
    n_mapped_proteins = pd.Series([len(i.split(';')) for i in df['Protein IDs']])
    df = df[n_mapped_proteins == 1] # return only uniquely mapped proteins
df = df[~df['Protein IDs'].str.contains('REV__')] # remove decoy mappings
df = df[~df['Protein IDs'].str.contains('CON__')] # remove contaminant mappings

# attribute cleanup
df.set_index('Protein IDs', inplace=True)
df.columns = [i.replace('iBAQ ' , '') for i in df.columns]
df.drop(columns='peptides', inplace=True)

df = np.log10(df+1)
df.fillna(0, inplace=True)

df_dda = df.copy()

#%% Plot MaxQuant

import seaborn as sns
import matplotlib.pyplot as plt

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 1.0)

data = df.copy()#.sample(1000)
data.columns = [f'{i} ({len(df)-(df[i] == 0).sum()})' for i in data.columns]

data = data.T
g = sns.clustermap(data, figsize=(20,8), dendrogram_ratio=(.075,.2),
                   xticklabels=False, yticklabels=True,
                   cmap='magma',
                   cbar_pos=(-0.01, .2, .02, .5),
                   vmin=-0.1,
                   )

cax = g.cax
cax.yaxis.set_ticks_position('left')
cax.set_yticks([0,3,6,9])
cax.tick_params(size=12, labelsize=30)
cax.text(x=-3.5,y=0.5,s='log$_{10}$ iBAQ', size=35, transform=cax.transAxes, rotation=90, va='center')

ax = g.ax_heatmap
ax.tick_params(axis='y', size=15, pad=10)
ax.set_xlabel('')

#%% read DIANN 

path = 'datasets/proteomics/mehta_hela_dia/report.pg_matrix.tsv'

df = pd.read_csv(path, sep='\t')
df.drop(columns=['Protein.Group', 'Protein.Names', 'Genes',
                 'First.Protein.Description'],
        inplace=True)
df.columns = [i.split('\\')[-1] for i in df.columns]

if only_unique_map:
    n_mapped_proteins = pd.Series([len(i.split(';')) for i in df['Protein.Ids']])
    df = df[n_mapped_proteins == 1] # return only uniquely mapped proteins

df.set_index('Protein.Ids', inplace=True)
df.columns = [f'HeLa R{i.replace(".raw","").split("_")[-1]}' for i in df.columns]

df = np.log10(df+1)
df.fillna(0, inplace=True)

df_dia = df.copy()

#%% Plot DIANN

data = df.copy()#.sample(1000)
data.columns = [f'{i} ({len(df)-(df[i] == 0).sum()})' for i in data.columns]

data = data.T
g = sns.clustermap(data, figsize=(20,6), dendrogram_ratio=(.075,.2),
                   xticklabels=False, yticklabels=True,
                   cmap='magma',
                   cbar_pos=(-0.01, .205, .02, .55),
                   vmin=-0.1,
                   )

cax = g.cax
cax.yaxis.set_ticks_position('left')
cax.set_yticks([0,3,6,9])
cax.tick_params(size=12, labelsize=30)
cax.text(x=-3.5,y=0.5,s='log$_{10}$ MaxLFQ', size=35, transform=cax.transAxes, rotation=90, va='center')

ax = g.ax_heatmap
ax.tick_params(axis='y', size=15, pad=10)
ax.set_xlabel('')

#%% Aggregate DIA-DIANN identification heatmap

agg = pd.DataFrame(index=set(df_dda.index.union(df_dia.index)))

for i in df_dda.columns:
    observed = df_dda[df_dda[i]!=0].index
    agg.loc[observed, f'DDA {i}'] = 1
    

for i in df_dia.columns:
    observed = df_dia[df_dia[i]!=0].index
    agg.loc[observed, f'DIA {i}'] = 1
    
agg.fillna(0, inplace=True)

#%% Plot aggregate IDs

data = agg.copy()#.sample(1000)

data = data.T
g = sns.clustermap(data, figsize=(25,9), dendrogram_ratio=(.075,.2),
                   xticklabels=False, yticklabels=True,
                   cmap='magma',
                   vmin=-0.1,
                   )

cax = g.cax
cax.set_visible(False)

ax = g.ax_heatmap
ax.tick_params(axis='y', size=15, pad=10)
ax.set_xlabel('Protein IDs', size=50)

# agg.to_csv('source_data/figure3_aggregated.csv')

#%%

ranked_prots = {}

for i in df_dda.columns:
    prots = df_dda[i]
    nonzero_prots = prots[prots != 0].sort_values(ascending=False)
    ranked_prots[f'DDA_{i.replace(" ", "_")}'] = nonzero_prots
    
for i in df_dia.columns:
    prots = df_dia[i]
    nonzero_prots = prots[prots != 0].sort_values(ascending=False)
    ranked_prots[f'DIA_{i.replace(" ", "_")}'] = nonzero_prots
    
# import pickle
# with open('datasets/proteomics/fig2_proteinlists.pkl', 'wb') as handle:
#     pickle.dump(ranked_prots, handle, protocol=pickle.HIGHEST_PROTOCOL)