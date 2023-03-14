# -*- coding: utf-8 -*-
"""
Perform analysis on CPTAC dataset (Figure 6)
"""

import pandas as pd
import pickle 
import os.path
import numpy as np
import glob

import seaborn as sns
import matplotlib.pyplot as plt

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 1.0)

#%% calculate auc and f1

from sklearn.metrics import roc_auc_score, f1_score

def find_auc(result):
    res = result.dropna().copy()
    res = res[res['y_true'] != -1]
    auc = round(roc_auc_score(res['y_true'], res['score']), 3)
    return auc
    pass

def find_f1(result):
    res = result.dropna().copy()
    res = res[res['y_true'] != -1]
    f1 = round(f1_score(res['y_true'], res['y_pred']), 3)
    return f1
    pass

#%% prose

from utils.pyprose_static import pyprose

with open('utils/feature_space/panel_corr.pkl', 'rb') as handle:
    panel_corr = pickle.load(handle)
    
#%% get housekeepers

from utils.convert_ids import symbol2uniprot, ensembl2uniprot
conv_symbol_to_uniprot =  symbol2uniprot()
conv_ensembl_to_uniprot = ensembl2uniprot()

# load datasets 
with open('datasets/proteomics/fig2_proteinlists.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

# get housekeeper genes
housekeeper_db = pd.read_csv('databases/Housekeeping_GenesHuman.csv', sep=';')
hk_symbol = set([i for i in housekeeper_db['Gene.name']])
hk = [conv_symbol_to_uniprot[i] for i in hk_symbol if i in conv_symbol_to_uniprot]

#%% Read Krug et al. (2020) CPTAC proteomics data

md = pd.read_csv('datasets/proteomics/krug_cptac2_brca_tmt/md.csv').set_index('Sample.ID')

df = pd.read_csv('datasets/proteomics/krug_cptac2_brca_tmt/tmt_data.csv')

id_symbols = set(df['geneSymbol'])

other_cols = ['id', 'geneSymbol', 'numColumnsProteinObserved',
              'numSpectraProteinObserved', 'protein_mw', 'percentCoverage',
              'numPepsUnique', 'scoreUnique', 'species', 'orfCategory',
              'accession_number', 'accession_numbers', 'subgroupNum', 'entry_name']

prot_md = df[other_cols].set_index('geneSymbol')

df.set_index('geneSymbol', drop=False, inplace=True)
df.drop(columns=other_cols, inplace=True)

# remove multiple isoforms, keep highest spectral matches
df = df[~df.index.duplicated(keep='first')]
gene_symbols = df.index
df.index = [conv_symbol_to_uniprot[i] if i in conv_symbol_to_uniprot else np.nan for i in df.index]
applied_map = pd.DataFrame(zip(gene_symbols, df.index), columns=['geneSymbol', 'uniprot'])

# remove housekeepers
df = df[~df.index.isin(hk)]

# remove NA indices
df = df[~df.index.isna()]

#%% PROSE P1 on all identifications

n = 1000
holdout_n = 200

for i, samplename in enumerate(df.columns):
    
    filename = f'source_data/figure7_krug_cptac2/prose_{samplename}_{n}_1rhk.csv'
    
    if os.path.exists(filename):
        print(f'{filename} found! skipping...')
        
    else:
    
        sub = df[samplename].sort_values(ascending=False).dropna()
        obs = sub[:n+holdout_n].index.to_list()
        unobs = sub[-(n+holdout_n):].index.to_list()
        result = pyprose(obs, unobs, panel_corr, holdout_n=holdout_n)
    
        result.to_csv(filename)

  
fs = glob.glob('source_data/figure7_krug_cptac2/prose_*_1000_1rhk.csv')

agg = pd.DataFrame()
agg_metric = pd.DataFrame(columns=['auc_train', 'auc_test', 'f1_train', 'f1_test'])

for f in fs:
    identifiers = f.split('_')
    samplename = identifiers[-3]
    result = pd.read_csv(f, index_col=0) 
    agg[samplename] = result.score_norm
    auc_train = find_auc(result[result['is_test_set'] == 0])
    auc_test = find_auc(result[result['is_test_set'] == 1])
    f1_train = find_f1(result[result['is_test_set'] == 0])
    f1_test = find_f1(result[result['is_test_set'] == 1])
    agg_metric.loc[samplename] = [auc_train, auc_test, f1_train, f1_test]
    
#%% Plot metrics

a = 0.3

fig, axes = plt.subplots(figsize=[10,9], nrows=2, ncols=1, sharex=True)

ax = axes[0]
sns.histplot(agg_metric, x='auc_train', alpha=a, kde=True, ax=ax, bins=20,
             line_kws=dict(linewidth=4))
ax.set_xlabel(''); ax.set_ylabel('')
ax.text(s='train', x=.05, y=.6, transform=ax.transAxes)

ax = axes[1]
sns.histplot(agg_metric, x='auc_test', alpha=a, kde=True, ax=ax, bins=20,
             line_kws=dict(linewidth=4))
ax.text(s='test', x=.05, y=.6, transform=ax.transAxes)

sns.despine()
ax.set_xlabel('auROC')
ax.set_ylabel('')


fig, axes = plt.subplots(figsize=[10,9], nrows=2, ncols=1, sharex=True)

ax = axes[0]
sns.histplot(agg_metric, x='f1_train', alpha=a, kde=True, ax=ax, bins=20,
             line_kws=dict(linewidth=4))
ax.set_xlabel(''); ax.set_ylabel('')
ax.text(s='train', x=.05, y=.6, transform=ax.transAxes)

ax = axes[1]
sns.histplot(agg_metric, x='f1_test', alpha=a, kde=True, ax=ax, bins=20,
             line_kws=dict(linewidth=4))
ax.text(s='test', x=.05, y=.6, transform=ax.transAxes)

sns.despine()
ax.set_xlabel('F1 score')
ax.set_ylabel('')

#%% Plot PCA + UMAP reduction

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap


x = StandardScaler().fit_transform(agg.T)
pcadf = PCA(n_components=40, random_state=0).fit_transform(x)
umapdf = umap.UMAP(n_neighbors=10, random_state=0).fit_transform(pcadf)
umapdf = pd.DataFrame(umapdf, columns=['UMAP-1','UMAP-2'])

umapdf.set_index(agg.T.index, inplace=True)

for col in md.columns:
    umapdf.loc[md.index, col] = md[col]


import matplotlib.pyplot as plt
import seaborn as sns

fig, ax = plt.subplots(figsize=[20,20])
g = sns.scatterplot(data=umapdf, x='UMAP-1', y='UMAP-2', 
                    palette='tab10',
                    hue='NMF.Cluster',
                    s=2000,#umapdf['NMF.Cluster.Membership.Score']**2*2000,
                    alpha=.8,
                    )#s=1200)

legend = ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

for i in legend.legendHandles:
    i._sizes = [1200]

# plt.savefig(f'plots/fig7_krug_cptac2_umap.png',
#             bbox_inches='tight',
#             dpi=600,)


# ['Sample.IDs', 'TMT.Plex', 'TMT.Channel', 'Tumor.Stage',
#  'Ischemia.Time.in.Minutes', 'PAM50', 'NMF.Cluster',
#  'NMF.Cluster.Membership.Score'] # metadata

#%%
# d = pd.DataFrame(umap.UMAP(n_neighbors=10, random_state=0).fit_transform(PCA(n_components=40, random_state=0).fit_transform(df.dropna().T)))
# d.index = df.columns

# for col in md.columns:
#     d.loc[md.index, col] = md[col]

# fig, ax = plt.subplots(figsize=[20,20])

# g = sns.scatterplot(data=d, x=0, y=1, 
#                     hue='NMF.Cluster',
#                     s=2000,
#                     alpha=1,
#                     legend=False
#                     )


#%% Plot clustermap for top 5000 proteins by variance

results = pd.DataFrame(index = agg.index)

for grouping in md['NMF.Cluster'].unique():    
    scores = agg[md[md['NMF.Cluster'] == grouping].index]
    meanscores = scores.mean(axis=1).sort_values(ascending=False)
    results.loc[meanscores.index, grouping] = meanscores

results = results.loc[results.var(axis=1).sort_values(ascending=False).index]

data = results[:5000].T
data.index = [i.split('-')[0] for i in data.index]


unobserved_cptac = data.columns.difference(df.index)


g = sns.clustermap(data=data, figsize=[25,6],
                   cbar_kws=dict(orientation='horizontal'),
                   cbar_pos=[.65,-0.05,.225,.04],
                   vmax=2, vmin=-2,
                   dendrogram_ratio=[.1,.2],
                   cmap='coolwarm', xticklabels=False)

cax = g.cax
cax.yaxis.set_ticks_position('left')
cax.tick_params(size=12, labelsize=30)
cax.text(x=-.1,y=0.5,s='PROSE score', size=30, transform=cax.transAxes,  
         va='center', ha='right')

ax = g.ax_heatmap
ax.tick_params(axis='both', size=15, pad=10, labelsize=30)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

g.savefig(f'plots/fig7_krug_cptac2_top5000variance.png',
          bbox_inches='tight',
          dpi=600,)

#%%
#plot PROSE scores vs rank of unobserved proteins in original screen

from utils.convert_ids import uniprot2symbol
conv = uniprot2symbol()


cgc = pd.read_csv('databases/cancer_gene_census.csv')
cgc.fillna(0, inplace=True)
oncogenes = cgc[cgc['Role in Cancer'].str.contains('oncogene').fillna(False)]['Gene Symbol']
tsgs = cgc[cgc['Role in Cancer'].str.contains('TSG').fillna(False)]['Gene Symbol']
cgs = cgc['Gene Symbol']

unobserved_data = data.T.loc[unobserved_cptac]
unobserved_data.index = [conv[i] if i in conv else np.nan for i in unobserved_data.index]
unobserved_data = unobserved_data[~unobserved_data.index.isin(id_symbols)]

unobserved_basal = unobserved_data[['Basal']].sort_values(by='Basal',ascending=False)
unobserved_basal['rank'] = unobserved_basal.rank()
unobserved_basal['s'] = [550 if i in oncogenes.values else 20 for i in unobserved_basal.index]
# unobserved_basal['s'] = [550 if i in tsgs.values else 20 for i in unobserved_basal.index]
unobserved_basal['cgs'] = [1 if i in cgs.values else 0 for i in unobserved_basal.index]


fig, ax = plt.subplots(figsize=[30,10])
g=sns.scatterplot(data=unobserved_basal, x='rank', y='Basal',
                  s=unobserved_basal['s'], linewidth=0, color='black')
plt.xlabel('Ranked proteins\n(originally unobserved in screen)', size=50)
plt.ylabel('PROSE score,\nNMF-annotated Basal', labelpad=40, size=50)

g.axhline(0, linestyle='--', lw=4, color='black')

sns.despine()

j = 0; ytex=4; xtex = 2000
for i, row in unobserved_basal[unobserved_basal['s'] >= 100].iterrows(): 
    x, y = row['rank'], row['Basal']
    ytex -= 0.54
    ax.annotate(xy=(x,y), xytext=(xtex,ytex), text=row.name, size=40, ha='left', color='darkorange',
                arrowprops=dict(facecolor='black', arrowstyle='-',relpos=[0,.5], color='darkorange'))
    j += 1
    if j >= 15:
        break

plt.tight_layout()


j = 0; ytex=4; xtex = 1000
for i, row in unobserved_basal[unobserved_basal['cgs'] == 0].iterrows(): 
    x, y = row['rank'], row['Basal']
    ytex -= 0.54
    ax.annotate(xy=(x,y), xytext=(xtex,ytex), text=row.name, size=40, ha='left', color='cornflowerblue',
                arrowprops=dict(arrowstyle='-',relpos=[1,.5], color='cornflowerblue'))
    j += 1
    if j >= 5:
        break

plt.tight_layout()

#%% Plot correlation map

lut = dict(zip(md['NMF.Cluster'].unique(), sns.color_palette('tab10',4)))
row_colors = md['NMF.Cluster'].map(lut)
row_colors.name = None

g = sns.clustermap(agg.corr(), figsize=[11,10],
                   cbar_kws=dict(orientation='horizontal'),
                   cbar_pos=[.65,-0.05,.225,.04],
                   dendrogram_ratio=[.1,0],
                   row_colors = row_colors,
                   col_colors = row_colors,
                   cmap='coolwarm',xticklabels=False,yticklabels=False)

cax = g.cax
cax.yaxis.set_ticks_position('left')
cax.tick_params(size=12, labelsize=30)
cax.text(x=-.1,y=0.5,s='Linear correlation (r)\n by PROSE score',
         size=30, transform=cax.transAxes,  
         va='center', ha='right')

#%% GSEA analysis of BRCA subtypes

import gseapy

for subtype in data.index:
    rnk = data.loc[subtype].sort_values(ascending=False)
    rnk.index = [conv[i] if i in conv else np.nan for i in rnk.index]
    rnk = rnk[~rnk.index.isna()]
    
    gseapy.prerank(rnk, gene_sets='MSigDB_Hallmark_2020',
                   no_plot=True,outdir=f'GSEA_Prerank/PROSE/{subtype}')
    


#%%

agg_nes = pd.DataFrame()

for subtype in data.index:
    subtype_nes = pd.read_csv(f'GSEA_Prerank/PROSE/{subtype}/gseapy.prerank.gene_sets.report.csv',
                              index_col=0)
    subtype_nes = subtype_nes[['nes']]
    
    for i, row in subtype_nes.iterrows():
        if row.name not in agg_nes.index:
            agg_nes.loc[row.name, subtype] = row.nes
        else:
            agg_nes.loc[row.name, subtype] = row.nes
            
g = sns.clustermap(data = agg_nes.T,
                   figsize=[40,15],
                   cbar_kws=dict(orientation='horizontal'),
                   cbar_pos=[.65,.1,.225,.04],
                   dendrogram_ratio=[.1,.3],
                   tree_kws=dict(linewidth=4),
                   cmap='rocket',xticklabels=True,yticklabels=True)

cax = g.cax
cax.yaxis.set_ticks_position('left')
cax.tick_params(size=12, labelsize=50)
cax.text(x=-.1,y=0.5,s='NES',
         size=60, transform=cax.transAxes,  
         va='center', ha='right')

ax = g.ax_heatmap
ax.tick_params(axis='x', size=15, pad=5, labelsize=30)
ax.tick_params(axis='y', size=15, pad=10, labelsize=50)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='left',)




