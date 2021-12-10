# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 18:38:52 2021

@author: Bertrand Jern Han Wong
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import umap
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

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%% Dictionary for gene-to-protein ID conversion

conv=pd.read_csv("databases/ensembl_uniprot_conversion.tsv",
               sep='\t',
               comment='#',
               )

conv = conv.rename(columns={'ID':'gene',
                            'Entry': 'uniprot'})
conv = conv[['gene','uniprot']]
conv = dict(zip(conv.gene,conv.uniprot))
validGenes = conv.keys() #set of genes with associated protein names

#%% Restrict to genes with corresponding UniProt protein IDs

#Load Klijn et al. (2015) RNA-seq dataset
df=pd.read_csv("klijn_rna_seq/E-MTAB-2706-query-results.tpms.tsv",
               sep='\t',
               comment='#',
               )

genes = list(df['Gene ID'].values)
df = df.drop(['Gene ID', 'Gene Name'], axis=1).T
df.reset_index(inplace=True)
df.columns = ['source']+genes
metacols = ['tissue','cancer','cell line']
df[metacols] = df.source.str.split(', ',n=2,expand=True)
metacols_df = df[metacols] #df containing cell line metadata

#restrict list of valid genes to those in the RNA-seq data
validGenes = list(df[genes].columns.intersection(validGenes))

#gene-to-protein ID conversion
df = df[validGenes]
df = df.fillna(0)
df = df.rename(columns=conv)

#%% Restrict to testable genes with TPM max > 10

df_max = df.max()
df_testable = df[df_max.loc[df_max > 10].index]

#%% Get correlation matrix

if os.path.isfile('interim_files/klijn_complete_spearmanCorr.tsv') == False:
    print('Corr matrix file not found... generating file...')
    df_corr = df_testable.corr(method='spearman')
    df_corr.to_csv('interim_files/klijn_complete_spearmanCorr.tsv', sep='\t')
    
else:
    print('Existing corr matrix found!')
    df_corr = pd.read_csv('interim_files/klijn_complete_spearmanCorr.tsv',sep='\t',
                          index_col=0)

#%% Define panel genes with desirable expression profiles

#Genes with median TPM > 1
df_med = df_testable.median()
df_med = df[df_med.loc[df_med > 1].index]+1

#log2(TPM+1) transformation
df_med = np.log2(df_med + 1)

#Get CV(log2(TPM+1)) and SD(corr) statistics
cv_tpm = (df_med.std()/df_med.mean()).rename('cv_tpm')
sd_spearman = (df_corr.std()[cv_tpm.index]).rename('sd_spearman')
joint_dispersion = pd.merge(cv_tpm, sd_spearman, on = cv_tpm.index).set_index('key_0')

#Critical quantiles for panel selection
tpm_crit = 0.75
sm_crit = 0.75
joint_dispersion['accept'] =\
joint_dispersion.apply(lambda x: 1 if (x.cv_tpm > cv_tpm.quantile(tpm_crit)\
                       and x.sd_spearman > sd_spearman.quantile(sm_crit))\
                       else 0 ,axis=1)

#Filter correlation matrix to only include panel genes
panel_df = joint_dispersion[joint_dispersion.accept == 1]
panel_proteins = set(panel_df.index)
panel_corr = df_corr[panel_proteins]
panel_tpm = df_med[panel_proteins]

if os.path.isfile('interim_files/klijn_panel_spearmanCorr.tsv') == False:
    print('Panel matrix file not found... generating file...')
    df_corr[panel_proteins].to_csv('interim_files/klijn_panel_spearmanCorr.tsv', sep='\t')

else:
    print('Panel matrix file already generated')

#%% Diagnostic plots (Panel selection)

#Scatterplot of panel gene expression profile
fig, ax = plt.subplots(figsize=[12,12])
g = sns.scatterplot(data=joint_dispersion,x='sd_spearman',y='cv_tpm',hue='accept',
                    alpha=0.5,s=200,palette=['gray','tomato'],lw=0)
plt.xlabel(r'SD($ρ_{i,j}$)',labelpad=15)
plt.ylabel(r'CV(log$_{2}$(TPM +1))',labelpad=15)
g.axhline(cv_tpm.quantile(tpm_crit), color='black',lw=4)
g.axvline(sd_spearman.quantile(sm_crit), color='black',lw=4)
plt.legend().remove()

plt.savefig('plots/panel_selection.png',
            format='png', dpi=600, bbox_inches='tight')
plt.show()

joint_dispersion.to_csv('source_data/Fig S7a (Panel selection, scatterplot).csv')

#%% Diagnostic plots (UMAP)

#UMAP showing tissue clustering using only panel gene TPMs
reducer = umap.UMAP(n_neighbors=25, min_dist=0.95, random_state=20)
scaled_panel = StandardScaler().fit_transform(panel_tpm)
u = reducer.fit_transform(scaled_panel)

umap_df = pd.DataFrame(u)
umap_df.columns = ['UMAP-1','UMAP-2']
umap_df[metacols] = metacols_df

#Get palette for individual tissues
tissues = metacols_df['tissue']
cmap = cm.get_cmap('seismic', len(tissues.unique()))
lut = dict(zip(tissues.unique(), [cmap(i)[:3] for i in range(cmap.N)]))
row_colors = tissues.map(lut)
umap_palette = sns.color_palette("husl", len(tissues.unique()))

#Complete UMAP plot
fig, ax = plt.subplots(figsize=[30,30])
g=sns.scatterplot(data=umap_df, x='UMAP-1', y='UMAP-2', hue='tissue',
                  alpha=0.9, s=1350, palette=umap_palette)
plt.xlabel('UMAP-1',size=100,labelpad=20)
plt.ylabel('UMAP-2',size=100,labelpad=20)
ax.legend(markerscale=6).remove()
umap_xlim, umap_ylim = ax.get_xlim(), ax.get_ylim()
plt.xticks([]); plt.yticks([])
plt.savefig('plots/panel_tpm_umap.png',
            format='png', dpi=300, bbox_inches='tight')
plt.show()

umap_df.to_csv('source_data/Fig S7b (Panel selection, UMAP).csv',index=False)

#Individual tissue plots
fig, axes = plt.subplots(nrows = 6, 
                          ncols = int(ceil(tissues.nunique())/6)+1,
                          figsize=[32,32])

for i, ax in enumerate(axes.flat):
    
    if i < tissues.nunique():
        tissue = tissues.unique()[i]
        g0=sns.scatterplot(data=umap_df[umap_df.tissue!=tissue],
                            x='UMAP-1', y='UMAP-2',
                            alpha=0.05, s=80, color='gray',
                            ax=ax)
        g=sns.scatterplot(data=umap_df[umap_df.tissue==tissue],
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
    
plt.savefig('plots/panel_tpm_individualTissue_umap.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show() 

#%% Diagnostic plots (tissue-specific clustering)

#Get tissue colors
tissues = metacols_df['tissue'].rename()
lut = dict(zip(tissues.unique(), sns.color_palette("husl", tissues.nunique())))
row_colors = tissues.map(lut)

#Binarized panel_tpm 
panel_binary_tpm = panel_tpm.where(panel_tpm > 2**1, other = 0)
panel_binary_tpm = panel_binary_tpm.where(panel_binary_tpm == 0, other = 1)


#Normal log2 TPM clustermap
g=sns.clustermap(panel_tpm,
                 z_score=1,
                 cmap='viridis',
                 figsize=[30,15],
                 xticklabels=False,yticklabels=False,row_colors=row_colors)
g.cax.set_visible(False)

plt.savefig('plots/panel_tpm_clustering.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show() 

#Binarized log2 TPM clustermap
g=sns.clustermap(panel_binary_tpm,
                 cmap='viridis',
                 figsize=[30,15],
                 xticklabels=False,yticklabels=False,row_colors=row_colors)

g.cax.set_visible(False)
plt.savefig('plots/panel_binarizedTpm_clustering.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show()

panel_tpm.set_index(metacols_df['cell line']).to_csv(
    'source_data/Fig S7c (Tissue-specific panel TPM pattern).csv')

panel_binary_tpm.set_index(metacols_df['cell line']).to_csv(
    'source_data/Fig S7d (Tissue-specific panel TPM pattern, binarized).csv')


#%% Diagnostic plots (protein correlatability)

fig, ax = plt.subplots(figsize=[10,10])
g=sns.histplot(data=df_corr[panel_proteins].abs().sum(axis=1),bins=100,
               alpha=0.2,lw=0,kde=True)

plt.xlabel(r'Correlatability, $Σ_{j}$|$ρ_{i,j}$|',labelpad=15)



