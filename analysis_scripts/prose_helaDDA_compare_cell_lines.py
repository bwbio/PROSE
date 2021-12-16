# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 18:38:52 2021

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

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%% Dictionary for gene-to-protein ID conversion

conv=pd.read_csv('databases/ensembl_uniprot_conversion.tsv',
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

#%% Restrict to testable genes with TPM max > 1

df_max = df.max()
df_testable = df[df_max.loc[df_max > 1].index]

#%% Get HeLa DDA protein lists

with open('interim_files/HeLa_DDA_sample.pkl', 'rb') as handle:
    testdata = pickle.load(handle)
    
panel_corr = pd.read_csv('interim_files/klijn_panel_spearmanCorr.tsv', sep='\t',index_col=0)


#%%
obs = testdata['HeLa R1']['two peptide']
unobs = testdata['HeLa R1']['no evidence']
q = pgx.prose(obs, unobs, panel_corr, holdout=False)
data = q.summary

#%% Diagnostic plot (HeLa TPM correlates)
result = []
initial = False
palette = ['#464646', '#3DB2FF', '#FF7F0E'] 

lines = 'HeLa', 'EJM'

fig, axes = plt.subplots(nrows=3, ncols=1,
          figsize=[17.5,30],
          gridspec_kw={'height_ratios': [1.5, 3, 3]})

for i, row in df_testable.iterrows():
    
    meta_row = metacols_df.iloc[i]
    tissue = meta_row['tissue']
    cell_line = meta_row['cell line']
    tpm = row[data.protein]
    rho = scipy.stats.spearmanr(tpm, data.score)[0]
    r = scipy.stats.pearsonr(tpm, data.score)[0]
    result.append([tissue,rho,r,cell_line])
  
    if cell_line in lines:
        data['tpm'] = np.log2(tpm+1).values
        data = data.dropna()                
        if initial == False:
            ax=axes[0]
            g1=sns.kdeplot(data=data,x='score_norm',hue='y_true',common_norm = False,
                           lw=4, palette=palette,
                           ax=ax)
            ax.get_legend().remove()
            ax.set_xlabel('')
            ax.set_xticks([])
            initial = True

        if cell_line == lines[0]:
            ax=axes[1]
            g2=sns.scatterplot(data=data, x='score_norm', y='tpm', hue='y_true',
                            alpha=.8, s=50, palette=palette,lw=0.001,
                            ax=ax)
            ax.set_ylabel(r'log$_{2}$(TPM +1)',labelpad=15)
            ax.set_xlabel('')
            ax.set_xticks([])
            ax.get_legend().remove()
            ax.text(.98, .98, cell_line+r', $ρ$ = '+str(round(rho,3)),
                 horizontalalignment='right',
                 verticalalignment='top',
                 size=50,
                 transform = ax.transAxes)
                        
        if cell_line == lines[1]:
            ax=axes[2]
            g3=sns.scatterplot(data = data, x = 'score_norm', y = 'tpm', hue = 'y_true',
                            alpha=.8,s=50,palette=palette,lw=0.001,
                            ax=ax)
            ax.set_ylabel(r'log$_{2}$(TPM +1)',labelpad=15)
            ax.set_xlabel('PROSE Score')
            ax.get_legend().remove()
            ax.text(.98, .98, cell_line+r', $ρ$ = '+str(round(rho,3)),
                 horizontalalignment='right',
                 verticalalignment='top',
                 size=50,
                 transform = ax.transAxes)
            
        data.to_csv('source_data/Fig S3a ({} PROSE-TPM scatterplot).csv'.format(cell_line))
            
plt.subplots_adjust(hspace=0.1)
plt.savefig('plots/HeLaR1_TPM_SVM.png',
    format='png', dpi=600, bbox_inches='tight') 
plt.show() 
    
result = pd.DataFrame(result, columns = ['tissue','rho','r','cell_line'])

hela_r = result[result.cell_line == 'HeLa'].rho.values[0]
k562_r = result[result.cell_line == 'K-562'].rho.values[0]

tissue_ms = {}
for tissue in result.tissue.unique():
    m = result[result.tissue == tissue]['rho'].mean()
    tissue_ms[tissue] = m

result['tissue_mean'] = result.apply(lambda x: tissue_ms[x.tissue], axis=1)    
result=result.sort_values(by='tissue_mean',ascending=False)

#%% Diagnostic plot (tissue specificity of HeLa R1 predictions)

fig, ax = plt.subplots(figsize=[10, 25])
g=sns.boxplot(data=result, x='rho', y ='tissue',showfliers=False)
g2=sns.stripplot(data=result, x='rho', y='tissue',size=6,color='black',alpha=0.9)
g.axvline(hela_r, lw=3,linestyle='--',color='black')
plt.text(hela_r,-1,'HeLa', ha='center',size=50)
plt.grid()
plt.ylabel('')
plt.xlabel(r'$ρ_{\rm\ TPM, score}$',labelpad=15,size=55)

oro_cloacal = ['chordate pharynx',
               'urinary bladder',
               'rectum',
               'esophagus',
               'oral cavity']

for i in ax.get_yticklabels():
    if i._text in oro_cloacal:
        i.set_weight('bold')
    if i._text == 'uterine cervix':
        i.set_weight('bold'); i.set_color("red")


plt.savefig('plots/HeLaR1_SVM_tissueSpecificity.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show() 

result.drop(columns=['r']).to_csv('source_data/Fig S3b (HeLa tissue-specific boxplot).csv',index=False)