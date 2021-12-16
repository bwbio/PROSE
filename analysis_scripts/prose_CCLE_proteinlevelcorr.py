# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 15:43:52 2021

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


tpm = pd.read_csv('ccle/ccle_tpm_formatted.tsv.gz', sep='\t').drop_duplicates('cell_line')
tmt = pd.read_csv('ccle/ccle_tmt_formatted_withMissingVals.tsv.gz', sep='\t').drop_duplicates('cell_line')
score = pd.read_csv('ccle/ccle_prose_formatted.tsv.gz', sep='\t').drop_duplicates('cell_line')
modules = pd.read_csv('fastica_files/fastica_report_prose_6.tsv', sep='\t')

#%%

tmtdf = tmt.dropna(axis=1)
tmtdf = tmtdf.set_index('cell_line')
tmtdf = tmtdf.drop(columns=['tissue'])
scoredf = score.set_index('cell_line')
tpmdf = tpm.set_index('cell_line')

common_cl = tmtdf.index.intersection(scoredf.index).intersection(tpmdf.index)
common_prot = tmtdf.columns.intersection(scoredf.columns).intersection(tpmdf.columns)

tmtdf = tmtdf.loc[common_cl][common_prot].T
scoredf = scoredf.loc[common_cl][common_prot].T
tpmdf = tpmdf.loc[common_cl][common_prot].T


df_pp = []
for protein in tmtdf.index:
    rho_tmt = scipy.stats.spearmanr(tmtdf.loc[protein],scoredf.loc[protein])[0]
    rho_tpm = scipy.stats.spearmanr(tpmdf.loc[protein],scoredf.loc[protein])[0]
    rho_nonprose = scipy.stats.spearmanr(tpmdf.loc[protein],tmtdf.loc[protein])[0]
    df_pp.append([protein, rho_tmt, rho_tpm, rho_nonprose])
df_pp = pd.DataFrame(df_pp, columns=['protein','PROSE : TMT',
                               'PROSE : TPM',
                               'TPM : TMT'])
df_pp = df_pp.sort_values(by='PROSE : TMT',ascending=False).reset_index(drop=True)
df_pp_melt = pd.melt(df_pp, id_vars='protein',value_vars=['PROSE : TMT',
                                                    'PROSE : TPM',
                                                    'TPM : TMT'])

df_pp.to_csv('source_data/Fig 2a (CCLE protein correlation KDE).csv')


tmtdf = tmtdf.T
scoredf = scoredf.T
tpmdf = tpmdf.T

df_cc = []
for cl in tmtdf.index:
    rho_tmt = scipy.stats.spearmanr(tmtdf.loc[cl],scoredf.loc[cl])[0]
    rho_tpm = scipy.stats.spearmanr(tpmdf.loc[cl],scoredf.loc[cl])[0]
    rho_nonprose = scipy.stats.spearmanr(tpmdf.loc[cl],tmtdf.loc[cl])[0]
    df_cc.append([cl, rho_tmt, rho_tpm, rho_nonprose])
    
df_cc = pd.DataFrame(df_cc, columns=['protein','PROSE : TMT',
                               'PROSE : TPM',
                               'TPM : TMT'])
df_cc = df_cc.sort_values(by='PROSE : TMT',ascending=False).reset_index(drop=True)
df_cc_melt = pd.melt(df_cc, id_vars='protein',value_vars=['PROSE : TMT',
                                                    'PROSE : TPM',
                                                    'TPM : TMT'])

df_cc.to_csv('source_data/Fig 2b (CCLE cell line correlation KDE).csv')


#%%

fig, axes = plt.subplots(figsize=[9,8],nrows=3,ncols=1,sharex=True)

cats = df_pp_melt.variable.unique()
for i in range(len(cats)):
    ax = axes[i]
    group = cats[i]
    subdf = df_pp_melt[df_pp_melt.variable == group]
    
    sns.kdeplot(data=subdf, x='value',ax=ax,
                color='#d4edff',lw=0.3,fill=True,alpha=0.4,bw_adjust=0.5,)
    g=sns.kdeplot(data=subdf, x='value',ax=ax,
                color='black',lw=0.7,fill=False,alpha=0.4,bw_adjust=0.5,)
    
    g.axvline(subdf.value.mean(), 0, 1, color='black', alpha=0.5)
    print(subdf.value.mean())
    
    ax.set_xlim(-0.3,1.1)
    ax.set_ylabel('')
    ax.set_yticks([])    
    ax.text(s=group,x=0,y=0.5,transform=ax.transAxes,size=28, ha='left',weight='bold')


    ax.tick_params(axis = 'both', which = 'major', labelsize = 25)

    if i == 0:
        ax.text(s='$n$ = {} proteins'.format(len(common_prot)),
                x=1.05,y=0.8,transform=ax.transAxes, ha='right',
                size=25)

    if i == 2:        
        ax.set_xlabel('Protein self-correlation ($ρ$)',
                      size=28, labelpad=20)
    
sns.despine(left=True)
plt.subplots_adjust(hspace=0.2)

plt.savefig('plots/CCLE_correlation_proteins_histKDE.png',format='png', 
            dpi=600, bbox_inches='tight') 

#%%

fig, axes = plt.subplots(figsize=[9,8],nrows=3,ncols=1,sharex=True)

cats = df_cc_melt.variable.unique()
for i in range(len(cats)):
    ax = axes[i]
    group = cats[i]
    subdf = df_cc_melt[df_cc_melt.variable == group]
    
    sns.kdeplot(data=subdf, x='value',ax=ax,
                color='#d4edff',lw=0.3,fill=True,alpha=0.4,bw_adjust=0.5,)
    g=sns.kdeplot(data=subdf, x='value',ax=ax,
                color='black',lw=0.7,fill=False,alpha=0.4,bw_adjust=0.5,)
    
    g.axvline(subdf.value.mean(), 0, 1, color='black', alpha=0.5)
    print(subdf.value.mean())
    
    ax.set_xlim(-0.1,0.7)
    ax.set_ylabel('')
    ax.set_yticks([])    

    ax.tick_params(axis = 'both', which = 'major', labelsize = 25)
    
    if i == 0:
        ax.text(s='$n$ = {} cell lines'.format(len(common_cl)),
                x=1,y=0.8,transform=ax.transAxes, ha='right',
                size=25)
        
    if i == 2:        
        ax.set_xlabel('Cell line self-correlation ($ρ$)',
                      size=28, labelpad=20)
sns.despine(left=True)
plt.subplots_adjust(hspace=0.2)

plt.savefig('plots/CCLE_correlation_cellLines_histKDE.png',format='png', 
            dpi=600, bbox_inches='tight') 



#%%

fig, ax = plt.subplots(figsize=[20,10])
protein=df.iloc[12].protein
sns.scatterplot(scoredf.loc[protein],tpmdf.loc[protein],
                s=200)
plt.show()