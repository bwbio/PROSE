# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 13:47:55 2021

@author: bw98j
"""

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


#%% Generate stability plots for k = 100 and 1000 estimators

for k in [100,1000]:
    result = {}
    for i in testdata.keys():
        obs = testdata[i]['two peptide']
        unobs = testdata[i]['no evidence']
        q = pgx.prose(obs, unobs, panel_corr, bag_kwargs = {'n_estimators':k})
        data = q.summary
        result[i] = data

        print(i, k, 'Done A')
    
    indep_result = {}
    for i in testdata.keys():
        obs = testdata[i]['two peptide']
        unobs = testdata[i]['no evidence']
        q = pgx.prose(obs, unobs, panel_corr, bag_kwargs = {'n_estimators':k})
        data = q.summary
        indep_result[i] = data    

        print(i, k, 'Done B')
        

    fig, axes = plt.subplots(nrows=4, ncols=4, figsize=[25,25])
    
    for r1, r2 in itertools.combinations(result.keys(),2):                     
        
        ax = axes[int(r1.split(' R')[-1])-1,int(r2.split(' R')[-1])-1]
        g = sns.scatterplot(x=result[r1].score_norm,y=result[r2].score_norm,ax=ax)  
        x = np.linspace(-3.4,3.4); y=x
        
        r=scipy.stats.pearsonr(result[r1].score_norm,result[r2].score_norm)[0]
        rho=scipy.stats.spearmanr(result[r1].score_norm,result[r2].score_norm)[0]
        
        g2 = sns.lineplot(x=x,y=y, lw=5,ax=ax,color='black',linestyle = ':')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
    
        ax.text(.02, .98, r'$ρ$ = '+str(round(rho,4))+'\n'+
                r'$r$ =  '+str(round(r,4)),
                horizontalalignment='left',
                verticalalignment='top',
                size=30,
                transform = ax.transAxes)
        
        g.axvline(0,color='grey',lw=2,linestyle = '--',alpha=0.5)
        g.axhline(0,color='grey',lw=2,linestyle = '--',alpha=0.5)
    
        if r1 == 'HeLa R1':
            ax.set_xlabel(r2)
            ax.xaxis.set_label_position('top')
        else:
            ax.set_title('')
    
        
        if r2 == 'HeLa R4':
            ax.set_ylabel(r1, rotation=270,labelpad=40)
            ax.yaxis.set_label_position('right')
        else: 
            ax.set_ylabel('')

        
        print(r1, r2, k, 'Done comparison')
        data=pd.DataFrame(zip(result[r1].protein,result[r1].score_norm,result[r2].score_norm),
                              columns = ['protein','score_norm_1','score_norm_2'])
        data.to_csv('source_data/Fig S1 ({}, {}, k{} stability comparison).csv'.format(r1,r2,k),
                    index=False)
        
        
    
    for i in result.keys():
        ax = axes[int(i.split(' R')[-1])-1,int(i.split(' R')[-1])-1]
        g = sns.scatterplot(x=result[i].score_norm,y=indep_result[i].score_norm,ax=ax)
        r=scipy.stats.pearsonr(result[i].score_norm,indep_result[i].score_norm)[0]
        rho=scipy.stats.spearmanr(result[i].score_norm,indep_result[i].score_norm)[0]
        g2 = sns.lineplot(x=x,y=y, lw=5,ax=ax,color='black',linestyle = ':')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
    
        ax.text(.02, .98, r'$ρ$ = '+str(round(rho,4))+'\n'+
                r'$r$ =  '+str(round(r,4)),
                horizontalalignment='left',
                verticalalignment='top',
                size=30,
                transform = ax.transAxes)
        
        g.axvline(0,color='grey',lw=2,linestyle = '--',alpha=0.5)
        g.axhline(0,color='grey',lw=2,linestyle = '--',alpha=0.5)
    
    
        if i == 'HeLa R1':
            ax.set_xlabel(i)
            ax.xaxis.set_label_position('top')
        else:
            ax.set_title('')
            
        if i == 'HeLa R4':
            ax.set_ylabel(i, rotation=270,labelpad=40)
            ax.yaxis.set_label_position('right')
        else: 
            ax.set_ylabel('')
            
        
    for i in (4,8,9,12,13,14):
        fig.delaxes(axes.flatten()[i])
    
    
    plt.subplots_adjust(hspace=0.1,wspace=0.1)
    plt.savefig('plots/HeLaR1_crossrep_stability_k'+str(k)+'.png',
                format='png', dpi=600, bbox_inches='tight') 
    plt.show() 

