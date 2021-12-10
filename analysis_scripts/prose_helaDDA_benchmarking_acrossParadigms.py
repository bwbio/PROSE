# -*- coding: utf-8 -*-
"""
Created on Fri Oct 29 09:20:13 2021

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
import pickle
from sklearn.decomposition import PCA
import umap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from operator import itemgetter

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%%

conv=pd.read_csv('databases/ensembl_uniprot_conversion.tsv',
               sep='\t',
               comment='#',
               )

conv = conv.rename(columns={'ID':'gene',
                            'Entry': 'uniprot'})
conv = conv[['gene','uniprot']]
conv = dict(zip(conv.gene,conv.uniprot))
validGenes = conv.keys() #set of genes with associated protein names

tpm = pd.read_csv('klijn_rna_seq/E-MTAB-2706-query-results.tpms.tsv', sep='\t',comment='#')
tpm.columns = [i.split(', ')[-1] for i in tpm.columns]
tpm = tpm.fillna(0)

tpm['protein'] = tpm.apply(lambda x: conv[x['Gene ID']] if x['Gene ID'] in validGenes else np.nan, axis=1)
tpm = tpm.dropna()
hela = tpm[['HeLa','protein']].set_index('protein')


#%%
ibaq = pd.read_csv('klijn_rna_seq/bekker_jensen_2017_ibaq_s3_mmc4.csv', skiprows=2)
ibaq = ibaq[['Protein IDs','Median HeLa iBAQ']]


ibaq['Protein IDs'] = ibaq.apply(lambda x: list(set([i.split('-')[0] for i in x['Protein IDs'].split(';')])),axis=1)
ibaq['matches'] = ibaq.apply(lambda x: len(x['Protein IDs']),axis=1)
ibaq = ibaq[ibaq.matches == 1]
ibaq['Protein IDs'] = ibaq.apply(lambda x: x[0][0], axis=1)
ibaq = ibaq.set_index('Protein IDs').drop(columns=['matches'])
ibaq = ibaq.dropna().drop_duplicates()
ibaq = np.log10(ibaq)
ibaq = ibaq[~ibaq.index.duplicated(keep='first')]

#%% Get HeLa DDA protein lists

with open('interim_files/HeLa_DDA_sample.pkl', 'rb') as handle:
    testdata = pickle.load(handle)

#%%

panel_corr = pd.read_csv('interim_files/klijn_panel_spearmanCorr.tsv', sep='\t',index_col=0)
panel_corr_scaled = pd.DataFrame(StandardScaler().fit_transform(panel_corr),
                                 columns = panel_corr.columns,
                                 index = panel_corr.index)

#%% Generate PCA and UMAP projections of panel_corr
pca = PCA(n_components=2)
pca.fit(panel_corr_scaled.T)
df_pca = pd.DataFrame(pca.components_.T, index = panel_corr_scaled.index, columns = ['PC1', 'PC2'])
df_pca.PC1 = df_pca.apply(lambda x: x.PC1*pca.explained_variance_ratio_[0],axis=1)
df_pca.PC2 = df_pca.apply(lambda x: x.PC2*pca.explained_variance_ratio_[1],axis=1)

reducer = umap.UMAP(min_dist=0, random_state=42)
u = reducer.fit_transform(panel_corr_scaled)
df_umap = pd.DataFrame(u, index = panel_corr_scaled.index, columns = ['UMAP-1', 'UMAP-2'])

#%% define KNN for prediction

class knn:
    def __init__(self, obs, unobs, df):
        df['y'] = df.apply(lambda row: pgx.labeller(row, obs, unobs),axis=1)
        subset = df[df.y != -1]
        split = lambda x:  train_test_split(x.drop(columns=['y']),
                                            x.y,test_size=200,
                                            stratify=x.y)
    
        X_train, X_test, Y_train, Y_test = split(subset)
        X_train, Y_train = RandomUnderSampler().fit_resample(X_train, Y_train)
        knn = KNeighborsClassifier()
        k_range = list(range(5,51))
        grid = GridSearchCV(knn, dict(n_neighbors=k_range), cv=5, scoring='accuracy')
        grid.fit(X_train, Y_train)
        knn=grid.best_estimator_
        print('KNN: Optimal k = ', grid.best_params_)
        
        model = knn.fit(X_train, Y_train)
        
        Y_pred = knn.predict(X_test)
        Y_scores = knn.predict_proba(X_test).T[1]
        Y_self = knn.predict(X_train)
        Y_self_scores = knn.predict_proba(X_train).T[1]
        self.f1 = round(f1_score(Y_test,Y_pred),4)
        self.f1_tr = round(f1_score(Y_train,Y_self),4)
        
        self.fpr, self.tpr, thresholds = roc_curve(Y_test, Y_scores, pos_label = 1)
        self.fpr_tr, self.tpr_tr, thresholds_tr = roc_curve(Y_train, Y_self_scores, pos_label = 1)
        
        self.auc = round(auc(self.fpr,self.tpr),3)
        self.auc_tr = round(auc(self.fpr_tr,self.tpr_tr),3)
        
        tested_proteins = np.array(df.index.to_list())
        probs = knn.predict_proba(df.drop(columns='y'))
        score = [i[1] for i in probs]
        score_norm = scipy.stats.zscore(score)
        
        self.summary = pd.DataFrame(zip(tested_proteins,
                                        score,
                                        score_norm,
                                        ),
                            
                                    columns = ['protein',         
                                               'score',
                                               'score_norm',
                                               ],
                            )

from pathlib import Path
if Path('source_data/benchmark_HeLa.tsv').is_file():
    data = pd.read_csv('source_data/benchmark_HeLa.tsv',sep='\t')   
else:  
    result = []
    for i in range(10):
        #random generation of sets for each simulation
        sets = dict() 
        for rep in testdata.keys():
            if rep not in sets.keys():
                sets[rep] = dict()
            import random
            obs, unobs = list(testdata[rep]['two peptide']), list(testdata[rep]['no evidence']) 
            sets[rep]['baseline'] = obs, unobs
            sets[rep]['downsamp_1k'] = random.sample(obs,1000), random.sample(unobs,1000)
            sets[rep]['downsamp_2k'] = random.sample(obs,2000), random.sample(unobs,2000)
            
            for dropout in [100,200,500,1000]:
                drop = random.sample(obs, dropout)
                sets[rep]['dropout_'+str(dropout)] = [i for i in obs if (i not in drop)], unobs
        
            mix = random.sample(obs,2000) + random.sample(unobs,2000)
            unobs_rand = random.sample(mix, 2000)
            sets[rep]['random'] = [i for i in mix if (i not in unobs_rand)], unobs_rand    
        reps = testdata.keys()    
        groupings = list(sets[rep].keys())
        
        for rep in reps:
            for group in groupings:
                print(i, rep, group)
                obs, unobs = sets[rep][group]
                container ={'knn_umap':knn(obs,unobs,df_umap),
                            'knn_pca':knn(obs,unobs,df_pca),
                            'prose':pgx.prose(obs,unobs,panel_corr,)} 
                for method in container.keys():
                    q = container[method]
                    score = q.summary[['protein','score_norm']].set_index('protein')
                    common_prot_tpm = scores.index.intersection(hela.index)
                    common_prot_ibaq = scores.index.intersection(ibaq.index)
                    
                    rhotpm = scipy.stats.spearmanr(score.loc[common_prot_tpm],
                                                   hela.loc[common_prot_tpm])[0]
                    rhoibaq = scipy.stats.spearmanr(score.loc[common_prot_ibaq],
                                                     ibaq.loc[common_prot_ibaq])[0]
                    
                    print(rhotpm, rhoibaq)
                    result.append([rep, group, i, method,
                                   q.f1, q.f1_tr,
                                   q.auc, q.auc_tr,
                                   rhotpm,rhoibaq])
    
    data = pd.DataFrame(result)
    data.columns = ['rep', 'group', 'i', 'method',
                    'f1', 'f1_tr',
                    'auc', 'auc_tr',
                    'rho_tpm','rho_ibaq']
    data.to_csv('source_data/Fig 1c,d (Benchmarking on HeLa DDA).tsv',sep='\t',index=False)

#%%

nrows = len(data.group.unique())
ncolumns = len(data.method.unique())
auc_df=pd.DataFrame()
auc_tr_df=pd.DataFrame()
f1_df=pd.DataFrame()
f1_tr_df=pd.DataFrame()
rho_tpm_df=pd.DataFrame()
rho_ibaq_df=pd.DataFrame()


renamedict = {'baseline':'Baseline',
             'downsamp_1k': '1,000',
             'downsamp_2k': '2,000',
             'dropout_10': '10',
             'dropout_50': '50',
             'dropout_100': '100',       
             'dropout_200': '200',
             'dropout_500': '500',
             'dropout_1000': ' 1,000',
             'random':'Random'}

for i in data.group.unique():
    for j in data.method.unique():
        subdf = data[data.group == i][data.method == j]
        
        i0=renamedict[i]

        auc_df.loc[i0,j] = subdf.auc.mean()
        auc_tr_df.loc[i0,j] = subdf.auc_tr.mean()
        f1_df.loc[i0,j] = subdf.f1.mean()
        f1_tr_df.loc[i0,j] = subdf.f1_tr.mean()
        rho_tpm_df.loc[i0,j] = subdf.rho_tpm.mean()
        rho_ibaq_df.loc[i0,j] = subdf.rho_ibaq.mean()
#%%

fig, axes = plt.subplots(nrows=1,ncols=8,figsize=[29, 10],
                         gridspec_kw=dict(width_ratios=[2,2,2,2,2,0.5,2,2]),
                         )
cmap = sns.diverging_palette(9, 255, as_cmap=True)
cbar_kws={"orientation": "horizontal"} 

cmap='mako_r'
vmin, vmax, center = 0.25, 1, 0.8
cbar_ax = fig.add_axes([0.35, 0.05, 0.18, 0.03])

cmap1='rocket_r'
vmin1, vmax1, center1 = -0.1, 1, 0.5
cbar_ax1 = fig.add_axes([0.708, 0.05, 0.18, 0.03])

f = lambda df:[list(df.values[0].round(2).astype(str))] +\
    [['','',df.values[i+1][2].round(2).astype(str)] for i in range((len(df)-1))]

ax=axes[0]
g=sns.scatterplot();sns.despine(left=True,bottom=True)
ax.set_xticks([]); ax.set_yticks([])
ax.axvline(0.5,0.16,0.59,clip_on=False,lw=3,color='black')
ax.axvline(0.5,0.66,0.84,clip_on=False,lw=3,color='black')

ax=axes[1]
g=sns.heatmap(auc_df,xticklabels=True,
              cmap=cmap,vmax=vmax,vmin=vmin,center=center,
              square=True, linewidths=2.5,ax=ax,cbar_ax=cbar_ax1,
              cbar_kws=cbar_kws,
              annot=f(auc_df),fmt = '',annot_kws={"size": 30})
ax.set_title('AUC, test', size=40)
ax.set_xticklabels(['UMAP-KNN','PCA-KNN','PROSE'],rotation=45,ha='right',size=35)
ax.text(y=0.73,x=-1.8,ha='left',s='Downsample',transform=ax.transAxes)
ax.text(y=0.35,x=-1.4,ha='left',s='Dropout',transform=ax.transAxes)



ax=axes[2]
g=sns.heatmap(auc_tr_df,xticklabels=False,yticklabels=False,
              cmap=cmap,vmax=vmax,vmin=vmin,center=center,
              square=True, linewidths=2.5,ax=ax,cbar_ax=cbar_ax,
              cbar_kws=cbar_kws,
              annot=f(auc_tr_df),fmt = '',annot_kws={"size": 30})
ax.set_title('AUC, train', size=40)

ax=axes[3]
g=sns.heatmap(f1_df,xticklabels=False,yticklabels=False,
              cmap=cmap,vmax=vmax,vmin=vmin,center=center,
              square=True, linewidths=2.5,ax=ax,cbar_ax=cbar_ax,
              cbar_kws=cbar_kws,
              annot=f(f1_df),fmt = '',annot_kws={"size": 30})
ax.set_title('F1, test', size=40)

ax=axes[4]
g=sns.heatmap(f1_tr_df,xticklabels=False,yticklabels=False,
              cmap=cmap,vmax=vmax,vmin=vmin,center=center,
              square=True, linewidths=2.5,ax=ax,cbar_ax=cbar_ax,
              cbar_kws=cbar_kws,
              annot=f(f1_tr_df),fmt = '',annot_kws={"size": 30})
ax.set_title('F1, train', size=40)

ax=axes[5]
g=sns.scatterplot();#sns.despine(left=True,bottom=True)
ax.set_xticks([]); ax.set_yticks([])


ax=axes[6]
g=sns.heatmap(rho_tpm_df,xticklabels=False,yticklabels=False,
              cmap=cmap1,vmax=vmax1,vmin=vmin1,center=center1,
              square=True, linewidths=2.5,ax=ax,cbar_ax=cbar_ax1,
              cbar_kws=cbar_kws,
              annot=f(rho_tpm_df),fmt = '',annot_kws={"size": 30})
ax.set_title('Gene expr.\n(TPM)', size=35,pad=10)


ax=axes[7]
g=sns.heatmap(rho_ibaq_df,xticklabels=False,yticklabels=False,
              cmap=cmap1,vmax=vmax1,vmin=vmin1,center=center1,
              square=True, linewidths=2.5,ax=ax,cbar_ax=cbar_ax1,
              cbar_kws=cbar_kws,
              annot=f(rho_ibaq_df),fmt = '',annot_kws={"size": 30})
ax.set_title('Protein expr.\n(iBAQ)', size=35,pad=10)
plt.subplots_adjust(wspace = 0.1)

ax.axhline(-1.3,-0.9,0.84,clip_on=False,lw=3,color='black')
ax.text(y=1.2,x=0,ha='center',s=r'Correlation ($ρ$) with:',transform=ax.transAxes)

plt.savefig('plots/HeLa_bench_summary.png',
            format='png', dpi=600, bbox_inches='tight')
