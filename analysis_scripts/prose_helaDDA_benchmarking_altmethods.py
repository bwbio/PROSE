# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 22:30:28 2021

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

#%% Get HeLa DDA protein lists

with open('interim_files/HeLa_DDA_sample.pkl', 'rb') as handle:
    testdata = pickle.load(handle)
    
panel_corr = pd.read_csv('interim_files/klijn_panel_spearmanCorr.tsv', sep='\t',index_col=0)

#%% Generate PCA nd UMAP projections of panel_corr with observed/unobserved proteins

panel_corr_scaled = pd.DataFrame(StandardScaler().fit_transform(panel_corr),
                                 columns = panel_corr.columns,
                                 index = panel_corr.index)

obs = obs = testdata['HeLa R1']['two peptide']
unobs = testdata['HeLa R1']['no evidence']

k = panel_corr_scaled.apply(lambda x: pgx.labeller(x, obs, unobs),axis=1)
subset = panel_corr_scaled.loc[k[k!=-1].index,:]

pca = PCA(n_components=2)
pca.fit(panel_corr_scaled.T)
df_pca = pd.DataFrame(pca.components_.T, index = panel_corr_scaled.index, columns = ['PC1', 'PC2'])
df_pca['hue'] = k

reducer = umap.UMAP(min_dist=0, random_state=42)
u = reducer.fit_transform(panel_corr_scaled)
df_umap = pd.DataFrame(u, index = panel_corr_scaled.index, columns = ['UMAP-1', 'UMAP-2'])
df_umap['hue'] = k


#%% Plot PCA and UMAP of panel_corr with observed/unobserved proteins

palette = sns.color_palette(itemgetter(0,1)(sns.color_palette("tab10", 10)))


#PCA plot
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=[10,10], 
                         gridspec_kw=dict(width_ratios=[6,1],
                                          height_ratios=[1,6],
                                          ),
                         )

ax = axes[1][0]
g=sns.scatterplot(data=df_pca[df_pca.hue!=1],x='PC1',y='PC2',hue='hue',
                  palette=palette,legend=False,alpha=0.8,ax=ax,edgecolor=None,s=10)
ax.set_xlabel('PC1 ({}%)'.format(round(pca.explained_variance_ratio_[0],2)*100))
ax.set_ylabel('PC2 ({}%)'.format(round(pca.explained_variance_ratio_[1],3)*100))

ax = axes[0][0]
g=sns.kdeplot(data=df_pca[df_pca.hue!=1],x='PC1',hue='hue',
              palette=palette,legend=False,ax=ax,common_norm=False,lw=4)
ax.axis('off')

ax = axes[1][1]
g=sns.kdeplot(data=df_pca[df_pca.hue!=1],y='PC2',hue='hue',
              palette=palette,legend=False,ax=ax,common_norm=False,lw=4)
ax.axis('off')

axes[0][1].remove()
plt.subplots_adjust(hspace = 0, wspace = 0)

plt.savefig('plots/HeLaR1_coexp_PCA_projection.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show()

#UMAP plot
fig, axes = plt.subplots(nrows=2,ncols=2,figsize=[10,10], 
                         gridspec_kw=dict(width_ratios=[6,1],
                                          height_ratios=[1,6],
                                          ),
                         )

ax = axes[1][0]
g=sns.scatterplot(data=df_umap[df_umap.hue != -1],x='UMAP-1',y='UMAP-2',hue='hue',
                  palette=palette,legend=False,alpha=0.8,ax=ax,edgecolor=None,s=10)
ax.set_xlabel('UMAP-1')
ax.set_ylabel('UMAP-2')

ax = axes[0][0]
g=sns.kdeplot(data=df_umap[df_umap.hue != -1],x='UMAP-1',hue='hue',
              palette=palette,legend=False,ax=ax,common_norm=False,lw=4)
ax.axis('off')

ax = axes[1][1]
g=sns.kdeplot(data=df_umap[df_umap.hue != -1],y='UMAP-2',hue='hue',
              palette=palette,legend=False,ax=ax,common_norm=False,lw=4)
ax.axis('off')

axes[0][1].remove()
plt.subplots_adjust(hspace = 0, wspace = 0)

plt.savefig('plots/HeLaR1_coexp_UMAP_projection.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show()


#save source data
df_pca[df_pca.hue!=1].to_csv('source_data/Fig S2 (PCA).csv')
df_umap[df_umap.hue!=1].to_csv('source_data/Fig S2 (UMAP).csv')

#%% KNN

#train-test split
split = lambda x:  train_test_split(x.drop(columns=['hue']),df.hue,test_size=200,stratify=df.hue,random_state=1)

#transform PCs by explained variance (dimensional weighting)
sub_pca = df_pca[df_pca.hue != -1]
sub_pca.PC1 = sub_pca.apply(lambda x: x.PC1*pca.explained_variance_ratio_[0],axis=1)
sub_pca.PC2 = sub_pca.apply(lambda x: x.PC2*pca.explained_variance_ratio_[1],axis=1)

sub_umap = df_umap[df_umap.hue != -1]
sub_pca.name = 'PCA + KNN'
sub_umap.name = 'UMAP + KNN'
knn = KNeighborsClassifier(n_neighbors=int(np.sqrt(len(panel_corr)-200)))

aucDict = {}
f1Dict = {}
result = pd.DataFrame()


#get KNN performance statistics
for df in (sub_pca, sub_umap):
    X_train, X_test, Y_train, Y_test = split(df)
    X_train, Y_train = RandomUnderSampler().fit_resample(X_train, Y_train)
    
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
    
    fpr, tpr, thresholds = roc_curve(Y_test, Y_scores, pos_label = 1)
    fpr_tr, tpr_tr, thresholds_tr = roc_curve(Y_train, Y_self_scores, pos_label = 1)    
    result = result.append(pd.DataFrame(np.array([fpr,tpr,[df.name]*len(fpr)]).T,
                                        columns = ['fpr','tpr', 'method']))
    aucDict[df.name] = round(auc(fpr,tpr),3)
    f1Dict[df.name] = round(f1_score(Y_test,Y_pred),4)

method = 'PROSE'
q = pgx.prose(obs, unobs, panel_corr)   
aucDict[method] = q.auc
result = result.append(pd.DataFrame(np.array([q.fpr,q.tpr,[method]*len(q.fpr)]).T,
                                    columns = ['fpr','tpr', 'method']))

#%% Plot ROC
        
result = result.reset_index(drop=True)
result.fpr = result.fpr.astype(float)
result.tpr = result.tpr.astype(float)

fig, ax = plt.subplots(figsize=[10,10])
sns.lineplot(data=result, x='fpr', y='tpr', hue='method',
             drawstyle='steps-post', ci=None,
             palette='rocket', lw=4
             )
x=y=np.linspace(0,1,2)
sns.lineplot(x=x, y=y,linestyle='--', color='black')

plt.xlabel('1 - Specificity')
plt.ylabel('Sensitivity')

l = ax.get_legend_handles_labels()[1]
plt.legend(labels=[i + ' ({})'.format(str(aucDict[i])) for i in l],
           loc='lower center', bbox_to_anchor=(0.5, -0.1-0.15*len(l)),
           frameon=False, prop={'size': 35})


plt.savefig('plots/HeLaR1_bench_ROC.png',
            format='png', dpi=600, bbox_inches='tight') 
plt.show()


#%% Generate 10 simulations of PROSE and KNN for F1 score and AUC plotting

obs = testdata['HeLa R1']['two peptide']
unobs = testdata['HeLa R1']['no evidence']

auc_result = pd.DataFrame()
f1_result = pd.DataFrame()

for i in tqdm(range(10)):
    
    #generate statistics for PROSE
    q = pgx.prose(obs, unobs, panel_corr)
    auc_result = auc_result.append(pd.DataFrame([q.auc,q.auc_tr,'PROSE']).T)
    f1_result = f1_result.append(pd.DataFrame([q.f1,q.f1_tr,'PROSE']).T)
    
    #generate statistics for KNN
    for df in (sub_pca, sub_umap):
        X_train, X_test, Y_train, Y_test = split(df)
        X_train, Y_train = RandomUnderSampler().fit_resample(X_train, Y_train)
    
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
        knn_f1 = round(f1_score(Y_test,Y_pred),4)
        knn_f1_tr = round(f1_score(Y_train,Y_self),4)
        
        fpr, tpr, thresholds = roc_curve(Y_test, Y_scores, pos_label = 1)
        fpr_tr, tpr_tr, thresholds_tr = roc_curve(Y_train, Y_self_scores, pos_label = 1)
        knn_auc = round(auc(fpr,tpr),3)
        knn_auc_tr = round(auc(fpr_tr,tpr_tr),3)
        
        auc_result = auc_result.append(pd.DataFrame([knn_auc,knn_auc_tr,df.name]).T)
        f1_result = f1_result.append(pd.DataFrame([knn_f1,knn_f1_tr,df.name]).T)

auc_result.columns = ['test','train','method'] 
f1_result.columns = ['test','train','method']  
    
#%% Plot simulated score benchmarks

auc_data = pd.melt(auc_result,id_vars='method',value_vars=['test','train'])
f1_data = pd.melt(f1_result,id_vars='method',value_vars=['test','train'])

#AUC benchmark
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[15,10],sharex=True)

ax = axes[0]
g = sns.boxplot(data=auc_data[auc_data.variable == 'test'],
                x='value',y='variable',hue='method',
                palette='rocket_r',ax=ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(axis='x')
ax.legend().remove()

ax = axes[1]
g = sns.boxplot(data=auc_data[auc_data.variable == 'train'],
                x='value',y='variable',hue='method',
                palette='rocket_r',ax=ax)
ax.set_ylabel('')
ax.grid(axis='x')

plt.xlabel('AUC')

handles, labels = ax.get_legend_handles_labels()
l_format = lambda df,i: str(round(df[df.method == i][df.variable == 'test'].value.mean(),3))+\
                        ' | '+\
                        str(round(df[df.method == i][df.variable == 'train'].value.mean(),3)) 
labels = [i + ' ({})'.format(l_format(auc_data,i)) for i in labels]
ax.legend(handles[0:3], labels[0:3],
          loc='lower center', bbox_to_anchor=(0.5,-1.3),
          frameon=False)

plt.savefig('plots/HeLaR1_benchmark_AUC.png',
            format='png', dpi=600, bbox_inches='tight') 


#F1 benchmark
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=[15,10],sharex=True)

ax = axes[0]
g = sns.boxplot(data=f1_data[f1_data.variable == 'test'],
                x='value',y='variable',hue='method',
                palette='rocket_r',ax=ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.grid(axis='x')
ax.legend().remove()

ax = axes[1]
g = sns.boxplot(data=f1_data[f1_data.variable == 'train'],
                x='value',y='variable',hue='method',
                palette='rocket_r',ax=ax)
ax.set_ylabel('')
ax.grid(axis='x')

plt.xlabel('F1 score')

handles, labels = ax.get_legend_handles_labels() 
labels = [i + ' ({})'.format(l_format(f1_data,i)) for i in labels]
ax.legend(handles[0:3], labels[0:3],
          loc='lower center', bbox_to_anchor=(0.5,-1.3),
          frameon=False)

plt.savefig('plots/HeLaR1_benchmark_F1.png',
            format='png', dpi=600, bbox_inches='tight') 