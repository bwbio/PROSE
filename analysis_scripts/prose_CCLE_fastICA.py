# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 08:43:02 2021

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

from sklearn.decomposition import FastICA
#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%% Read formatted matrices

score = pd.read_csv('ccle/ccle_prose_formatted.tsv.gz', sep='\t').drop_duplicates('cell_line')
mapper = score[['cell_line','tissue']]
score = score.set_index('cell_line').drop(columns='tissue').T

tpm = pd.read_csv('ccle/ccle_tpm_formatted.tsv.gz', sep='\t').drop_duplicates('cell_line')

tmt = pd.read_csv('ccle/ccle_tmt_formatted.tsv.gz', sep='\t').drop_duplicates('cell_line')
tmt = tmt.set_index('cell_line').drop(columns='tissue').T


tmt_cleaned = pd.read_csv('ccle/ccle_tmt_formatted_withMissingVals.tsv.gz', sep='\t').drop_duplicates('cell_line')
tmt_cleaned = tmt_cleaned.set_index('cell_line').drop(columns='tissue').T
tmt_cleaned = tmt_cleaned.dropna()


#%% ICA

#200, 0 default
ica = FastICA(n_components=200, random_state=0).fit_transform(score)
df = pd.DataFrame(ica, index=score.index)

ica_tmt = FastICA(n_components=200, random_state=0).fit_transform(tmt)
df_tmt = pd.DataFrame(ica_tmt, index=tmt.index)

#%% Generate ICA p-value matrices (Supplementary File 5)

ica_pval_prose = df.apply(lambda x: norm.cdf(x,x.mean(),x.std()))
ica_pval_prose.to_csv('fastica_files/Supplementary File 5 PROSE_FastICA_pvals.tsv',sep='\t')

#%%


tpmmat = tpm.set_index('cell_line').T
common_cl = list(set(tpm.cell_line).intersection(score.columns))


#returns result (df), moddict (dict)
def identify_modules(icaDf, source, tpmmat, common_cl, alpha=10**-6):

    #only get common elements for concordance estimation
    icaDf = icaDf.loc[icaDf.index.intersection(tpmmat.index)]    
    source = source.loc[source.index.intersection(tpmmat.index)]

    from scipy.stats import norm
    moddict = {}
    temp_moddict = {}
    for i in icaDf.columns:
        data = icaDf[i]
        upper=norm.ppf(1-alpha/2, data.mean(), data.std())
        lower=norm.ppf(0+alpha/2, data.mean(), data.std())
        temp_moddict[i] = data[data > upper].append(data[data < lower])
    
    result = []

    for i in list(temp_moddict.keys()):
        mod = temp_moddict[i]
        if len(mod) >= 10 and len(mod) < 1000:
            matrix = source.loc[mod.index].T        
            r = matrix.corr(method='spearman').abs().mean()
            rr = r.abs().mean()
            spread = matrix.std().mean()
            core = r.sort_values(ascending=False)
            core_name = core.index[0]
            core_r = core[0]      
            rhos = []
            for protein in mod.index:
                rho = scipy.stats.spearmanr(tpmmat.loc[protein][common_cl], source.loc[protein][common_cl])[0]
                rhos.append(rho)
            rhos = np.mean(rhos)
            proteins = ','.join(mod.index)
            
            result.append([i, len(mod), rr, spread, core_name, core_r, rhos, proteins])            
    result=pd.DataFrame(result, columns = ['temp','size','rr', 'spread', 'core','core_r','concordance','proteins'])
    result=result.sort_values(by='concordance',ascending=False)
    result['id'] = range(1, len(result)+1)
    moddict = {row['id']:temp_moddict[row['temp']] for (i, row) in result.iterrows()}
    result = result[['id','size','concordance','core','core_r','rr','spread','proteins']]
    result = result.reset_index(drop=True)
    print(result.drop(columns='proteins'))
    return result


for sig in [6]: 
    alpha = 10**(-sig)
    print(alpha)
    result = identify_modules(df, score, tpmmat, common_cl,alpha=alpha)
    result_tmt = identify_modules(df_tmt, tmt, tpmmat, common_cl,alpha=alpha)
    result.to_csv('fastica_files/fastica_report_prose_{}.tsv'.format(sig),index=False,sep='\t')
    result_tmt.to_csv('fastica_files/fastica_report_tmt_{}.tsv'.format(sig),index=False,sep='\t')

#%% compare concordance

#significance label
def sig(x):
    if x <= 0.001: return '***'
    if 0.001 < x <= 0.01: return '**'
    if 0.01 < x <= 0.05: return '*'
    else: return '(n.s.)'

#boostrap test
def bs(a,b,k=5000,s=100):
    import random
    out = 0
    for i in range(k):
        a_ = np.median(random.sample(a, s))
        b_ = np.median(random.sample(b, s))
        if a_ >= b_:
            out +=1
    
    return 1-(out/k)

data_prose = pd.DataFrame(result.concordance)
data_tmt = pd.DataFrame(result_tmt.concordance)
data_prose['method'] = 'PROSE'; data_tmt['method'] = 'TMT'
data = pd.concat([data_prose, data_tmt])
# Violin plot,concordance
fig, ax = plt.subplots(figsize=[6,10])
g=sns.violinplot(data=data,x='method', y='concordance',color='#d4edff')
for violin in ax.collections[::2]: violin.set_alpha(0.7)
sns.despine()

pval=bs(a=list(data_prose.concordance),
        b=list(data_tmt.concordance),
        k=100000, s=10)

print('p-value = {}'.format(pval))
plt.text(s=sig(pval),x=0.5,y=max(data.concordance)+0.1,ha='center')
g.axhline(max(data.concordance)+0.1,.25,.75,color='black',lw=3)
plt.xlabel('')
plt.ylabel(r'Concordance with RNA-seq' '\n' r'(mean $ρ_{\rmTPM}$)',
           labelpad=40)
plt.grid(axis='y')

plt.savefig('plots/ICA_module/concordance_violin.png',format='png', 
            dpi=600, bbox_inches='tight')

data.to_csv('source_data/Fig 4b (concordance violin).csv')



#compare connectivity
data_prose = pd.DataFrame(result.rr)
data_tmt = pd.DataFrame(result_tmt.rr)
data_prose['method'] = 'PROSE'; data_tmt['method'] = 'TMT'
data = pd.concat([data_prose, data_tmt])

# Violin plot, connectivity
fig, ax = plt.subplots(figsize=[6,10])
g=sns.violinplot(data=data,x='method', y='rr',color='#d4edff')
for violin in ax.collections[::2]: violin.set_alpha(0.7)
sns.despine()

pval=bs(a=list(data_prose.rr),
        b=list(data_tmt.rr),
        k=100000, s=10)

print('p-value = {}'.format(pval))
plt.text(s=sig(pval),x=0.5,y=max(data.rr)+0.2,ha='center')
g.axhline(max(data.rr)+0.2,.25,.75,color='black',lw=3)
plt.xlabel('')
plt.ylabel(r'Module connectivity' '\n' r'(mean $ρ$ between proteins)',
           labelpad=40)
plt.grid(axis='y')

plt.savefig('plots/ICA_module/connectivity_violin.png',format='png', 
            dpi=600, bbox_inches='tight')

data.to_csv('source_data/Fig 4c (connectivity violin).csv')



#%%
# Scatterplot, module size
fig, axes = plt.subplots(figsize=[10,10],nrows=2,ncols=1,sharex=True)

ax=axes[0]
g=sns.scatterplot(data=result_tmt, x='size',y='rr',color='#d4edff',
                  s=200, edgecolor='black',linewidth=0.4,ax=ax)
ax.set_xlabel('')
ax.set_ylabel('')
ax.text(x=1,y=0.95,s='TMT',ha='right',va='top',size=40, transform=ax.transAxes)
ax.grid()

plt.xlabel('Module size')
ax=axes[1]
g=sns.scatterplot(data=result, x='size',y='rr',color='#d4edff',
                  s=200, edgecolor='black',linewidth=0.4,ax=ax)
plt.xlabel('Module size')
plt.ylabel('')
ax.text(x=1,y=0.95,s='PROSE',ha='right',va='top',size=40, transform=ax.transAxes)
sns.despine()

plt.subplots_adjust(hspace=0.3)
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.ylabel('Concordance',labelpad=60,size=45)
ax.grid()

plt.savefig('plots/ICA_module/concordance_scatter.png',format='png', 
            dpi=600, bbox_inches='tight')

#%%

#get cell line and tf labels
for i in ['placeholder']:
    relabeler = pd.read_csv('databases/ensembl_uniprot_conversion.tsv',sep='\t')
    relabeler = relabeler[['ID','Entry','Entry name']]
    relabeler['name'] = relabeler.apply(lambda x: x['Entry name'].split('_')[0], axis=1)
    
    tfs = pd.read_csv('databases/the_human_transcription_factors.csv',index_col=0)
    tfs = tfs[tfs['Is TF?'] == 'Yes']['Ensembl ID']
    
    relabeler = relabeler[relabeler.ID.isin(tfs)]
    #print(relabeler)
    coi = mapper.sort_values(by='tissue')
    coi = coi.cell_line.drop_duplicates()
    coi = coi.to_list()
    tissues = mapper.tissue
    lut = dict(zip(tissues.unique(), sns.color_palette("husl", len(tissues.unique()))))
    row_colors = pd.DataFrame(zip(coi, tissues.map(lut)),
                              columns=['cell_line','tissues'],
                              ).drop_duplicates().set_index('cell_line')
    row_colors = row_colors.loc[coi]
    row_colors.columns=['']

#get top n modules and plot concordance
n = 20
for i, row in result[:20].iterrows():
    mod = row['id']
    proteins = row['proteins'].split(',')
    data = score.loc[proteins][common_cl]
    cmap = sns.diverging_palette(255, 9, as_cmap=True)
    
    ##############################################################################
    #PROSE Clustermap with shared ordering as PROSE clustering above
    
    g = sns.clustermap(data=data, 
                       cmap='mako',
                       vmin=-2,vmax=2,center=0,
                       figsize=[15,15],
                       xticklabels=False,yticklabels=False,
                       col_colors=row_colors,
                       dendrogram_ratio=(0.15,0.15),
                       cbar_kws={"orientation": "horizontal"},
                       tree_kws=dict(linewidths=1),
                       cbar_pos=None if mod != 1 else [.56, 0, .4, .02])
    ax = g.ax_heatmap
    ax.set_xlabel('')
    ax.text(x=0,y=1.260,s=r'Module {} (Core: {})'.format(mod,row.core), 
            ha='left',size=60, weight='bold',
            transform=ax.transAxes)

    if mod == 1:
        ax.text(x=0.45,y=-0.11,s='PROSE score ',ha='right',size=40, transform=ax.transAxes)
        
    g.savefig('plots/ICA_module/CCLE_module_{}_{}_score.png'.format(mod,row.core),
        format='png', dpi=600, bbox_inches='tight') 
    plt.show()

    order_col, order_row = g.dendrogram_col.reordered_ind, g.dendrogram_row.reordered_ind
    cols = dict(enumerate(data.columns))
    rows = dict(enumerate(data.index))            
    reverse_rows = {v:order_row[k]  for k,v in rows.items()}
    
    ##############################################################################
    #TPM Clustermap with shared ordering as PROSE clustering above

    reord_tpm = tpm.set_index('cell_line').T.loc[data.index, data.columns]
    reord_tpm = reord_tpm.loc[[rows[i] for i in order_row], [cols[j] for j in order_col]]

    g = sns.clustermap(data=reord_tpm,
                       z_score=0,
                       cmap='rocket',
                       vmin=-2,vmax=2,center=0,
                       figsize=[17,15],
                       xticklabels=False,yticklabels=True,
                       col_colors=row_colors,
                       col_cluster=False,row_cluster=False,
                       dendrogram_ratio=0.15,
                       cbar_kws={"orientation": "horizontal"},
                       cbar_pos=None if mod != 1 else [.49, 0, .34, .02],
                       )
    ax = g.ax_heatmap
    ax.set_xlabel('')
    ax.text(x=0,y=1.06,s='Concordance = {}'.format(str(round(row.concordance,3))), 
            ha='left',size=50,
            transform=ax.transAxes)
    
    
    ###
    tf_set = list(relabeler[relabeler.Entry.isin(data.index)].Entry)
    texts = []   
    
    for i in range(len(ax.get_yticklabels())):
        s = ax.get_yticklabels()[i].get_text()
        if s in tf_set:
            obj = relabeler[relabeler.Entry == s]
            label = '{} ({})'.format(obj.Entry.values[0], obj.name.values[0])
            x, y = ax.get_yticklabels()[i].get_position()
            y = 1-(y/len(data+1))
            texts.append([y,y,x,label])
    
    import math
    texts = pd.DataFrame(texts, columns = ['y0','y','x','s'])
    if len(texts) != 0:
        if len(texts) > 1:
            for iteration in tqdm(range(500)):
                for i, rw in  texts.iterrows():
                    comparator = texts[texts.index!=rw.name]
                    idx = comparator.y.sub(rw.y).abs().idxmin()
                    push = 1 if  (rw.y - texts.iloc[idx].y) > 0 else -1                 
                    dist=min(comparator.apply(lambda x: abs(x.y-rw.y),axis=1))
                    texts.loc[i,'dist']=dist
                    texts.loc[i,'push'] = (1/dist)*push*0.0001*random() if dist < 0.05 else 0
                mindist = min(texts.dist.abs())
                if mindist > 0.05:
                    print('convergence after {} iterations'.format(iteration))
                    break
                else:
                    texts.y = texts.apply(lambda x: x.y + x.push,axis=1)
                if iteration == 1000:
                    print('Max iterations reached!')
                    
        for i, rw in texts.iterrows():
            x=rw.x
            y0 = rw.y0
            y1 = rw.y
            if abs(y0-y1) <= 0.008:
                y1 = y0
                
            for xdot,ydot in zip(np.linspace(x-0.001,x+0.007,40), np.linspace(y0, y0,40)):
                plt.text(y=ydot, x=xdot, s= '●',transform=ax.transAxes,size=8,alpha=0.7)     
            
            px=30+int(math.dist([x+0.007,y0],[x+0.039, y1])/(0.039-0.007)*20)
            for xdot,ydot in zip(np.linspace(x+0.007,x+0.039,px), np.linspace(y0, y1,px)):
                plt.text(y=ydot, x=xdot, s= '●',transform=ax.transAxes,size=8,alpha=0.7)
                
            for xdot,ydot in zip(np.linspace(x+0.04,x+0.055,40), np.linspace(y1, y1,40)):
                plt.text(y=ydot, x=xdot, s= '●',transform=ax.transAxes,size=8,alpha=0.7)
                
            plt.text(y=y1, x=x+0.065, s = rw.s,
                  transform=ax.transAxes,ha='left',va='center')
    
    
    ax.set_yticks([])

    if mod == 1:
        ax.text(x=0.45,y=-0.11,s='Relative gene expression' ,ha='right',size=40, transform=ax.transAxes)
    
    g.savefig('plots/ICA_module/CCLE_module_{}_{}_tpm.png'.format(mod,row.core),
            format='png', dpi=600, bbox_inches='tight')
    plt.show()
    
    if mod == 1:
        data.to_csv('source_data/Fig4d (module {}, PROSE).csv'.format(mod))
        reord_tpm.to_csv('source_data/Fig4d (module {}, TPM).csv'.format(mod))
    if mod == 2:
        data.to_csv('source_data/Fig4e (module {}, PROSE).csv'.format(mod))
        reord_tpm.to_csv('source_data/Fig4e (module {}, TPM).csv'.format(mod))
    if 3 <= mod <= 8:
        data.to_csv('source_data/FigS5 (module {}, PROSE).csv'.format(mod))
        reord_tpm.to_csv('source_data/FigS5 (module {}, TPM).csv'.format(mod))
    if mod == 19:
        data.to_csv('source_data/FigS6 (module {}, PROSE).csv'.format(mod))
        reord_tpm.to_csv('source_data/FigS6 (module {}, TPM).csv'.format(mod))
