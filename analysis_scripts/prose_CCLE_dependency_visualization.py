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

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%% Read formatted matrices

score = pd.read_csv('ccle/ccle_prose_formatted.tsv.gz', sep='\t').drop_duplicates('cell_line')
tmt = pd.read_csv('ccle/ccle_tmt_formatted_withMissingVals.tsv.gz', sep='\t').drop_duplicates('cell_line')
tpm = pd.read_csv('ccle/ccle_tpm_formatted.tsv.gz', sep='\t').drop_duplicates('cell_line')
drive = pd.read_csv('ccle/ccle_drive_formatted_withGeneName.tsv.gz', sep='\t')
mapper = score[['cell_line','tissue']]

#%%

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

#%% Scatterplots, TMT

cois = ['MDAMB468','THP1','A549','CAKI1']
renamedict = {'MDAMB468':'MDA-MB-468',
              'THP1':'THP-1',
              'HEPG2':'HepG2',
              'NCIH460':'NCI-H460',
              'CAKI1':'Caki-1',
              'A549': 'A549'}

cgc = pd.read_csv('databases/cancer_gene_census.csv').fillna('NA')
cancer_genes = cgc[cgc['Role in Cancer'].str.contains('oncogene')]
oncos = [i.upper() for j in [i.split(',') for i in cancer_genes.Synonyms] for i in j]

#plot dependency against PROSE score
fig, axes = plt.subplots(nrows=2, ncols=4,
      figsize=[45,13],
      gridspec_kw={'height_ratios': [1, 4]})

n=0
for cell_line in cois:
    dep_coi=drive[['gene','protein',cell_line]].set_index('protein')
    score_coi=score[score.cell_line == cell_line].drop(columns=['cell_line','tissue']).T
    dep_coi.columns = ['gene','dep']; score_coi.columns = ['score_norm']
    
    matched_cl = dep_coi.join(score_coi).dropna()
    matched_cl.dep = -matched_cl.dep
    
    matched_cl['hue'] = matched_cl.apply(lambda x: 0 if x.dep < matched_cl.dep.quantile(0.95) else 1, axis=1)
    matched_cl['size'] = matched_cl.apply(lambda x: 1 if (x.gene in oncos)&(x.hue==1) else 0, axis=1)
    matched_cl['alpha'] = matched_cl.apply(lambda x: 0.95 if(x.gene in oncos)&(x.hue==1) else 0.35, axis=1)

    
    ax=axes[0][n]
    g1=sns.kdeplot(data=matched_cl,x='score_norm',hue='hue',
                  lw=3,palette='tab10',ax=ax,common_norm=False,legend=False)
    ax.axis('off')
    # ax.get_legend().remove()
    # ax.set_xlabel('')
    # ax.set_xticks([])
    ax.set_ylim(top=ax.get_ylim()[1]*1.25)

    upper = matched_cl[matched_cl.dep >= matched_cl.dep.quantile(0.95)]
    lower = matched_cl[matched_cl.dep < matched_cl.dep.quantile(0.95)]
    diff = upper.score_norm.median()-lower.score_norm.median()  
    p = bs(set(upper.score_norm.values),set(lower.score_norm.values))
    ax.text(s='Δ = {} {}'.format(str(round(diff,3)), sig(p)),
            x=0.03,y=0.98,
            transform = ax.transAxes, ha='left',va='top')   
    
    if n==0:
        ax.set_ylabel('Density',labelpad=20)

    else:
        ax.set_ylabel('')
        
    initial = True
    
    
    ax=axes[1][n]
    g2=sns.scatterplot(data=matched_cl, x='score_norm',y='dep',
                       hue='hue',size="size", sizes=(600, 100),
                       alpha=matched_cl.alpha.to_list(),legend=False,ax=ax)
    ax.text(s=renamedict[cell_line],size=50,
            x=0.04,y=0.98,
            transform = ax.transAxes, ha='left',va='top')
    ax.set_xlabel('')
    
    if n==0:
        ax.set_ylabel('Dependency',labelpad=20)
    else:
        ax.set_ylabel('')
    
    matched_cl.drop(columns=['size','alpha']).to_csv(
        'source_data/Fig 3e ({} PROSE-dependency distribution).csv'.format(cell_line)
        )
    
    n+=1

plt.subplots_adjust(hspace=0)
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('PROSE score',labelpad=30,size=50)

plt.savefig('plots/CCLE_oncogeneDependencies_proseCorr.png',format='png', 
            dpi=600, bbox_inches='tight') 
plt.show() 

#%%

#plot dep against TMT quant
fig, axes = plt.subplots(nrows=2, ncols=4,
      figsize=[45,13],
      gridspec_kw={'height_ratios': [1, 3.3]})

n=0
for cell_line in cois:
    dep_coi=drive[['gene','protein',cell_line]].set_index('protein')
    tmt_coi=tmt[tmt.cell_line == cell_line].drop(columns=['cell_line','tissue']).T
    dep_coi.columns = ['gene','dep']; tmt_coi.columns = ['tmt']
    tmt_coi.tmt = StandardScaler().fit_transform(tmt_coi) #Z-norm TMT quants
    tmt_coi=tmt_coi[tmt_coi.tmt >= -5]    
    matched_cl = dep_coi.join(tmt_coi).dropna()
    matched_cl.dep = -matched_cl.dep
    
    matched_cl['hue'] = matched_cl.apply(lambda x: 0 if x.dep < matched_cl.dep.quantile(0.95) else 1, axis=1)
    matched_cl['size'] = matched_cl.apply(lambda x: 1 if (x.gene in oncos)&(x.hue==1) else 0, axis=1)
    matched_cl['alpha'] = matched_cl.apply(lambda x: 0.95 if(x.gene in oncos)&(x.hue==1) else 0.35, axis=1)

    ax=axes[0][n]
    g1=sns.kdeplot(data=matched_cl,x='tmt',hue='hue',
                  lw=3,palette='tab10',ax=ax,common_norm=False,legend=False)
    ax.axis('off')
    # ax.get_legend().remove()
    # ax.set_xlabel('')
    # ax.set_xticks([])
    ax.set_ylim(top=ax.get_ylim()[1]*1.25)

    upper = matched_cl[matched_cl.dep >= matched_cl.dep.quantile(0.95)]
    lower = matched_cl[matched_cl.dep < matched_cl.dep.quantile(0.95)]
    diff = upper.tmt.median()-lower.tmt.median()
    p = bs(set(upper.tmt.values),set(lower.tmt.values))
    ax.text(s='Δ = {} {}'.format(str(round(diff,3)), sig(p)),
            x=0.03,y=0.98,
            transform = ax.transAxes, ha='left',va='top')   
    
    if n==0:
        ax.set_ylabel('Density',labelpad=20)

    else:
        ax.set_ylabel('')
        
    initial = True
    
    
    ax=axes[1][n]
    g2=sns.scatterplot(data=matched_cl, x='tmt',y='dep',
                       hue='hue',size="size", sizes=(600, 100),
                       alpha=matched_cl.alpha.to_list(),legend=False,ax=ax)
    ax.text(s=renamedict[cell_line],size=50,
            x=0.04,y=0.98,
            transform = ax.transAxes, ha='left',va='top')
    ax.set_xlabel('')
    
    if n==0:
        ax.set_ylabel('Dependency',labelpad=20)
    else:
        ax.set_ylabel('')
    
    matched_cl.drop(columns=['size','alpha']).to_csv(
    'source_data/Fig 3f ({} TMT-dependency distribution).csv'.format(cell_line)
    )
    
    n+=1
    

plt.subplots_adjust(hspace=0)
fig.add_subplot(111, frame_on=False)
plt.tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('TMT quant',labelpad=30,size=50)

plt.savefig('plots/CCLE_oncogeneDependencies_tmtCorr.png',format='png', 
            dpi=600, bbox_inches='tight') 

#%%

n=0

fig, axes = plt.subplots(nrows=1,ncols=4,figsize=[40,10])
for cell_line in cois:
    ax=axes[n]
    dep_coi=drive[['gene','protein',cell_line]].set_index('protein')
    tmt_coi=tmt[tmt.cell_line == cell_line].drop(columns=['cell_line','tissue']).T
    score_coi=score[score.cell_line == cell_line].drop(columns=['cell_line','tissue']).T
    dep_coi.columns = ['gene','dep']; tmt_coi.columns = ['tmt']; score_coi.columns = ['score_norm']
    tmt_coi.tmt = StandardScaler().fit_transform(tmt_coi) #Z-norm TMT quants
    tmt_coi=tmt_coi[tmt_coi.tmt >= -5]    
    matched_cl = dep_coi.join(tmt_coi).join(score_coi).dropna()
    matched_cl.dep = -matched_cl.dep
    
    matched_cl['hue'] = matched_cl.apply(lambda x: 0 if x.dep < matched_cl.dep.quantile(0.95) else 1, axis=1)
    matched_cl['size'] = matched_cl.apply(lambda x: 1 if (x.gene in oncos)&(x.hue==1) else 0, axis=1)
    matched_cl['alpha'] = matched_cl.apply(lambda x: 0.95 if(x.gene in oncos)&(x.hue==1) else 0.75, axis=1)

    matched_cl = matched_cl.sort_values(by=['hue','size'])
    
    g=sns.scatterplot(data=matched_cl,x='score_norm',y='tmt',hue='hue',size='size',sizes=[100,600],
                      alpha=matched_cl.alpha.to_list(),ax=ax)
    g.axvline(matched_cl.score_norm.median(), color='black', linestyle='--')
    g.axhline(matched_cl.tmt.median(), color='black', linestyle='--')
    
    rho = scipy.stats.spearmanr(matched_cl.score_norm, matched_cl.tmt)[0]
    ax.text(s = renamedict[cell_line]\
            + '\n' + r'$ρ$ = {}'.format(round(rho,3)),
            size=40,
            x=0.04,y=0.98,
        transform = ax.transAxes, ha='left',va='top')
    ax.get_legend().remove()
    ax.set_xlabel('')
    
    if n == 0:
        ax.set_ylabel('TMT quant')
    else:
        ax.set_ylabel('')
    
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor="none", bottom=False, left=False)
    plt.xlabel(r"PROSE score",labelpad=30,size=50)
    
    matched_cl.drop(columns=['size','alpha']).to_csv(
    'source_data/Fig S4c ({} TMT-PROSE distribution).csv'.format(cell_line)
    )
    
    
    n+=1

plt.savefig('plots/CCLE_oncogeneDependencies_tmtScoreCorr.png',format='png', 
            dpi=600, bbox_inches='tight') 

#%% Verification 

# df = pd.read_csv('ccle/D2_DRIVE_gene_dep_scores.csv')
# df['gene'] = df.apply(lambda x: x['Unnamed: 0'].split(' (')[0], axis=1)
# df.columns = [i.split('_')[0] for i in df.columns]

# df1 = pd.read_csv('ccle/protein_quant_current_normalized.csv.gz')
# df1['id']=  df1.apply(lambda x: x['Protein_Id'].split('|')[1], axis=1)
# df1.columns = [i.split('_')[0] for i in df1.columns]

# #verify that plotted data matches corresponding base data
# genes = ['COPB2','PXDN']
# for gene in genes:
#     pro = conv0[gene]
#     tmtcheck = tmt[tmt.cell_line == cell_line][pro].values[0]
#     depcheck = -drive[drive.protein == pro][cell_line].values[0]
#     print(('{}\t'*5).format(pro,
#                             round(tmtcheck,3),
#                             round(depcheck,3),
#                             matched_cl.loc[pro].tmt == tmtcheck,
#                             matched_cl.loc[pro].dep == depcheck
#                             )
#               )
#     print(gene, conv0[gene], df[df.gene == gene][cell_line])
#     print(matched_cl.sort_values(by='dep',ascending=False).loc[pro])

#%% Summary violinplot

result = []

common_cell_lines = drive.columns.drop(['gene','protein'])\
                    .intersection(tmt.cell_line)\
                    .intersection(score.cell_line)

for cell_line in tqdm(common_cell_lines):
    dep_coi=drive[['gene','protein',cell_line]].set_index('protein')
    score_coi=score[score.cell_line == cell_line].drop(columns=['cell_line','tissue']).T
    tmt_coi=tmt[tmt.cell_line == cell_line].drop(columns=['cell_line','tissue']).T.dropna()
    dep_coi.columns = ['gene','dep']
    score_coi.columns = ['score_norm']
    tmt_coi.columns = ['tmt']
    tmt_coi.tmt = StandardScaler().fit_transform(tmt_coi)
    matched_cl = dep_coi.join(tmt_coi).join(score_coi).dropna()
    matched_cl.dep = -matched_cl.dep
    
    upper = matched_cl[matched_cl.dep >= matched_cl.dep.quantile(0.95)]
    lower = matched_cl[matched_cl.dep < matched_cl.dep.quantile(0.95)]
    
    diff_tmt = upper.tmt.median()-lower.tmt.median()
    diff_score = upper.score_norm.median()-lower.score_norm.median()
    
    if cell_line in cois:
        print(cell_line, diff_score, diff_tmt)
    
    ks_tmt = scipy.stats.ks_2samp(upper.tmt,lower.tmt)[0]
    ks_score = scipy.stats.ks_2samp(upper.score_norm,lower.score_norm)[0]
    
    result.append([diff_tmt,diff_score,ks_tmt,ks_score,cell_line])
    
result = pd.DataFrame(result, columns=['diff_tmt','diff_score',
                                       'ks_tmt','ks_score',
                                       'cell_line'])

result.to_csv('source_data/Fig 3a-d (Separability of essential genes, violin and scatter).tsv', sep='\t')

result_melt = result.melt(id_vars='cell_line',value_vars=['diff_tmt','diff_score',
                                                          'ks_tmt','ks_score'])
    
#%% Aggregate plotting across cell lines

# Violin plot, KS
data = result_melt[result_melt.variable.isin(['ks_tmt','ks_score'])]
fig, ax = plt.subplots(figsize=[6,10])
g=sns.violinplot(data=data,x='variable', y='value',color='#d4edff')
plt.ylim(0,max(data.value)+0.2)
g.axhline(max(data.value)+0.1,.25,.75,color='black',lw=3)
for violin in ax.collections[::2]: violin.set_alpha(0.7)

pval = scipy.stats.ttest_rel(result.ks_score, result.ks_tmt)[1]
print('p-value = {}'.format(pval))
plt.text(s=sig(pval),x=0.5,y=max(data.value)+0.1,ha='center')

g.set_xticklabels(['TMT','PROSE'])
plt.xlabel('')
plt.ylabel(r'Separability, $D_{\rmKS}$',
           labelpad=20)
plt.grid(axis='y')
sns.despine()

plt.savefig('plots/CCLE_oncogeneDependencies_distribution_violin.png',format='png', 
            dpi=600, bbox_inches='tight') 

# Violin plot, Difference
data = result_melt[result_melt.variable.isin(['diff_tmt','diff_score'])]
fig, ax = plt.subplots(figsize=[6,10])
g=sns.violinplot(data=data,x='variable', y='value',color='#d4edff')
plt.ylim(min(data.value)-0.1,max(data.value)+0.5)
g.axhline(max(data.value)+0.3,.25,.75,color='black',lw=3)
for violin in ax.collections[::2]: violin.set_alpha(0.7)

pval = scipy.stats.ttest_rel(result.diff_score, result.diff_tmt)[1]
print('p-value = {}'.format(pval))
plt.text(s=sig(pval),x=0.5,y=max(data.value)+0.3,ha='center')
g.set_xticklabels(['TMT','PROSE'])
plt.xlabel('')
plt.ylabel('Difference of medians, Δ',
           labelpad=20)
plt.grid(axis='y')
plt.axhline(0, color='black')
sns.despine()


plt.savefig('plots/CCLE_oncogeneDependencies_medianDiff_violin.png',format='png', 
            dpi=600, bbox_inches='tight') 

#%%
# Scatterplot, KS
data = result
data['hue'] = data.apply(lambda x: 1 if x.cell_line in cois else 0, axis=1)
data['size'] = data.apply(lambda x: 1 if x.cell_line in cois else 0, axis=1)
data=data.sort_values(by='hue')
fig, ax = plt.subplots(figsize=[10,10])
g=sns.scatterplot(data=data,x='ks_score', y='ks_tmt',s=200,
                   hue='hue',palette=['#d4edff','black'],size='size',sizes=[200,400],alpha=0.8,
                   legend=False,edgecolor='black',linewidth=0.4)
plt.plot(np.linspace(0,0.6),np.linspace(0,0.6), linestyle='--', color='black')
plt.ylim(0.06,0.348)
plt.xlabel(r'$D_{\rmKS}$ by PROSE score',labelpad = 20)
plt.ylabel(r'$D_{\rmKS}$ by TMT quant', labelpad = 20)
sns.despine()
ax.tick_params(axis='both', which='major', labelsize=35)

posdict = {'MDAMB468':[0.32,0.14],
           'THP1':[0.44,0.185],
           'CAKI1':[0.14,0.17],
           'A549':[0.377,0.25]}

for cell_line in cois:
    i = data[data.cell_line==cell_line]
    xp, yp = posdict[cell_line]

    plt.text(s='  {}'.format(renamedict[cell_line]),x=xp, y=yp, 
             size=40,color='black',ha='center',va='center')

plt.savefig('plots/CCLE_oncogeneDependencies_distribution_scatter.png',format='png', 
            dpi=600, bbox_inches='tight') 


# Scatterplot, Difference
data = result
data['hue'] = data.apply(lambda x: 1 if x.cell_line in cois else 0, axis=1)
data['size'] = data.apply(lambda x: 1 if x.cell_line in cois else 0, axis=1)
data=data.sort_values(by='hue')
fig, ax = plt.subplots(figsize=[10,10])
g=sns.scatterplot(data=data,x='diff_score', y='diff_tmt',s=200,
                   hue='hue',palette=['#d4edff','black'],size='size',sizes=[200,400],alpha=0.8,
                   legend=False,edgecolor='black',linewidth=0.4)
plt.plot(np.linspace(-0.5,0.6),np.linspace(-0.5,0.6), linestyle='--', color='black')
plt.ylim(-0.45,0.6)
plt.xlabel(r'Δ by PROSE score',labelpad = 20)
plt.ylabel(r'Δ by TMT quant', labelpad = 20)
sns.despine()
ax.tick_params(axis='both', which='major', labelsize=35)

posdict = {'MDAMB468':[0.53,-0.08],
           'THP1':[0.82,0.05],
           'CAKI1':[-0.07,-0.165],
           'A549':[0.9,0.17]}

for cell_line in cois:
    i = data[data.cell_line==cell_line]
    xp, yp = posdict[cell_line]

    plt.text(s='  {}'.format(renamedict[cell_line]),x=xp, y=yp, 
             size=40,color='black',ha='left',va='center')
    
plt.savefig('plots/CCLE_oncogeneDependencies_medianDiff_scatter.png',format='png', 
            dpi=600, bbox_inches='tight') 
