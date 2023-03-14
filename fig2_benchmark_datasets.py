# -*- coding: utf-8 -*-
"""
Perform benchmarking analysis (Figure 2)
"""

import pandas as pd
import pickle 
import os.path
import numpy as np
from Bio import SeqIO

#%% benchmarking tools

import utils.bench_network_scoring as bn
from utils.pyprose_static import pyprose

with open('utils/feature_space/panel_corr.pkl', 'rb') as handle:
    panel_corr = pickle.load(handle)
    
#%%

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

# get all proteins examined with fasta
fa = SeqIO.parse(open('databases/uniprot-proteome_UP000005640_reviewed.fasta'),'fasta')
all_prots = [i.id.split('|')[1] for i in fa]

#%% 


# get list of observed proteins using either 'top_n' or 'random' method
def get_observed(samplename, n=1000, method='top_n', remove_housekeepers=True,
                 datasets=datasets, random_state=0):
    
    ranked_prots = datasets[samplename]
    
    if remove_housekeepers:
        ranked_prots.loc[[i for i in ranked_prots.index if i not in hk]]    
    
    if method == 'top_n':
        obs = ranked_prots.index[:n].to_list()
    elif method == 'random':
        if random_state != None:
            obs = ranked_prots.sample(n, random_state=random_state).index.to_list()
        else:
            obs = ranked_prots.sample(n).index.to_list()
        
    return obs

def get_unobserved(samplename, obs, n=1000, remove_housekeepers=True,
                   random_state=0, all_prots=all_prots):
    
    if remove_housekeepers:
        considered_prots = set(all_prots).difference(set(hk))
    else:
        considered_prots = set(all_prots)
        
    unobs_all = list(set(considered_prots.difference(set(obs))))
    
    if random_state != None:
        np.random.seed(random_state)
        
    unobs = list(np.random.choice(unobs_all, n, replace=False))
        
    return unobs
    
# get sample data as df
samplenames = datasets.keys()
sampledata = pd.DataFrame([i.split('_') for i in samplenames],
                          index = samplenames,
                          columns = ['method', 'sample', 'rep'])

#%% create methods matrix

mm = bn.methodmatrix()
mm.loc['PROSE'] = [pyprose, dict(panel_corr=panel_corr, n_estimators=200)]
mm.loc['PROSE (random)'] = [pyprose, dict(panel_corr=panel_corr,  n_estimators=200)]
mm.loc['ToppGene']['kwargs']['feature_space'] = None
mm.loc['ToppGene (coexpr.)']['kwargs']['feature_space'] = None

# REMOVE TOPPGENE FOR FAST RUN
# mm=mm[~mm.index.str.contains('ToppGene')]

#%% Generate network benchmark results

import itertools as it
from sklearn.metrics import roc_auc_score

def find_auc(result):
    res = result.dropna().copy()
    res = res[res['y_true'] != -1]
    auc = round(roc_auc_score(res['y_true'], res['score']), 3)
    return auc
    pass


# for full dataset
opt_remove_housekeepers = [True, False]
opt_n = ['full']
opt_obs_method = ['full']
opt_samplename = samplenames
opt_random_state = range(1)

# add additional values for holdout
holdout_n = 200
for i in mm.index: mm.loc[i,'kwargs']['holdout_n'] = holdout_n

grid_params = list(it.product(opt_remove_housekeepers, opt_n, opt_obs_method, opt_samplename, opt_random_state))

for params in grid_params:
    
    print(params)
    remove_housekeepers, n, obs_method, samplename, random_state = params
    
    if remove_housekeepers == True: 
        obs = list(datasets[samplename].index.difference(hk))
    elif remove_housekeepers == False:
        obs = list(datasets[samplename].index)
        
    actual_n = len(obs)
   
    unobs = get_unobserved(samplename, obs, n=actual_n, remove_housekeepers=remove_housekeepers, random_state=random_state)
    
    for i, mm_row in mm.iterrows():
        methodname = mm_row.name
        
        run_identifier = f'{samplename}_{n}_{int(remove_housekeepers)}rhk_{methodname}_{obs_method}_k{random_state}'
        filename = f'source_data/figure2/{run_identifier}.csv.gz'
        
        if not os.path.exists(filename):
            if methodname != 'PROSE (random)':
                result = mm_row.func(obs, unobs, **mm_row.kwargs)
            else:
                U = list(set(obs).union(set(unobs)))
                rand_obs, rand_unobs = U[:int(len(U)/2)], U[int(len(U)/2):]
                result = mm_row.func(rand_obs, rand_unobs, **mm_row.kwargs)
                
            result.to_csv(filename, compression='gzip')
            print(samplename, methodname, find_auc(result[result['is_test_set'] == 1]))
            
        else:
            print(f'{filename} already found!')



# for downsampling analysis 

# iterate over various parameters
opt_remove_housekeepers = [True, False]
opt_n = [1000, 2000, 1500, 500, 200, 100]
opt_obs_method = ['random','top_n']
opt_samplename = samplenames
opt_random_state = range(1)

grid_params = list(it.product(opt_remove_housekeepers, opt_n, opt_obs_method, opt_samplename, opt_random_state))

for params in grid_params:
    
    print(params)
    remove_housekeepers, n, obs_method, samplename, random_state = params
    
    actual_n = n+holdout_n    
    
    obs = get_observed(samplename, n=actual_n, method=obs_method, remove_housekeepers=remove_housekeepers, random_state=random_state)
    unobs = get_unobserved(samplename, obs, n=actual_n, remove_housekeepers=remove_housekeepers, random_state=random_state)
    
    for i, mm_row in mm.iterrows():
        methodname = mm_row.name
        
        run_identifier = f'{samplename}_{n}_{int(remove_housekeepers)}rhk_{methodname}_{obs_method}_k{random_state}'
        filename = f'source_data/figure2/{run_identifier}.csv.gz'
        
        if not os.path.exists(filename):
            if methodname != 'PROSE (random)':
                result = mm_row.func(obs, unobs, **mm_row.kwargs)
            else:
                U = list(set(obs).union(set(unobs)))
                rand_obs, rand_unobs = U[:int(len(U)/2)], U[int(len(U)/2):]
                result = mm_row.func(rand_obs, rand_unobs, **mm_row.kwargs)
                
            result.to_csv(filename, compression='gzip')
            print(samplename, methodname, find_auc(result[result['is_test_set'] == 1]))
            
        else:
            print(f'{filename} already found!')
        

#%% Load orthogonal HeLa TPM

tpm = pd.read_csv('datasets/transcriptomics/klijn_cancerlines_tpm/E-MTAB-2706-query-results.tpms.tsv', comment='#', sep='\t')
tpm = tpm[['Gene ID', 'uterine cervix, cervical adenocarcinoma, HeLa']]
tpm.set_index('Gene ID', inplace=True)
tpm.index = [conv_ensembl_to_uniprot[i] if i in conv_ensembl_to_uniprot else np.nan for i in tpm.index]
tpm = tpm.loc[tpm.index.dropna()]
tpm = tpm[~tpm.index.duplicated(keep='first')]
tpm.dropna(inplace=True)
tpm.columns = ['tpm']   

#%% Generate metrics dataframe

import glob
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

def find_correlation(result, vector, method='spearmanr', result_attribute='score',
                      test_set=1):
        
    res = result.copy()

    if test_set == 1:
        res = res[res['is_test_set'] == 1]
        
    elif test_set == 0:
        res = res[res['is_test_set'] == 0]
        
    elif test_set == 'exclude_test':
        res = res[res['is_test_set'] != 1]
    
    res = res.dropna()[result_attribute]
    vec = vector.dropna()
    matched_indices = res.index.intersection(vec.index)
    
    if method == 'pearsonr':
        r = pearsonr(res.loc[matched_indices], vec.loc[matched_indices])[0]
    elif method == 'spearmanr':
        r = spearmanr(res.loc[matched_indices], vec.loc[matched_indices])[0]                                                     
        
    return round(r,3)


fs = glob.glob('source_data/figure2/*.csv.gz')

df = pd.DataFrame(columns=['r_ibaq', 'r_tpm',
                           'auc_train','r_ibaq_train','r_tpm_train',
                           'auc_test','r_ibaq_test','r_tpm_test',
                           'acquisition','biosample','rep','n','remove_hk','method',
                           'samplemethod','simulation'])

for i, f in tqdm(enumerate(fs)):
    fname_clean =  f.replace('.csv.gz','').replace('top_n', 'topn').split('\\')[-1]
    identifiers = fname_clean.split('_')
    samplename = '_'.join(identifiers[:3])
    result = pd.read_csv(f, index_col=0)
    
    auc_test = find_auc(result[result['is_test_set'] == 1])
    try:
        auc_train = find_auc(result[result['is_test_set'] == 0])
    except:
        auc_train = np.NaN
    
    # if ('DDA_HeLa' in fname_clean) or ('DIA_HeLa' in fname_clean):   
    ibaq = datasets[samplename]
    r_ibaq = find_correlation(result, ibaq, test_set=None)
    r_ibaq_test = find_correlation(result, ibaq, test_set=1)
    r_ibaq_train = find_correlation(result, ibaq, test_set=0)
        
    if 'HeLa' in fname_clean:   
        r_tpm = find_correlation(result, tpm['tpm'], test_set=None)
        r_tpm_test = find_correlation(result, tpm['tpm'], test_set=1)
        r_tpm_train = find_correlation(result, tpm['tpm'], test_set=0)

    df.loc[i] = [r_ibaq, r_tpm,
                 auc_train, r_ibaq_train, r_tpm_train,
                 auc_test, r_ibaq_test, r_tpm_test] + identifiers
    
df.to_csv('source_data/figure2_aggregated.csv')
            
#%% Plot Figure 2 results

import itertools as it
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm
    
#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 1.0)

df = pd.read_csv('source_data/figure2_aggregated.csv', index_col=0)
df['dataset'] = [f"{row['acquisition']}_{row['biosample']}" for i,row in df.iterrows()]
df['method'] = ['UMAP-KNN' if i == 'UMAPKNN' else i for i in df['method']]
df['method'] = ['ToppGene (default)' if i == 'ToppGene' else i for i in df['method']]

for n, remove_hk, samplemethod in list(it.product([2000,1500,1000,500,200],['1rhk','0rhk'],['random','topn']))+\
                                  list(it.product(['full'],['1rhk','0rhk'],['full'])):
    # change defaults to generate plots for other parameters
    defaults = dict(n=str(n), remove_hk=remove_hk, samplemethod=samplemethod)
    n, remove_hk, samplemethod = defaults['n'], defaults['remove_hk'], defaults['samplemethod']
    # plot for default implementation
    
    data = df[(df['n']==n)&
              (df['remove_hk']==remove_hk) &
              (df['samplemethod']==samplemethod)]
    
    
    method_order = ['PROSE', 'MaxLink', 'MaxLink (norm.)',
                    'PROTREC',
                    'Node2Vec', 'UMAP-KNN',
                    'RWRH', 'ToppGene (default)', 'ToppGene (coexpr.)']
    
    titles = dict(auc='auROC',
                  r_ibaq='Rank correlation ($ρ$)\nProtein expression',
                  r_tpm='Rank correlation ($ρ$)\nGene expression',
                  )
    
    data.loc[data['method'].isin(['UMAP-KNN', 'RWRH']),'auc_train'] = np.nan
    
    maxcolors=10
    cmap = matplotlib.cm.get_cmap('tab10_r')
    colors = [cmap(i/maxcolors) for i in range(maxcolors)]
    
    for metric in ['auc','r_ibaq','r_tpm']:
        
        kws_training_set = dict(capsize=0.05, alpha=0.2, width=1, y=metric+'_train')
        kws_test_set = dict(capsize=0.05, alpha=0.8, width=1, y=metric+'_test')
        kws_main = dict(capsize=0.05, alpha=0.8, width=1, y=metric)
    
        for dataset, sub in data.groupby('dataset'):
            
            if ('THP1' in dataset) and (metric == 'tpm'):
                continue
    
            fig, ax = plt.subplots(figsize=[8,10])
            
            if (metric == 'auc'):
        
                g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
                                palette=colors, errorbar='sd',
                                **kws_test_set)
                
            else:
                
                g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
                    palette=colors, errorbar='sd',
                    **kws_test_set)
                
                # g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
                #     palette=colors, errorbar='sd',
                #     **kws_training_set)              
                
            if metric == 'auc': 
                plt.ylim(0.4)
                g.axhline(0.5, color='black', linestyle=':')
            else: 
                plt.ylim(0)
                g.axhline(0, color='black', linestyle=':')
            
            plt.legend().set_visible(False)
            plt.xlim(-0.6)
            ax.set_title(titles[metric], pad=25)
            ax.set_ylabel(''); ax.set_xlabel('')
            ax.set_xticklabels([])
            ax.tick_params(axis='x', size=0)
            ax.tick_params(axis='y', size=20, pad=15)
            sns.despine()
            
            plt.tight_layout()
            
            directory = f'plots/fig2/{remove_hk}/{n}/{samplemethod}/{dataset}'
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            plt.savefig(f'{directory}/{metric}.png',
                        bbox_inches='tight',
                        dpi=600,)
    
#%% plot legend

fig, ax = plt.subplots(figsize=[8,12])
g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
                palette=colors,
                **kws_test_set)
ax.legend(loc='center left', bbox_to_anchor=(1, .5),frameon=False)
plt.tight_layout()
plt.savefig(f'plots/fig2/legendbox.png',
                bbox_inches='tight',
                dpi=600,)

#%% Plot metrics across different levels of missingness (with housekeepers removed)

for dataset, samplemethod in list(it.product(['DDA_HeLa', 'DIA_HeLa','DDA_THP1'], ['random', 'topn'])):
    
    data = df[df['samplemethod'] == samplemethod].copy()
    data = data[data['remove_hk'] == '1rhk']
    data = data[data['dataset'] == dataset]
    data['n'] = data['n'].astype(int)
    
    method_order = ['PROSE', 'MaxLink', 'MaxLink (norm.)',
                    'PROTREC',
                    'Node2Vec', 'UMAP-KNN',
                    'RWRH', 'ToppGene (default)', 'ToppGene (coexpr.)']
    
    
    fig, axes = plt.subplots(figsize=[20,40], nrows=3, ncols=1, sharex=True)
    
    
    kws = dict(lw=10, markers=True,
               markersize=40,
               err_style=None,
               marker='o',
               )

    
    ax=axes[0]
    g=sns.lineplot(data=data, x='n', y='auc_test', hue='method', hue_order=method_order,
                   palette=colors, ax=ax, **kws)
    ax.legend().set_visible(False)
    ax.set_ylabel('auROC', labelpad=20)
    ax.set_xlabel('')
    
    ax=axes[1]
    g=sns.lineplot(data=data, x='n', y='r_ibaq_test', hue='method', hue_order=method_order,
                   palette=colors, ax=ax, **kws)
    ax.legend().set_visible(False)
    ax.set_ylabel('Rank correlation ($ρ$)\nProtein expression', labelpad=20)
    ax.set_xlabel('')
    
    ax=axes[2]
    g=sns.lineplot(data=data, x='n', y='r_tpm_test', hue='method', hue_order=method_order,
                   palette=colors, ax=ax, **kws)
    ax.legend().set_visible(False)
    ax.set_ylabel('Rank correlation ($ρ$)\nGene expression', labelpad=20)
    ax.set_xlabel('')
    
    plt.xticks([2000,1500,1000,500,200], size=50)
    
    plt.gca().invert_xaxis()
    
    sns.despine()
    
    dataset_label = ' '.join(dataset.split('_')[::-1])
    
    samplemethod_label = {'random':'Random sampled proteins',
                          'topn':'Top n proteins'}[samplemethod]
    
    savefilename = f'{dataset_label}_{samplemethod_label}.png'
    
    axes[0].set_title(f'{dataset_label}, {samplemethod_label}',
                      size=80, pad=50)
    
    plt.savefig(f'plots/fig2/dropout_metric/{savefilename}',
                bbox_inches='tight',
                dpi=600,)
    
    plt.show()



#%% plot legend

fig, ax = plt.subplots(figsize=[10,20])
g=sns.lineplot(data=data, x='n', y='auc_test', hue='method', hue_order=method_order,
                   palette=colors, ax=ax, **kws)
ax.legend().set_visible(True)
ax.legend(loc='center left', bbox_to_anchor=(1, .5),frameon=False,
          markerscale=5)
plt.tight_layout()

for lh in ax.legend_.legendHandles: 
    lh.set_lw(kws['lw'])
    lh.set_marker('o')


plt.savefig(f'plots/fig2/legendline.png',
                bbox_inches='tight',
                dpi=600,)
plt.show()







#%%






























#%%
# #%% Plot results, including housekeepers

# import seaborn as sns
# import matplotlib.pyplot as plt

# #plot parameters
# plt.rcParams['mathtext.fontset'] = 'custom'
# plt.rcParams['mathtext.it'] = 'Arial:italic'
# plt.rcParams['mathtext.rm'] = 'Arial'
# plt.rc('font',family='arial',size=40)
# plt.rc('hatch',linewidth = 1.0)

# df = pd.read_csv('source_data/figure2_aggregated.csv', index_col=0)
# df['dataset'] = [f"{row['acquisition']}_{row['biosample']}" for i,row in df.iterrows()]
# df['method'] = ['UMAP-KNN' if i == 'UMAPKNN' else i for i in df['method']]
# df['method'] = ['ToppGene (default)' if i == 'ToppGene' else i for i in df['method']]


# defaults = dict(n=1000, remove_hk='0rhk', samplemethod='random')
# # plot for default implementation

# data = df[(df['n']==defaults['n'])&
#           (df['remove_hk']==defaults['remove_hk'])&
#           (df['samplemethod']==defaults['samplemethod'])]


# method_order = ['PROSE', 'PROSE (random)', 'MaxLink', 'MaxLink (norm.)',
#                 'PROTREC',
#                 'Node2Vec', 'UMAP-KNN',
#                 'RWRH', 'ToppGene (default)', 'ToppGene (coexpr.)']

# titles = dict(auc='auROC',
#               r_ibaq='Rank correlation ($ρ$)\nProtein expression',
#               r_tpm='Rank correlation ($ρ$)\nGene expression',
#               )

# # replace NA values with 0
# data = data.fillna(-1)
# data.loc[data['method'].isin(['UMAP-KNN', 'RWRH']),'auc_train'] = -1

# maxcolors=10
# cmap = matplotlib.cm.get_cmap('tab10_r')
# colors = [cmap(i/maxcolors) for i in range(maxcolors)]

# for metric in ['auc','r_ibaq','r_tpm']:
    
#     kws_training_set = dict(capsize=0.05, alpha=0.1, width=1, y=metric+'_train')
#     kws_test_set = dict(capsize=0.05, alpha=0.8, width=1, y=metric+'_test')
#     kws_main = dict(capsize=0.05, alpha=0.8, width=1, y=metric)

#     for dataset, sub in data.groupby('dataset'):
        
#         if ('THP1' in dataset) and (metric == 'tpm'):
#             continue
        
        
#         fig, ax = plt.subplots(figsize=[8,10])
        
#         if (metric == 'auc'):#or (metric == 'r_ibaq')
    
#             g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
#                             palette=colors, errorbar='sd',
#                             **kws_test_set)
            
#         else:
            
#             g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
#                 palette=colors, errorbar='sd',
#                 **kws_main)
            
            
#         if metric == 'auc': 
#             plt.ylim(0.4)
#             g.axhline(0.5, color='black', linestyle=':')
#         else: 
#             plt.ylim(0)
        
#         plt.legend().set_visible(False)
#         plt.xlim(-0.6)
#         ax.set_title(titles[metric], pad=25)
#         ax.set_ylabel(''); ax.set_xlabel('')
#         ax.set_xticklabels([])
#         ax.tick_params(axis='x', size=0)
#         ax.tick_params(axis='y', size=20, pad=15)
#         sns.despine()
        
#         plt.tight_layout()
        
#         plt.savefig(f'plots/fig2_withhk_1000_random_{dataset}_{metric}.png',
#                     bbox_inches='tight',
#                     dpi=600,)


            
# # #%% Plot Figure 2 results (using top_n approach)

# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # #plot parameters
# # plt.rcParams['mathtext.fontset'] = 'custom'
# # plt.rcParams['mathtext.it'] = 'Arial:italic'
# # plt.rcParams['mathtext.rm'] = 'Arial'
# # plt.rc('font',family='arial',size=40)
# # plt.rc('hatch',linewidth = 1.0)

# # defaults = dict(n=1000, remove_hk='1rhk', samplemethod='topn')
# # # plot for default implementation

# # data = df[(df['n']==defaults['n'])&
# #           (df['remove_hk']==defaults['remove_hk'])&
# #           (df['samplemethod']==defaults['samplemethod'])]


# # method_order = ['PROSE', 'PROSE (random)', 'MaxLink', 'MaxLink (norm.)',
# #                 'PROTREC',
# #                 'Node2Vec', 'UMAP-KNN',
# #                 'RWRH', 'ToppGene (default)', 'ToppGene (coexpr.)']

# # titles = dict(auc='auROC',
# #               r_ibaq='Rank correlation ($ρ$)\nProtein expression',
# #               r_tpm='Rank correlation ($ρ$)\nGene expression',
# #               )

# # # replace NA values with 0
# # data = data.fillna(-1)
# # data.loc[data['method'].isin(['UMAP-KNN', 'RWRH']),'auc_train'] = -1

# # maxcolors=10
# # cmap = matplotlib.cm.get_cmap('tab10_r')
# # colors = [cmap(i/maxcolors) for i in range(maxcolors)]

# # for metric in ['auc','r_ibaq','r_tpm']:
    
# #     kws_training_set = dict(capsize=0.05, alpha=0.1, width=1, y=metric+'_train')
# #     kws_test_set = dict(capsize=0.05, alpha=0.8, width=1, y=metric+'_test')
# #     kws_main = dict(capsize=0.05, alpha=0.8, width=1, y=metric)

# #     for dataset, sub in data.groupby('dataset'):
        
# #         if ('THP1' in dataset) and (metric == 'tpm'):
# #             continue
        
        
# #         fig, ax = plt.subplots(figsize=[8,10])
        
# #         if (metric == 'auc'):#or (metric == 'r_ibaq')
    
# #             g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
# #                             palette=colors, errorbar='sd',
# #                             **kws_test_set)
            
# #         else:
            
# #             g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
# #                 palette=colors, errorbar='sd',
# #                 **kws_main)
            
            
# #         if metric == 'auc': 
# #             plt.ylim(0.4)
# #             g.axhline(0.5, color='black', linestyle=':')
# #         else: 
# #             plt.ylim(0)
        
# #         plt.legend().set_visible(False)
# #         plt.xlim(-0.6)
# #         ax.set_title(titles[metric], pad=25)
# #         ax.set_ylabel(''); ax.set_xlabel('')
# #         ax.set_xticklabels([])
# #         ax.tick_params(axis='x', size=0)
# #         ax.tick_params(axis='y', size=20, pad=15)
# #         sns.despine()
        
# #         plt.tight_layout()
        
# #         plt.savefig(f'plots/fig2_nohk_1000_topn_{dataset}_{metric}.png',
# #                     bbox_inches='tight',
# #                     dpi=600,)

        
# # #%% plot legend

# # fig, ax = plt.subplots(figsize=[8,12])
# # g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
# #                 palette=colors,
# #                 **kws_test_set)
# # ax.legend(loc='center left', bbox_to_anchor=(1, .5),frameon=False)
# # plt.tight_layout()
# # plt.savefig(f'plots/fig2_legendbox.png',
# #                 bbox_inches='tight',
# #                 dpi=600,)

# # #%% Plot results, including housekeepers

# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # #plot parameters
# # plt.rcParams['mathtext.fontset'] = 'custom'
# # plt.rcParams['mathtext.it'] = 'Arial:italic'
# # plt.rcParams['mathtext.rm'] = 'Arial'
# # plt.rc('font',family='arial',size=40)
# # plt.rc('hatch',linewidth = 1.0)

# # df = pd.read_csv('source_data/figure2_aggregated.csv', index_col=0)
# # df['dataset'] = [f"{row['acquisition']}_{row['biosample']}" for i,row in df.iterrows()]
# # df['method'] = ['UMAP-KNN' if i == 'UMAPKNN' else i for i in df['method']]
# # df['method'] = ['ToppGene (default)' if i == 'ToppGene' else i for i in df['method']]


# # defaults = dict(n=1000, remove_hk='0rhk', samplemethod='topn')
# # # plot for default implementation

# # data = df[(df['n']==defaults['n'])&
# #           (df['remove_hk']==defaults['remove_hk'])&
# #           (df['samplemethod']==defaults['samplemethod'])]


# # method_order = ['PROSE', 'PROSE (random)', 'MaxLink', 'MaxLink (norm.)',
# #                 'PROTREC',
# #                 'Node2Vec', 'UMAP-KNN',
# #                 'RWRH', 'ToppGene (default)', 'ToppGene (coexpr.)']

# # titles = dict(auc='auROC',
# #               r_ibaq='Rank correlation ($ρ$)\nProtein expression',
# #               r_tpm='Rank correlation ($ρ$)\nGene expression',
# #               )

# # # replace NA values with 0
# # data = data.fillna(-1)
# # data.loc[data['method'].isin(['UMAP-KNN', 'RWRH']),'auc_train'] = -1

# # maxcolors=10
# # cmap = matplotlib.cm.get_cmap('tab10_r')
# # colors = [cmap(i/maxcolors) for i in range(maxcolors)]

# # for metric in ['auc','r_ibaq','r_tpm']:
    
# #     kws_training_set = dict(capsize=0.05, alpha=0.1, width=1, y=metric+'_train')
# #     kws_test_set = dict(capsize=0.05, alpha=0.8, width=1, y=metric+'_test')
# #     kws_main = dict(capsize=0.05, alpha=0.8, width=1, y=metric)

# #     for dataset, sub in data.groupby('dataset'):
        
# #         if ('THP1' in dataset) and (metric == 'tpm'):
# #             continue
        
        
# #         fig, ax = plt.subplots(figsize=[8,10])
        
# #         if (metric == 'auc'):#or (metric == 'r_ibaq')
    
# #             g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
# #                             palette=colors, errorbar='sd',
# #                             **kws_test_set)
            
# #         else:
            
# #             g = sns.barplot(data=sub, x='n', hue='method', hue_order=method_order, 
# #                 palette=colors, errorbar='sd',
# #                 **kws_main)
            
            
# #         if metric == 'auc': 
# #             plt.ylim(0.4)
# #             g.axhline(0.5, color='black', linestyle=':')
# #         else: 
# #             plt.ylim(0)
        
# #         plt.legend().set_visible(False)
# #         plt.xlim(-0.6)
# #         ax.set_title(titles[metric], pad=25)
# #         ax.set_ylabel(''); ax.set_xlabel('')
# #         ax.set_xticklabels([])
# #         ax.tick_params(axis='x', size=0)
# #         ax.tick_params(axis='y', size=20, pad=15)
# #         sns.despine()
        
# #         plt.tight_layout()
        
# #         plt.savefig(f'plots/fig2_withhk_1000_topn_{dataset}_{metric}.png',
# #                     bbox_inches='tight',
# #                     dpi=600,)