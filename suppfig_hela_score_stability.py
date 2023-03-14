# -*- coding: utf-8 -*-
"""
Replot phenotypic specificity using new DDA dataset
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

#%%
import itertools as it
import os

remove_housekeepers = True
n = 2000
obs_method = 'random'
holdout_n = 0

opt_samplename = [i for i in samplenames if 'DDA_HeLa' in i]
opt_random_state = [1,2]
opt_n_estimators = [10,50,100,200,500,1000]

grid_params = list(it.product(opt_random_state, opt_n_estimators, opt_samplename))

for params in grid_params:
    random_state, n_estimators, samplename = params
    
    filename = f'source_data/suppfig_score_stability/{samplename}_{n_estimators}k_{random_state}.csv.gz'
    
    if not os.path.exists(filename):
    
        obs = get_observed(samplename, n=n, method=obs_method, remove_housekeepers=remove_housekeepers)
        unobs = get_unobserved(samplename, obs, n=n, remove_housekeepers=remove_housekeepers)
    
        result = pyprose(obs, unobs, panel_corr=panel_corr,
                         n_estimators=n_estimators,
                         random_state=random_state,
                         bag_kws={'random_state':random_state})
        
        result.to_csv(filename, compression='gzip')
        
    else:
        print(f'{filename} already exists!')

    
#%% 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 1.0)


for n_estimators in opt_n_estimators:
    
    fig, axes = plt.subplots(figsize=[20,20], nrows=4, ncols=4)
    
    for i,j in it.product(opt_samplename, opt_samplename):
        rep1, rep2 = i[-1], j[-1]
        row, col = int(rep1)-1, int(rep2)-1

        if col<row:
            fig.delaxes(axes[row, col])
            continue

        data_i = pd.read_csv(f'source_data/suppfig_score_stability/{i}_{n_estimators}k_1.csv.gz', index_col=0)
            
        if rep1==rep2:
            data_j = pd.read_csv(f'source_data/suppfig_score_stability/{j}_{n_estimators}k_2.csv.gz', index_col=0)
        else:
            data_j = pd.read_csv(f'source_data/suppfig_score_stability/{j}_{n_estimators}k_1.csv.gz', index_col=0)
        
        print(rep1, rep2, pearsonr(data_i.score_norm, data_j.score_norm))
    
        ax = axes[row, col]
        data = pd.DataFrame(zip(data_i.score_norm, data_j.score_norm))   
        sns.scatterplot(data=data, x=0, y=1, ax=ax, alpha=0.5, s=10)  
        ax.set_xlabel('') ; ax.set_ylabel('')
        ax.set_yticks([]); ax.set_xticks([])
        
        if row==0:
            ax.text(x=0.5, y=1.03, s=j.replace('DDA_', '').replace('_', ' '),
                    transform=ax.transAxes, ha='center')
        
        if col==3:
            ax.text(x=1.03, y=0.5, s=i.replace('DDA_', '').replace('_', ' '),
                    transform=ax.transAxes, va='center', rotation=270)
            
        ax.plot([-4,4],[-4,4], color='black', linestyle=':', alpha=0.4, linewidth=4)
        
        r = round(pearsonr(data_i.score_norm, data_j.score_norm)[0],3)
        #rho = round(spearmanr(data_i.score_norm, data_j.score_norm)[0],3)
        
        ax.text(x=0.05, y=0.95, s=f'$r$ = {r}',
                transform=ax.transAxes, va='top', size=40)
        
        # break

    plt.text(-3,0.5,s=f'$k$ = {n_estimators}', transform=ax.transAxes, size=80)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    # break

    plt.savefig(f'plots/suppfig_score_stability_helaDDA/{n_estimators}_estimators.png',
                bbox_inches='tight',
                dpi=600,)
    

    