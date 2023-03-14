# -*- coding: utf-8 -*-
"""
See patterns of protein recovery based on PROSE scores
"""

import pandas as pd
import pickle 
import os.path
import numpy as np


# load datasets 
with open('datasets/proteomics/fig2_proteinlists.pkl', 'rb') as handle:
    datasets = pickle.load(handle)

# get sample data as agg_id
samplenames = datasets.keys()
sampledata = pd.DataFrame([i.split('_') for i in samplenames],
                          index = samplenames,
                          columns = ['method', 'sample', 'rep'])

agg_id = pd.read_csv('source_data/figure3_aggregated.csv', index_col=0)

#%%

import glob

fs = glob.glob('source_data/figure2/*_1000_1rhk_PROSE_random_k0.csv.gz')

# for f in fs:
#     df = pd.read_csv(f, index_col=0)

# dda_samp = 'DDA HeLa R1'
# f = f'source_data/figure2/{"_".join(dda_samp.split(" "))}_1000_1rhk_PROSE_random_k0.csv.gz'

# dia_ids = agg_id[(agg_id['DIA HeLa R1'] == 1) &
#                  (agg_id['DIA HeLa R2'] == 1) &
#                  (agg_id['DIA HeLa R3'] == 1) &
#                  (agg_id['DIA HeLa R4'] == 1)].index

# mismatch_ids = agg_id[(agg_id['DDA THP1 R1'] == 1) &
#                       (agg_id['DDA THP1 R2'] == 1) &
#                       (agg_id['DDA THP1 R3'] == 1)].index

# missing_dda = agg_id[(agg_id[dda_samp] == 0) &
#                      (agg_id.index.isin(dia_ids))].index

# mismatch_dda = agg_id[(agg_id[dda_samp] == 0) &
#                      (agg_id.index.isin(mismatch_ids))].index

# result = pd.read_csv(f, index_col=0)

# missing_scores = result[result.index.isin(missing_dda)]\
#                  .sort_values(by='score_norm', ascending=False)
                 
# mismatch_scores = result[result.index.isin(mismatch_dda)]\
#                  .sort_values(by='score_norm', ascending=False)
                 
#%% Plot missing recovery

import seaborn as sns
import matplotlib.pyplot as plt

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 1.0)





fig, ax = plt.subplots(figsize=[10,10])

for dda_samp in ['DDA HeLa R1', 'DDA HeLa R2', 'DDA HeLa R3', 'DDA HeLa R4']:
# dda_samp = 'DDA HeLa R1'
    f = f'source_data/figure2/{"_".join(dda_samp.split(" "))}_1000_1rhk_PROSE_random_k0.csv.gz'
    
    dia_ids = agg_id[(agg_id['DIA HeLa R1'] == 1) |
                     (agg_id['DIA HeLa R2'] == 1) |
                     (agg_id['DIA HeLa R3'] == 1) |
                     (agg_id['DIA HeLa R4'] == 1)].index
    
    mismatch_ids = agg_id[(agg_id['DDA THP1 R1'] == 1) |
                          (agg_id['DDA THP1 R2'] == 1) |
                          (agg_id['DDA THP1 R3'] == 1)].index

    dia_specific = dia_ids.difference(mismatch_ids)
    mismatch_specific = mismatch_ids.difference(dia_ids)
    
    missing_dda = agg_id[(agg_id[dda_samp] == 0) &
                         (agg_id.index.isin(dia_specific))].index
    
    mismatch_dda = agg_id[(agg_id[dda_samp] == 0) &
                         (agg_id.index.isin(mismatch_specific))].index

    
    result = pd.read_csv(f, index_col=0)
    
    missing_scores = result[result.index.isin(missing_dda)]\
                     .sort_values(by='score_norm', ascending=False)
                     
                     
    mismatch_scores = result[result.index.isin(mismatch_dda)]\
                     .sort_values(by='score_norm', ascending=False)
                     
    
    sns.kdeplot(data = missing_scores, x='prob', color='orange', cut=1)
    sns.kdeplot(data = mismatch_scores, x='prob', color='cornflowerblue', cut=1)
    
    sns.despine()


plt.xlabel('LR Prob(protein)')
plt.ylabel('HeLa DDA \nunobserved proteins', labelpad=20)



#%%


# fig, ax = plt.subplots(figsize=[10,10])

# for dda_samp in ['DDA THP1 R1', 'DDA THP1 R2', 'DDA THP1 R3']:
# # dda_samp = 'DDA HeLa R1'
#     f = f'source_data/figure2/{"_".join(dda_samp.split(" "))}_1000_1rhk_PROSE_top_n_k0.csv.gz'
    
#     mismatch_ids = agg_id[(agg_id['DIA HeLa R1'] == 1) &
#                           (agg_id['DIA HeLa R2'] == 1) &
#                           (agg_id['DIA HeLa R3'] == 1) &
#                           (agg_id['DIA HeLa R4'] == 1)].index

#     other_samps = [i for i in ['DDA THP1 R1', 'DDA THP1 R2', 'DDA THP1 R3'] if i != dda_samp]
    
#     missing_ids = agg_id[(agg_id[other_samps[0]] == 1) &
#                           (agg_id[other_samps[1]] == 1)].index
    
#     missing_dda = agg_id[(agg_id[dda_samp] == 0) &
#                           (agg_id.index.isin(dia_ids))].index
    
    
#     dia_specific = dia_ids.difference(mismatch_ids)
#     mismatch_specific = mismatch_ids.difference(dia_ids)
    
#     #%%
    
#     missing_dda = agg_id[(agg_id[dda_samp] == 0) &
#                          (agg_id.index.isin(dia_specific))].index
    
#     mismatch_dda = agg_id[(agg_id[dda_samp] == 0) &
#                          (agg_id.index.isin(mismatch_specific))].index

    
#     result = pd.read_csv(f, index_col=0)
    
#     missing_scores = result[result.index.isin(missing_dda)]\
#                      .sort_values(by='score_norm', ascending=False)
                     
                     
#     mismatch_scores = result[result.index.isin(mismatch_dda)]\
#                      .sort_values(by='score_norm', ascending=False)
                     
    
#     sns.ecdfplot(data = missing_scores, x='prob', color='orange')
#     sns.ecdfplot(data = mismatch_scores, x='prob', color='cornflowerblue')
    
#     sns.despine()

# plt.xlabel('LR Prob(protein)')
# plt.ylabel('Proportion of HeLa DDA \nmissing proteins', labelpad=20)
