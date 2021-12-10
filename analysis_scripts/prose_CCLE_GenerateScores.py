# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:54:13 2021

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
from sklearn.preprocessing import quantile_transform
import pickle
import re
import umap

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%%

df_temp = pd.read_csv('ccle/protein_quant_current_normalized.csv.gz')
expmat = df_temp[df_temp.columns[48:]]
ids = df_temp[df_temp.columns[0]].apply(lambda x: x.split('|')[1]).rename('')
expmat.index = ids

cell_lines = pd.Series([i.split('_')[0] for i in expmat.columns]).rename('cell_line')
tissues = pd.Series([i.split('_',1)[-1].split('_TenPx')[0].lower().replace('_', ' ')\
                     for i in expmat.columns]).rename('tissue')
md=pd.concat([cell_lines,tissues],axis=1)
mdconv = dict(zip(md.cell_line,md.tissue))
metacols = ['cell_line','tissue']
expmat = expmat.T.reset_index(drop=True)
expmat = np.exp(expmat).fillna(0)
proteins = expmat.columns.to_list()
df=pd.concat([md, expmat],axis=1)

#%%

occurences = {}
for i, row in df.iterrows():
    low = row[proteins].quantile(0.5)
    high = row[proteins].quantile(0.5)
    
    obs = row[proteins][row[proteins] > high].index.to_list()
    unobs = row[proteins][row[proteins] <= low].index.to_list()
    occurences[row.cell_line] = {'obs':obs, 'unobs':unobs, 'ratio':len(obs)/len(unobs)}

#%%

panel_corr = pd.read_csv('interim_files/klijn_panel_spearmanCorr.tsv', sep='\t',index_col=0)

#%% Generate PROSE scores for cell lines. n_estimators = 500.

completed = [i.split('\\')[-1].split('.pkl')[0] for i in glob.glob('ccle/prose_result_cell_line/*.pkl')]
prose_result = {}

for cell_line in df.cell_line:
    path = 'ccle/prose_result_cell_line/' + cell_line + '.pkl'
    obs = occurences[cell_line]['obs']
    unobs = occurences[cell_line]['unobs']
    
    if cell_line not in completed:
        print('Processing', cell_line+' ...', end='')
        q=pgx.prose(obs, unobs, panel_corr, bag_kwargs = dict(n_estimators=500))
        with open(path, 'wb') as handle:
            pickle.dump(q, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(' Done')
    else:
        with open(path, 'rb') as handle:
            q = pickle.load(handle)
    prose_result[cell_line] = q 


#%%

result = []
for cell_line in cell_lines:
    tissue = mdconv[cell_line]
    score = prose_result[cell_line].summary.score_norm.to_list()
    result.append([cell_line, tissue]+score)

result = pd.DataFrame(result)
result.columns = ['cell_line', 'tissue'] + q.summary.protein.to_list()
result = result.drop_duplicates('cell_line')
result.to_csv('ccle/ccle_prose_formatted.tsv.gz', index=False, sep='\t', compression = 'gzip')
    