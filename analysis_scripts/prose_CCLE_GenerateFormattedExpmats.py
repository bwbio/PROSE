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

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

#%% Get TMT quants matrix

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
expmat_withMV = expmat.copy()

#matrix with missing values
tmt_withMV = pd.concat([md, expmat],axis=1).drop_duplicates('cell_line')
tmt_withMV.to_csv('ccle/ccle_tmt_formatted_withMissingVals.tsv.gz', index=False, sep='\t', compression = 'gzip')

# fill NA values with cell-line specific TMT minimum
expmat = expmat.apply(lambda x: x.fillna(x.min()),axis=1)
proteins = expmat.columns.to_list()

tmt = pd.concat([md, expmat],axis=1).drop_duplicates('cell_line')
tmt.to_csv('ccle/ccle_tmt_formatted.tsv.gz', index=False, sep='\t', compression = 'gzip')

#%% Dictionary for gene-to-protein ID conversion

conv_base=pd.read_csv("databases/ensembl_uniprot_conversion.tsv",
                      sep='\t',
                      comment='#',
                      )

conv_base = conv_base.rename(columns={'ID':'gene',
                                      'Entry': 'uniprot',
                                      'Gene names': 'names'})
conv_base[['Entry name']] = conv_base.apply(lambda x: x['Entry name'].split('_')[0],axis=1)

conv = conv_base[['gene','uniprot']]
conv = dict(zip(conv.gene,conv.uniprot))
validGenes = conv.keys() #set of genes with associated protein names

conv_commonName = dict(zip(conv_base.uniprot, conv_base['Entry name']))

conv0 = {}
temp = conv_base[['uniprot','names']]
for i, row in conv_base.iterrows():
    if type(row.names) == str:
        for i in row.names.split(' '):
            conv0[re.sub(r'[^a-zA-Z0-9]', '', i)] = row.uniprot


#%% Get TPM matrix

tpm=pd.read_csv('ccle/E-MTAB-2770-query-results.tpms.tsv',
                 sep='\t',
                 skiprows=4
                 )

genes = list(tpm['Gene ID'].values)
tpm = tpm.drop(['Gene ID', 'Gene Name'], axis=1).T
tpm.reset_index(inplace=True)
tpm.columns = ['source']+genes
metacols = ['cancer','cell_line']
tpm[metacols] = tpm.source.str.split(', ',n=1,expand=True)

#format metadata
tpm['cell_line'] = tpm.apply(lambda x: x.cell_line.split(',')[-1],axis=1)
tpm['cell_line'] = tpm.apply(lambda x: re.sub(r'[^a-zA-Z0-9]', '', x.cell_line),axis=1)
tpm['cell_line'] = tpm.apply(lambda x: x.cell_line.upper(),axis=1)

metacols_tpm = tpm[metacols] #df containing cell line metadata

#restrict list of valid genes to those in the RNA-seq data
validGenes = list(tpm[genes].columns.intersection(validGenes))
tpm = tpm[validGenes]
tpm = tpm.fillna(0)

#log2+1 transform
tpm = np.log2(tpm+1)
tpm = pd.concat([metacols_tpm, tpm],axis=1)

#restrict to cell lines with matching identifiers in the proteomic screen
tpm = tpm.rename(columns=conv)
tpm = tpm.drop(columns='cancer').drop_duplicates('cell_line')
tpm.to_csv('ccle/ccle_tpm_formatted.tsv.gz', index=False, sep='\t', compression = 'gzip')

#%% Get DRIVE dependency matrix

drive = pd.read_csv('ccle/D2_DRIVE_gene_dep_scores.csv')
drive = drive.rename(columns={'Unnamed: 0':'gene'})

#gene renaming
drive.gene = drive.apply(lambda x: re.split('&|-',x.gene.split(' (')[0]), axis=1)
drive = drive[drive.gene.apply(lambda x: len(x)) == 1]
drive.gene = drive.apply(lambda x: x.gene[0], axis=1)
drive = drive[drive.gene.isin(conv0.keys())]
drive['protein'] = drive.apply(lambda x: conv0[x.gene], axis=1)
drive = drive.reset_index(drop = True)

identifiers = drive[['gene','protein']].copy()
drive_expmat = drive.drop(columns=['gene','protein'])
drive_expmat.columns = [i.split('_')[0] for i in drive_expmat.columns]
drive_expmat = drive_expmat[drive_expmat.columns.intersection(cell_lines)]
drive_expmat = drive_expmat.fillna(0)
drive_cell_lines = drive_expmat.columns
drive = pd.concat([identifiers,drive_expmat],axis=1)

drive.to_csv('ccle/ccle_drive_formatted_withGeneName.tsv.gz', index=False, sep='\t', compression = 'gzip')

drive = drive.drop(columns='gene')
drive = drive.set_index('protein')
drive = drive.T
drive = drive.reset_index()
drive = drive.rename(columns={'index':'cell_line'}).drop_duplicates('cell_line')

drive.to_csv('ccle/ccle_drive_formatted.tsv.gz', index=False, sep='\t', compression = 'gzip')