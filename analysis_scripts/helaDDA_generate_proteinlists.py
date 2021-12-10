# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:31:19 2021

@author: bw98j
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import umap
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


#%%

conv=pd.read_csv("databases/ensembl_uniprot_conversion.tsv",
               sep='\t',
               comment='#',
               )
conv = conv.rename(columns={'ID':'gene',
                            'Entry': 'uniprot'})
conv = conv[['gene','uniprot']]
conv = dict(zip(conv.gene,conv.uniprot))
validGenes = conv.keys()
validProts = conv.values()

#%% read HeLa peptide list

DDA_results = {}
naiive_barchart_data = []
inference_barchart_data = []

file_paths = glob.glob('mehta_dda_samples/*/*')
for path in file_paths:
    tags = path.split('\\')
    method = tags[1]
    sample = tags[2].split('.tsv')[0]
    sample_tags = sample.split('_')
    cell_line = sample_tags[0]
    rep = 'R'+str(sample_tags[-1])
    label = cell_line + ' ' + rep 
    
    if label not in DDA_results.keys():
        DDA_results[label] = dict()


    with open(path,"r") as fi:
        
        
        #main analysis for protein assembly from MSGF-FDR peptides
        if method == 'naiive':
            hela_peptide = []
            hela_protein = []
            for ln in fi:
                if ln.startswith("PEPTIDE"):
                    hela_peptide.append(ln[:-1].split('\t'))
                if ln.startswith("PROTEIN") and 'DECOY' not in ln:
                    hela_protein.append(ln[:-1].split('\t'))
            df_pep = pd.DataFrame(hela_peptide)
            df_prot = pd.DataFrame(hela_protein)
            df_pep.columns = ['0','1','2','FDR','4','5','6','7','8','9','10','proteins','12','13']
            df_prot.columns = ['0','1','2','id','4','5','6']
            
            df_pep = df_pep[['FDR','proteins']]
            df_pep.proteins = df_pep.apply(lambda x: [i.split('|')[1] for i in x['proteins'].split(';') if 'DECOY' not in i],axis=1)
            df_pep['shared'] = df_pep.apply(lambda x: len(x.proteins),axis=1)
            
            df_prot['id'] = df_prot.apply(lambda x: x['id'].split('|')[1],axis=1)
            proteins = df_prot['id'].to_list()
            
            FDR_threshold = 0.01
            FDR_lowEvd_threshold = 0.3
            
            df_unique_pep = df_pep[(df_pep.shared == 1) & (df_pep.FDR.astype(float) < FDR_threshold)]
            df_ambig_pep = df_pep[(df_pep.shared > 1) & (df_pep.FDR.astype(float) < FDR_threshold)]
            df_high_pep = df_pep[(df_pep.FDR.astype(float) < FDR_lowEvd_threshold)]
            df_low_pep = df_pep[(df_pep.FDR.astype(float) >= FDR_lowEvd_threshold)]
            
            uniqueSupport = set([a for c in df_unique_pep.proteins.to_list() for a in c])
            ambigSupport = set([a for c in df_ambig_pep.proteins.to_list() for a in c]).difference(uniqueSupport)
            highEvdSupport = set([a for c in df_high_pep.proteins.to_list() for a in c])
            lowEvdSupport = set([a for c in df_low_pep.proteins.to_list() for a in c]).difference(highEvdSupport)
            weakSupport = set([a for c in df_high_pep.proteins.to_list() for a in c]).difference(uniqueSupport).difference(ambigSupport)
            noSupport = set(validProts).difference(highEvdSupport) #all database proteins - highEvd
            
            peptCount = (df_unique_pep.apply(lambda x: x.proteins[0], axis = 1)).value_counts()
            onepept = peptCount[peptCount == 1]
            twopept = peptCount[peptCount >= 2]
            onepeptSupport = set(onepept.index.to_list())
            twopeptSupport = set(twopept.index.to_list())
                                 
            evidence_type = {'unique':uniqueSupport,
                             'ambiguous':ambigSupport,
                             'high evidence':highEvdSupport,
                             'low evidence':lowEvdSupport,
                             'weak evidence':weakSupport,
                             'no evidence':noSupport,
                             'one peptide':onepeptSupport,
                             'two peptide':twopeptSupport}
            
            # print(label, [(i,len(evidence_type[i])) for i in evidence_type.keys()])
            
            naiive_barchart_data.append([len(twopeptSupport),len(onepeptSupport),len(ambigSupport),
                                  len(weakSupport),len(noSupport), rep])

            for i in evidence_type.keys():
                DDA_results[label][i] = evidence_type[i]

        #auxilliary FDR results from other inference methods
        else:
            hela_protein = []
            for ln in fi:
                if ln.startswith("PROTEIN") and 'DECOY' not in ln:
                    hela_protein.append(ln[:-1].split('\t'))
            df_prot = pd.DataFrame(hela_protein)
            df_prot.columns = ['0','FDR','2','id','4','5','6']
            df_prot['id'] = df_prot.apply(lambda x: x['id'].split('|')[1],axis=1)
            proteins = set(df_prot[df_prot.FDR.astype(float) < 0.01].id.to_list())
            df_prot = df_prot[['FDR','id']]
            
            DDA_results[label][method] = proteins
            inference_barchart_data.append([len(proteins),method, rep, label])
    
import pickle
with open('interim_files/HeLa_DDA_sample.pkl', 'wb') as handle:
    pickle.dump(DDA_results, handle, protocol=pickle.HIGHEST_PROTOCOL)