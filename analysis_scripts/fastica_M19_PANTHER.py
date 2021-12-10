# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 09:12:40 2021

@author: bw98j
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#plot parameters
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rc('font',family='arial',size=40)
plt.rc('hatch',linewidth = 2.0)

df = pd.read_csv('fastica_files/panther/HX_module_19.txt', sep='\t', skiprows=11)
df.columns = ['go','reflist','n','expect','over','fc','p','fdr']
df['fdr'] = -np.log10(df['fdr'])
df['go_name'] = df.apply(lambda x: x.go.split(' (GO')[0], axis=1)

df = df.sort_values(by='fdr', ascending=False)


subclasses = {'development':'specification|morpho|development|embryo',
              'gene regulation':'DNA|RNA|transcription',
              'metabolism':'biosynthetic|metabo',
              }

fig, axes = plt.subplots(nrows=3,ncols=1,figsize=[20,45], sharex=True,
                         gridspec_kw=dict(height_ratios=[1.8,1,3.8]))


colors = sns.husl_palette(3)
n = 0
for subclass in subclasses.keys():
    ax=axes[n]
    subdf = df[df.go.str.contains(subclasses[subclass], regex=True)]
    subdf = subdf.reset_index(drop=True)
    g=sns.barplot(data=subdf,x='fdr',y='go_name',color=colors[n], ax=ax)
    ax.set_ylabel('')
    ax.set_xlabel('')
    
    g.axvline(subdf.fdr.max()+5.5, 0.02,0.98, clip_on = False,
              lw=30, color=colors[n],alpha=0.2)
    g.text(s=subclass,y=len(subdf)/2-0.5,x=subdf.fdr.max()+7,
           rotation=270,ha='center',va='center',size=60,
           clip_on=False)
    
    for i, row in subdf.iterrows():
        s = '{}'.format(row.fc)
        if s == ' > 100': s = '>100'
        
        g.text(s=s,
               x=row.fdr+0.5, y=i+0.1,
               size=40,
               ha='left', va='center', clip_on=False)
    
    if n == 0:
        ax.text(s='fold enrichment', x=subdf.fdr.max()-11, y=len(subdf)-3,
                size=45)
        ax.plot([subdf.fdr.max()-11,subdf.fdr.max()-8],
                [len(subdf)-8,len(subdf)-4.5],lw=3,color='black')
    
    if n == 2:
        ax.set_xlabel('-log$_{10}$(FDR)', size=60)
        
    
    n+=1


plt.xlim(0,35)
sns.despine()
sns.despine(ax=axes[0],bottom=True)
sns.despine(ax=axes[1],bottom=True)
plt.subplots_adjust(hspace = 0.05)

plt.savefig('plots/FastICA_M19_PANTHER.png',
            format='png', dpi=600, bbox_inches='tight') 

