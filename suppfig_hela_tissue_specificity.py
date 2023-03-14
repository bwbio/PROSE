# -*- coding: utf-8 -*-
"""
Visualize tissue-specific correlation
"""

from utils.convert_ids import ensembl2uniprot
import numpy as np
from scipy.stats import pearsonr, spearmanr
conv = ensembl2uniprot()

#%% [PLOTTING] HeLa, PROSE-TPM/iBAQ correlation

# load independent iBAQ
ibaq = pd.read_csv('datasets/proteomics/bekker_jensen_hela_ibaq/bekker_jensen_2017_ibaq_s3_mmc4.csv', skiprows=2)
ibaq = ibaq[['Protein IDs','Median HeLa iBAQ']]
ibaq['Protein IDs'] = ibaq.apply(lambda x: list(set([i.split('-')[0] for i in x['Protein IDs'].split(';')])),axis=1)
ibaq['matches'] = ibaq.apply(lambda x: len(x['Protein IDs']),axis=1)
ibaq = ibaq[ibaq.matches == 1]
ibaq['Protein IDs'] = ibaq.apply(lambda x: x[0][0], axis=1)
ibaq = ibaq.set_index('Protein IDs').drop(columns=['matches'])
ibaq = ibaq.dropna().drop_duplicates()
ibaq = np.log10(ibaq)
ibaq_hela = ibaq[~ibaq.index.duplicated(keep='first')]['Median HeLa iBAQ']

# load TPM
tpm = pd.read_csv('datasets/transcriptomics/klijn_cancerlines_tpm/E-MTAB-2706-query-results.tpms.tsv', comment='#', sep='\t')
genes = list(tpm['Gene ID'].values)
tpm_sub = tpm.drop(['Gene ID', 'Gene Name'], axis=1).T
tpm_sub.reset_index(inplace=True)
tpm_sub.columns = ['source']+genes
metacols = ['tissue','cancer','cell_line']
tpm_sub[metacols] = tpm_sub.source.str.split(', ',n=2,expand=True)
md=tpm_sub[metacols]
md=md.drop_duplicates(subset='cell_line')

tpm.columns = [i.split(', ')[-1] for i in tpm.columns]
tpm = tpm.fillna(0)
tpm['protein'] = tpm.apply(lambda x: conv[x['Gene ID']] if x['Gene ID'] in conv else np.nan, axis=1)
tpm = tpm.dropna()

alternate = 'NCI-H345'

tpm_hela = tpm[['HeLa','protein']].set_index('protein')
tpm_alternate = tpm[[alternate,'protein']].set_index('protein')

tpm_hela = np.log2(tpm_hela[~tpm_hela.index.duplicated()]+1)['HeLa']
tpm_alternate = np.log2(tpm_alternate[~tpm_alternate.index.duplicated()]+1)[alternate]

#%%

import scipy.stats

# load 
path='source_data/figure2/DDA_HeLa_R1_2000_1rhk_PROSE_random_k0.csv.gz'
result = pd.read_csv(path, index_col=0)

common_ibaq = ibaq_hela.index.intersection(result.index)
common_tpm = tpm_hela.index.intersection(result.index)
common_alt_tpm = tpm_alternate.index.intersection(result.index)

result['alpha'] = result.apply(lambda row: 1 if row.y_true != -1 else 0.05, axis=1)
result.loc[common_ibaq, 'ibaq'] = ibaq_hela.loc[common_ibaq]
result.loc[common_tpm, 'tpm'] = tpm_hela.loc[common_tpm]
result.loc[common_alt_tpm, 'alt_tpm'] = tpm_alternate.loc[common_alt_tpm]

#%%

r_ibaq = pearsonr(ibaq_hela.loc[common_ibaq], result.loc[common_ibaq]['score_norm'])[0]
rho_ibaq = spearmanr(ibaq_hela.loc[common_ibaq], result.loc[common_ibaq]['score_norm'])[0]

r_tpm = pearsonr(tpm_hela.loc[common_tpm], result.loc[common_tpm]['score_norm'])[0]
rho_tpm = spearmanr(tpm_hela.loc[common_tpm], result.loc[common_tpm]['score_norm'])[0]

r_tpm_alternate = pearsonr(tpm_alternate.loc[common_alt_tpm], result.loc[common_alt_tpm]['score_norm'])[0]
rho_tpm_alternate = spearmanr(tpm_alternate.loc[common_alt_tpm], result.loc[common_alt_tpm]['score_norm'])[0]


#%%

data=pd.DataFrame()

palette =  ['#FF7F0E','#464646','#3DB2FF']
fig,axes=plt.subplots(figsize=[18,32],nrows=4,ncols=2,
                      gridspec_kw=dict(height_ratios=[.3,1,1,1],
                                       width_ratios=[1,.2]))

xlim=[-2.9,3.5]

ax=axes[0][0]
g0=sns.kdeplot(data=result,x='score_norm',hue='y_true',hue_order=[1,-1,0],ax=ax,common_norm=False,
               palette=palette,linewidth=5)
sns.despine(left=True,bottom=False,ax=ax)
ax.set_ylabel('');ax.set_xlabel('');ax.set_yticks([]);ax.set_xticks([])

ax=axes[1][0]
g0=sns.scatterplot(data=result.loc[common_tpm],x='score_norm',y='tpm',hue='y_true',ax=ax,palette=palette,
                   alpha=result.loc[common_tpm].alpha.to_list(),
                   s=100,lw=0.001,hue_order=[1,-1,0])

ax.text(.01,.98,s='HeLa transcriptome\n$r$ = {}\n$ρ$ = {}'\
        .format(round(r_tpm,3),round(rho_tpm,3)),
        transform=ax.transAxes,va='top')
    
ax.set_ylabel('Gene expression\nlog$_{2}$(TPM+1)')
ax.set_xlim(xlim)


ax=axes[2][0]
g1=sns.scatterplot(data=result.loc[common_alt_tpm],x='score_norm',y='alt_tpm',hue='y_true',ax=ax,palette=palette,
                   alpha=result.loc[common_alt_tpm].alpha.to_list(),
                   s=100,lw=0.001,hue_order=[1,-1,0])
ax.text(.01,.98,s='NCI-H345 transcriptome\n$r$ = {}\n$ρ$ = {}'\
        .format(round(r_tpm_alternate,3),round(rho_tpm_alternate,3)),
        transform=ax.transAxes,va='top')
ax.set_ylabel('Gene expression\nlog$_{2}$(TPM+1)')
ax.set_xlim(xlim)

ax=axes[3][0]
g1=sns.scatterplot(data=result.loc[common_ibaq],x='score_norm',y='ibaq',hue='y_true',ax=ax,palette=palette,
                   alpha=result.loc[common_ibaq].alpha.to_list(),
                   s=200,lw=0.001,hue_order=[1,-1,0])
ax.text(.01,.98,s='HeLa proteome\n$r$ = {}\n$ρ$ = {}'\
        .format(round(r_ibaq,3),round(rho_ibaq,3)),
        transform=ax.transAxes,va='top')
ax.set_ylabel('Protein expression\niBAQ, HeLa')
ax.set_xlim(xlim)

ax=axes[1][1]
g0=sns.kdeplot(data=result.loc[common_tpm],y='tpm',hue='y_true',hue_order=[1,-1,0],ax=ax,common_norm=False,
               palette=palette,linewidth=5)
ax.set_ylim(axes[1][0].get_ylim())

ax=axes[2][1]
g0=sns.kdeplot(data=result.loc[common_alt_tpm],y='alt_tpm',hue='y_true',hue_order=[1,-1,0],ax=ax,common_norm=False,
               palette=palette,linewidth=5)
ax.set_ylim(axes[2][0].get_ylim())

ax=axes[3][1]
g0=sns.kdeplot(data=result.loc[common_ibaq],y='ibaq',hue='y_true',hue_order=[1,-1,0],ax=ax,common_norm=False,
               palette=palette,linewidth=5)
ax.set_ylim(axes[3][0].get_ylim())

for i,ax in enumerate(axes):
    
    ax[0].set_xlim(-2.7,3.5)
    
    try:
        ax[0].get_legend().remove()
        ax[1].get_legend().remove()
    except: pass
    
    try:
        ax[1].set_ylabel('');ax[1].set_xlabel('')
    except: pass
    
    sns.despine(ax=ax[1],left=True,bottom=True)
    ax[1].set_xticklabels([]);ax[1].set_yticklabels([])
    
    if i<3:
        ax[0].set_xlabel('');ax[0].set_xticks([])
    else:
        ax[0].set_xlabel('PROSE score')
        ax[0].set_xticks(range(-2,4))
        ax[0].tick_params(axis='x',length=15)
        
plt.subplots_adjust(wspace=0.01,hspace=0.05)

plt.savefig('plots/HeLa_R1_feature_correlation.png',
            format='png', dpi=600, bbox_inches='tight') 



#%% [PLOTTING] HeLa, tissue-specific correlation

data=[]

sub_tpm = tpm.set_index('protein').loc[common_tpm].drop(columns=['Gene ID','Gene Name'])
sub_obj = result.loc[common_tpm]
obj_tpm = sub_obj.score_norm.to_list()

for i, row in md.iterrows():
    tissue,cancer,cell_line=row.tissue,row.cancer,row.cell_line
    
    try:
        cell_line_tpm=sub_tpm[cell_line]
        rho_tpm = scipy.stats.spearmanr(cell_line_tpm, obj_tpm)[0]
    except:
        continue
    
    tissue=tissue.split(' tissue')[0]
        
    data.append([cell_line,tissue,rho_tpm])
    
data=pd.DataFrame(data,columns=['cell_line','tissue','rho'])
print(data)


sorter=[]

for i, df in data.groupby('tissue'):
    sorter.append([i,df.rho.mean()])
    
sorter=pd.DataFrame(sorter,columns=['tissue','rho_mean']).sort_values(by='rho_mean',ascending=False)

fig,ax=plt.subplots(figsize=[10,25])
g0=sns.boxplot(data=data,x='rho',y='tissue',order=sorter.tissue,fliersize=0)
g1=sns.stripplot(data=data,x='rho',y='tissue',order=sorter.tissue,color='black',size=5,alpha=0.4)
ax.grid()
ax.set_ylabel('')
ax.set_xlabel('Rank correlation ($\itρ$)\n(TPM, PROSE score)')

g0.axvline(data[data.cell_line=='HeLa'].rho.values[0], lw=3,linestyle='--',color='black')
plt.text(data[data.cell_line=='HeLa'].rho.values[0],-1,'HeLa', ha='center',size=50)
g0.axvline(data[data.cell_line=='NCI-H345'].rho.values[0], lw=3,linestyle='--',color='black')
plt.text(data[data.cell_line=='NCI-H345'].rho.values[0],-1,'NCI-H345', ha='center',size=50)


oro_cloacal = ['chordate pharynx',
               'urinary bladder',
               'rectum',
               'esophagus']

for i in ax.get_yticklabels():
    if i._text in oro_cloacal:
        i.set_weight('bold')
    if i._text == 'uterine cervix':
        i.set_weight('bold'); i.set_color("red")

plt.savefig('plots/HeLa_R1_tissue_correlation.png',
            format='png', dpi=600, bbox_inches='tight') 

