# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 01:05:26 2021

@author: bw98j
"""

#%% import

import pkg_resources
import os
import datetime
import pickle
import pandas as pd
import numpy as np
import scipy.stats
from timeit import default_timer as timer
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)

#sklearn
import sklearn
import sklearn.metrics
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression

#%%

def labeller(row, pos_indices, neg_indices):
    
    """
Generates labels for proteins based on list. 

Arguments:
row: (DataFrame row, for internal use)
pos_indices: (list/set; 1D-like) observed proteins
neg_indices: (list/set; 1D-like) unobserved proteins

    """
    if row.name in pos_indices:
        return 1
    elif row.name in neg_indices:
        return 0
    else:
        return -1

class prose:
    def __init__(self,
                 obs, unobs, panel_corr,
                 imb = 'reweight',
                 downsample_n = 1000, 
                 holdout_n = 100,
                 verbose = True,
                 logging_path = '',
                 expt_label = '',
                 svm_kwargs={},
                 bag_kwargs={},
                 logistic_kwargs={},
                 ):
        
        args = locals()
        start = timer()  
    
        if 'Handle default kwargs':
            svm_kwargs = {**svm_kwargs}
            logistic_kwargs = {**logistic_kwargs}
            bag_kwargs = {'n_estimators':1000,
                          'max_samples':100,
                          'max_features':50,
                          **bag_kwargs}
    
        df = panel_corr.dropna()
        df['y'] = df.apply(lambda x: labeller(x, obs, unobs), axis=1)
        Y_true = df['y']
        df_sub = df[df.y != -1]
        df_sub = df_sub.sample(frac=1)

        if 'Handle imbalanced learning (imb)':
            if imb == 'downsample':
                df_sub = pd.concat([df_sub[df_sub.y==1].sample(downsample_n),
                                    df_sub[df_sub.y==0].sample(downsample_n)]
                                   )
            
            elif imb == 'reweight':
                svm_kwargs = {'class_weight':'balanced',
                              **svm_kwargs}
            
            elif imb == None:
                pass
    
    
            test = pd.concat([df_sub[df_sub.y==1][:holdout_n],
                               df_sub[df_sub.y==0][:holdout_n]]
                              )            
            train = pd.concat([df_sub[df_sub.y==1][holdout_n:],
                               df_sub[df_sub.y==0][holdout_n:]]
                              )
        
        if 'Generate scores':
            Y_train = train.y
            Y_test = test.y
            X_train = train.drop(columns='y')
            X_test = test.drop(columns='y')
            
            clf = BaggingClassifier(base_estimator=LinearSVC(**svm_kwargs), **bag_kwargs)
            clf.fit(X_train, Y_train)
            score = clf.decision_function(panel_corr)
            score_norm = scipy.stats.zscore(score)
            tested_proteins = np.array(panel_corr.index.to_list())
            X_train_lr = scipy.stats.zscore(clf.decision_function(X_train)).reshape(-1,1)
            
            lr = LogisticRegression(**logistic_kwargs)
            lr.fit(X_train_lr, Y_train)
        
            Y_pred = clf.predict(panel_corr)
            X_lr = score_norm.reshape(-1,1) 
            lr_pred = lr.predict(X_lr)
            lr_prob = lr.predict_proba(X_lr)
            lr_prob = [i[1] for i in lr_prob]

        if 'Generate summary frame':
            data = zip(tested_proteins,Y_pred,Y_true,score,score_norm,lr_prob)
            columns = ['protein','y_pred','y_true','score','score_norm','prob']
            self.summary = pd.DataFrame(data, columns=columns)

        if 'Generate training set scores':
            Y_train_pred = clf.predict(X_train)
            Y_train_scores = clf.decision_function(X_train)
            self.cr_tr = sklearn.metrics.classification_report(Y_train, Y_train_pred)
            self.f1_tr = sklearn.metrics.f1_score(Y_train, Y_train_pred)
            self.ac_tr = sklearn.metrics.accuracy_score(Y_train, Y_train_pred)
            self.cm_tr = sklearn.metrics.confusion_matrix(Y_train, Y_train_pred)   
        
            fpr_tr, tpr_tr, thresholds_tr = sklearn.metrics.roc_curve(Y_train, Y_train_scores, pos_label = 1)
            self.auc_tr = round(sklearn.metrics.auc(fpr_tr, tpr_tr), 3)
    
        if 'Generate test set scores':
            if holdout_n > 0:
                Y_test_pred =  clf.predict(X_test)
                Y_test_scores = clf.decision_function(X_test)
                self.cr = sklearn.metrics.classification_report(Y_test, Y_test_pred)
                self.f1 = sklearn.metrics.f1_score(Y_test, Y_test_pred)
                self.ac = sklearn.metrics.accuracy_score(Y_test, Y_test_pred)
                self.cm = sklearn.metrics.confusion_matrix(Y_test, Y_test_pred)   
                
                fpr_tr, tpr_tr, thresholds_tr = sklearn.metrics.roc_curve(Y_test, Y_test_scores, pos_label = 1)
                self.auc = round(sklearn.metrics.auc(fpr_tr, tpr_tr), 3)
        
        if 'Auxilliary attributes':
            self.clf = clf
            self.lr = lr
            self.runtime = round(timer()-start,3)
        
        if 'Generate report':
            if verbose == True:
                
                if 'Debug string':
                    obs_overlap = set(tested_proteins).intersection(obs)
                    unobs_overlap = set(tested_proteins).intersection(unobs)
                    
                    str0 = 'Run {} completed in {}s'
                    str0 = str0.format(datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                                       self.runtime)
                    
                    str1 = '{} of {} input observed proteins in feature space\n'+\
                           '{} of {} input unobserved proteins in feature space\n'+\
                           ''
                    str1 = str1.format(len(obs_overlap),len(obs),
                                       len(unobs_overlap),len(unobs))
                    
                    if imb == 'downsample':
                        imb = imb +\
                              '\n[downsample_n] Downsample size: {} per class'.format(downsample_n)
                    
                    str2 = 'Arguments:\n' +\
                           '[imb] Imbalance correction: {}\n'.format(imb) +\
                           '[holdout_n] Holdout proteins: {} per class\n'.format(holdout_n) +\
                           '[svm_kwargs]: {}\n'.format(svm_kwargs) +\
                           '[bag_kwargs]: {}\n'.format(bag_kwargs) +\
                           '[logistic_kwargs]: {}\n'.format(logistic_kwargs)   
                             
                    
                    debugstr = '\n'.join([str0,str1,str2])
                
                if 'Metric string':    
                    try:
                        str0 = 'Test\t{:.4f}\t{:.4f}\t{:.4f}'\
                                .format(self.ac, self.f1, self.auc)
                    except:
                        str0 = 'no test set defined'
                        
                    str1 = 'Train\t{:.4f}\t{:.4f}\t{:.4f}'\
                            .format(self.ac_tr, self.f1_tr, self.auc_tr)
                    
                    try:
                        str2=self.cm
                    except:
                        str2=self.cm_tr
                    str2 = '\nConfusion matrix:\n'+str(str2)

                    metrichead = 'Performance:\nSet\tAcc.\tF1\tAUC'
                    
                    metricstr = '\n'.join(['{}']*4).format(metrichead,str0, str1, str2)
                          
                if 'Create log directory':
                    logstr = '\n'.join([debugstr,metricstr])
                    basepath = os.getcwd()
                               
                    if logging_path == '':
                        path = basepath + '\\prose_output'
                    else:
                        path = logging_path
            
                    os.makedirs(path, exist_ok=True)
            
            
                    existing_folders = os.listdir(path)
                    if expt_label == '':
                        for i in range(10000):
                            expt_label = 'prose_run_{}'.format(i)

                            newfolder = path+'\\{}'.format(expt_label)
                            
                            if expt_label not in existing_folders:
                                path = newfolder
                                break
                    else:            
                        newfolder = path+'\\'+expt_label
                        if expt_label not in existing_folders:
                            path = newfolder
                        else:
                            for i in range(10000):
                                temp_expt_label = '{}_{}'.format(expt_label,i)
                                newfolder = path+'\\{}'.format(temp_expt_label)
                                if temp_expt_label not in existing_folders:
                                    path = newfolder
                                    break

                    print(newfolder)
                    os.makedirs(newfolder, exist_ok=True)
                    
                if 'Write log.txt':
                        with open("{}\\log.txt".format(newfolder), "w") as txt:
                         txt.write(logstr)
                                             
                if 'Write distribution.png':
                        try:
                            import matplotlib.pyplot as plt
                            import seaborn as sns
                            palette = ['#bababa', '#1f77b4', '#FF7F0E'] 
    
                            fig, axes = plt.subplots(nrows=2,ncols=1,figsize=[7,4])
                            
                            ax=axes[0]
                            g = sns.kdeplot(data=self.summary, x='score_norm',hue='y_true',
                                            common_norm=False,ax=ax,
                                            palette=palette,legend=False,lw=2)
                            ax.set_xlabel('PROSE score')
                            ax.set_ylabel('')
                            
                            ax=axes[1]
                            g = sns.kdeplot(data=self.summary, x='prob',hue='y_true',common_norm=False,ax=ax,
                                            palette=palette,lw=2)
                            ax.set_xlabel(r'$P_{\rmLR}$ (protein exists)')
                            ax.set_ylabel('')
                            
                            plt.subplots_adjust(hspace=0.5)
                            fig.supylabel('Density',x=0.01)
                            
                            plt.savefig("{}\\distribution.png".format(newfolder),
                                        format='png', dpi=100, bbox_inches='tight') 
                            plt.show() 
                        except:
                            print('Missing plotting libraries (matplotlib.pyplot or seaborn')

                if 'Write summary.tsv':
                    condensed = self.summary.round(3)
                    condensed.to_csv("{}\\summary.tsv".format(newfolder),
                                     sep = '\t',
                                     index=False)

class vignette:
    def __init__(self):
        
        strm = pkg_resources.resource_stream(__name__, 'vignette/HeLa_DDA_sample.pkl')
        testdata = pickle.load(strm)
        self.obs = testdata['HeLa R1']['two peptide']
        self.unobs = testdata['HeLa R1']['no evidence']      

        strm = pkg_resources.resource_stream(__name__, 'vignette/klijn_panel_spearman.csv.gz')
        self.panel_corr = pd.read_csv(strm, compression='gzip', index_col=0)


#%% Test code
# q=prose(obs,unobs,panel_corr,
#         imb='reweight',
#         holdout_n=100,
#         bag_kwargs=dict(n_estimators=5),
#         expt_label='HeLa',
#         verbose=False,
#         )


