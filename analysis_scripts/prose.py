# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 11:57:51 2021

@author: bw98j
"""


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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#imbalanced learning
from imblearn.over_sampling import SMOTE

#%%

def labeller(row,pos_indices,neg_indices):
    
    """
Generates labels for proteins based on list. (observed/unobserved/not considered)

Attributes:
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
    
    """
PROSE generates scores from a set of observed and unobserved proteins.

=========================================================

Returns a prose object with the following attributes:

Attributes:
summary: (pandas.DataFrame) a summary of classifier results
clf: fitted sklearn.SVM.LinearSVC object
lr: fitted sklearn.linear_model.LogisticRegression object

Diagnostics:
clf_report_train: classification metrics on training set
cm_train: confusion matrix on training set
f1_train: F1 score on training set
clf_report: classification metrics on test set (requires holdout=True)
cm: confusion matrix on test set (requires holdout=True)
f1: F1 score on test set (requires holdout=True)
runtime: (float) runtime in seconds

=========================================================

Required arguments:
obs: (set/list/1D-like) observed proteins
unobs: (set/list/1D-like) unobserved proteins
corr_mat: (pandas.DataFrame) df with panel protein IDs as columns and tested protein IDs as indices
    
Optional arguments:
downsample: (int) the number of proteins the majority class will be downsampled to. Default = None
downsample_seed: (int) random seed for downsampling. Default = 0
smote: (bool) whether to carry out synthetic minority oversampling. Default = True
holdout: (bool) whether to holdout a test set for model validation. Default = True
holdout_n: (int) number of holdout proteins in each class. Default = 100

Optional kwargs (dict format):
svm_kwargs: pass to sklearn.svm.LinearSVC()
bag_kwargs: pass to sklearn.ensemble.BaggingClassifier()
train_test_kwargs: pass to sklearn.model_selection_train_test_split()
logistic_kwargs: pass to sklearn.linear_model.LogisticRegression()
smote_kwargs: pass to imblearn.oversampling.SMOTE()

Default kwargs:

logistic_kwargs = {}
svm_kwargs = {}
bag_kwargs = {'n_estimators':100, 'max_samples':1000, 'max_features':100}
train_test_kwargs = {'test_size':holdout_n*2, 'shuffle':True, 'random_state':}

    """
    
    def __init__(self, obs, unobs, corr_mat,
                 downsample=None,
                 downsample_seed=0,
                 smote=True,
                 holdout=True,
                 holdout_n=100,
                 svm_kwargs={},
                 bag_kwargs={},
                 logistic_kwargs={},
                 train_test_kwargs={},
                 smote_kwargs={}):
        
        start = timer()  
    
        svm_kwargs = {**svm_kwargs}
    
        bag_kwargs = {'n_estimators':1000,
                      'max_samples':100,
                      'max_features':50,
                      **bag_kwargs}
    
        train_test_kwargs = {'test_size':holdout_n*2,
                             'shuffle':True,
                             'random_state':1,
                             **train_test_kwargs}
    
        logistic_kwargs = {}
    
        smote_kwargs = {**smote_kwargs}
    
        df = corr_mat.dropna()
    
        
        df['y'] = df.apply(lambda x: labeller(x, obs, unobs), axis=1)
        labelcol = df['y']
        df_labels = df[df.y != -1]
        
        if type(downsample) == int: 
                df_labels = df_labels.reindex(np.random.RandomState(seed=downsample_seed).permutation(df_labels.index))
                df_labels = pd.concat([df_labels[df_labels.y==1][:downsample],
                                       df_labels[df_labels.y==0][:downsample]])
        
        Y = df_labels.y
        X = StandardScaler().fit_transform(df_labels.drop(columns='y'))
        
        if holdout == True:
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y,
                                                                **train_test_kwargs)
        else:
            X_train, Y_train = X, Y
        
        if smote == True:
            sm = SMOTE(**smote_kwargs)
            X_train, Y_train = sm.fit_resample(X_train, Y_train.ravel())          

        clf = BaggingClassifier(base_estimator=LinearSVC(**svm_kwargs), **bag_kwargs).fit(X_train, Y_train)
        score = clf.decision_function(corr_mat)
        score_norm = scipy.stats.zscore(score)
        Y_true = np.array(labelcol.to_list())
        tested_proteins = np.array(corr_mat.index.to_list())
        
        corr_mat_lr = score_norm.reshape(-1,1)
        X_lr = scipy.stats.zscore(clf.decision_function(X_train)).reshape(-1,1)
        
        lr = LogisticRegression(**logistic_kwargs).fit(X_lr, Y_train)
        lr_score = lr.decision_function(corr_mat_lr)
        lr_score_norm = scipy.stats.zscore(lr_score)
        Y_pred = lr.predict(corr_mat_lr)
        Y_pred = clf.predict(corr_mat)
        lr_prob = lr.predict_proba(corr_mat_lr)

        
        self.summary = pd.DataFrame(zip(tested_proteins,
                                        Y_pred,
                                        Y_true,
                                        score,
                                        score_norm,
                                        lr_prob
                                        ),
                                    
                                    columns = ['protein',
                                               'y_pred',
                                               'y_true',
                                               'score',
                                               'score_norm',
                                               'prob',
                                               ],
                                    )

        self.summary.prob = self.summary.apply(lambda x: x.prob[1],axis=1)

        self.clf, self.lr = clf, lr    
           
        Y_train_pred = lr.predict(X_lr)
        self.clf_report_tr = sklearn.metrics.classification_report(Y_train, Y_train_pred)
        self.f1_tr = sklearn.metrics.f1_score(Y_train, Y_train_pred)
        self.cm_tr = sklearn.metrics.confusion_matrix(Y_train, Y_train_pred)   
        
        self.X_train_lr = scipy.stats.zscore(clf.decision_function(X_train)).reshape(-1,1)
        Y_scores_tr = lr.predict_proba(self.X_train_lr).T[1]
        self.fpr_tr, self.tpr_tr, self.thresholds_tr = sklearn.metrics.roc_curve(Y_train, Y_scores_tr, pos_label = 1)
        self.auc_tr = round(sklearn.metrics.auc(self.fpr_tr,self.tpr_tr),3)
    
        if holdout == True:
            self.X_test_lr = scipy.stats.zscore(clf.decision_function(X_test)).reshape(-1,1)
            self.Y_test_lr = Y_test
            
            Y_test_pred = lr.predict(self.X_test_lr)
            self.clf_report = sklearn.metrics.classification_report(Y_test, Y_test_pred)
            self.f1 = sklearn.metrics.f1_score(Y_test, Y_test_pred)
            self.cm = sklearn.metrics.confusion_matrix(Y_test, Y_test_pred)
            
            Y_scores = lr.predict_proba(self.X_test_lr).T[1]
            self.fpr, self.tpr, self.thresholds = sklearn.metrics.roc_curve(Y_test, Y_scores, pos_label = 1)
            self.auc = round(sklearn.metrics.auc(self.fpr,self.tpr),3)
        
        self.runtime = round(timer()-start,3)
        
#%%

# q0 = prose(obs, unobs, panel_corr, bag_kwargs={'n_estimators':100})
