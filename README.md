# PROSE
Given a gene/protein list, PROSE identifies similarly enriched genes/proteins from a co-expression matrix. 

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
