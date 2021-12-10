# PROSE
![Schematic](https://github.com/bwbio/PROSE/blob/assets/Schematic.jpg)

Given a list of observed or unobserved gene/protein list, PROSE identifies similarly enriched genes/proteins from a co-expression matrix. PROSE can also use a list of upregulated/downregulated elements as the input.

Read the preprint here: https://doi.org/10.1101/2021.11.13.468488

# Example Usage

Begin with the following files in the same working directory:
- prose.py
- klijn_panel_spearmanCorr.tsv
- HeLa_DDA_sample.pkl

```
#Load the example data (from Mehta et al. (2021), HeLa DDA)
with open('HeLa_DDA_sample.pkl', 'rb') as handle:
    testdata = pickle.load(handle)
obs = testdata['HeLa R1']['two peptide'] #set of observed proteins
unobs = testdata['HeLa R1']['no evidence'] #set of unobserved proteins


#Load the correlation matrix
panel_corr = pd.read_csv('klijn_panel_spearmanCorr.tsv', sep='\t',index_col=0)

import prose

result = prose.prose(obs, unobs, panel_corr)

print(result.summary)
```


# Documentation

class prose(_obs_, _unobs_, _corr_mat_, _downsample_=None, _downsample_seed_=0, _smote_=True, _holdout_=True, _holdout_n_=100, \**kwargs)


<details><summary>Attributes</summary>
   
- _**summary**_: (pandas.DataFrame) a summary of classifier results
- _**clf**_: fitted sklearn.SVM.LinearSVC object
- _**lr**_: fitted sklearn.linear_model.LogisticRegression object
</details>

    
<details><summary>Diagnostics</summary>
   
- _**clf_report_train**_: classification metrics on training set
- _**cm_train**_: confusion matrix on training set
- _**f1_train**_: F1 score on training set
- _**clf_report**_: classification metrics on test set (requires holdout=True)
- _**cm**_: confusion matrix on test set (requires holdout=True)
- _**f1**_: F1 score on test set (requires holdout=True)
- _**runtime**_: runtime in seconds
</details>
    
<details><summary>Required arguments</summary>
    
- _**obs**_: (set/list/1D-like) observed proteins
- _**unobs**_: (set/list/1D-like) unobserved proteins
- _**corr_mat**_: (pandas.DataFrame) df with panel protein IDs as columns and tested protein IDs as indices
</details>
   
<details><summary>Optional arguments</summary>
    
- _**downsample**_: (int) the number of proteins the majority class will be downsampled to. Default = None
- _**downsample_seed**_: (int) random seed for downsampling. Default = 0
- _**smote**_: (bool) whether to carry out synthetic minority oversampling. Default = True
- _**holdout**_: (bool) whether to holdout a test set for model validation. Default = True
- _**holdout_n**_: (int) number of holdout proteins in each class. Default = 100
</details>
    
<details><summary>Optional kwargs (as dictionaries)</summary>
    
- _**svm_kwargs**_: pass to sklearn.svm.LinearSVC()
- _**bag_kwargs**_: pass to sklearn.ensemble.BaggingClassifier()
- _**train_test_kwargs**_: pass to sklearn.model_selection_train_test_split()
- _**logistic_kwargs**_: pass to sklearn.linear_model.LogisticRegression()
- _**smote_kwargs**_: pass to imblearn.oversampling.SMOTE()
</details>

<details><summary>Default kwargs</summary>
    
- _**logistic_kwargs**_ = {}
- _**svm_kwargs**_ = {}
- _**bag_kwargs**_ = {'n_estimators':100, 'max_samples':100, 'max_features':50}
- _**train_test_kwargs**_ = {'test_size':holdout_n*2, 'shuffle':True, 'random_state':}
</p></details>
