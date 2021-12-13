# PROSE
<img src="https://github.com/bwbio/PROSE/blob/assets/Schematic.jpg" width="100%" height="100%">

Given a list of well-defined observed/unobserved proteins, PROSE rapidly learns their underlying co-regulation patterns in a related gene co-expression matrix, before generating an enrichment score for all proteins in the proteome, including _missing proteins_ that were not considered in the initial lists.

In general, these enrichment scores correspond to the _importance_ of the individual proteins in the sample phenotype. As we expect a considerable number of proteins to be missing in any proteomic screen, we can use a high enrichment score as a threshold for recoving some of these missing proteins. PROSE scores can also be directly applied to downstream analyses, with the increased proteome coverage leading to more robust results.

If you use PROSE in your work, please consider citing us: 

Wong, B. J. H., Kong, W., & Goh, W. W. B. (2021). _Single-sample proteome enrichment enables missing protein recovery and phenotype association_. bioRxiv. https://doi.org/10.1101/2021.11.13.468488


## Installation

PROSE can be installed directly from PyPI as follows:
```
pip install pyprose
```

Or, to update PROSE to its latest version:
```
pip install pyprose -U
```

## Example Usage
PROSE can be easily run as a command line tool or in your IDE of choice. The following Python code demonstrates how to generate PROSE enrichment scores for proteins in a given correlation matrix. 

**_pyprose.vignette_()** contains our example dataset. This includes [HeLa protein sets from Mehta et al. (2021)](https://doi.org/10.1101/2020.11.07.372276), as well as a downcast correlation matrix generated from the [Klijn et al. (2015) RNA-Seq dataset](https://doi.org/10.1038/nbt.3080).
```
import pyprose

#load the test data
test_data = pyprose.vignette()

#get the observed/unobserved protein lists
obs = test_data.obs
unobs = test_data.unobs

#get the correlation matrix
panel_corr = test_data.panel_corr 

#use this setting train only 5 sub-classifiers (much faster)
bag_kwargs = {'n_estimators':5}

#run PROSE
result = pyprose.prose(obs, unobs, panel_corr, bag_kwargs = bag_kwargs)

#show the resultant DataFrame
print(result.summary)
```


**result.summary** returns a pandas DataFrame object, similar to below:

|   protein  	| y_pred 	| y_true 	|  score 	| score_norm 	|  prob 	|
|:----------:	|:------:	|:------:	|:------:	|:----------:	|:-----:	|
|   O43657   	|    1   	|   -1   	|  1.096 	|    0.923   	|  0.65 	|
|   Q9H2S6   	|    0   	|    0   	| -1.675 	|   -0.279   	| 0.187 	|
|   O60762   	|    1   	|    1   	|  1.73  	|    1.198   	|  0.75 	|
|   Q8IZE3   	|    0   	|   -1   	| -1.539 	|    -0.22   	| 0.203 	|
|     ...    	|   ...  	|   ...  	|   ...  	|     ...    	|  ...  	|
| A0A0B4J2D5 	|    0   	|   -1   	|  -1.35 	|   -0.139   	| 0.227 	|
| A0A0B4J2H0 	|    1   	|    0   	|  0.813 	|     0.8    	|  0.6  	|
| A0A0D9SF12 	|    0   	|   -1   	|  -3.77 	|   -1.188   	| 0.045 	|
|   Q9BTK6   	|    0   	|    1   	| -2.781 	|    -0.76   	| 0.091 	|

- **_protein_** gives individual protein IDs, taken from the index of the correlation matrix used
- **_y_pred_** is the predicted label
- **_y_true_** is the true label (1 for observed, 0 for unobserved, -1 for unknown)
- **_score_** measures how strongly a protein is close to the 'observed' group (positive values are closer)
- **_score_norm_** is the normalized score (ideally, use this for downstream analysis)
- **_prob_** is the probability that the protein is considered 'observed'

## Logging
By default, PROSE also creates a log folder to store its output (you can disable this by setting **_verbose_** = False).
- **_log.txt_** contains the main arguments used for the run, as well as prediction metrics
- **_summary.tsv_** contains a human-readable version of the above DataFrame
- **_prose_object.pkl_** contains the pickled PROSE output
- **_distribution.png_** shows the score and probability distributions of the individual classes
<img src="https://github.com/bwbio/PROSE/blob/assets/Distribution.jpg" width="75%" height="75%">
