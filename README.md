# RecRank
This software was developed to demonstrate that the rank aggregation methods can be successfully applied to recommendation systems. Three main libraries were used during development:
1. [LensKit](https://github.com/lenskit/lkpy) - is a set of Python tools for experimenting with and studying recommender systems. It provides support for training, running, and evaluating recommender algorithms in a flexible fashion suitable for research and education.
2. [Rankx](https://github.com/AmenRa/ranx) - is a library of fast ranking evaluation metrics implemented in Python.
3. [Optuna](https://github.com/optuna/optuna) - is an automatic hyperparameter optimization software framework particularly designed for machine learning.

Aggregating results of recommendation algorithms can be quite complex,
so this process was divided into 8 stages (as shown in the `main.py` file):
* Stage 1 - Load parameters for experiment.
* Stage 2 - Split dataset into train and test sets (can be cached).
* Stage 3 - Tune the parameters of recommendation algorithms (can be cached).
* Stage 4 - Initialize recommendation algorithms with the best set of parameters found in stage 3.
* Stage 5 - Generate recommendations in the form of rankings (can be cached).
* Stage 6 - Tune supervised aggregation methods on the training set (can be cached).
* Stage 7 - Final aggregation, using supervised and unsupervised aggregation methods.
* Stage 8 - Evaluate results (using test set).

#### Please note:
* Some stages can be cached to speed up the process of aggregation. This will save some files to `cache` folder. If you want to disable cache, go to section below [run without cache](#run-without-cache)
* `test_set` should be used only for final evaluation in *Stage 8*.
* Results are printed on the console and saved to a file `results/results_for_MovieLens_X.txt` in the form of latex table.

## What is rank aggregation?
As pointed out in [[1, page 417]](#1), this is a relatively unexplored approach in the context of 
recommendation systems, where instead of a single algorithm, 
a certain set of algorithms is used that generate recommendations for a given user, 
and then the results of these algorithms are aggregated to create
a new recommendation. Aggregation is not a trivial problem,
as there is no single universal method for combining such rankings.

## Installation
This software uses 
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python3 -m pip install tensorflow

## Reproducing results from the paper
If you want to reproduce results from the paper "*A comparative study of rank aggregation methods in recommendation systems*", just run script `main.py` with default settings. 
It will use some cached files like: training and test sets, learned parameters, etc.

## Run without cache
If you want to tune algorithms and generate recommendations on your own, you have to:
* Download datasets (MovieLens 100k or MovieLens 1M) and put them in `data` folder.
* Set `use_cached_files` to `False` in `parameters.py` file. 
* Run `main.py` script.

*Please note: This process can take a while*

Datasets can be downloaded from:
* MovieLens 100k - https://grouplens.org/datasets/movielens/100k/
* MovieLens 1M - https://grouplens.org/datasets/movielens/1m/

## Performance
If you want better performance, please read about Numba (used by Ranx) and Tensorflow (used by LensKit). More details here:
- Numba - https://numba.pydata.org/numba-doc/0.46.0/user/threading-layer.html#numba-threading-layer-setting-mech
- Tensorflow - https://www.tensorflow.org/install/pip

## Citation
If you use RecRank in your scientific publication, please consider citing our paper:

## License
RecRank is an open-sourced software licensed under the MIT license.

## References
<a id="1">[1]</a>
Aggarwal, C. C.,
Advanced Topics in Recommender Systems.
In: *Recommender Systems: The Textbook*.
Springer International Publishing:
Cham, 2016;
pp. 411-448