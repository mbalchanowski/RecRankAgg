# RecRankAgg

## Reproducing results from the paper
If you just want to reproduce results from the paper [[2]](#2) you will have to download this repository, [install some packages](#installation)
and run script `main.py`. With default settings it will use some cached files like: training and test sets, learned parameters, etc.
If you want to generate everything from scratch, go to section below: [run without cache](#run-without-cache)

## Introduction
This software was developed to conduct experiments for the paper [[2]](#2) 
which demonstrates that the [rank aggregation](#what-is-rank-aggregation) methods can be successfully applied to recommendation systems. 
To keep this software relatively simple, it uses only **5 recommendation algorithms** and **20 aggregation methods**.

Three main libraries were used during development:
1. [LensKit](https://github.com/lenskit/lkpy) - is a set of Python tools for experimenting with and studying recommender systems. It provides support for training, running, and evaluating recommender algorithms in a flexible fashion suitable for research and education.
2. [Ranx](https://github.com/AmenRa/ranx) - is a library of fast ranking evaluation metrics implemented in Python.
3. [Optuna](https://github.com/optuna/optuna) - is an automatic hyperparameter optimization software framework particularly designed for machine learning.

Aggregating rankings generated by different recommendation algorithms can be pretty complex,
so this process was divided into 8 stages (as shown in the `main.py` file):
```python
# Stage 1: Load parameters for experiment.
parameters = ParametersForMovieLens100k()

# Stage 2: Split dataset into train and test sets (can be cached).
training_set, test_set = train_test_split(parameters)

# Stage 3: Tune the parameters of recommendation algorithms (can be cached).
best_parameters_rec_algorithms = rec_tuner.tune_recommendations_algorithms(parameters, training_set)

# Stage 4: Initialize recommendation algorithms with the best set of parameters found in stage 3.
rec_algorithms = generator.init_recommendation_algorithms(parameters, best_parameters_rec_algorithms)

# Stage 5: Generate recommendations in the form of rankings (can be cached).
recommendations = generator.generate_recommendations(parameters, training_set, rec_algorithms)

# Stage 6: Tune supervised aggregation methods on the training set (can be cached).
fusion_methods_parameters = fusion_tuner.tune_fusion_methods(parameters, training_set, rec_algorithms)

# Stage 7:  Final aggregation, using supervised and unsupervised aggregation methods.
aggregated_results = aggregator.aggregate_recommendations(parameters, recommendations, fusion_methods_parameters)

# Stage 8: Evaluate results (using test set).
helpers.evaluate_and_save_results(parameters.name, aggregated_results, best_parameters_rec_algorithms, test_set)
```

#### Please note:
* Some stages can be cached to speed up the process of aggregation. This will save some files to `cache` folder. If you want to disable cache, go to section below [run without cache](#run-without-cache)
* `test_set` should be used only for final evaluation in *Stage 8*.
* Results are printed on the console and saved to a file `results/results_for_MovieLens_X.txt` in the form of latex table.
* All used fusion methods can be found here: https://amenra.github.io/ranx/fusion/#supported-fusion-algorithms

Parameters for experiments can be set in the file `parameters.py`:
```python
name = "experiments_on_MovieLens_100k"
dataset = ML100K()  # dataset for experiments
rec_number = 10     # number of recommendations per algorithm
number_of_trails = 100    # trails for Optuna parameters tuning
use_cached_files = True   # use files saved in "cache" directory

rec_algorithms = [ItemkNN, ImplicitMF, UserkNN, MostPopular, BPR]
unsupervised_fusion_methods = ["min", "med", "anz", "log_isr", "bordafuse", "condorcet", "max", "sum", "mnz", "isr"]
supervised_fusion_methods = ["gmnz", "rrf", "slidefuse", "bayesfuse", "wmnz", "rbc", "logn_isr", "posfuse", "wsum",
                             "w_bordafuse"]
```

## Installation
RecRankAgg need some packages to run. You can install them with `pip`:
```python
pip install ranx
pip install lenskit
pip install lenskit-tf
pip install optuna
pip install matplotlib
```

## What is rank aggregation?
As pointed out in [[1, page 417]](#1), this is a relatively unexplored approach in the context of 
recommendation systems, where instead of a single algorithm, 
a certain set of algorithms is used that generate recommendations for a given user, 
and then the results of these algorithms are aggregated to create
a new recommendation. Aggregation is not a trivial problem,
as there is no single universal method for combining such rankings.

Check out my other repository for more information: [Rank aggregation basic informations](https://github.com/mbalchanowski/Rank-aggregation-basic-informations)

## Run without cache
If you want to tune algorithms and generate recommendations on your own, you have to:
* Set `use_cached_files` to `False` in `parameters.py` file. 
* Run `main.py` script.

Datasets should be downloaded automatically, but you can also download them from:
* MovieLens 100k - https://grouplens.org/datasets/movielens/100k/
* MovieLens 1M - https://grouplens.org/datasets/movielens/1m/

## Performance
If you want better performance, please read about Numba (used by Ranx) and Tensorflow (used by LensKit). More details here:
- Numba - https://numba.pydata.org/numba-doc/0.46.0/user/threading-layer.html#numba-threading-layer-setting-mech
- Tensorflow - https://www.tensorflow.org/install/pip

To run tensorflow with CUDA you will need:
```
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
python3 -m pip install tensorflow
```

## Citation
If you use RecRankAgg in your scientific publication, please consider citing our paper:

```bibtex
@article{Balchanowski2023,
	AUTHOR = {Bałchanowski, Michał and Boryczka, Urszula},
	TITLE = {A Comparative Study of Rank Aggregation Methods in Recommendation Systems},
	JOURNAL = {Entropy},
	VOLUME = {25},
	YEAR = {2023},
	NUMBER = {1},
	ARTICLE-NUMBER = {132},
	ISSN = {1099-4300},
	DOI = {10.3390/e25010132}
}
```

## License
RecRankAgg is an open-sourced software licensed under the [MIT license](LICENSE.md).

## References
<a id="1">[1]</a>
Aggarwal, C. C.,
Advanced Topics in Recommender Systems.
In: *Recommender Systems: The Textbook*.
Springer International Publishing:
Cham, 2016;
pp. 411-448

<a id="2">[2]</a>
Bałchanowski M, Boryczka U. A Comparative Study of Rank Aggregation Methods in Recommendation Systems. Entropy. 2023; 25(1):132. https://doi.org/10.3390/e25010132
