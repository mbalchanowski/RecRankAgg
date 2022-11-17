import optuna
from optuna.visualization.matplotlib import plot_slice
from optuna.samplers import TPESampler
import pandas as pd
import matplotlib.pyplot as plt
from lenskit import topn, crossfold as xf
from lenskit.algorithms import als, item_knn, user_knn, tf
from rec_rank.generator import eval
import os
import pathlib
global global_training_set
import pickle


def evaluate(trainset, model):
    all_recs = []
    validation_data = []
    for train, validation in xf.partition_users(trainset[['user', 'item', 'rating', 'timestamp']], 1, xf.LastFrac(0.2)):
        validation_data.append(validation)
        all_recs.append(eval(model.name, model, train, 10))

    all_recs = pd.concat(all_recs, ignore_index=True)
    test_data = pd.concat(validation_data, ignore_index=True)
    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(all_recs, test_data)

    return results.ndcg.mean()


def implicit_mf_tuning(trial):
    # For hyperparametr tuning we only use training_data
    trainset = global_training_set

    features = trial.suggest_int('features', 2, 50, step=1)
    reg = trial.suggest_float("reg", 0, 1, step=0.01)
    method = trial.suggest_categorical("method", ["lu", "cg"])
    weight = trial.suggest_float("weight", 0, 10, step=0.1)

    model = als.ImplicitMF(features=features,
                           reg=reg,
                           weight=weight,
                           method=method,
                           use_ratings=True,
                           iterations=200)
    model.name = "ImplicitMF"
    trial.set_user_attr("name", model.name)

    return evaluate(trainset, model)


def bpr_tuning(trial):
    # For hyperparametr tuning we only use training_data
    trainset = global_training_set

    features = trial.suggest_int('features', 2, 50, step=1)
    reg = trial.suggest_float('reg', 0.00, 1, step=0.01)
    neg_count = trial.suggest_int('neg_count', 1, 20, step=1)

    model = tf.BPR(features=features,
                   reg=reg,
                   neg_count=neg_count,
                   epochs=200,
                   neg_weight=False)
    model.name = "BPR"
    trial.set_user_attr("name", model.name)

    return evaluate(trainset, model)


def item_item_tuning(trial):
    # For hyperparametr tuning we only use training_data
    trainset = global_training_set

    nnbrs = trial.suggest_int('nnbrs', 2, 50, step=1)
    min_nbrs = trial.suggest_int("min_nbrs", 1, 10, step=1)

    if min_nbrs > nnbrs:
        min_nbrs = nnbrs

    model = item_knn.ItemItem(nnbrs=nnbrs,
                              min_nbrs=min_nbrs,
                              aggregate="sum",
                              center=False,
                              feedback="explicit")
    model.name = "ItemkNN"
    trial.set_user_attr("name", model.name)

    return evaluate(trainset, model)


def user_user_tuning(trial):
    # For hyperparametr tuning we only use training_data
    trainset = global_training_set

    nnbrs = trial.suggest_int('nnbrs', 2, 50, step=1)
    min_nbrs = trial.suggest_int("min_nbrs", 1, 10, step=1)

    if min_nbrs > nnbrs:
        min_nbrs = nnbrs

    model = user_knn.UserUser(nnbrs=nnbrs, min_nbrs=min_nbrs, center=False, aggregate='sum', feedback="explicit")
    model.name = "UserkNN"
    trial.set_user_attr("name", model.name)

    return evaluate(trainset, model)


def tune_recommendations_algorithms(parameters, training_set):
    print("Stage 3: Tune recommendations algorithms...")
    global global_training_set
    global_training_set = training_set

    cwd = os.getcwd()
    cache_files_path = cwd + "\\cache\\" + parameters.name + "\\parameters\\"
    pathlib.Path(cache_files_path).mkdir(parents=True, exist_ok=True)
    recommendation_algorithms_best_parameters_path = pathlib.Path(cache_files_path + "recommendation_algorithms_best_parameters")

    # Load parameters from cache files if needed
    if recommendation_algorithms_best_parameters_path.exists() and parameters.use_cached_files is True:
        with open(recommendation_algorithms_best_parameters_path, 'rb') as handle:
            print("Loaded from cache")
            best_parameters_rec_algorithms = pickle.load(handle)
            return best_parameters_rec_algorithms

    models_for_tuning = [implicit_mf_tuning, user_user_tuning, item_item_tuning, bpr_tuning]
    seed = 10

    best_parameters_rec_algorithms = {}
    for model in models_for_tuning:
        sampler = TPESampler(seed=seed)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(model, n_trials=parameters.number_of_trails)

        algorithm_name = study.best_trial.user_attrs["name"]
        best_parameters_rec_algorithms[algorithm_name] = {"params": study.best_trial.params,
                                                "name": algorithm_name,
                                                "NDCG": study.best_trial.value}

        plot_slice(study, target_name="MAP")
        pathlib.Path(cache_files_path + "graphs\\").mkdir(parents=True, exist_ok=True)
        plt.savefig(cache_files_path + "graphs\\" + model.__name__ + "_plot_slice.pdf", format="pdf")

    # Save best parameters found to file
    with open(recommendation_algorithms_best_parameters_path, 'wb') as handle:
        pickle.dump(best_parameters_rec_algorithms, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return best_parameters_rec_algorithms
