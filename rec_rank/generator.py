from lenskit import batch, util
from lenskit.algorithms import Recommender
import pandas as pd
import os
from lenskit import topn


def eval(aname, algo, train, number_of_recommendations_to_generate):
    fittable = util.clone(algo)
    fittable = Recommender.adapt(fittable)
    fittable.fit(train)
    users = sorted(train.user.unique())
    recs = batch.recommend(fittable, users, number_of_recommendations_to_generate)
    recs['Algorithm'] = aname
    return recs


def generate_recommendations(parameters, training_set, algorithms):
    print("Stage 5: Generate recommendations...")
    cwd = os.getcwd()
    saved_files_path = cwd + "\\cache\\" + parameters.name + "\\"

    if parameters.use_cached_files is True:
        print("Loaded from cache")
        return pd.read_pickle(saved_files_path + "recommendations")

    recommendations = []
    for algorithm in algorithms:
        recommendations.append(
            eval(algorithm.name, algorithm, training_set, parameters.rec_number)
        )

    all_recommendations = pd.concat(recommendations, ignore_index=True)
    all_recommendations.to_pickle(saved_files_path + "recommendations")

    return all_recommendations


def generate_recommendations_for_fusion_tuning(training_set, algorithms, number_of_recommendations_to_generate):
    recommendations = []
    for algorithm in algorithms:
        recommendations.append(
            eval(algorithm.name, algorithm, training_set, number_of_recommendations_to_generate)
        )

    return pd.concat(recommendations, ignore_index=True)


def init_recommendation_algorithms(params, best_parameters_rec_algorithms):
    print("Stage 4: Initializing recommendation algorithms...")
    inited_rec_algorithms = []
    for algorithm in params.rec_algorithms:
        inited_algorithm = algorithm(algorithm.__name__, best_parameters_rec_algorithms)
        inited_rec_algorithms.append(inited_algorithm)

    return inited_rec_algorithms
