from lenskit.algorithms import als, item_knn, user_knn, basic, tf

seed = 1 # for some reproducibility

def UserkNN(name, best_parameters_rec_algorithms):
    parameters = best_parameters_rec_algorithms[name]["params"]
    algorithm = user_knn.UserUser(**parameters, center=False, aggregate='sum', feedback="explicit", seed=seed)
    algorithm.name = "UserkNN"

    return algorithm

def ItemkNN(name, best_parameters_rec_algorithms):
    parameters = best_parameters_rec_algorithms[name]["params"]
    algorithm = item_knn.ItemItem(**parameters, center=False, aggregate="sum", feedback="explicit", seed=seed)
    algorithm.name = "ItemkNN"

    return algorithm

def ImplicitMF(name, best_parameters_rec_algorithms):
    parameters = best_parameters_rec_algorithms[name]["params"]
    algorithm = als.ImplicitMF(**parameters, use_ratings=True, iterations=200, rng_spec=seed)
    algorithm.name = name

    return algorithm

def MostPopular(name, best_parameters_rec_algorithms):
    algorithm = basic.Popular()
    algorithm.name = name

    return algorithm

def BPR(name, best_parameters_rec_algorithms):
    parameters = best_parameters_rec_algorithms[name]["params"]

    algorithm = tf.BPR(**parameters,
                       epochs=200,
                       rng_spec=seed,
                       neg_weight=False)
    algorithm.name = name

    return algorithm

def Random(parameters):
    algorithm = basic.Random()
    algorithm.name = "Random"

    return algorithm
