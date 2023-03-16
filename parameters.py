from rec_rank.recommendation_algorithms import UserkNN, MostPopular, BPR, ImplicitMF, ItemkNN
from lenskit.datasets import ML100K, ML1M
import numba


class ParametersForMovieLens100k:
    # If you want better performance, you need to config numba, for example
    # numba.config.THREADING_LAYER = "tbb"

    name = "experiments_on_MovieLens_100k"
    dataset = ML100K()
    rec_number = 10     # number of recommendations
    number_of_trails = 100    # trails for Optuna parameters tuning
    use_cached_files = True
    fusion_norm = "min-max"    # normalization strategy: https://amenra.github.io/ranx/normalization/#Normalization

    rec_algorithms = [ItemkNN, ImplicitMF, UserkNN, MostPopular, BPR]
    # names and aliases of fusion methods can be found here: https://amenra.github.io/ranx/fusion/#supported-fusion-algorithms
    unsupervised_fusion_methods = ["min", "med", "anz", "log_isr", "bordafuse", "condorcet", "max", "sum", "mnz", "isr"]
    supervised_fusion_methods = ["gmnz", "rrf", "slidefuse", "bayesfuse", "wmnz", "rbc", "logn_isr", "posfuse", "wsum",
                                 "w_bordafuse"]


class ParametersForMovieLens1M:
    # If you want better performance, you need to config numba, for example
    # numba.config.THREADING_LAYER = "tbb"

    name = "experiments_on_MovieLens_1M"
    dataset = ML1M()
    rec_number = 10 # number of recommendations
    number_of_trails = 100    # trails for Optuna parameters tuning
    use_cached_files = True
    fusion_norm = "min-max"    # normalization strategy: https://amenra.github.io/ranx/normalization/#Normalization

    rec_algorithms = [ItemkNN, ImplicitMF, UserkNN, MostPopular, BPR]
    # names and aliases of fusion methods can be found here: https://amenra.github.io/ranx/fusion/#supported-fusion-algorithms
    unsupervised_fusion_methods = ["min", "med", "anz", "log_isr", "bordafuse", "condorcet", "max", "sum", "mnz", "isr"]
    supervised_fusion_methods = ["gmnz", "rrf", "slidefuse", "bayesfuse", "wmnz", "rbc", "logn_isr", "posfuse", "wsum",
                                 "w_bordafuse"]
