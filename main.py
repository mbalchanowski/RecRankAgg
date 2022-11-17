from rec_rank import aggregator, generator
import rec_rank.parameters_tuning.recommendation_algorithms_tuning as rec_tuner
import rec_rank.parameters_tuning.fusion_algorithms_tuning as fusion_tuner
from rec_rank.helpers import helpers
from rec_rank.dataset import train_test_split
from parameters import ParametersForMovieLens100k, ParametersForMovieLens1M


if __name__ == '__main__':
    # Stage 1: Load parameters for experiments
    parameters = ParametersForMovieLens1M()

    # Stage 2: Split dataset into train and test sets (can be cached).
    training_set, test_set = train_test_split(parameters)

    # Stage 3: Get the best parameters for recommendation algorithms (can be cached).
    best_parameters_rec_algorithms = rec_tuner.tune_recommendations_algorithms(parameters, training_set)

    # Stage 4: Initialize recommendation algorithms, with the best set of parameters found in stage 3.
    rec_algorithms = generator.init_recommendation_algorithms(parameters, best_parameters_rec_algorithms)

    # Stage 5: Generate recommendations in the form of rankings (can be cached).
    recommendations = generator.generate_recommendations(parameters, training_set, rec_algorithms)

    # Stage 6: Tune supervised aggregation algorithms (can be cached).
    fusion_methods_parameters = fusion_tuner.tune_fusion_methods(parameters, training_set, rec_algorithms)

    # Stage 7: Final aggregation, using supervised and unsupervised aggregation methods.
    aggregated_results = aggregator.aggregate_recommendations(parameters, recommendations, fusion_methods_parameters)

    # Stage 8: Evaluation of results, using a test set.
    helpers.evaluate_and_save_results(parameters.name, aggregated_results, best_parameters_rec_algorithms, test_set)
