from rec_rank import generator
from ranx import Qrels, Run, optimize_fusion, fuse
import os
from pathlib import Path
import pickle
from rec_rank.dataset import train_validation_split
import pathlib


def fusion_methods_optimization(validation_set, runs_collection, fusion_methods):
    fusion_runs = []
    best_parameters_dict = {}

    # "Qrels", or "query relevance judgments", stores the ground truth for conducting evaluations.
    qrels = Qrels.from_df(validation_set, q_id_col="user", doc_id_col="item", score_col="rating")

    for method in fusion_methods:
        best_params = optimize_fusion(
            qrels=qrels,
            runs=runs_collection,
            method=method,
            metric="ndcg@10"
        )

        best_parameters_dict[method] = best_params
        fused_run = fuse(runs=runs_collection, method=method, params=best_params)
        fusion_runs.append(fused_run)

    return fusion_runs, best_parameters_dict


def tune_fusion_methods(parameters, full_training_set, algorithms):
    print("Stage 6: Tuning fusion methods...")
    cwd = os.getcwd()
    cache_files_path = cwd + "\\cache\\" + parameters.name + "\\parameters\\"
    pathlib.Path(cache_files_path).mkdir(parents=True, exist_ok=True)
    fusion_algorithms_best_parameters_path = Path(cache_files_path + "fusion_algorithms_best_parameters")

    # Load parameters from cache files if needed
    if fusion_algorithms_best_parameters_path.exists() and parameters.use_cached_files is True:
        with open(fusion_algorithms_best_parameters_path, 'rb') as handle:
            print("Loaded from cache")
            return pickle.load(handle)

    # Create partial training set and validation set from full training set and generate recommendations
    partial_training_set, validation_set = train_validation_split(full_training_set)
    all_recommendations = generator.generate_recommendations_for_fusion_tuning(partial_training_set, algorithms, 10)

    # Cast columns to specific type
    all_recommendations = all_recommendations.astype({"user": str, "item": str})
    validation_set = validation_set.astype({"user": str, "item": str, "rating": int})

    # For every algorithm, we need to create separate "Run" object
    runs_collection = []
    for all_recs_grouped in all_recommendations.groupby("Algorithm"):
        run = Run.from_df(all_recs_grouped[1], q_id_col="user", doc_id_col="item", score_col="score")
        run.name = all_recs_grouped[0]
        runs_collection.append(run)

    fusion_runs, best_parameters_dict = fusion_methods_optimization(validation_set, runs_collection, parameters.supervised_fusion_methods)
    runs_collection.extend(fusion_runs)

    # Save best parameters found to file
    with open(fusion_algorithms_best_parameters_path, 'wb') as handle:
        pickle.dump(best_parameters_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return best_parameters_dict
