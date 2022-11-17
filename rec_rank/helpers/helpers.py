from ranx import compare, Qrels
import os
import pprint
import pathlib


def evaluate_and_save_results(experiment_name, runs_collection, best_parameters_rec_algorithms, test_set):
    print("Stage 8: Calculating results... This process can take around 10 minutes for ML-100k and 1 hour for ML-1M due to statistical tests calculation")
    test_set = test_set.astype({"user": str, "item": str, "rating": int})

    # "Qrels", or "query relevance judgments", stores the ground truth for conducting evaluations.
    qrels = Qrels.from_df(test_set, q_id_col="user", doc_id_col="item", score_col="rating")

    results = compare(
        qrels,
        runs_collection,
        ["ndcg@10", "map@10", "precision@1", "precision@10", "recall@10"],
        stat_test="fisher",
        max_p=0.05
    )

    # print results to console
    print(results)

    cwd = os.getcwd()
    results_path = cwd + "\\results\\"
    pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

    # save results to text file (as latex table)
    with open(results_path + "results_for_" + experiment_name + ".txt", 'w') as handle:
        print(results.to_latex(), file=handle)

    # save the best parameters to text file
    with open(results_path + "best_paramaters_for_" + experiment_name + ".txt", 'w') as handle:
        pprint.pprint(best_parameters_rec_algorithms, handle, width=1)

    print()
    print("The results were saved to the directory 'results'")
