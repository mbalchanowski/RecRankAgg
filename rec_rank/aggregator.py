from ranx import fuse
from ranx import Run


def fusion(runs_collection, fusion_methods, fusion_methods_parameters=None):
    fusion_runs = []
    for method in fusion_methods:
        if fusion_methods_parameters is None:
            fused_run = fuse(runs=runs_collection,
                             method=method)
            fusion_runs.append(fused_run)
        else:
            fused_run = fuse(runs=runs_collection,
                             method=method,
                             params=fusion_methods_parameters[method])
            fusion_runs.append(fused_run)

    return fusion_runs


def aggregate_recommendations(parameters, all_recommendations, fusion_methods_parameters):
    print("Stage 7: Generating aggregations...")
    all_recs_df = all_recommendations.astype({"user": str, "item": str})

    # For every algorithm, we need to create separate "Run" object
    rec_algorithms = []
    for all_recs_grouped in all_recs_df.groupby("Algorithm"):
        run = Run.from_df(all_recs_grouped[1], q_id_col="user", doc_id_col="item", score_col="score")
        run.name = all_recs_grouped[0]
        rec_algorithms.append(run)

    # Fusion methods without optimization
    fused_by_unsupervised_fusion_methods = fusion(rec_algorithms, parameters.unsupervised_fusion_methods)

    # Fusion methods with optimization
    fused_by_supervised_fusion_methods = fusion(rec_algorithms, parameters.supervised_fusion_methods, fusion_methods_parameters)

    # Put all algorithms in one collection (Run objects)
    all_runs = []
    all_runs.extend(rec_algorithms)
    all_runs.extend(fused_by_unsupervised_fusion_methods)
    all_runs.extend(fused_by_supervised_fusion_methods)

    return all_runs
