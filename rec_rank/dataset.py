from lenskit import crossfold as xf
import pandas as pd
import os
import pathlib
from rec_rank.helpers.helpers import ask_to_download_dataset


def train_test_split(parameters):
    print("Stage 2: Train/test split...")
    cwd = os.getcwd()
    saved_files_path = cwd + "\\cache\\" + parameters.name + "\\"
    pathlib.Path(saved_files_path).mkdir(parents=True, exist_ok=True)

    if parameters.use_cached_files is True:
        print("Loaded from cache")
        training_set = pd.read_pickle(saved_files_path + "training_set")
        test_set = pd.read_pickle(saved_files_path + "test_set")

        return training_set, test_set

    ask_to_download_dataset(parameters.dataset)

    # We are using "partition_users" and "LastFrac" methods, since we will be using last 20% of users ratings as testset
    for training_set, test_set in xf.partition_users(parameters.dataset.ratings[['user', 'item', 'rating', 'timestamp']],
                                          partitions=1,
                                          method=xf.LastFrac(0.2)):

        training_set.to_pickle(saved_files_path + "training_set")
        test_set.to_pickle(saved_files_path + "test_set")

        return training_set, test_set

# This method is exclusively used for train-validation set split, when training fusion algorithms
def train_validation_split(training_set):
    # We are using "partition_users" and "LastFrac" methods, since we will be using last 20% of users ratings as testset
    for training_set, validation_set in xf.partition_users(training_set[['user', 'item', 'rating', 'timestamp']],
                                          partitions=1,
                                          method=xf.LastFrac(0.25)):
        return training_set, validation_set
