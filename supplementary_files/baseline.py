"""Train the baseline model i.e. a logistic regression on the average of the resnet features and
and make a prediction.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=False, type=Path, default="DataChallenge_resnet_only/",
                    help="directory where data is stored")
parser.add_argument("--num_runs", required=False, type=int, default=10,
                    help="Number of runs for the cross validation")
parser.add_argument("--num_splits", default=5, type=int,
                    help="Number of splits for the cross validation")


def get_average_features(filenames):
    """Load and aggregate the resnet features by the average.

    Args:
        filenames: list of filenames of length `num_patients` corresponding to resnet features

    Returns:
        features: np.array of mean resnet features, shape `(num_patients, 2048)`
    """
    # Load numpy arrays
    features = []
    for f in filenames:
        patient_features = np.load(f)

        # Remove location features (but we could use them?)
        patient_features = patient_features[:, 3:]

        aggregated_features = np.mean(patient_features, axis=0)
        features.append(aggregated_features)

    features = np.stack(features, axis=0)
    return features


if __name__ == "__main__":
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load the data
    assert args.data_dir.is_dir()

    train_dir = args.data_dir / "train_input" / "resnet_features"
    test_dir = args.data_dir / "test_input"  / "resnet_features"

    train_output_filename = args.data_dir / "train_output.csv"

    train_output = pd.read_csv(train_output_filename)

    # Get the filenames for train
    filenames_train = [train_dir / "{}.npy".format(idx) for idx in train_output["ID"]]
    for filename in filenames_train:
        assert filename.is_file(), filename

    # Get the labels
    labels_train = train_output["Target"].values

    assert len(filenames_train) == len(labels_train)


    # Get the numpy filenames for test
    filenames_test = sorted(test_dir.glob("*.npy"))
    for filename in filenames_test:
        assert filename.is_file(), filename
    ids_test = [f.stem for f in filenames_test]


    # Get the resnet features and aggregate them by the average
    features_train = get_average_features(filenames_train)
    features_test = get_average_features(filenames_test)

    # -------------------------------------------------------------------------
    # Use the average resnet features to predict the labels

    # Multiple cross validations on the training set
    aucs = []
    for seed in tqdm(range(args.num_runs)):
        # Use logistic regression with L2 penalty
        estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="liblinear")

        cv = sklearn.model_selection.StratifiedKFold(n_splits=args.num_splits, shuffle=True,
                                                     random_state=seed)

        # Cross validation on the training set
        auc = sklearn.model_selection.cross_val_score(estimator, X=features_train, y=labels_train,
                                                      cv=cv, scoring="roc_auc", verbose=0)

        aucs.append(auc)

    aucs = np.array(aucs)

    print("Predicting weak labels by mean resnet")
    print("AUC: mean {}, std {}".format(aucs.mean(), aucs.std()))


    # -------------------------------------------------------------------------
    # Prediction on the test set

    # Train a final model on the full training set
    estimator = sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0, solver="liblinear")
    estimator.fit(features_train, labels_train)

    preds_test = estimator.predict_proba(features_test)[:, 1]

    # Check that predictions are in [0, 1]
    assert np.max(preds_test) <= 1.0
    assert np.min(preds_test) >= 0.0

    # -------------------------------------------------------------------------
    # Write the predictions in a csv file, to export them in the suitable format
    # to the data challenge platform
    ids_number_test = [i.split("ID_")[1] for i in ids_test]
    test_output = pd.DataFrame({"ID": ids_number_test, "Target": preds_test})
    test_output.set_index("ID", inplace=True)
    test_output.to_csv(args.data_dir / "preds_test_baseline.csv")
