#!/usr/bin/env python
import os
import sys
import pandas as pd
import numpy as np
from parsezeeklogs import ParseZeekLogs
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def load(source_file: str):
    parser = ParseZeekLogs(source_file, batchsize=65536)
    parser_fields = parser.get_fields()
    assert isinstance(parser_fields, list)

    def _generator(parser: ParseZeekLogs):
        for log_record in parser:
            if log_record is not None:
                yield log_record

    return pd.DataFrame(_generator(parser), columns=parser_fields)


def trim(df: pd.DataFrame):
    df.drop(
        inplace=True,
        columns=[
            "ts",
            "uid",
            "id.orig_h",
            "id.orig_p",
            "id.resp_h",
            "proto",
            "service",
            "conn_state",
            "local_orig",
            "local_resp",
            "missed_bytes",
            "history",
            "tunnel_parents",
            "detailed-label",
        ],
    )

    df["label"] = (
        df["label"]
        .replace(to_replace="Benign", value=0)
        .replace(to_replace="benign", value=0)
        .replace(to_replace="Malicious", value=1)
        .replace(to_replace="malicious", value=1)
    )
    df["duration"] = pd.to_numeric(df["duration"], downcast="float").fillna(value=0)
    for column in [
        "id.resp_p",
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "orig_ip_bytes",
        "resp_pkts",
        "resp_ip_bytes",
        "label",
    ]:
        df[column] = (
            pd.to_numeric(df[column], downcast="unsigned")
            .fillna(value=0)
            .astype("uint8")
        )

    # drop empty entries if any
    df = df.drop(
        df[
            (df.duration == 0)
            & (df.orig_bytes == 0)
            & (df.resp_bytes == 0)
            & (df.orig_pkts == 0)
            & (df.orig_ip_bytes == 0)
            & (df.resp_pkts == 0)
            & (df.resp_ip_bytes == 0)
        ].index
    )
    return df


def split(df: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop(columns=["label"], inplace=False).values,
        df.label,
        test_size=0.2,
        train_size=0.8,
        stratify=df.label,
        random_state=64,
        shuffle=True,
    )
    return X_train, X_test, y_train, y_test


def scale(X_train, X_test):
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    minmax_scaler.fit(X_train)
    X_train = minmax_scaler.transform(X_train)
    X_test = minmax_scaler.transform(X_test)
    return X_train, X_test


def pca(X_train, X_test):
    pca = PCA(n_components=3, random_state=10)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_test = pca.transform(X_test)
    return X_train, X_test


def process(source_file: str):
    df = load(source_file)
    df = trim(df)
    X_train, X_test, y_train, y_test = split(df)
    X_train, X_test = scale(X_train, X_test)
    X_train, X_test = pca(X_train, X_test)
    return X_train, X_test, y_train, y_test


def process_to_files(output_dir="processed", output_file="output.npy"):
    source_files = []
    for file in sys.argv[1:]:
        source_files.append(file)
        if not os.path.exists(file):
            raise FileNotFoundError

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    elif os.path.exists(os.path.join(output_dir, output_file)):
        raise FileExistsError

    training_data, testing_data, training_labels, testing_labels = process(
        source_files[0]
    )
    for source_file in source_files[1:]:
        X_train, X_test, y_train, y_test = process(source_file)
        training_data = np.concatenate((training_data, X_train))
        testing_data = np.concatenate((testing_data, X_test))
        training_labels = np.concatenate((training_labels, y_train))
        testing_labels = np.concatenate((testing_labels, y_test))

    np.save(file=os.path.join(output_dir, "train.npy"), arr=training_data)
    np.save(file=os.path.join(output_dir, "test.npy"), arr=testing_data)
    with open(os.path.join(output_dir, "train.npy.labels"), "w") as fd:
        for sample in training_labels:
            fd.write(str(sample))
    with open(os.path.join(output_dir, "test.npy.labels"), "w") as fd:
        for sample in testing_labels:
            fd.write(str(sample))


if __name__ == "__main__":
    sys.exit(process_to_files())
