#!/usr/bin/env python
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import argparse
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datetime import datetime
from collections import Counter

mp.set_sharing_strategy("file_system")
device = torch.device("cpu")


def _time_now():
    return datetime.now().strftime("%H:%M:%S")


class ConnlogDataset(Dataset):
    def __init__(self, path, label_suffix=".labels"):
        self.path = path
        self.data = torch.tensor(np.load(path), dtype=torch.float32, device=device)
        with open(path + label_suffix) as fd:
            self.labels_string = fd.read()
            self.labels = torch.tensor(
                [[int(n)] for n in self.labels_string],
                dtype=torch.float32,
                device=device,
            )

    def get_class_counts(self):
        return Counter(self.labels_string)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def classify(args):
    num_epochs = args.epochs
    learning_rate = args.learn_rate
    stat_interval = args.stat_interval

    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=3, out_features=64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=64, out_features=32),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=32, out_features=16),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=16, out_features=1),
    ).to(device)

    train_dataloader = DataLoader(
        dataset=ConnlogDataset(
            path=os.path.join(args.input_dir, args.train_file),
            label_suffix=args.label_suffix,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_dataloader = DataLoader(
        dataset=ConnlogDataset(
            path=os.path.join(args.input_dir, args.test_file),
            label_suffix=args.label_suffix,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    # attain class weights accounting for class imbalance
    assert isinstance(train_dataloader.dataset, ConnlogDataset)
    class_counts = train_dataloader.dataset.get_class_counts()

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [class_counts["0"] / class_counts["1"]], dtype=torch.float32, device=device
        )
    )

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    training_loss, test_predictions, test_labels = [], [], []

    for epoch in range(num_epochs):
        model.train()
        batch_cnt = 0
        for inputs, labels in train_dataloader:
            batch_cnt += 1

            predictions = model(inputs)
            loss = loss_function(predictions, labels)
            training_loss.append(loss)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if stat_interval:
                if batch_cnt % stat_interval == 0:
                    print(
                        f"{_time_now()}: "
                        f"Training Loss: {(sum(training_loss[-stat_interval:])/stat_interval):.4f} "
                        f"Batch: {batch_cnt} "
                        f"Epoch [{epoch+1}/{num_epochs}] "
                    )

    model.eval()
    for inputs, labels in test_dataloader:
        predictions = model(inputs)
        predictions = torch.sigmoid(predictions)
        test_labels.extend(labels)
        test_predictions.extend(np.round(predictions.detach().numpy()))

    return test_predictions, test_labels


def classify_with_metrics(args):
    predictions, labels = classify(args)

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    report = classification_report(
        y_true=labels,
        y_pred=predictions,
        target_names=["Benign", "Malicious"],
        zero_division=0,
    )
    true_neg, false_neg, false_pos, true_pos = confusion_matrix(
        y_true=labels, y_pred=predictions
    ).ravel()
    cm = {"TN": true_neg, "FN": false_neg, "FP": false_pos, "TP": true_pos}
    return accuracy, cm, report


def main(args):
    time_start = datetime.now()
    accuracy, cm, report = classify_with_metrics(args)
    print("\nAccuracy:", accuracy)
    print("Confusion matrix: ", cm)
    print("Report:\n", report)
    print(f"Total running time: {datetime.now() - time_start}")


def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_dir",
        type=str,
        help="Path to directory containing preprocessed dataset",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=5,
        type=int,
        help="Number of passes performed on dataset",
    )
    parser.add_argument(
        "-l",
        "--learn-rate",
        default=0.1,
        type=float,
        help="Learning rate per batch for optimizer function",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=2**8,
        type=int,
        help="Number of samples per batch used before updating weights",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        default=4,
        type=int,
        help="Number of workers to spawn for dataloader",
    )
    parser.add_argument(
        "-i",
        "--stat-interval",
        default=500,
        type=int,
        help="Interval (in batches) at which to print statistics",
    )
    parser.add_argument(
        "--train-file",
        default="train.npy",
        type=str,
        help="Name of file within input dir containing training data",
    )
    parser.add_argument(
        "--test-file",
        default="test.npy",
        type=str,
        help="Name of file within input dir containing testing data",
    )
    parser.add_argument(
        "--label-suffix",
        default=".labels",
        type=str,
        help="Suffix of files within input dir containing labels",
    )
    parser.add_argument(
        "--save-plots", default=False, type=bool, help="Save plots to working directory"
    )
    return parser.parse_args(*args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    sys.exit(main(args))
