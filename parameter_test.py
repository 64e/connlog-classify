#!/usr/bin/env python
import os
import sys
import classify


def get_log_file(path="log.txt"):
    if os.path.exists(path):
        raise FileExistsError
    return open(path, "w")


def main(input_dir: str):
    args = {}
    epochs = [1, 5, 10, 25, 50, 100]
    learning_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1]

    log = get_log_file()

    for epoch in epochs:
        for learning_rate in learning_rates:
            args = classify.parse_args(
                [input_dir, "--epochs", str(epoch), "--learn-rate", str(learning_rate)]
            )
            accuracy, cm, _ = classify.classify_with_metrics(args)
            log.write(
                f"[epochs: {epoch}, learning_rate: {learning_rate}] accuracy: {accuracy}, cm: {cm}\n"
            )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
