#!/usr/bin/env python
import os
import sys
import preprocess


def process(source):
    df = preprocess.load(source)
    df = preprocess.trim(df)
    return df


def main(source):
    if not os.path.exists(source):
        raise FileNotFoundError
    output = source + ".csv"
    if os.path.exists(output):
        raise FileExistsError

    df = process(source)
    df.to_csv(output)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
