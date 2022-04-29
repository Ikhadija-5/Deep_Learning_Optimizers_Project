from importlib.metadata import metadata


import argparse
from importlib.metadata import metadata
args = argparse.Namespace(
    lr = 0.9,
    epoch = 2000,
    bs = 8,
    train_size = 0.8,
    path = "./data",
    metadata = "./data/data.xlsx",
    wd = 1.0
)
