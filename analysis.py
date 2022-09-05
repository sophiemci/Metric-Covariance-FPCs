import csv
import multiprocessing
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import itertools
import time
from dataprep import all_adjacency, all_laplacians, smaller_laplacians
from metrics import frobenius, procrustes, square_root
from analysis_helpers import smoother, standard_plot, eigenfunctions

if __name__ == "__main__":
    file = "autocovs-pairs/chat-frobenius-2019-all_adj.pkl"
    metric = frobenius
    space = "adjacency matrices"

    with open(file,'rb') as f:
        c_hat = pickle.load(f)/ (365 * 364 * 4)

    standard_plot(smoother(c_hat), samples=240, title = f"Smoothed metric autocovarance surface,\n 2019 data, {metric.__name__} metric, {space}")
    evals, efuncs = eigenfunctions(smoother(c_hat, s=0), K=3, metric=frobenius, space=space)