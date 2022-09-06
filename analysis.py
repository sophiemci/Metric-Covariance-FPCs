import csv
import multiprocessing
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import matplotlib.pyplot as plt
from functools import reduce
import itertools
import time
from numpy.linalg import norm, svd, eigh, eig
from dataprep import all_adjacency, all_laplacians, smaller_laplacians, smaller_adjacency
from metrics import frobenius, procrustes, square_root
from analysis_helpers import smoother, standard_plot, eigenfunctions, sqrt_mean, procrustes_mean

if __name__ == "__main__":
    file = "autocovs-pairs/chat-frobenius-2019-2019.pkl"
    metric = frobenius
    space = "Laplacian matrices"
    laplacians = all_laplacians
    laplacians.pop("name")

    laplacians = {key : item for key, item in laplacians.items() if key.year == 2019}

    colours = ['blue','green','yellow','red','purple','magenta','black']

    with open(file,'rb') as f:
        c_hat = pickle.load(f)/(4*364*365)

    standard_plot(smoother(c_hat), samples=240, title = f"Smoothed metric autocovarance surface,\n 2019 data, {metric.__name__} metric, {space}")
    evals, efuncs = eigenfunctions(smoother(c_hat, s=0), K=23, metric=metric, space=space, plot= True)

    if metric.__name__ == "frobenius":
        mean = reduce(lambda x,y: x+y, laplacians.values())/ len(laplacians)
    elif metric.__name__ == "square_root":
        mean = sqrt_mean(laplacians)
    else:
        mean = procrustes_mean(laplacians)

    scores = {'date': laplacians.keys()}
    for i in range(4):
        D_i = lambda x: list(map(lambda y: np.sqrt(metric(y[0], y[1])), zip(x, mean)))
        scores.update({f'score {i}': pd.Series(map(lambda x: np.inner(D_i(x), efuncs[i][::10])/24, laplacians.values()))})

    all_scores = pd.DataFrame(scores)
    all_scores.sort_values(by = 'date', inplace=True)
    all_scores['day'] = all_scores['date'].apply(lambda x: x.strftime('%A'))
    all_scores['year'] = all_scores['date'].apply(lambda x: x.strftime('%y'))
    all_scores = all_scores[all_scores['year']=='19']

    #extreme values
    weekday_scores = all_scores[all_scores['day']!= "Sunday"]
    weekday_scores = weekday_scores[weekday_scores['day']!= "Saturday"]
    
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday','Saturday','Sunday']
    plt.rcParams["figure.figsize"] = (6,6)
    for i,day in enumerate(days):
        sub_frame = all_scores[all_scores['day']==day]
        plt.scatter(sub_frame["score 0"],sub_frame["score 1"],color=colours[i],label=day)

    plt.legend(loc='lower right')
    plt.xlabel('FPOC 1 score')
    plt.ylabel('FPOC 2 score')
    plt.title(f'1st vs 2nd Frechet for the Cycling Data \n 2019, {metric.__name__} metric, {space} space.')
    plt.show()

    

    