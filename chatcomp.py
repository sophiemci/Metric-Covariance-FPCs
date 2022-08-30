import csv
import multiprocessing
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import itertools
import time
from dataprep import all_laplacians
from metrics import frobenius, procrustes, square_root

def day_to_day(A,B,metric):
    '''
    A,B are (24,44,44) tuples for each day of data 
    works fastest with non-vectorized version
    APPROACH 1: iterators and maps
    Could optimise to only compute for upper diag? need diff iterator structure
    '''
    items = itertools.product(A,B)
    applied = map(lambda x: metric(x[0],x[1]), items)
    return np.fromiter(applied,float).reshape(24,24)

## Get the year by year matrix
def c_hat_years(year1, year2, metric):
    keys1 = filter(lambda x: x.year == year1, all_laplacians.keys())
    keys1b = filter(lambda x: x.year == year1, all_laplacians.keys())
    keys2 = filter(lambda x: x.year == year2, all_laplacians.keys())
    keys2b = filter(lambda x: x.year == year2, all_laplacians.keys())
    dict1 = dict(zip(keys1, map(lambda x: all_laplacians[x],keys1b)))
    dict2 = dict(zip(keys2, map(lambda x: all_laplacians[x],keys2b)))
    print(f"starting for {year1} and {year2}")
    ## apply the itertools map etc to the pairwise terms 
    pairs = itertools.product(dict1.values(), dict2.values(), repeat=1)
    C_hat = np.zeros((24,24))
    
    D1 = np.sum(tuple(map(lambda x: day_to_day(x[0],x[1],metric),pairs)),axis=0)
    D3 = np.sum(tuple(map(lambda x: day_to_day(x,x,metric), dict1.values())),axis=0)
    D4 = np.sum(tuple(map(lambda x: day_to_day(x,x,metric), dict2.values())),axis=0)
    
    #D1.transpose() is D2
    chat = D1 + D1.transpose() - (D3 * len(dict2.keys()) + D4 * len(dict1.keys()))
    with open(f'autocovs-pairs/chat-{metric.__name__}-{year1}-{year2}.pkl','wb') as f:
        pickle.dump(chat,f)

    print(f'success for {year1} and {year2}!')
    return chat

if __name__ == "__main__":
    metric = frobenius
    years = np.arange(2018,2023,1)

    pairs = list(itertools.combinations(years,2)) + [(year,year) for year in years]
    args = [(x[0],x[1],metric) for x in pairs]
    print('starting multiprocess')
    
    pool = multiprocessing.Pool()
    results = pool.starmap(c_hat_years,args)
    
    all = np.sum(results[0:-len(years)], axis = 0)*2 + np.sum(results[-len(years):], axis = 0)
    all_s = all / (4 * len(all_laplacians) * (len(all_laplacians)-1))
    with open(f'autocovs-pairs/chat-{metric.__name__}-all.pkl','wb') as f:
        pickle.dump(all_s, f)
    