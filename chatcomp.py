import csv
import multiprocessing
import pandas as pd
import numpy as np
import pickle
import os
import datetime
import itertools
import time
from dataprep import all_adjacency, all_laplacians, smaller_laplacians, smaller_adjacency
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
def c_hat_years(year1, year2, metric, laplacians):
    keys1 = filter(lambda x: x.year == year1, laplacians.keys())
    keys1b = filter(lambda x: x.year == year1, laplacians.keys())
    keys2 = filter(lambda x: x.year == year2, laplacians.keys())
    keys2b = filter(lambda x: x.year == year2, laplacians.keys())
    dict1 = dict(zip(keys1, map(lambda x: laplacians[x],keys1b)))
    dict2 = dict(zip(keys2, map(lambda x: laplacians[x],keys2b)))
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

def c_hat_fast(yr, metric, laplacians):
    name = laplacians.pop("name")
    keys1 = filter(lambda x: x.year == yr, laplacians.keys())
    keys1b = filter(lambda x: x.year == yr, laplacians.keys())
    dict1 = dict(zip(keys1, map(lambda x: laplacians[x],keys1b)))
    print(f"starting for {yr}")
    ## apply the itertools map etc to the pairwise terms 

    pairs = itertools.product(dict1.values(), dict1.values(), repeat=1)
    C_hat = np.zeros((24,24))
    
    pool = multiprocessing.Pool()
    D1_temp = pool.starmap(day_to_day,map(lambda x: (x[0],x[1],metric), pairs))
    D1 = np.sum(D1_temp, axis = 0)

    D3_temp = pool.starmap(day_to_day,map(lambda x: (x,x,metric), dict1.values()))
    D3 = np.sum(D3_temp, axis = 0)
    
    #D1.transpose() is D2, D3 = D4 
    chat = D1 + D1.transpose() - 2 * D3 * len(dict1.keys())
    with open(f'autocovs-pairs/chat-{metric.__name__}-{yr}-{name}.pkl','wb') as f:
        pickle.dump(chat/(4 * 364 * 365),f)
    
    print(f'success for {yr}!')
    return chat


if __name__ == "__main__":
    #edit these inputs
    metric = frobenius
    laplacians = smaller_laplacians
    approach = "fast"   #"fast" or "yearly"
    years = np.arange(2019,2020,1)

    if approach == "fast":
        #Compute year only c-hats
        for year in years:
            c_hat_fast(year, metric, laplacians)

    else:
        pairs = list(itertools.combinations(years,2)) + [(year,year) for year in years]
        args = [(x[0],x[1],metric,laplacians) for x in pairs]
        print('starting multiprocess')
        
        pool = multiprocessing.Pool()
        results = pool.starmap(c_hat_years,args)
        
        all = np.sum(results[0:-len(years)], axis = 0)*2 + np.sum(results[-len(years):], axis = 0)
        all_s = all / (4 * len(all_laplacians) * (len(all_laplacians)-1))
        with open(f'autocovs-pairs/chat-{metric.__name__}-{laplacians["name"]}-all.pkl','wb') as f:
            pickle.dump(all_s, f)
