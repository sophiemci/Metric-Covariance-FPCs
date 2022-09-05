import numpy as np
import pickle
import datetime

with open('daily_networks_hourly_2018-22.pkl','rb') as f:
    data = pickle.load(f)

def get_adjacency(X_day):
    X_day = np.vectorize(lambda x: 0.5*(x + np.transpose(x)),
                       signature= '(n,n)->(n,n)')(X_day)
    X_scaled = X_day/np.max(X_day)
    return X_scaled

    
def get_laplacians(X_day):
    '''
    For a day of data, 
        - take the symmetric part of the adjacency matrix
        - scale entries to [0,1] accross the day
        - compute Laplacian for each hourly time slice
    '''
    X_day = np.vectorize(lambda x: 0.5*(x + np.transpose(x)),
                       signature= '(n,n)->(n,n)')(X_day)
    X_scaled = X_day/np.max(X_day)
    l = np.vectorize(lap, signature= '(n,n)->(n,n)')
    return l(X_scaled)
    
def lap(X):
    D = np.sum(X,axis=1)
    n = len(D)
    L = np.zeros((n,n))
    np.fill_diagonal(L,D)
    return L - X

#all_laplacians contains the hourly laplacians for each day
all_adjacency = dict(zip(data.keys(), map(get_adjacency, data.values())))
all_adjacency['name'] = "all_adj"

#all_laplacians contains the hourly laplacians for each day
all_laplacians = dict(zip(data.keys(), map(get_laplacians, data.values())))
all_laplacians['name'] = "all_lap"

#the 10 x 10 laplacians we look at for the Procrustes and Square Root metric
smaller_laplacians = dict(zip(data.keys(), map(lambda x: get_laplacians(x[:,:10,:10]), data.values())))
smaller_laplacians['name'] = "small_lap"