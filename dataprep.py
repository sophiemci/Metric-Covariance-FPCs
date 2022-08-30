import numpy as np
import pickle
import datetime

with open('daily_networks_hourly_2018-22.pkl','rb') as f:
    data = pickle.load(f)
    
def laplacians(X_day):
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
    #44 is the number of stations we consider for the Frobenius metric 
    L = np.zeros((44,44))
    np.fill_diagonal(L,D)
    return L - X

#all_laplacians contains the hourly laplacians for each day
all_laplacians = dict(zip(data.keys(), map(laplacians, data.values())))