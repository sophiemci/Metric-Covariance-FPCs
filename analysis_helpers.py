import csv
import multiprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import datetime
import itertools
import time
from dataprep import all_adjacency, all_laplacians, smaller_laplacians
from metrics import frobenius, procrustes, square_root
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy import interpolate
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy.linalg import norm, svd, eigh, eig
import scipy.linalg as la
from scipy.optimize import minimize

def standard_plot(c_hat, samples = 24, title = None):
    #standard plotter for the estimated/ smoothed C-hat
    X, Y = np.mgrid[0:samples:1, 0:samples:1]
    lines = max(24, samples/4)
    fig = plt.figure(figsize=(20,20))
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,c_hat, 
                    cmap='jet',rcount = lines,ccount = lines, 
                    alpha=1, antialiased=False,edgecolor='black',lw=0.01)
    ax.view_init(35, 250)
    ax.set_title(f"{title}",size=16)
    plt.xlabel("Hour of day",size=14,labelpad=20)
    plt.ylabel("Hour of day",size=14,labelpad=20)
    plt.xticks(np.arange(0,samples,samples/6), [f'{item}00hrs' for item in np.arange(0,24,4)],size=10)
    plt.yticks(np.arange(0,samples,samples/6), [f'{item}00hrs' for item in np.arange(0,24,4)],size=10)
    ax.set_zlabel('C_hat(s,t)', fontsize=14, labelpad=20)
    for t in ax.zaxis.get_major_ticks(): t.label.set_fontsize(10)
    plt.show()

def smoother(c_hat, samples = 24, s=0):
    '''
    Fits a 2D rectangular spline to an estimated autocovariance surface.
    Defaults to hourly (24) samples
    s parameter may need to be changed if using different frequency
    returns a 10x finer grid sampled c_hat
    '''
    X, Y = np.mgrid[0:samples:1, 0:samples:1]
    Z = c_hat
    xnew, ynew = np.mgrid[0:samples:0.1, 0:samples:0.1]

    spline = interpolate.RectBivariateSpline(X[:,0], Y[0,:], Z,s=s)
    c_hat_new = spline(xnew[:,0], ynew[0,:])
    return c_hat_new

def eigenfunctions(c_hat, K, metric, space = None):
    '''
    Computes the first K eigenfunctions of C_hat by constrained maximisation
    Constraints - evecs have 2-norm 1 and are orthogonal
    '''
    
    #objective for minimization is v -> -v^TCv
    evec_obj = lambda v,C : np.matmul(np.transpose(v),np.matmul(-1*C,v))
    
    #initialise to the first computed np.linalg eigenfunction
    v_0 = eigh(c_hat)[1][0]

    constraints = [{'type':'eq', 'fun': lambda v: 1 - np.inner(v,v),
                            'jac':lambda v: -2*v
                          }]
    print(f"getting 1st Eigenfunction")
    opt = minimize(evec_obj,v_0,(c_hat),
                   jac = lambda x, c_hat: -2*np.matmul(c_hat,x),
            constraints = constraints,
            method = 'SLSQP',
            options={'maxiter':2000})

    lambdas = [opt.fun]
    opts = [opt.x]
    print(opt.status, opt.message)
    
    
    for k in range(K):
        print(f"getting {k+2}th Eigenfunction")
        new_constr = {'type':'eq',
                     'fun': lambda v,u: np.inner(u,v),
                     'args':[opts[k]]
                          }
        constraints.append(new_constr)

        next_opt = minimize(evec_obj,v_0,(c_hat),
                   jac = lambda x, c_hat: -2*np.matmul(c_hat,x),
            constraints = constraints,
            method='SLSQP',
            options={'maxiter':2000
                    })
        opts.append(next_opt.x)
        lambdas.append(next_opt.fun) 
        print(next_opt.status, next_opt.message)
        
    #plotting
    perc_exp = 100*lambdas/np.sum(lambdas)
    J = len(opts[0])
    for i in range((K+1)//4):
        plt.clf()
        for j in range(4):
            plt.plot(opts[i*4+j],label=f'ef {i*4+j+1}, var_expl= {np.round(100*perc_exp[i*4+j],2)} %')
        plt.legend(loc='best')
        plt.title(f'eigenfunctions {i*4+1} to {i*4+4} of C(s,t)\n fitted by maximisation, 5 observations per hour\n {metric.__name__} metric, {space}')
        plt.xlabel('t')
        plt.xticks(np.linspace(0,J+1,7),np.linspace(0,24,7))
        plt.show()
        
    return lambdas, opts




