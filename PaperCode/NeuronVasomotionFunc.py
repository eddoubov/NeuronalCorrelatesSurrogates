# Standard data science libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import math
import copy as cp
import time

plt.close('all')
sns.set(color_codes=True)

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize, least_squares

from sklearn.linear_model import Lasso, LinearRegression

import scipy.stats as st
from scipy import signal
import scipy
from scipy.interpolate import interp1d
from scipy.fftpack import fft, ifft, rfft, irfft

import statsmodels.api as sm

from collections import Counter

import itertools
from itertools import permutations

import sys

import pyunicorn
from pyunicorn import timeseries
from pyunicorn.timeseries.surrogates import Surrogates

from matplotlib import cm

import os
from os import listdir
from os.path import isfile, join

import random

def createLagMat(X, tau_list):
    X = np.array(X)
    num_tau = len(tau_list)
    start_ind = np.max(tau_list)
    num_rows_X = X.shape[0]
    num_rows_lag = num_rows_X - start_ind
    
    lag_X_matrix = np.zeros((num_rows_lag, num_tau))

    for i in np.arange(num_tau):        
        temp_tau = tau_list[i]
        
        lag_X_matrix[:,i] = X[(start_ind-temp_tau):(num_rows_X-temp_tau)]
    
    return lag_X_matrix

def CMA_Sync(X, N, eig_values, eig_vecs, alpha, smooth):

    import numpy as np
    import scipy.stats as st
    
    import pyunicorn
    from pyunicorn import timeseries
    from pyunicorn.timeseries.surrogates import Surrogates
    
    m = len(eig_values)
    lambda_mat_surr = np.zeros((N, len(eig_values)))

    ts = timeseries.surrogates.Surrogates(X.T, silence_level=2)
    for i in np.arange(N):
        surr_ts = ts.AAFT_surrogates(X.T)
        surr_ts = surr_ts.T
        
        if smooth[0] == 1:
            surr_ts = smooth_mat(surr_ts, smooth[1])
        
        surr_cov_mat = np.cov(surr_ts.T)
        surr_eig_values, surr_eig_vecs = np.linalg.eig(surr_cov_mat);
    
        temp_sort_ind = np.argsort(surr_eig_values)
        sorted_surr_eig_values = surr_eig_values[temp_sort_ind]
    
        lambda_mat_surr[i,:] = sorted_surr_eig_values
    
    mean_lambdas_surr = np.mean(lambda_mat_surr, axis=0)
    std_lambdas_surr = np.std(lambda_mat_surr, axis=0)

    alpha_bonf = alpha/m
    
    K = st.norm.ppf(1-alpha_bonf/2)
    
    thresh = K*std_lambdas_surr
    
    sync_ind = np.zeros(m)
    for t in np.arange(m):
        if eig_values[t] > (thresh[t] + mean_lambdas_surr[t]):
            sync_ind[t] = (eig_values[t] - mean_lambdas_surr[t])/(m - mean_lambdas_surr[t])
        else:
            sync_ind[t] = 0
    
    PI_mat = np.zeros((m,m))

    for i in np.arange(m):
        PI_mat[m-i-1,:] = eig_values[i]*(np.power(eig_vecs[:,i], 2))
    
    return [sync_ind, PI_mat, mean_lambdas_surr, thresh]

def createTimeWindows(data, T, w):
    
    data = np.array(data);
    
    if len(data.shape) > 1:
        num_dim = data.shape[1];
    else:
        num_dim = 1;
        data = data.reshape((len(data),1))
        
    num_obs = data.shape[0];
    
    ref_beg_ind = 0;
    
    static_init_mat = np.zeros((T,num_dim,1))
    
    T_vec = []
    
    while ref_beg_ind < num_obs-T:
        
        if ref_beg_ind == 0:
            tw_mat = static_init_mat;
        else:
            tw_mat = np.concatenate((tw_mat, static_init_mat), axis=2);
        
        ref_end_ind = ref_beg_ind + T;
        
        #print(ref_end_ind)
        
        tw_mat[:,:,-1] = data[ref_beg_ind:ref_end_ind,:];
        
        ref_beg_ind = ref_end_ind - w;
    
        T_vec.append(ref_end_ind)
    
    return [tw_mat, np.array(T_vec)]

def detCorrSigTimeLags(X, L, tau_list):
    L_strd = (L - np.mean(L))/np.std(L)

    lag_X_matrix = createLagMat(X, tau_list)
    
    max_tau_shift = np.max(tau_list)
    
    L_temp = L_strd[max_tau_shift:]
    
    corr_vec = np.zeros(len(tau_list));
    
    for i in np.arange(len(tau_list)):
        
        temp_corr = np.corrcoef(L_temp, lag_X_matrix[:,i]);
        corr_vec[i] = temp_corr[0,1]
        
    return [corr_vec, lag_X_matrix]

def moving_average(x, w):

    import numpy as np

    return np.convolve(x, np.ones(w), 'valid') / w

def smooth_mat(neur_data_raw, ma_w):

    import numpy as np

    ta_adj = ma_w-1
    
    temp_mov_av = np.zeros(neur_data_raw[ta_adj:,:].shape)
    
    for i in np.arange(neur_data_raw.shape[1]):
        temp_mov_av[:,i] = moving_average(neur_data_raw[:,i], ma_w)
        
    return temp_mov_av

def genNull_Distr(X_mat, L_mat, tau_list, num_iter):

    m = X_mat.shape[1]
    
    null_corr_values = np.zeros((num_iter, m))
    corr_pvalues = np.zeros((m,1))
        
    for k in np.arange(num_iter):

        temp_ts = timeseries.surrogates.Surrogates(X_mat.T, silence_level=2)

        surr_X = temp_ts.AAFT_surrogates(X_mat.T)
        surr_X = surr_X.T

        temp_ts2 = timeseries.surrogates.Surrogates(L_mat.T, silence_level=2)

        surr_L = temp_ts2.AAFT_surrogates(L_mat.T)
        surr_L = surr_L.T
        surr_L = surr_L.ravel()

        for z in np.arange(m):
            surr_X_series = surr_X[:,z]

            temp_surr_corrs, _ = detCorrSigTimeLags(surr_X_series, surr_L, tau_list)

            sig_corr_ind = np.argmax(temp_surr_corrs)
            null_corr_values[k,z] = temp_surr_corrs[sig_corr_ind]
    
    return null_corr_values

def det_pvalue(X_mat, L_mat, target_corr, tau_list, num_iter):
    
    m = X_mat.shape[1]
    
    null_corr_values = genNull_Distr(X_mat, L_mat, tau_list, num_iter)
    
    corr_pvalues = np.zeros((m,1))
    
    for z in np.arange(m):
        temp_corr_distr = null_corr_values[:,z]
        temp_mean = np.mean(temp_corr_distr)
        temp_std = np.std(temp_corr_distr)
        temp_zscore = (target_corr[z] - temp_mean)/temp_std
        
        corr_pvalues[z] = 2*(1-st.norm.cdf(abs(temp_zscore)))
        
    return [corr_pvalues, null_corr_values]

def create_Surr_data(X, L):
    
    if len(X.shape) <= 1:
        X = X.reshape((len(X), 1))
    
    L = L.reshape((len(L), 1))
    
    ## Create Surrogate PO2 Data
    
    ts = timeseries.surrogates.Surrogates(L.T, silence_level=2)

    L_surr_ts = ts.AAFT_surrogates(L.T)
    L_surr_ts = L_surr_ts.T
    
    ## Create Surrogate Neuron Data

    ts = timeseries.surrogates.Surrogates(X.T, silence_level=2)

    X_surr_ts = ts.AAFT_surrogates(X.T)
    X_surr_ts = X_surr_ts.T

    temp_num_neuron = X_surr_ts.shape[1]
    
    return X_surr_ts, L_surr_ts
