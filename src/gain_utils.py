import numpy as np
import torch

# Normalization function
def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.'''
    _, dim = data.shape
    norm_data = data.copy()
    
    if parameters is None:
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)
        
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)
        
        norm_parameters = {'min_val': min_val, 'max_val': max_val}
    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']
        
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)
        
        norm_parameters = parameters
    
    return norm_data, norm_parameters

# Renormalization function
def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.'''
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']
    
    _, dim = norm_data.shape
    renorm_data = norm_data.copy()
    
    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]
    
    return renorm_data

# Rounding function
def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.'''
    _, dim = data_x.shape
    rounded_data = imputed_data.copy()
    
    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])
    
    return rounded_data

# RMSE Loss calculation
def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data.'''
    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)
    
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)
    
    rmse = np.sqrt(nominator / float(denominator))
    
    return rmse

# Xavier initialization
def xavier_init(size):
    '''Xavier initialization.'''
    in_dim = size[0]
    xavier_stddev = 1. / np.sqrt(in_dim / 2.)
    return torch.normal(mean=0, std=xavier_stddev, size=size)

# Binary Sampler
def binary_sampler(p, rows, cols):
    '''Sample binary random variables.'''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

# Uniform Sampler
def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.'''
    return np.random.uniform(low, high, size=[rows, cols])

# Sample Batch Index
def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.'''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx
