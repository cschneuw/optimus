import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pandas as pd

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


class Generator(nn.Module):
    def __init__(self, input_dim, output_dim, h_dim):
        super(Generator, self).__init__()
        self.layer1 = nn.Linear(input_dim * 2, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, output_dim)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        inputs = torch.cat([x, m], dim=1)
        h1 = self.activation(self.layer1(inputs))
        h2 = self.activation(self.layer2(h1))
        output = self.sigmoid(self.layer3(h2))
        return output

class Discriminator(nn.Module):
    def __init__(self, input_dim, h_dim):
        super(Discriminator, self).__init__()
        self.layer1 = nn.Linear(input_dim * 2, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, input_dim)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, h):
        inputs = torch.cat([x, h], dim=1)
        h1 = self.activation(self.layer1(inputs))
        h2 = self.activation(self.layer2(h1))
        output = self.sigmoid(self.layer3(h2))
        return output
    

def gain(data_x, gain_parameters):
    data_m = 1 - np.isnan(data_x)
    batch_size = gain_parameters['batch_size']
    hint_rate = gain_parameters['hint_rate']
    alpha = gain_parameters['alpha']
    iterations = gain_parameters['iterations']
    no, dim = data_x.shape
    h_dim = int(dim)
    
    norm_data, norm_parameters = normalization(data_x)
    norm_data_x = np.nan_to_num(norm_data, 0)
    
    # GAIN architecture
    G = Generator(dim, dim, h_dim)
    D = Discriminator(dim, h_dim)
    
    optimizer_G = optim.Adam(G.parameters())
    optimizer_D = optim.Adam(D.parameters())
    
    X = torch.tensor(norm_data_x, dtype=torch.float32)
    M = torch.tensor(data_m, dtype=torch.float32)
    H = torch.tensor(binary_sampler(hint_rate, no, dim), dtype=torch.float32)
    
    for it in tqdm(range(iterations)):
        batch_idx = sample_batch_index(no, batch_size)
        X_mb = torch.tensor(norm_data_x[batch_idx, :], dtype=torch.float32)
        M_mb = torch.tensor(data_m[batch_idx, :], dtype=torch.float32)
        Z_mb = torch.tensor(uniform_sampler(0, 0.01, batch_size, dim), dtype=torch.float32)
        H_mb_temp = torch.tensor(binary_sampler(hint_rate, batch_size, dim), dtype=torch.float32)
        H_mb = M_mb * H_mb_temp
        
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
        
        optimizer_D.zero_grad()
        D_loss_temp = -torch.mean(M_mb * torch.log(D(X_mb, H_mb) + 1e-8) + (1 - M_mb) * torch.log(1. - D(X_mb, H_mb) + 1e-8))
        D_loss_temp.backward()
        optimizer_D.step()
        
        optimizer_G.zero_grad()
        G_loss_temp = -torch.mean((1 - M_mb) * torch.log(D(G(X, M), H_mb) + 1e-8))
        MSE_loss = torch.mean((M_mb * X_mb - M_mb * G(X, M))**2) / torch.mean(M_mb)
        G_loss = G_loss_temp + alpha * MSE_loss
        G_loss.backward()
        optimizer_G.step()
    
    Z_mb = torch.tensor(uniform_sampler(0, 0.01, no, dim), dtype=torch.float32)
    M_mb = torch.tensor(data_m, dtype=torch.float32)
    X_mb = torch.tensor(norm_data_x, dtype=torch.float32)
    X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
    
    imputed_data = G(X_mb, M_mb).detach().numpy()
    imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
    imputed_data = renormalization(imputed_data, norm_parameters)
    imputed_data = rounding(imputed_data, data_x)
    
    return imputed_data

# Make sure to replace the utils functions like binary_sampler, uniform_sampler, sample_batch_index,
# normalization, renormalization, and rounding with their PyTorch equivalents.

class GAINImputer:
    def __init__(self, hint_rate, alpha, iterations):
        self.hint_rate = hint_rate
        self.alpha = alpha
        self.iterations = iterations
        self.batch_size = None
        self.G = None
        self.D = None
        self.norm_parameters = None

    def fit(self, X, y=None):
        # Create missingness mask
        self.batch_size = X.shape[0]
        if isinstance(X, pd.DataFrame):
            X = X.copy().values
        data_m = 1 - np.isnan(X)
        no, dim = X.shape
        h_dim = int(dim)
        
        # Normalize the data
        norm_data, self.norm_parameters = normalization(X)
        norm_data_x = np.nan_to_num(norm_data, 0)
        
        # Initialize the generator and discriminator
        self.G = Generator(dim, dim, h_dim)
        self.D = Discriminator(dim, h_dim)
        
        optimizer_G = optim.Adam(self.G.parameters())
        optimizer_D = optim.Adam(self.D.parameters())
        
        # Convert data to tensors
        X_tensor = torch.tensor(norm_data_x, dtype=torch.float32)
        M_tensor = torch.tensor(data_m, dtype=torch.float32)
        
        # Training loop
        for it in tqdm(range(self.iterations)):
            # Sample a mini-batch
            batch_idx = sample_batch_index(no, self.batch_size)
            X_mb = torch.tensor(norm_data_x[batch_idx, :], dtype=torch.float32)
            M_mb = torch.tensor(data_m[batch_idx, :], dtype=torch.float32)
            Z_mb = torch.tensor(uniform_sampler(0, 0.01, self.batch_size, dim), dtype=torch.float32)
            H_mb_temp = torch.tensor(binary_sampler(self.hint_rate, self.batch_size, dim), dtype=torch.float32)
            H_mb = M_mb * H_mb_temp
            
            # Combine observed data and random noise for missing values
            X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
            
            # Train the Discriminator
            optimizer_D.zero_grad()
            D_loss = -torch.mean(M_mb * torch.log(self.D(X_mb, H_mb) + 1e-8) +
                                 (1 - M_mb) * torch.log(1. - self.D(X_mb, H_mb) + 1e-8))
            D_loss.backward()
            optimizer_D.step()
            
            # Train the Generator
            optimizer_G.zero_grad()
            G_loss_temp = -torch.mean((1 - M_mb) * torch.log(self.D(self.G(X_tensor, M_tensor), H_mb) + 1e-8))
            MSE_loss = torch.mean((M_mb * X_mb - M_mb * self.G(X_tensor, M_tensor))**2) / torch.mean(M_mb)
            G_loss = G_loss_temp + self.alpha * MSE_loss
            G_loss.backward()
            optimizer_G.step()

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy().values
        data_m = 1 - np.isnan(X)
        no, dim = X.shape
        
        # Normalize the input data
        norm_data, _ = normalization(X, parameters=self.norm_parameters)
        norm_data_x = np.nan_to_num(norm_data, 0)
        
        # Impute missing values
        Z_mb = torch.tensor(uniform_sampler(0, 0.01, no, dim), dtype=torch.float32)
        M_mb = torch.tensor(data_m, dtype=torch.float32)
        X_mb = torch.tensor(norm_data_x, dtype=torch.float32)
        X_mb = M_mb * X_mb + (1 - M_mb) * Z_mb
        
        # Generate imputed data
        imputed_data = self.G(X_mb, M_mb).detach().numpy()
        imputed_data = data_m * norm_data_x + (1 - data_m) * imputed_data
        
        # Renormalize the data back to the original scale
        imputed_data = renormalization(imputed_data, self.norm_parameters)
        imputed_data = rounding(imputed_data, X)
        
        return imputed_data

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)