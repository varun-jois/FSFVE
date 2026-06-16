import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import logging

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30): 
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30., dropout=0, norm=None): # i added dropout
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))
        if dropout:
            self.net.append(nn.Dropout(dropout))
        if norm and norm == 'bn':
            num_features = 15625
            self.net.append(nn.BatchNorm1d(num_features))
        if norm and norm == 'ln':
            self.net.append(nn.LayerNorm([1024, 512]))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))
            if dropout:
                self.net.append(nn.Dropout(dropout))
            if norm and norm == 'bn':
                num_features = 15625
                self.net.append(nn.BatchNorm1d(num_features))
            if norm and norm == 'ln':
                self.net.append(nn.LayerNorm([1024, 512]))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)

    
    def forward(self, x):
        output = self.net(x) + x
        return output      

"""
with buffer
"""
# class Siren(nn.Module):
#     def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
#                  first_omega_0=30, hidden_omega_0=30., dropout=0, norm=None): # i added dropout
#         super().__init__()
        
#         self.net = []
#         self.net.append(SineLayer(in_features, hidden_features, 
#                                   is_first=True, omega_0=first_omega_0))
#         if dropout:
#             self.net.append(nn.Dropout(dropout))
#         if norm and norm == 'bn':
#             num_features = 15625
#             self.net.append(nn.BatchNorm1d(num_features))
#         if norm and norm == 'ln':
#             self.net.append(nn.LayerNorm([1024, 512]))

#         for i in range(hidden_layers):
#             self.net.append(SineLayer(hidden_features, hidden_features, 
#                                       is_first=False, omega_0=hidden_omega_0))
#             if dropout:
#                 self.net.append(nn.Dropout(dropout))
#             if norm and norm == 'bn':
#                 num_features = 15625
#                 self.net.append(nn.BatchNorm1d(num_features))
#             if norm and norm == 'ln':
#                 self.net.append(nn.LayerNorm([1024, 512]))

#         if outermost_linear:
#             final_linear = nn.Linear(hidden_features, out_features)
            
#             with torch.no_grad():
#                 final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
#                                               np.sqrt(6 / hidden_features) / hidden_omega_0)
                
#             self.net.append(final_linear)
#         else:
#             self.net.append(SineLayer(hidden_features, out_features, 
#                                       is_first=False, omega_0=hidden_omega_0))
        
#         self.net = nn.Sequential(*self.net)
#         # self.codebook = nn.Parameter(torch.rand((1024, in_features)).mul(2).sub(1))
#         # self.codebook = nn.Parameter(torch.empty((256, in_features)))
#         self.confidence = nn.Sequential(*[
#             nn.Linear(256, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         ])

#     def init_codebook(self, train_data):
#         batch_size, patches, features = train_data.shape
#         X = train_data.view((batch_size*patches, features)).numpy()
#         X = X[::4, :]
#         logging.info(f'Train size for kmeans: {X.shape}')
#         clt = KMeans(256, n_init=1)
#         clt.fit(X)
#         logging.info(f'Number of iters for kmeans: {clt.n_iter_}')
#         # with torch.no_grad():
#         #     self.codebook.copy_(torch.tensor(clt.cluster_centers_))
#         self.codebook = torch.tensor(clt.cluster_centers_).to('cuda')


#     def forward(self, x):
#         # min_dist = torch.cdist(x, self.codebook).min(dim=-1).values  # shape (b,f)
#         # min_dist = torch.exp(-min_dist.unsqueeze(-1) * 2)
#         # output = self.net(x) * min_dist + x
#         dist = torch.cdist(x, self.codebook)
#         output = x + self.net(x) * self.confidence(dist)
#         return output     

