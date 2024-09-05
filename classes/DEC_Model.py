# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 11:38:30 2024
@author: djy41
"""
import torch.nn as nn
import torch

# Define DEC model
class ClusterLayer(nn.Module):
    def __init__(self, z, n_clusters):
        super(ClusterLayer, self).__init__()
        self.z = z
        self.n_clusters = n_clusters
        self.cluster_centers = nn.Parameter(torch.Tensor(n_clusters, z.size(dim=0)))
        self.cluster_centers.data.uniform_(-1, 1)
    
    def forward(self, z):
        q = self.soft_assign(z)
        return  q
    
    def soft_assign(self, z):
        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_centers)**2, dim=2))
        q = q ** ((self.cluster_centers.shape[1] + 1.0) / 2.0)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q
#*****************************************************************************