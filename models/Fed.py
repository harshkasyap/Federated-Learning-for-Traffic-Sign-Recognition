#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import torch
from torch import nn


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def TrimmedMean(w, beta=0.05):
    """
    Perform the Trimmed Mean aggregation on client weights.
    
    Args:
        w (list of dicts): A list of client weight dictionaries.
        beta (float): Fraction of values to trim from both ends (0 <= beta < 0.5).
                      For example, beta=0.1 means trim 10% of smallest and largest values.
    
    Returns:
        dict: Aggregated weights after applying the trimmed mean.
    """
    num_clients = len(w)
    trim_count = int(beta * num_clients)  # Number of smallest and largest values to trim
    w_avg = copy.deepcopy(w[0])
    
    for k in w_avg.keys():
        # Collect the values for the current key from all clients
        values = torch.stack([w[i][k] for i in range(num_clients)])
        
        # Sort the values along the first dimension (clients)
        sorted_values, _ = torch.sort(values, dim=0)
        
        # Trim the smallest and largest `trim_count` values
        trimmed_values = sorted_values[trim_count: num_clients - trim_count]
        
        # Compute the mean of the remaining values
        w_avg[k] = torch.mean(trimmed_values, dim=0)
    
    return w_avg
