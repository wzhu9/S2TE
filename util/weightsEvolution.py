# coding=utf-8

import networkx as nx
import torch
import numpy as np
import random

def initialize_ba_network(layer_sizes, sparsity_factor):
    """
    Barabási–Albert style bipartite graph
    
    Parameters
    ----------
    layer_sizes : (int, int)
        Tuple of (input_size, output_size).
    sparsity_factor : float
        Fraction (0–1) of total possible connections to create.

    Returns
    -------
    ba_weights : ndarray, shape (input_size, output_size)
        Binary adjacency matrix (float dtype for compatibility).
    """
    input_size, output_size = layer_sizes
    total_possible = input_size * output_size

    # Edge budget, rounded to nearest integer, at least 1 when sparsity_factor>0
    if sparsity_factor <= 0:
        return np.zeros((input_size, output_size), dtype=float)
    target_edges = int(round(total_possible * sparsity_factor))
    target_edges = max(1, min(target_edges, total_possible))
    rng = np.random.default_rng()
    adj = np.zeros((input_size, output_size), dtype=np.uint8)  # 0/1 matrix
    degrees = np.ones(output_size, dtype=np.int32)             # start at 1
    degree_sum = degrees.sum()

    # ---------- main loop ----------
    edges_added = 0
    while edges_added < target_edges:
        src = rng.integers(input_size)                         # random input node

        # weighted choice for output node (O(out_size) cumsum in C, fast)
        r = rng.integers(degree_sum)
        tgt = np.searchsorted(np.cumsum(degrees), r, side='right')

        # If edge exists, pick again (expected few retries under sparse regimes)
        if adj[src, tgt]:
            continue

        # add edge and update bookkeeping
        adj[src, tgt] = 1
        degrees[tgt] += 1
        degree_sum += 1
        edges_added += 1

    return adj.astype(float)


def find_first_pos(weight_tensor, value):
    diff = torch.abs(weight_tensor - value)
    idx = torch.argmin(diff)
    return idx.item()


def find_last_pos(tensor, value):
    diff = torch.abs(tensor - value)
    reversed_diff = torch.flip(diff, [0])  
    idx = torch.argmin(reversed_diff)
    last_pos = tensor.shape[0] - idx.item()
    return last_pos



def rewire_mask(weights, zeta, rf, epoch, stop_epoch, sparse_para_num, epoch_growth, epoch_stable):
    # Rewire weight matrix
    no_weights = torch.sum(weights != 0).item()
    # Flatten and sort the weights array
    values = torch.flatten(weights)
    values, _ = torch.sort(values)

    # Calculate the threshold values for removing weights
    first_zero_pos = find_first_pos(values, 0)
    last_zero_pos = find_last_pos(values, 0)
    largest_negative = values[int((1 - zeta) * first_zero_pos)]
    smallest_positive = values[
        int(min(values.shape[0] - 1, last_zero_pos + zeta * (values.shape[0] - last_zero_pos)))]

    # Rewire the weights
    rewired_weights = weights.clone()
    rewired_weights[rewired_weights > smallest_positive] = 1
    rewired_weights[rewired_weights < largest_negative] = 1
    rewired_weights[rewired_weights != 1] = 0
    weight_mask_core = rewired_weights.clone()

    if (torch.sum(rewired_weights == 0).item() == 0 and stop_epoch == 0):
        stop_epoch = epoch-1
    nr_add = 0
    no_rewires = 0
    if (stop_epoch !=0 and stop_epoch < epoch < epoch_growth):
        no_rewires = no_weights - torch.sum(rewired_weights).item()
    elif (epoch_growth<epoch<epoch_stable):
        no_rewires = int(no_weights * rf)
        if ((torch.sum(rewired_weights).item() + no_rewires) < sparse_para_num):
            no_rewires = sparse_para_num - torch.sum(rewired_weights).item()
    else:
        no_rewires = int(no_weights * rf)
        if rf == zeta:
            no_rewires = no_weights - torch.sum(rewired_weights).item()

    # no_rewires = no_weights - torch.sum(rewired_weights).item()
    while nr_add < no_rewires:
        if (torch.sum(rewired_weights == 0).item() == 0):
            break
        zero_positions = torch.nonzero(rewired_weights == 0)
        random_position = zero_positions[random.randint(0, len(zero_positions) - 1)]
        rewired_weights[random_position[0], random_position[1]] = 1
        nr_add += 1

    fixed_stop_epoch = stop_epoch

    return rewired_weights, weight_mask_core, fixed_stop_epoch

# dynamic adjustment of regeneration factor
def cal_rf(epoch, epoch_growth, epoch_stable, k_sig, zeta, rl):
    if epoch < epoch_growth:
        rf = 1 / (1 + np.exp(-k_sig * (epoch - epoch_growth)))
        rf = zeta + zeta - ((zeta - rl) + 2 * rl * rf)
        return rf
    if epoch_growth <= epoch < epoch_stable:
        rf = 1 / (1 + np.exp(-0.8 * k_sig * (epoch - epoch_growth)))
        rf = zeta + zeta - ((zeta - rl) + 2 * rl * rf)
        return rf
    if epoch >= epoch_stable:
        return zeta




