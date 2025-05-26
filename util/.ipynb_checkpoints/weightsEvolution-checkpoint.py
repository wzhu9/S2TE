import networkx as nx
import torch
import numpy as np
import random

# def initialize_ba_network(layer_sizes, m0):
#     G = nx.barabasi_albert_graph(layer_sizes[0], m0)
#     adjacency_matrix = nx.adjacency_matrix(G).toarray()
#     input_size, output_size = layer_sizes[0], layer_sizes[1]
#     ba_weights = np.zeros((input_size, output_size))

#     for i in range(input_size):
#         connected_nodes = np.nonzero(adjacency_matrix[i])[0]
#         for j in range(output_size):
#             ba_weights[i, j] = 1 if j in connected_nodes else 0

#     noParameters = np.count_nonzero(ba_weights == 1)

#     return noParameters, ba_weights

def initialize_ba_network(layer_sizes, sparsity_factor):
    """
    使用BA算法将两层神经网络单元连接起来，通过稀疏度超参数控制连接的数量。
    :param input_size: 输入层单元数量
    :param output_size: 输出层单元数量
    :param sparsity_factor: 稀疏度超参数，控制连接数量占总数的百分比
    :return: 一个表示连接的邻接矩阵
    """

    input_size, output_size = layer_sizes
    # 计算目标连接数
    total_possible_connections = input_size * output_size
    target_connections = int(total_possible_connections * sparsity_factor)

    # 初始化图
    G = nx.Graph()

    # BA模型不直接适用于“两层”模型，但我们可以模拟通过逐步添加边来体现优先连接
    # 添加所有节点
    G.add_nodes_from(range(input_size + output_size))

    added_edges = 0
    while added_edges < target_connections:
        # 从输入层随机选择一个节点
        source = np.random.randint(0, input_size)

        # 计算输出层每个节点的连接概率
        degrees = np.array([max(G.degree(node), 1) for node in range(input_size, input_size + output_size)])
        probabilities = degrees / degrees.sum()

        # 选择目标节点
        target = np.random.choice(range(input_size, input_size + output_size), p=probabilities)

        # 添加边
        if not G.has_edge(source, target):
            G.add_edge(source, target)
            added_edges += 1

    # 获取邻接矩阵
    ba_weights = nx.to_numpy_array(G)[0:input_size, input_size:input_size + output_size]
    noParameters = np.count_nonzero(ba_weights == 1)

    return noParameters, ba_weights

def initialize_er_network(epsilon, noRows, noCols):
    # Generate an Erdos Renyi sparse weights mask
    mask_weights = torch.rand((noRows, noCols))
    prob = 1 - (epsilon * (noRows + noCols)) / (noRows * noCols)  # normal tp have 8x connections
    mask_weights = torch.where(mask_weights < prob, torch.zeros_like(mask_weights), torch.ones_like(mask_weights))
    noParameters = torch.sum(mask_weights)
    return noParameters.item(), mask_weights

def find_first_pos(weight_tensor, value):
    diff = torch.abs(weight_tensor - value)
    idx = torch.argmin(diff)
    return idx.item()


def find_last_pos(tensor, value):
    diff = torch.abs(tensor - value)
    reversed_diff = torch.flip(diff, [0])  # 反转张量
    idx = torch.argmin(reversed_diff)
    last_pos = tensor.shape[0] - idx.item()
    return last_pos


# def rewire_mask(weights, no_weights, zeta, rf, epoch):
#     # Rewire weight matrix3
#     no_weights = torch.nonzero(weights).size(0)
#     if epoch == 50 or epoch == 100 or epoch == 150 or epoch == 200 or epoch == 299:
#         print("weights number of epoch{} is {}".format(epoch, no_weights))  
#     # Flatten and sort the weights array
#     values = torch.flatten(weights)
#     values, _ = torch.sort(values)

#     # Calculate the threshold values for removing weights
#     first_zero_pos = find_first_pos(values, 0)
#     last_zero_pos = find_last_pos(values, 0)
#     largest_negative = values[int((1 - zeta) * first_zero_pos)]
#     smallest_positive = values[
#         int(min(values.shape[0] - 1, last_zero_pos + zeta * (values.shape[0] - last_zero_pos)))]

#     # Rewire the weights
#     rewired_weights = weights.clone()
#     rewired_weights[rewired_weights > smallest_positive] = 1
#     rewired_weights[rewired_weights < largest_negative] = 1
#     rewired_weights[rewired_weights != 1] = 0
#     weight_mask_core = rewired_weights.clone()

#     # Add random weights
#     nr_add = 0
#     no_rewires = int(no_weights * rf)
#     if rf == zeta:
#         no_rewires = no_weights - torch.sum(rewired_weights)
#     while nr_add < no_rewires:
#         i = np.random.randint(0, rewired_weights.shape[0])
#         j = np.random.randint(0, rewired_weights.shape[1])
#         if rewired_weights[i, j] == 0:
#             rewired_weights[i, j] = 1
#             nr_add += 1

#     return rewired_weights, weight_mask_core

# # dynamic adjustment of regeneration factor
# def cal_rf(epoch, epoch_max, k_sig, zeta, rl):
#     epoch_extremum = epoch_max // 3
#     epoch_stable = epoch_extremum * 2
#     if epoch < epoch_extremum:
#         rf = 1 / (1 + np.exp(-k_sig * (epoch - epoch_extremum)))
#         rf = zeta + zeta - ((zeta - rl) + 2 * rl * rf)
#         return rf
#     if epoch_extremum <= epoch < epoch_stable:
#         rf = 1 / (1 + np.exp(-0.8 * k_sig * (epoch - epoch_extremum)))
#         rf = zeta + zeta - ((zeta - rl) + 2 * rl * rf)
#         return rf
#     if epoch >= epoch_stable:
#         return zeta


def rewire_mask(weights, no_weights, zeta, rf, epoch, stop_epoch, sparse_para_num):  
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
    #     start_epoch = 200-epoch
    # if (stop_epoch != 0):
    #     start_epoch = 200 - (stop_epoch+1)
    # Add random weights
    nr_add = 0
    no_rewires = 0
    if (stop_epoch !=0 and stop_epoch < epoch < 100):
        no_rewires = no_weights - torch.sum(rewired_weights).item()
    elif (100<epoch<200):
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
def cal_rf(epoch, epoch_max, k_sig, zeta, rl):
    epoch_extremum = 100
    epoch_stable = 200
    if epoch < epoch_extremum:
        rf = 1 / (1 + np.exp(-k_sig * (epoch - epoch_extremum)))
        rf = zeta + zeta - ((zeta - rl) + 2 * rl * rf)
        return rf
    if epoch_extremum <= epoch < epoch_stable:
        rf = 1 / (1 + np.exp(-0.8 * k_sig * (epoch - epoch_extremum)))
        rf = zeta + zeta - ((zeta - rl) + 2 * rl * rf)
        return rf
    if epoch >= epoch_stable:
        return zeta




