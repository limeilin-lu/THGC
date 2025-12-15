# DRNL_HDE_4Node_Enhanced.py
# 基于DRNL_HDE_2Loss_update.py修改为支持4个节点类型的异构图
# 4个节点类型：被动元件、主动元件、电源元件、网络节点
#跑kicad+ltspice+ltspice数据集的时候，把device_type改成=34;把DATASET=改成对应的数据集
#跑spicenetlist+analoggenie+masalachai数据集的时候，把device_type改成=18;把DATASET=改成对应的数据集
import math
from itertools import chain
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse.csgraph import shortest_path
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score
from torch.nn import BCEWithLogitsLoss, Conv1d, MaxPool1d, ModuleList
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, GCNConv, global_sort_pool
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import k_hop_subgraph, to_scipy_sparse_matrix, negative_sampling
from tqdm import tqdm
import GPUtil
import time
import warnings
import matplotlib.pyplot as plt
import os
import networkx as nx
from collections import deque

# Set Device Here
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 修改的参数配置 ====================
# Dataset - 【修改1】更改为4节点异构图数据集
#DATASET = "SpiceNetlist"
#DATASET = "Masala-CHAI"
DATASET = "KiCad_github"
#DATASET = "LTspice_demos"
#DATASET = "LTspice_examples"
DATASET_ROOT_DIRECTORY = "./data/"
DATASET_PT = DATASET_ROOT_DIRECTORY + DATASET + "_4node_heterogeneous.pt"  # 【修改】数据集文件名
DATASET_PROCESSED = DATASET + "_4node_heterogeneous_processed.pt"  # 【修改】处理后文件名

# HDE Configuration - 【修改2】更新为4节点类型
USE_HDE = True
NODE_TYPES = 4  # 【修改】从2改为4：被动元件 + 主动元件 + 电源元件 + 网络节点
MAX_DIST = 3
# 【修改3】更新节点类型映射
HDE_TYPE_MAPPING = {
    'P': 0,  # 被动元件 (Passive)
    'A': 1,  # 主动元件 (Active)
    'S': 2,  # 电源元件 (Source)
    'N': 3  # 网络节点 (Network)
}


# ==================== 关键改进：更优的超参数（保持不变）====================
N_SPLITS = 5
MIN_NUM_EPOCHS = 8
RANDOM_STATE = 42
MAX_NUM_EPOCHS = 60  # 增加训练轮数
PATIENCE = 6  # 增加耐心值
MIN_IMPROVEMENT = 0.001
LEARNING_RATE = 1e-4  # 提高学习率
WEIGHT_DECAY = 1e-6  # 添加权重衰减
BATCH_SIZE = 6  # 增加批次大小（根据GPU内存调整）
HIDDEN_CHANNELS = 80  # 增加隐藏维度
NUM_LAYERS = 4  # 增加网络深度
DROPOUT_RATE = 0.5  # 添加Dropout
MAX_EPOCHS_WHERE_TEST_ACC_STUCK = 8

# 【修改4】模型保存目录
MODEL_SAVE_DIRECTORY = "./model-save-4node"
PLOT_SAVE_DIRECTORY = "./plot-4node"
os.makedirs(MODEL_SAVE_DIRECTORY, exist_ok=True)
os.makedirs(PLOT_SAVE_DIRECTORY, exist_ok=True)

warnings.filterwarnings('ignore')


def get_gpu_usage():
    GPUs = GPUtil.getGPUs()
    if len(GPUs) == 0:
        return "No GPU"
    gpu_info = []
    for gpu in GPUs:
        gpu_info.append(f"GPU {gpu.id}: {gpu.load * 100:.1f}% Load, {gpu.memoryUtil * 100:.1f}% Memory")
    return "; ".join(gpu_info)


class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self):
        return [DATASET_PROCESSED]

    def process(self):
        data_list = [torch.load(DATASET_PT, weights_only=False)]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def drnl_node_labeling(edge_index, src, dst, num_nodes=None):
    src, dst = (dst, src) if src > dst else (src, dst)
    adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
    idx = list(range(src)) + list(range(src + 1, adj.shape[0]))
    adj_wo_src = adj[idx, :][:, idx]
    idx = list(range(dst)) + list(range(dst + 1, adj.shape[0]))
    adj_wo_dst = adj[idx, :][:, idx]
    dist2src = shortest_path(adj_wo_dst, directed=False, unweighted=True, indices=src)
    dist2src = np.insert(dist2src, dst, 0, axis=0)
    dist2src = torch.from_numpy(dist2src)
    dist2dst = shortest_path(adj_wo_src, directed=False, unweighted=True, indices=dst - 1)
    dist2dst = np.insert(dist2dst, src, 0, axis=0)
    dist2dst = torch.from_numpy(dist2dst)
    dist = dist2src + dist2dst
    dist_over_2, dist_mod_2 = dist // 2, dist % 2
    z = 1 + torch.min(dist2src, dist2dst)
    z += dist_over_2 * (dist_over_2 + dist_mod_2 - 1)
    z[src] = 1.
    z[dst] = 1.
    z[torch.isnan(z)] = 0.
    return z.to(torch.long)


# 【修改5】HDE特征提取器 - 更新为4节点类型
class HDE_Enhanced_Subgraph_Extractor:
    def __init__(self, node_types=4, max_dist=3):  # 【修改】从2改为4
        self.node_types = node_types
        self.max_dist = max_dist
        self.type2idx = HDE_TYPE_MAPPING
        self.global_node_types = None

    def prepare_node_types(self, data):
        if hasattr(data, "node_types"):
            self.global_node_types = data.node_types
        else:
            raise ValueError("Data object must have 'node_types' attribute for heterogeneous graph")

    # 【修改6】更新节点类型推断函数
    def infer_node_type(self, global_node_idx):
        if self.global_node_types is None:
            raise RuntimeError("Call prepare_node_types() first")

        node_type_idx = self.global_node_types[global_node_idx].item()
        # 映射到对应的节点类型字符
        type_map = {0: 'P', 1: 'A', 2: 'S', 3: 'N'}
        return type_map.get(node_type_idx, 'P')  # 默认为被动元件

    def edge_index_to_networkx(self, edge_index, sub_nodes, data):
        G = nx.Graph()
        for i, node_idx in enumerate(sub_nodes):
            global_idx = node_idx.item()
            node_type = self.infer_node_type(global_idx)
            G.add_node(f"N{i}", type=node_type, original_idx=global_idx)
        for i in range(edge_index.size(1)):
            src, dst = edge_index[:, i].tolist()
            G.add_edge(f"N{src}", f"N{dst}")
        return G

    # 【修改7】更新HDE计算以支持4种节点类型
    def compute_node_hde(self, G, node_name, target_name):
        try:
            try:
                shortest_path = nx.shortest_path(G, node_name, target_name)
                if len(shortest_path) - 1 > self.max_dist + 1:
                    return np.zeros(self.node_types * (self.max_dist + 1), dtype=np.float32)
            except nx.NetworkXNoPath:
                return np.zeros(self.node_types * (self.max_dist + 1), dtype=np.float32)

            cnt = [self.max_dist] * self.node_types
            try:
                paths = []
                # 【修改8】初始化4种节点类型计数
                queue = deque([(node_name, [node_name], {'P': 0, 'A': 0, 'S': 0, 'N': 0})])

                while queue:
                    current, path, type_counts = queue.popleft()
                    if current == target_name:
                        paths.append((path, type_counts.copy()))
                        if len(paths) >= 3:
                            break
                        continue
                    if len(path) >= self.max_dist + 2:
                        continue
                    for neighbor in G.neighbors(current):
                        if neighbor not in path:
                            new_type_counts = type_counts.copy()
                            neighbor_type = G.nodes[neighbor].get('type', 'P')
                            if neighbor_type in self.type2idx:
                                new_type_counts[neighbor_type] += 1
                            queue.append((neighbor, path + [neighbor], new_type_counts))

                if not paths:
                    return np.zeros(self.node_types * (self.max_dist + 1), dtype=np.float32)
            except:
                # 【修改9】回退路径计算更新为4种节点类型
                paths = [(shortest_path,
                          {node_type: shortest_path.count(node_type) - (
                              1 if node_type == G.nodes[node_name].get('type', 'P') else 0)
                           for node_type in ['P', 'A', 'S', 'N']})]

            # 【修改10】处理4种节点类型的路径
            for path, type_counts in paths:
                res = [0] * self.node_types
                res[0] = type_counts.get('P', 0)  # 被动元件
                res[1] = type_counts.get('A', 0)  # 主动元件
                res[2] = type_counts.get('S', 0)  # 电源元件
                res[3] = type_counts.get('N', 0)  # 网络节点

                for k in range(self.node_types):
                    if res[k] > 0:
                        cnt[k] = min(cnt[k], res[k])

            one_hot_list = []
            for i in range(self.node_types):
                count_val = min(cnt[i], self.max_dist)
                one_hot = np.eye(self.max_dist + 1, dtype=np.float32)[count_val]
                one_hot_list.append(one_hot)

            return np.concatenate(one_hot_list)
        except Exception as e:
            print(f"HDE computation error: {e}")
            return np.zeros(self.node_types * (self.max_dist + 1), dtype=np.float32)

    def compute_subgraph_hde(self, sub_nodes, sub_edge_index, src, dst, data):
        try:
            G = self.edge_index_to_networkx(sub_edge_index, sub_nodes, data)
            node_mapping = {i: f"N{i}" for i in range(len(sub_nodes))}
            hde_matrix = []
            for i, node_idx in enumerate(sub_nodes):
                node_name = node_mapping[i]
                src_name = node_mapping[src]
                dst_name = node_mapping[dst]
                if node_name in G.nodes and src_name in G.nodes and dst_name in G.nodes:
                    dist_to_src = self.compute_node_hde(G, node_name, src_name)
                    dist_to_dst = self.compute_node_hde(G, node_name, dst_name)
                    hde_feature = np.concatenate([dist_to_src, dist_to_dst])
                else:
                    hde_feature = np.zeros(self.node_types * (self.max_dist + 1) * 2, dtype=np.float32)
                hde_matrix.append(hde_feature)
            return torch.FloatTensor(np.array(hde_matrix))
        except Exception as e:
            print(f"HDE computation failed: {e}")
            zero_feature = np.zeros((len(sub_nodes), self.node_types * (self.max_dist + 1) * 2), dtype=np.float32)
            return torch.FloatTensor(zero_feature)


def extract_enclosing_subgraphs(edge_index, edge_label_index, y, num_hops, data, global_max_z=None, use_hde=False):
    data_list = []
    local_max_z = 0
    hde_extractor = None
    if use_hde:
        hde_extractor = HDE_Enhanced_Subgraph_Extractor(node_types=NODE_TYPES, max_dist=MAX_DIST)
        hde_extractor.prepare_node_types(data)

    for src, dst in edge_label_index.t().tolist():
        sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
            [src, dst], num_hops, edge_index, relabel_nodes=True, num_nodes=data.x.size(0))
        src, dst = mapping.tolist()
        mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
        mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
        sub_edge_index = sub_edge_index[:, mask1 & mask2]
        z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))
        local_max_z = max(local_max_z, int(z.max()))
        node_features = data.x[sub_nodes]

        if use_hde and hde_extractor:
            hde_features = hde_extractor.compute_subgraph_hde(sub_nodes, sub_edge_index, src, dst, data)
            if hde_features is not None and hde_features.size(0) == node_features.size(0):
                node_features = torch.cat([node_features, hde_features], dim=1)
            else:
                zero_hde = torch.zeros((node_features.size(0), NODE_TYPES * (MAX_DIST + 1) * 2),
                                       dtype=node_features.dtype)
                node_features = torch.cat([node_features, zero_hde], dim=1)

        data_item = Data(x=node_features, z=z, edge_index=sub_edge_index, y=y)
        data_list.append(data_item)

    max_z = global_max_z if global_max_z is not None else local_max_z
    for data_item in data_list:
        data_item.z = torch.clamp(data_item.z, max=max_z)
        one_hot = F.one_hot(data_item.z, max_z + 1).to(torch.float)
        data_item.x = torch.cat([one_hot, data_item.x], dim=1)

    return data_list


# ==================== 【修改11】强化的损失函数 - 更新为4节点类型 ====================
class EnhancedHeterogeneousLoss(torch.nn.Module):
    """
    专为4节点异构图设计的强化损失函数
    充分利用18种设备类型和4种节点类型（被动、主动、电源、网络）异构特性
    """

    def __init__(self, alpha=0.4, beta=0.35, gamma=0.2, delta=0.05,
                 node_types=4, device_types=34, use_hde=True):  # 【修改】node_types从2改为4
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.node_types = node_types  # 【修改】现在是4个节点类型
        self.device_types = device_types
        self.use_hde = use_hde

        # 基础损失
        self.bce_loss = BCEWithLogitsLoss(reduction='none')

        # 可学习的设备类型重要性权重
        self.device_weights = torch.nn.Parameter(torch.ones(device_types) / device_types)

    def forward(self, predictions, targets, batch_data=None):
        device = predictions.device

        # 1. 自适应BCE损失
        base_loss = self._compute_adaptive_bce_loss(predictions, targets)

        # 2. 4节点设备类型感知损失 - 核心改进
        device_loss = torch.tensor(0.0, device=device)
        if batch_data is not None and self.use_hde:
            try:
                device_loss = self._compute_4node_device_type_loss(predictions, targets, batch_data)
            except Exception as e:
                print(f"4-node device type loss error (skipping): {e}")
                device_loss = torch.tensor(0.0, device=device)

        # 3. 4节点拓扑结构损失
        topology_loss = torch.tensor(0.0, device=device)
        if batch_data is not None and self.use_hde:
            try:
                topology_loss = self._compute_4node_topology_loss(predictions, targets, batch_data)
            except Exception as e:
                print(f"4-node topology loss error (skipping): {e}")
                topology_loss = torch.tensor(0.0, device=device)

        # 4. 简化的对比损失
        contrast_loss = torch.tensor(0.0, device=device)
        try:
            contrast_loss = self._compute_simple_contrast_loss(predictions, targets)
        except Exception as e:
            print(f"Contrast loss error (skipping): {e}")
            contrast_loss = torch.tensor(0.0, device=device)

        # 组合损失
        total_loss = (self.alpha * base_loss +
                      self.beta * device_loss +
                      self.gamma * topology_loss +
                      self.delta * contrast_loss)

        return total_loss

    def _compute_adaptive_bce_loss(self, predictions, targets):
        """自适应BCE损失"""
        bce = self.bce_loss(predictions.view(-1), targets.float())
        probs = torch.sigmoid(predictions.view(-1))
        uncertainty = 1.0 - torch.abs(probs - 0.5) * 2
        adaptive_weights = 1.0 + 2.0 * uncertainty
        return (bce * adaptive_weights).mean()

    # 【修改12】新增4节点设备类型感知损失
    def _compute_4node_device_type_loss(self, predictions, targets, batch_data):
        """4节点设备类型感知损失"""
        if not hasattr(batch_data, 'x') or batch_data.x.size(1) < 34:
            return torch.tensor(0.0, device=predictions.device)

        device_features = batch_data.x[:, :34]
        batch_indices = batch_data.batch
        unique_batches = torch.unique(batch_indices)

        device_loss = 0.0
        valid_count = 0

        for batch_idx in unique_batches:
            try:
                mask = batch_indices == batch_idx
                if mask.sum() == 0:
                    continue

                subgraph_devices = device_features[mask]
                device_dist = subgraph_devices.sum(dim=0) + 1e-8
                device_dist = device_dist / device_dist.sum()

                if batch_idx.item() >= len(predictions):
                    continue

                pred_prob = torch.sigmoid(predictions[batch_idx.item()])
                target_val = targets[batch_idx.item()].float()

                device_importance = torch.sum(self.device_weights * device_dist)
                weighted_error = torch.abs(pred_prob - target_val) * device_importance

                # 【修改13】4节点类型多样性奖励
                # 对于4种节点类型，增加类型多样性奖励
                entropy = -torch.sum(device_dist * torch.log(device_dist + 1e-8))
                # 4节点类型的多样性惩罚更加复杂
                diversity_penalty = entropy * torch.abs(pred_prob - target_val) * 0.15  # 增加权重

                device_loss += weighted_error + diversity_penalty
                valid_count += 1

            except Exception as e:
                continue

        return device_loss / max(valid_count, 1)

    # 【修改14】4节点拓扑一致性损失
    def _compute_4node_topology_loss(self, predictions, targets, batch_data):
        """4节点拓扑一致性损失"""
        if not hasattr(batch_data, 'x'):
            return torch.tensor(0.0, device=predictions.device)

        batch_indices = batch_data.batch
        unique_batches = torch.unique(batch_indices)
        topology_loss = 0.0
        valid_count = 0

        for batch_idx in unique_batches:
            try:
                mask = batch_indices == batch_idx
                if mask.sum() < 2:
                    continue

                subgraph_features = batch_data.x[mask]

                if self.use_hde and subgraph_features.size(1) > 19:
                    feature_dim = subgraph_features.size(1)

                    if feature_dim >= 32:
                        # 【修改15】假设4节点HDE特征在后32维 (4 * (3+1) * 2 = 32)
                        hde_start = feature_dim - 32
                        drnl_part = subgraph_features[:, :hde_start]
                        hde_part = subgraph_features[:, hde_start:]

                        if drnl_part.size(1) > 0 and hde_part.size(1) > 0:
                            drnl_summary = drnl_part.mean(dim=0)
                            hde_summary = hde_part.mean(dim=0)

                            if drnl_summary.size(0) != hde_summary.size(0):
                                min_dim = min(drnl_summary.size(0), hde_summary.size(0))
                                drnl_summary = drnl_summary[:min_dim]
                                hde_summary = hde_summary[:min_dim]

                            if batch_idx.item() >= len(predictions):
                                continue

                            pred_conf = torch.abs(torch.sigmoid(predictions[batch_idx.item()]) - 0.5) * 2

                            if drnl_summary.numel() > 0 and hde_summary.numel() > 0:
                                feature_sim = F.cosine_similarity(
                                    drnl_summary.unsqueeze(0),
                                    hde_summary.unsqueeze(0)
                                ).abs()

                                consistency = torch.abs(feature_sim - pred_conf)
                                # 【修改16】4节点的一致性损失权重调整
                                topology_loss += consistency * 1.2  # 增加4节点一致性重要性
                                valid_count += 1

            except Exception as e:
                continue

        return topology_loss / max(valid_count, 1)

    def _compute_simple_contrast_loss(self, predictions, targets):
        """简化的对比损失"""
        probs = torch.sigmoid(predictions.view(-1))

        pos_mask = targets == 1
        neg_mask = targets == 0

        if pos_mask.sum() == 0 or neg_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)

        pos_probs = probs[pos_mask]
        neg_probs = probs[neg_mask]

        # 简单的边界损失：正样本应该>0.6，负样本应该<0.4
        pos_loss = F.relu(0.6 - pos_probs).mean()
        neg_loss = F.relu(neg_probs - 0.4).mean()

        return pos_loss + neg_loss


# ==================== 强化的模型架构（保持EnhancedHDE_DGCNN不变）====================
class EnhancedHDE_DGCNN(torch.nn.Module):
    """完全修复版本 - 解决所有维度不匹配问题"""

    def __init__(self, hidden_channels, num_layers, num_features=None, k=0.6,
                 node_types=4, max_dist=3, use_hde=True, dropout=0.25):  # 【修改】node_types从2改为4
        super().__init__()
        if num_features is None:
            raise ValueError("num_features must be specified")

        self.use_hde = use_hde
        if k < 1:
            self.k = 15
        else:
            self.k = int(k)

        print(f"防弹模型初始化: k={self.k}, features={num_features}, hidden={hidden_channels}")
        print(f"4节点类型: 被动元件、主动元件、电源元件、网络节点")

        # 图卷积层 - 简单稳定的设计
        self.convs = ModuleList()
        self.convs.append(GCNConv(num_features, hidden_channels))
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, 1))  # 保持原设计

        self.dropout = torch.nn.Dropout(dropout)

        # 关键改进：延迟创建分类器，根据实际输出动态调整
        self.classifier = None
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout_rate = dropout

        print(f"防弹模型: 延迟初始化分类器")

    def _create_classifier(self, input_dim):
        """根据实际输入维度创建分类器"""
        print(f"动态创建分类器: input_dim={input_dim}")

        if input_dim >= 512:
            classifier = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 256),
                torch.nn.BatchNorm1d(256),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(256, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate * 0.5),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )
        elif input_dim >= 128:
            classifier = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 128),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 1)
            )
        elif input_dim >= 64:
            classifier = torch.nn.Sequential(
                torch.nn.Linear(input_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(self.dropout_rate),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
                torch.nn.Linear(32, 1)
            )
        else:
            classifier = torch.nn.Sequential(
                torch.nn.Linear(input_dim, max(16, input_dim // 2)),
                torch.nn.ReLU(),
                torch.nn.Linear(max(16, input_dim // 2), 1)
            )

        return classifier

    def forward(self, x, edge_index, batch):
        # 图卷积部分
        xs = [x]
        for conv in self.convs:
            xs += [conv(xs[-1], edge_index).tanh()]

        # 特征融合
        x = torch.cat(xs[1:], dim=-1)  # 跳过输入特征

        # 全局排序池化
        x = global_sort_pool(x, batch, self.k)

        # 关键：动态创建和使用分类器
        actual_dim = x.size(1)

        if self.classifier is None:
            self.classifier = self._create_classifier(actual_dim).to(x.device)
        elif self.classifier[0].in_features != actual_dim:
            # 如果维度变了，重新创建
            print(f"维度变化: {self.classifier[0].in_features} -> {actual_dim}")
            self.classifier = self._create_classifier(actual_dim).to(x.device)

        return self.classifier(x)


# ==================== 强化的训练函数（更新为4节点）====================
def enhanced_train(model, loader, optimizer, scheduler, criterion):
    model.train()
    total_loss = 0
    y_pred, y_true = [], []

    # 【修改17】使用4节点强化损失函数
    if USE_HDE:
        enhanced_criterion = EnhancedHeterogeneousLoss(
            alpha=0.4,  # 基础BCE
            beta=0.35,  # 4节点设备类型感知（重点）
            gamma=0.2,  # 4节点拓扑一致性
            delta=0.05,  # 难样本挖掘
            node_types=NODE_TYPES,  # 【修改】4节点类型
            device_types=34,
            use_hde=USE_HDE
        ).to(device)
        print("Using 4-node enhanced heterogeneous dual encoding loss")
    else:
        enhanced_criterion = criterion

    with tqdm(loader, desc="Training", unit="batch", mininterval=10) as tepoch:
        for batch_idx, data in enumerate(tepoch):
            data = data.to(device)
            optimizer.zero_grad()

            # 前向传播
            out = model(data.x, data.edge_index, data.batch)

            # 损失计算
            if USE_HDE:
                loss = enhanced_criterion(out.view(-1), data.y, batch_data=data)
            else:
                loss = criterion(out.view(-1), data.y.to(torch.float))

            # 反向传播 + 梯度裁剪
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # 学习率调度
            if scheduler is not None:
                scheduler.step()

            total_loss += float(loss) * data.num_graphs
            y_pred.append(out.view(-1).cpu().detach())
            y_true.append(data.y.view(-1).cpu().to(torch.float))

            # 更新进度条
            tepoch.set_postfix({
                'loss': f'{float(loss):.4f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'gpu': get_gpu_usage().split(':')[1].split('%')[0] + '%' if 'GPU' in get_gpu_usage() else 'CPU'
            })

    train_loss = total_loss / len(loader.dataset)
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    train_auc = roc_auc_score(y_true, y_pred)
    y_pred_binary = (torch.sigmoid(y_pred) >= 0.5).int()
    train_acc = accuracy_score(y_true, y_pred_binary)

    return train_loss, train_auc, train_acc


@torch.no_grad()
def test(model, loader, mode="Validation"):
    model.eval()
    y_pred, y_true = [], []
    with tqdm(loader, desc=mode, unit="batch", mininterval=10) as tepoch:
        for data in tepoch:
            data = data.to(device)
            logits = model(data.x, data.edge_index, data.batch)
            y_pred.append(logits.view(-1).cpu())
            y_true.append(data.y.view(-1).cpu().to(torch.float))
    y_pred = torch.cat(y_pred)
    y_true = torch.cat(y_true)
    roc_auc = roc_auc_score(y_true, y_pred)
    y_pred_binary = (torch.sigmoid(y_pred) >= 0.5).int()
    accuracy = accuracy_score(y_true, y_pred_binary)
    return roc_auc, accuracy


def compute_global_max_z(data, train_edges, test_edges):
    global_max_z = 0
    transform = RandomLinkSplit(num_val=0.1, num_test=0.0, is_undirected=True,
                                split_labels=True, add_negative_train_samples=True)
    train_data, val_data, _ = transform(Data(edge_index=train_edges, x=data.x))
    test_neg_edge_index = negative_sampling(edge_index=train_edges, num_nodes=data.x.size(0),
                                            num_neg_samples=test_edges.size(1), method="sparse")

    all_edge_pairs = [
        (train_data.edge_index, train_data.pos_edge_label_index),
        (train_data.edge_index, train_data.neg_edge_label_index),
        (val_data.edge_index, val_data.pos_edge_label_index),
        (val_data.edge_index, val_data.neg_edge_label_index),
        (train_edges, test_edges),
        (train_edges, test_neg_edge_index)
    ]

    for edge_index, edge_label_index in all_edge_pairs:
        for src, dst in edge_label_index.t().tolist():
            try:
                sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
                    [src, dst], 2, edge_index, relabel_nodes=True, num_nodes=data.x.size(0))
                src, dst = mapping.tolist()
                mask1 = (sub_edge_index[0] != src) | (sub_edge_index[1] != dst)
                mask2 = (sub_edge_index[0] != dst) | (sub_edge_index[1] != src)
                sub_edge_index = sub_edge_index[:, mask1 & mask2]
                z = drnl_node_labeling(sub_edge_index, src, dst, num_nodes=sub_nodes.size(0))
                global_max_z = max(global_max_z, int(z.max()))
            except Exception as e:
                print(f"Warning: Error computing z for edge ({src}, {dst}): {e}")
                continue
    return global_max_z


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """余弦退火学习率调度器"""

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ==================== 【修改18】完整的4节点K折实验函数 ====================
def run_enhanced_kfold_experiment(n_splits=5, num_epochs=60):
    """运行4节点增强版K折交叉验证实验"""
    print("=" * 60)
    print("启动4节点增强版异构图链路预测实验")
    print(f"4节点类型：被动元件、主动元件、电源元件、网络节点")
    print(f"目标：Validation AUC > 0.92, Test AUC > 0.90, Test Acc > 0.85")
    print("=" * 60)

    dataset = MyOwnDataset(DATASET_ROOT_DIRECTORY)
    data = dataset[0]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    edge_indices = data.edge_index.t().cpu().numpy()

    val_auc_scores = []
    test_auc_scores = []
    test_acc_scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(edge_indices)):
        restart_fold = True
        max_restarts = 5
        restart_count = 0

        while restart_fold and restart_count < max_restarts:
            restart_fold = False
            if restart_count > 0:
                print(f"\n重启第 {fold + 1}/{n_splits} 折 (第 {restart_count + 1} 次尝试)")
            else:
                print(f"\n第 {fold + 1}/{n_splits} 折实验开始")

            train_edges = data.edge_index[:, train_idx]
            test_edges = data.edge_index[:, test_idx]

            # 计算global_max_z
            global_max_z = compute_global_max_z(data, train_edges, test_edges)
            print(f"Global max z: {global_max_z}")

            # 数据分割
            transform = RandomLinkSplit(num_val=0.1, num_test=0.0, is_undirected=True,
                                        split_labels=True, add_negative_train_samples=True)
            train_data, val_data, _ = transform(Data(edge_index=train_edges, x=data.x))

            # 子图提取
            print("正在提取封闭子图（4节点HDE增强）...")
            train_pos_data = extract_enclosing_subgraphs(
                train_data.edge_index, train_data.pos_edge_label_index, 1, 2, data, global_max_z, use_hde=USE_HDE)
            train_neg_data = extract_enclosing_subgraphs(
                train_data.edge_index, train_data.neg_edge_label_index, 0, 2, data, global_max_z, use_hde=USE_HDE)
            val_pos_data = extract_enclosing_subgraphs(
                val_data.edge_index, val_data.pos_edge_label_index, 1, 2, data, global_max_z, use_hde=USE_HDE)
            val_neg_data = extract_enclosing_subgraphs(
                val_data.edge_index, val_data.neg_edge_label_index, 0, 2, data, global_max_z, use_hde=USE_HDE)
            test_pos_data = extract_enclosing_subgraphs(train_edges, test_edges, 1, 2, data, global_max_z,
                                                        use_hde=USE_HDE)
            neg_edge_index = negative_sampling(edge_index=train_edges, num_nodes=data.x.size(0),
                                               num_neg_samples=test_edges.size(1), method="sparse")
            test_neg_data = extract_enclosing_subgraphs(train_edges, neg_edge_index, 0, 2, data, global_max_z,
                                                        use_hde=USE_HDE)

            # 数据加载器
            train_dataset = train_pos_data + train_neg_data
            val_dataset = val_pos_data + val_neg_data
            test_dataset = test_pos_data + test_neg_data

            effective_batch_size = min(BATCH_SIZE, max(1, len(train_dataset) // 20))
            train_loader = DataLoader(train_dataset, batch_size=effective_batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=effective_batch_size)
            test_loader = DataLoader(test_dataset, batch_size=effective_batch_size)

            # 模型参数
            num_nodes_list = sorted([d.num_nodes for d in train_dataset])
            k = num_nodes_list[int(math.ceil(0.6 * len(num_nodes_list))) - 1]
            k = max(34, k)  # 增加k值
            num_features = train_dataset[0].x.size(1)

            print(f"4节点模型配置: features={num_features}, k={k}, hidden={HIDDEN_CHANNELS}")

            # 【修改19】创建4节点增强模型
            model = EnhancedHDE_DGCNN(
                hidden_channels=HIDDEN_CHANNELS,
                num_layers=NUM_LAYERS,
                num_features=num_features,
                k=k,
                node_types=NODE_TYPES,  # 【修改】4节点类型
                max_dist=MAX_DIST,
                use_hde=USE_HDE,
                dropout=DROPOUT_RATE
            ).to(device)

            # 模型参数初始化
            for name, param in model.named_parameters():
                if 'weight' in name and param.dim() > 1:
                    if 'conv' in name:
                        torch.nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
                    else:
                        torch.nn.init.xavier_normal_(param)
                elif 'bias' in name:
                    torch.nn.init.constant_(param, 0)

            # 优化器和调度器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                betas=(0.9, 0.999),
                eps=1e-8
            )

            total_steps = len(train_loader) * num_epochs
            warmup_steps = total_steps // 10
            scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

            criterion = BCEWithLogitsLoss()

            # 训练记录
            train_losses, train_aucs, train_accs = [], [], []
            val_aucs, val_accs = [], []
            test_aucs, test_accs = [], []

            best_val_auc = 0
            best_test_auc = 0
            best_test_acc = 0

            early_stop_best_val_acc = -float('inf')
            patience_counter = 0

            print(f"开始4节点训练 (目标: {num_epochs} epochs)")

            # 训练循环
            for epoch in range(1, num_epochs + 1):
                # 训练
                train_loss, train_auc, train_acc = enhanced_train(
                    model, train_loader, optimizer, scheduler, criterion
                )

                # 验证和测试
                val_auc, val_acc = test(model, val_loader, "Validation")
                test_auc, test_acc = test(model, test_loader, "Testing")

                # 早停检查
                if epoch == MAX_EPOCHS_WHERE_TEST_ACC_STUCK and abs(test_acc - 0.5) < 1e-4:
                    print(f"测试准确率停滞在0.5，重启第 {fold + 1} 折...")
                    restart_fold = True
                    restart_count += 1
                    break

                # 记录指标
                train_losses.append(train_loss)
                train_aucs.append(train_auc)
                train_accs.append(train_acc)
                val_aucs.append(val_auc)
                val_accs.append(val_acc)
                test_aucs.append(test_auc)
                test_accs.append(test_acc)

                # 保存最佳模型
                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_test_auc = test_auc
                    best_test_acc = test_acc
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'global_max_z': global_max_z,
                        'k': k,
                        'num_features': num_features,
                        'use_hde': USE_HDE,
                        'node_types': NODE_TYPES,  # 【修改】保存4节点类型信息
                        'best_val_auc': best_val_auc,
                        'best_test_auc': best_test_auc,
                        'best_test_acc': best_test_acc
                    }, f"{MODEL_SAVE_DIRECTORY}/4node_enhanced_model_fold{fold + 1}.pth")

                # 早停机制
                if val_acc > early_stop_best_val_acc + MIN_IMPROVEMENT:
                    early_stop_best_val_acc = val_acc
                    patience_counter = 0
                else:
                    if epoch > MIN_NUM_EPOCHS:
                        patience_counter += 1
                    if patience_counter > PATIENCE:
                        print(f"第 {epoch} 轮触发早停 (第 {fold + 1} 折)")
                        break

                # 【修改20】打印进度 - 4节点状态
                status = "4-Node-Enhanced" if USE_HDE else "Original"
                progress_bar = "█" * int(val_auc * 10) + "░" * (10 - int(val_auc * 10))
                print(f"[{status}] 第{fold + 1}折 Epoch {epoch:02d} | "
                      f"Train: L={train_loss:.4f} AUC={train_auc:.4f} Acc={train_acc:.4f} | "
                      f"Val: AUC={val_auc:.4f} Acc={val_acc:.4f} | "
                      f"Test: AUC={test_auc:.4f} Acc={test_acc:.4f} | "
                      f"{progress_bar}")

                # 达到目标检查
            #   if val_auc >= 0.92 and test_auc >= 0.90 and test_acc >= 0.85:
            #       print(f"达到目标指标！提前完成第 {fold + 1} 折")
            #       break

            if restart_fold:
                continue

            # 可视化结果（与原版相同）
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 3, 1)
            plt.plot(range(1, len(train_losses) + 1), train_losses, 'b-', label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(f'4-Node Training Loss - Fold {fold + 1}')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(range(1, len(train_aucs) + 1), train_aucs, 'g-', label='Train AUC')
            plt.plot(range(1, len(val_aucs) + 1), val_aucs, 'r-', label='Val AUC')
            plt.plot(range(1, len(test_aucs) + 1), test_aucs, 'b-', label='Test AUC')
            plt.axhline(y=0.90, color='orange', linestyle='--', label='Target AUC')
            plt.xlabel('Epoch')
            plt.ylabel('AUC Score')
            plt.title(f'4-Node AUC Scores - Fold {fold + 1}')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.plot(range(1, len(train_accs) + 1), train_accs, 'g-', label='Train Acc')
            plt.plot(range(1, len(val_accs) + 1), val_accs, 'r-', label='Val Acc')
            plt.plot(range(1, len(test_accs) + 1), test_accs, 'b-', label='Test Acc')
            plt.axhline(y=0.85, color='orange', linestyle='--', label='Target Acc')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title(f'4-Node Accuracy - Fold {fold + 1}')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f'{PLOT_SAVE_DIRECTORY}/4node_enhanced_results_fold{fold + 1}.png', dpi=300)
            plt.close()

            # 记录最佳结果
            val_auc_scores.append(best_val_auc)
            test_auc_scores.append(best_test_auc)
            test_acc_scores.append(best_test_acc)

            print(
                f"第 {fold + 1} 折完成 | 最佳4节点结果: Val AUC={best_val_auc:.4f}, Test AUC={best_test_auc:.4f}, Test Acc={best_test_acc:.4f}")

        if restart_count >= max_restarts:
            print(f"第 {fold + 1} 折在 {max_restarts} 次重试后仍未收敛，使用默认值")
            val_auc_scores.append(0.5)
            test_auc_scores.append(0.5)
            test_acc_scores.append(0.5)

    # 最终结果报告
    print("\n" + "=" * 80)
    print("FINAL 4-NODE ENHANCED RESULTS")
    print("=" * 80)

    mean_val_auc = np.mean(val_auc_scores)
    std_val_auc = np.std(val_auc_scores)
    mean_test_auc = np.mean(test_auc_scores)
    std_test_auc = np.std(test_auc_scores)
    mean_test_acc = np.mean(test_acc_scores)
    std_test_acc = np.std(test_acc_scores)

    print(f"Average Validation AUC: {mean_val_auc:.4f} ± {std_val_auc:.4f}")
    print(f"Average Test AUC:       {mean_test_auc:.4f} ± {std_test_auc:.4f}")
    print(f"Average Test Accuracy:  {mean_test_acc:.4f} ± {std_test_acc:.4f}")

    # 目标达成检查
    targets_met = []
    if mean_val_auc >= 0.92:
        targets_met.append("✅ Validation AUC ≥ 0.92")
    else:
        targets_met.append(f"❌ Validation AUC < 0.92 (差距: {0.92 - mean_val_auc:.4f})")

    if mean_test_auc >= 0.90:
        targets_met.append("✅ Test AUC ≥ 0.90")
    else:
        targets_met.append(f"❌ Test AUC < 0.90 (差距: {0.90 - mean_test_auc:.4f})")

    if mean_test_acc >= 0.85:
        targets_met.append("✅ Test Accuracy ≥ 0.85")
    else:
        targets_met.append(f"❌ Test Accuracy < 0.85 (差距: {0.85 - mean_test_acc:.4f})")

    print("\n目标达成情况:")
    for target in targets_met:
        print(f"   {target}")

    print(f"\n各折详细结果:")
    for i, (val_auc, test_auc, test_acc) in enumerate(zip(val_auc_scores, test_auc_scores, test_acc_scores)):
        status = "⭐" if val_auc >= 0.92 and test_auc >= 0.90 and test_acc >= 0.85 else "📊"
        print(f"   第{i + 1}折 {status}: Val AUC={val_auc:.4f}, Test AUC={test_auc:.4f}, Test Acc={test_acc:.4f}")

    print("=" * 80)

    return val_auc_scores, test_auc_scores, test_acc_scores


# ==================== 【修改21】主函数 ====================
if __name__ == "__main__":
    print("启动4节点增强版DRNL-HDE异构图链路预测实验")
    print(f"实验配置:")
    print(f"   - 使用4节点HDE增强: {USE_HDE}")
    print(f"   - 节点类型: {NODE_TYPES} (被动元件 + 主动元件 + 电源元件 + 网络节点)")
    print(f"   - spicenetlist/analoggenie/masalachai设备类型: 18种")
    print(f"   - kicad_github/ltspice_demos设备类型: 34种")
    print(f"   - 学习率: {LEARNING_RATE}")
    print(f"   - 隐藏维度: {HIDDEN_CHANNELS}")
    print(f"   - 网络层数: {NUM_LAYERS}")
    print(f"   - Dropout: {DROPOUT_RATE}")
    print(f"   - 最大轮数: {MAX_NUM_EPOCHS}")
    print(f"目标: Val AUC ≥ 0.92, Test AUC ≥ 0.90, Test Acc ≥ 0.85")
    print("=" * 80)
    print("4节点异构图修改总结:")
    print("1. 更新NODE_TYPES从2改为4")
    print("2. 更新HDE_TYPE_MAPPING包含P, A, S, N节点类型")
    print("3. 修改HDE_Enhanced_Subgraph_Extractor支持4节点类型")
    print("4. 增强EnhancedHeterogeneousLoss的4节点平衡")
    print("5. 更新模型架构处理4节点HDE特征")
    print("6. 修改数据集文件路径使用4node_heterogeneous数据")
    print("7. 添加4节点类型平衡损失计算")
    print("8. 增强4节点双重编码一致性损失")
    print("9. 更新所有日志和状态消息支持4节点")
    print("=" * 80)
    print("-" * 60)

    # 运行实验
    val_scores, test_auc_scores, test_acc_scores = run_enhanced_kfold_experiment(N_SPLITS, MAX_NUM_EPOCHS)

    print("\n实验完成！4节点结果已保存到 ./model-save-4node/ 和 ./plot-4node/ 目录")