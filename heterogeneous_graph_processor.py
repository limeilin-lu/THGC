# heterogeneous_4node_processor.py
# 创建包含4种节点类型的异构图：被动元件、主动元件、电源元件、网络节点
# 基于17种元器件类型

import json
import networkx as nx
import numpy as np
import pickle
import os
import torch
from collections import defaultdict
from torch_geometric.utils import from_networkx

# 配置
#json_folder = "./SpiceNetlist/JSON"
json_folder = "./Masala-CHAI/JSON"
#json_folder = "./AnalogGenie/JSON_unique_ID"

# 17种元器件分类定义
COMPONENT_CATEGORIES = {
    # 被动元件 (Passive Components)
    'passive': [
        'Cap', 'Capacitor',  # 电容
        'Ind', 'Inductor',  # 电感
        'Res', 'Resistor',  # 电阻
        'Diode', 'diode'  # 二极管 (大小写都支持)
    ],

    # 主动元件 (Active Components)
    'active': [
        'NMOS', 'PMOS', 'MOSFET',  # MOS晶体管
        'NPN', 'PNP',  # 双极晶体管
        'Op_amp',  # 运放
        'IC'  # 集成电路
    ],

    # 电源元件 (Source Components)
    'source': [
        'Voltage',  # 电压源
        'Current'  # 电流源
    ]
    # 网络节点 'NET' 单独处理
}

# 扁平化所有组件类型
ALL_COMPONENT_TYPES = set()
for category_components in COMPONENT_CATEGORIES.values():
    ALL_COMPONENT_TYPES.update(category_components)


component_port_mapping = {
    "PMOS": ["Drain", "Source", "Gate"],
    #"PMOS_normal": ["Drain", "Source", "Gate"],
    #"PMOS_cross": ["Drain", "Source", "Gate"],
    #"PMOS_bulk": ["Drain", "Source", "Gate", "Body"],
    "NMOS": ["Drain", "Source", "Gate"],
    #"NMOS_normal": ["Drain", "Source", "Gate"],
    #"NMOS_cross": ["Drain", "Source", "Gate"],
    #"NMOS_bulk": ["Drain", "Source", "Gate", "Body"],
    "Voltage": ["Pos", "Neg"],
    #"Voltage_1": ["Pos", "Neg"],
    #"Voltage_2": ["Pos", "Neg"],
    "Current": ["Pos", "Neg"],
    "NPN": ["Base", "Emitter", "Collect"],
    #"BJT_NPN": ["Base", "Emitter", "Collect"],
    #"BJT_NPN_cross": ["Base", "Emitter", "Collect"],
    "PNP": ["Base", "Emitter", "Collect"],
    #"BJT_PNP": ["Base", "Emitter", "Collect"],
    #"BJT_PNP_cross": ["Base", "Emitter", "Collect"],
    "Diode": ["In", "Out"],
    #"diode": ["In", "Out"],
    "Diso_amp": ["InN", "InP", "Out"],
    "Siso_amp": ["In", "Out"],
    #"Sisp_amp": ["In", "Out"],
    "Dido_amp": ["InN", "InP", "OutN", "OutP"],
    "Cap": ["Pos", "Neg"],
    #"Capacitor": ["Pos", "Neg"],
    #"Gnd": ["port"],
    #"gnd": ["port"],
    "Ind": ["Pos", "Neg"],
    #"inductor": ["Pos", "Neg"],
    "Res": ["Pos", "Neg"],
    #"Resistor_1": ["Pos", "Neg"],
    #"Resistor_2": ["Pos", "Neg"]
}


def get_component_category(component_type):
    """根据元件类型返回分类"""
    for category, components in COMPONENT_CATEGORIES.items():
        if component_type in components:
            return category
    # 改进的未知类型处理
    print(f"Warning: Unknown component type '{component_type}', defaulting to 'passive'")
    return 'passive'


def get_node_id(instance_id, category):
    """统一的节点ID生成函数"""
    prefix_map = {
        'passive': 'PASS',
        'active': 'ACT',
        'source': 'SRC',
        'network': 'NET'
    }
    prefix = prefix_map.get(category, 'UNK')
    return f"{prefix}_{instance_id}"


def create_4node_heterogeneous_graph_from_circuit(circuit):
    """
    创建改进的4种节点异构图
    """
    G = nx.Graph()

    # 收集信息
    all_nets = set()
    component_to_nets = defaultdict(dict)
    component_stats = defaultdict(int)

    # 第一遍：收集和验证信息
    for component in circuit:
        instance_id = component["instance_id"]
        component_type = component["component_type"]
        port_connections = component["port_connection"]

        component_stats[component_type] += 1

        for port_type, net_name in port_connections.items():
            if isinstance(net_name, dict):
                continue
            if net_name and str(net_name).strip():  # 更好的空值检查
                net_name_str = str(net_name).strip()
                all_nets.add(net_name_str)
                component_to_nets[instance_id][port_type] = net_name_str

    # 添加元件节点
    node_type_counts = {'P': 0, 'A': 0, 'S': 0}

    for component in circuit:
        instance_id = component["instance_id"]
        component_type = component["component_type"]
        category = get_component_category(component_type)

        # 生成节点ID和类型
        if category == 'passive':
            comp_id = get_node_id(instance_id, 'passive')
            node_type = 'P'  # Passive
            node_type_counts['P'] += 1
        elif category == 'active':
            comp_id = get_node_id(instance_id, 'active')
            node_type = 'A'  # Active
            node_type_counts['A'] += 1
        elif category == 'source':
            comp_id = get_node_id(instance_id, 'source')
            node_type = 'S'  # Source
            node_type_counts['S'] += 1
        else:
            # 默认处理
            comp_id = get_node_id(instance_id, 'passive')
            node_type = 'P'
            category = 'passive'
            node_type_counts['P'] += 1

        # 添加更丰富的节点属性
        G.add_node(comp_id,
                   node_type=node_type,
                   device_type=component_type,
                   category=category,
                   instance_id=instance_id,
                   # 可以添加更多属性，如参数值等
                   **{k: v for k, v in component.items()
                      if k not in ['instance_id', 'component_type', 'port_connection']})

    # 添加网络节点
    for net_name in all_nets:
        net_id = get_node_id(net_name, 'network')
        G.add_node(net_id,
                   node_type='N',  # Network
                   device_type="NET",
                   category="network",
                   net_name=net_name)

    # 添加连接关系
    edge_count = 0
    for component in circuit:
        instance_id = component["instance_id"]
        component_type = component["component_type"]
        category = get_component_category(component_type)

        # 获取对应的元件节点ID
        comp_id = get_node_id(instance_id, category)

        for port_type, net_name in component_to_nets[instance_id].items():
            net_id = get_node_id(net_name, 'network')

            # 添加边，包含更多信息
            G.add_edge(comp_id, net_id,
                       connection_type=port_type,
                       net_name=net_name,
                       component_category=category,
                       component_type=component_type,
                       component_instance=instance_id)
            edge_count += 1

    return G, component_stats, node_type_counts


def process_all_circuits():
    """
    处理所有电路的改进版本
    """
    circuits = []
    global_component_stats = defaultdict(int)
    global_node_stats = defaultdict(int)

    # 读取所有电路文件
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"Found {len(json_files)} circuit files")

    for filename in json_files:
        filepath = os.path.join(json_folder, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                circuit = json.load(f)

            # 添加instance_id
            type_counter = defaultdict(int)
            for component in circuit:
                ctype = component["component_type"]
                global_component_stats[ctype] += 1

                instance_id = f"{ctype}_{type_counter[ctype]}"
                component["instance_id"] = instance_id
                type_counter[ctype] += 1

            circuits.append((filename, circuit))

        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    print(f"Successfully loaded {len(circuits)} circuits")

    # 显示组件统计
    print("\n=== Component Type Statistics ===")
    for ctype in sorted(global_component_stats.keys()):
        category = get_component_category(ctype)
        print(f"  {ctype:15} ({category:7}): {global_component_stats[ctype]:4} instances")

    # 创建标签映射
    unique_device_types = set(global_component_stats.keys())
    unique_device_types.add("NET")
    unique_device_types.update(ALL_COMPONENT_TYPES)  # 确保包含所有定义的类型

    label_mapping = {device_type: idx for idx, device_type in enumerate(sorted(unique_device_types))}

    print(f"\n=== Device Type Label Mapping ({len(label_mapping)} types) ===")
    for k, v in label_mapping.items():
        category = get_component_category(k) if k != "NET" else "network"
        print(f"  {k:15} ({category:7}): {v:2}")

    # 创建图数据集
    graph_list = {'train_x': [], 'test_x': []}
    train_ratio = 0.8
    num_train = int(train_ratio * len(circuits))

    total_node_stats = {'P': 0, 'A': 0, 'S': 0, 'N': 0}
    failed_circuits = []

    for circuit_idx, (filename, circuit) in enumerate(circuits):
        try:
            G, comp_stats, node_counts = create_4node_heterogeneous_graph_from_circuit(circuit)

            # 统计节点数量
            network_count = len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'N'])
            node_counts['N'] = network_count

            # 累计统计
            for node_type, count in node_counts.items():
                total_node_stats[node_type] += count

            print(f"Circuit {circuit_idx + 1:3} ({filename:20}): "
                  f"{G.number_of_nodes():3} nodes "
                  f"(P:{node_counts['P']:2}, A:{node_counts['A']:2}, "
                  f"S:{node_counts['S']:2}, N:{node_counts['N']:2}), "
                  f"{G.number_of_edges():3} edges")

            # 分配到训练集或测试集
            if circuit_idx < num_train:
                graph_list['train_x'].append(G)
            else:
                graph_list['test_x'].append(G)

        except Exception as e:
            print(f"Error processing circuit {filename}: {e}")
            failed_circuits.append(filename)
            continue

    if failed_circuits:
        print(f"\nFailed to process {len(failed_circuits)} circuits: {failed_circuits}")

    print(f"\n=== Total Node Statistics ===")
    print(f"  Passive components: {total_node_stats['P']}")
    print(f"  Active components:  {total_node_stats['A']}")
    print(f"  Source components:  {total_node_stats['S']}")
    print(f"  Network nodes:      {total_node_stats['N']}")
    print(f"  Total nodes:        {sum(total_node_stats.values())}")


    """
    # spicenetlist保存数据
    os.makedirs("./data", exist_ok=True)
    with open("./data/SpiceNetlist_4node_heterogeneous.pkl", 'wb') as f:
        pickle.dump(graph_list, f)
    with open("./data/SpiceNetlist_4node_heterogeneous_label_mapping.pkl", 'wb') as f:
        pickle.dump(label_mapping, f)
    """
    # masalachai保存数据
    os.makedirs("./data", exist_ok=True)
    with open("./data/Masala-CHAI_4node_heterogeneous.pkl", 'wb') as f:
        pickle.dump(graph_list, f)
    with open("./data/Masala-CHAI_4node_heterogeneous_label_mapping.pkl", 'wb') as f:
        pickle.dump(label_mapping, f)
    print(f"\nSaved 4-node heterogeneous graph data:")
    print(f"  - Training graphs: {len(graph_list['train_x'])}")
    print(f"  - Test graphs: {len(graph_list['test_x'])}")
    print(f"  - Device types: {len(label_mapping)}")

    return graph_list, label_mapping


def convert_to_pyg_format():
    """
    改进的PyG格式转换
    """
    """
    # SPICENETLIST加载数据
    with open("./data/SpiceNetlist_4node_heterogeneous.pkl", 'rb') as f:
        dataset = pickle.load(f)
    with open("./data/SpiceNetlist_4node_heterogeneous_label_mapping.pkl", 'rb') as f:
        mapping = pickle.load(f)
    """
    # MASALACHAI加载数据
    with open("./data/Masala-CHAI_4node_heterogeneous.pkl", 'rb') as f:
        dataset = pickle.load(f)
    with open("./data/Masala-CHAI_4node_heterogeneous_label_mapping.pkl", 'rb') as f:
        mapping = pickle.load(f)
    train_x = dataset['train_x']

    X = []
    node_types = []
    all_graphs = []

    # 节点类型映射
    node_type_mapping = {'P': 0, 'A': 1, 'S': 2, 'N': 3}

    print(f"Converting {len(train_x)} training graphs to PyG format...")

    for graph_idx, g in enumerate(train_x):
        if graph_idx % 10 == 0:
            print(f"  Processing graph {graph_idx + 1}/{len(train_x)}")

        for node in g.nodes():
            node_data = g.nodes[node]
            device_type = node_data['device_type']
            node_type = node_data['node_type']

            # 创建one-hot特征
            device_index = mapping.get(device_type, -1)
            if device_index == -1:
                print(f"Warning: Unknown device type '{device_type}' in node {node}")
                device_index = len(mapping)  # 使用额外的索引

            feature_size = len(mapping)
            feat = np.zeros(feature_size, dtype=np.float32)
            if device_index < feature_size:
                feat[device_index] = 1.0
            X.append(feat)

            # 节点类型编码
            node_type_idx = node_type_mapping.get(node_type, 0)
            node_types.append(node_type_idx)

        # 创建无属性图用于PyG
        g_ = nx.Graph()
        g_.add_nodes_from(g.nodes())
        g_.add_edges_from(g.edges())
        all_graphs.append(g_)

    # 合并所有图
    print("Combining all graphs...")
    combined_graph = nx.disjoint_union_all(all_graphs)
    data = from_networkx(combined_graph, group_node_attrs=None, group_edge_attrs=None)

    data.x = torch.from_numpy(np.array(X)).float()
    data.node_types = torch.tensor(node_types, dtype=torch.long)

    # MASALACHAI保存
    torch.save(data, './data/Masala-CHAI_4node_heterogeneous.pt')

    # SPICENETLIST保存
    #torch.save(data, './data/SpiceNetlist_4node_heterogeneous.pt')
    print("Saved 4-node heterogeneous PyG dataset!")

    # 统计信息
    node_type_counts = torch.bincount(data.node_types)
    print(f"\n=== PyG Dataset Statistics ===")
    print(f"Node types distribution:")
    print(f"  - Passive components (0): {node_type_counts[0].item()}")
    if len(node_type_counts) > 1:
        print(f"  - Active components  (1): {node_type_counts[1].item()}")
    if len(node_type_counts) > 2:
        print(f"  - Source components  (2): {node_type_counts[2].item()}")
    if len(node_type_counts) > 3:
        print(f"  - Network nodes      (3): {node_type_counts[3].item()}")

    print(f"\nDataset properties:")
    print(f"  - Total nodes: {data.num_nodes}")
    print(f"  - Total edges: {data.num_edges}")
    print(f"  - Feature dimension: {data.x.size(1)}")
    print(f"  - Device types: {len(mapping)}")

    return data, mapping


def validate_dataset():
    """
    验证生成的数据集（兼容 PyTorch 2.6+ 安全机制）
    """
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

        # 添加所有必要的 PyG 类到安全列表
        torch.serialization.add_safe_globals([
            GlobalStorage,
            NodeStorage,
            EdgeStorage,
            DataEdgeAttr,
            DataTensorAttr,
            Data
        ])

        print("\n=== Dataset Validation ===")

        # 尝试使用 weights_only=True 加载
        try:
            """
            data = torch.load(
                './data/SpiceNetlist_4node_heterogeneous.pt',
                weights_only=True
            )
            """
            data = torch.load(
                './data/Masala-CHAI_4node_heterogeneous.pt',
                weights_only=True
            )
            print("✓ Loaded with weights_only=True (secure mode)")
        except Exception as e:
            print(f"Warning: weights_only=True failed: {e}")
            print("Falling back to weights_only=False...")
            # 如果安全模式失败，回退到传统模式
            """
            data = torch.load(
                './data/SpiceNetlist_4node_heterogeneous.pt',
                weights_only=False
            )
            """
            data = torch.load(
                './data/Masala-CHAI_4node_heterogeneous.pt',
                weights_only=False
            )
            print("✓ Loaded with weights_only=False")

        print(f"✓ Dataset loaded successfully")
        print(f"✓ Node features shape: {data.x.shape}")
        print(f"✓ Edge index shape: {data.edge_index.shape}")
        print(f"✓ Node types shape: {data.node_types.shape}")

        # 检查数据完整性
        assert data.x.shape[0] == data.node_types.shape[0], "Node count mismatch"
        assert torch.max(data.node_types) <= 3, "Invalid node type"
        assert torch.min(data.node_types) >= 0, "Invalid node type"

        # 检查边的有效性
        max_node_idx = data.x.shape[0] - 1
        assert torch.max(data.edge_index) <= max_node_idx, "Edge index out of range"
        assert torch.min(data.edge_index) >= 0, "Negative edge index"

        # 统计节点类型分布
        node_type_counts = torch.bincount(data.node_types)
        print(f"\n=== Node Type Distribution ===")
        type_names = ['Passive', 'Active', 'Source', 'Network']
        for i, count in enumerate(node_type_counts):
            if i < len(type_names):
                print(f"  {type_names[i]:8} (type {i}): {count.item():5} nodes")

        # 检查特征的有效性
        print(f"\n=== Feature Statistics ===")
        print(f"  Feature dimension: {data.x.shape[1]}")
        print(f"  Non-zero features per node: {(data.x > 0).sum(dim=1).float().mean():.2f}")
        print(f"  Feature sparsity: {(data.x == 0).float().mean():.3f}")

        print(f"\n✓ All validation checks passed")
        return True

    except Exception as e:
        print(f"✗ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def safe_load_dataset():
    """
    安全加载数据集的辅助函数
    """
    try:
        import torch
        from torch_geometric.data import Data
        from torch_geometric.data.storage import GlobalStorage, NodeStorage, EdgeStorage
        from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

        # 设置安全全局变量
        safe_globals = [
            GlobalStorage,
            NodeStorage,
            EdgeStorage,
            DataEdgeAttr,
            DataTensorAttr,
            Data
        ]

        # 使用上下文管理器确保安全加载
        with torch.serialization.safe_globals(safe_globals):
            """
            data = torch.load(
                './data/SpiceNetlist_4node_heterogeneous.pt',
                weights_only=True
            )
            """
            data = torch.load(
                './data/Masala-CHAI_4node_heterogeneous.pt',
                weights_only=True
            )

        return data

    except Exception as e:
        print(f"Safe loading failed, trying fallback: {e}")
        # 回退方案
        """
        return torch.load(
            './data/SpiceNetlist_4node_heterogeneous.pt',
            weights_only=False
        )
        """
        return torch.load(
            './data/Masala-CHAI_4node_heterogeneous.pt',
            weights_only=False
        )



if __name__ == "__main__":
    print("=== Improved 4-Node Heterogeneous Circuit Graph Processor ===")

    # 显示组件分类
    print("\n=== Component Categories ===")
    for category, components in COMPONENT_CATEGORIES.items():
        print(f"{category.upper()}: {', '.join(sorted(components))}")

    try:
        # 处理电路
        graph_list, label_mapping = process_all_circuits()

        # 转换为PyG格式
        print("\n=== Converting to PyG Format ===")
        data, mapping = convert_to_pyg_format()

        # 验证数据集
        validate_dataset()

        print(f"\n=== Process Completed Successfully ===")

    except Exception as e:
        print(f"Error in main process: {e}")
        import traceback

        traceback.print_exc()