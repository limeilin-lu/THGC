# IntRo: Integrated Topology and Relation-Specific Features for Drug-Drug Interaction Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Overview

IntRo is a deep learning framework for predicting drug-drug interactions (DDI) by integrating:
- **Molecular-level pretraining** with self-supervised learning
- **Topology features** from relational graph convolutional networks (RGCN)
- **Relation-specific semantic features** via an interpretable IPE module

The model achieves state-of-the-art performance on DDI prediction tasks through a two-stage pipeline.

### System Requirements
- **Python**: 3.7.10
- **CUDA**: 11.0
- **PyTorch**: 1.7.1+cu110
- **GPU**: NVIDIA GPU with at least 8GB memory (recommended)

---

## 🔬 Two-Stage Pipeline

### **Stage 1: Molecular Pretraining (MRL)**
Generate molecular graph embeddings using Multi-task Representation Learning with three key components:

- **MADR** (Masked Atom-bond Detection and Reconstruction): Self-supervised node/edge masking with copy-or-predict mechanism
- **GEN** (Graph Encoder Network): Multi-layer GNN encoder (GCN/GAT/GraphSAGE) for molecular graph encoding  
- **MSR** (Multi-Scale contrastive Representation learning): Contrastive learning across different graph augmentations

**Input**: Drug SMILES strings  
**Output**: 300-dimensional molecular embeddings

### **Stage 2: DDI Prediction**
Predict interaction types between drug pairs using:

- **RGCN Encoder**: Relational graph convolution to capture drug interaction topology
- **IPE Module** (Relation-Specific Feature Module): Learns interpretable semantic features based on relation component tables
- **MSC** (Multi-view Self-supervised Contrastive Learning): Two contrastive objectives:
  - Node feature corruption (structural view)
  - Edge type corruption (relational view)
- **MLP Classifier**: Concatenates topology features + IPE semantic features + molecular residual features

**Input**: Drug pairs + interaction graph  
**Output**: DDI type predictions (65 classes for Deng dataset, 86 for Ryu dataset)

---

## 🏗️ Model Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Stage 1: MRL Pretraining               │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐          │
│  │   MADR   │───▶│   GEN    │───▶│   MSR    │          │
│  │ (Det+Rec)│    │ (GNN×5)  │    │(Contrast)│          │
│  └──────────┘    └──────────┘    └──────────┘          │
│       ↓                                  ↓              │
│  Atom/Bond Masking            Graph Augmentation        │
└─────────────────────────────────────────────────────────┘
                        ↓
            300-dim Molecular Embeddings
                        ↓
┌─────────────────────────────────────────────────────────┐
│              Stage 2: DDI Prediction (IntRo)            │
│                                                          │
│  ┌──────────────┐      ┌──────────────┐                │
│  │ RGCN Encoder │      │  IPE Module  │                │
│  │  (Topology)  │      │  (Semantic)  │                │
│  └──────┬───────┘      └──────┬───────┘                │
│         │                     │                         │
│         └─────────┬───────────┘                         │
│                   │                                     │
│         ┌─────────▼─────────┐                          │
│         │   Concatenate     │◀── Molecular Residuals   │
│         │  (Topo+Sem+Res)   │                          │
│         └─────────┬─────────┘                          │
│                   │                                     │
│         ┌─────────▼─────────┐                          │
│         │    MLP Decoder    │                          │
│         └─────────┬─────────┘                          │
│                   │                                     │
│            DDI Type Prediction                          │
│                   +                                     │
│              MSC Losses                                 │
│    (Node Corruption + Edge Corruption)                 │
└─────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
IntRo/
├── MRL/                                    # Stage 1: Molecular Pretraining
│   ├── drug_mrl.py                        # Main script: pretraining & embedding extraction
│   ├── model_mrl.py                       # GNN architectures (GCN/GAT/GraphSAGE)
│   ├── checkpoints/                       # Pretrained models & embeddings
│   │   ├── Deng_drug_embeddings.npy      # Final embeddings for Deng dataset
│   │   ├── Ryu_drug_embeddings.npy       # Final embeddings for Ryu dataset
│   │   ├── *_WO_Det.npy                  # Ablation: without MADR
│   │   ├── *_WO_GEN.npy                  # Ablation: without encoder
│   │   └── *_WO_MSR.npy                  # Ablation: without MSR
│   └── data/
│       └── Deng_drugs_processed/          # Processed molecular graphs
│           └── processed/
│               ├── data.pt                # PyTorch Geometric graph data
│               ├── drug_id_map.pt         # Drug ID mappings
│               └── smiles.csv             # SMILES representations
│
├── DDIN/                                   # Stage 2: DDI Prediction
│   ├── codes/                             # Main DDI prediction code
│   │   ├── main.py                        # Entry point: loads data, trains model
│   │   ├── train.py                       # Training loop & evaluation
│   │   ├── layer.py                       # IntRo model (RGCN + IPE + MSC)
│   │   ├── data_preprocess.py             # Data loading & preprocessing
│   │   ├── instantiation.py               # Model initialization
│   │   ├── parms_setting.py               # Hyperparameter configuration
│   │   └── utils.py                       # Utility functions (normalization, etc.)
│   │
│   ├── Dengsdataset/                      # Deng et al. dataset (572 drugs, 65 DDI types)
│   │   ├── drug_list.csv                  # List of drug IDs
│   │   ├── drug_smiles.csv                # Drug SMILES for MRL input
│   │   ├── Deng_drug_embeddings.npy       # Precomputed molecular embeddings
│   │   ├── drug_ids.npy                   # Drug ID array
│   │   ├── newddixiao-1.csv               # Additional DDI data
│   │   └── normal/                        # Train/val/test splits
│   │       ├── ddi_training1.csv          # Training interactions
│   │       ├── ddi_validation1.csv        # Validation interactions
│   │       └── ddi_test1.csv              # Test interactions
│   │
│   └── Ryusdataset/                       # Ryu et al. dataset (548 drugs, 86 DDI types)
│       ├── drug_list.csv
│       ├── Ryu_drug_embeddings.npy
│       ├── drug_ids.npy
│       ├── DDI_event.csv
│       ├── newddixiao-1.csv
│       └── normal/
│           ├── ddi_training1.csv
│           ├── ddi_validation1.csv
│           └── ddi_test1.csv
│
└── checkpoints/                            # Final trained models
    └── IntRo_seed.pth                     # Best model checkpoint
```

### File Descriptions

#### Stage 1 (MRL) Files
- **`drug_mrl.py`**: Implements the full pretraining pipeline with MADR (masking), GEN (encoding), and MSR (contrastive learning). Extracts final molecular embeddings.
- **`model_mrl.py`**: Defines GNN layers (GCNConv, GATConv, GraphSAGEConv) and the graphcl contrastive learning wrapper.

#### Stage 2 (IntRo) Files
- **`layer.py`**: 
  - **IntRo model**: RGCN encoder + IPE module + MSC discriminators
  - **IPE module**: Constructs relation component tables, fuses semantic features, generates contrastive samples
  - **MSC discriminators**: Node/edge corruption detection for contrastive learning
- **`data_preprocess.py`**: Loads DDI data, creates graph structures, prepares DataLoaders
- **`train.py`**: Training/testing loops with mixed-precision training, early stopping, metric computation
- **`main.py`**: Orchestrates the full pipeline (data loading → model creation → training → checkpointing)
- **`parms_setting.py`**: Centralized hyperparameter management (learning rate, dropout, loss weights, etc.)
- **`utils.py`**: Matrix normalization and graph utilities

---

## 🔧 Environment Setup

### Requirements
```bash
Python == 3.7.10
PyTorch == 1.7.1+cu110
PyTorch Geometric == 2.0.0
CUDA == 11.0
```

### Installation

#### Option 1: Using requirements.txt (Recommended)
```bash
# Clone repository
git clone https://github.com/yourusername/IntRo.git
cd IntRo

# Create conda environment
conda create -n intro python=3.7.10
conda activate intro

# Install PyTorch with CUDA 11.0 first (required before other packages)
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Geometric and extensions
pip install torch-geometric==2.0.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.7.1+cu110.html

# Install remaining dependencies
pip install networkx==2.6.3 numpy==1.18.5 scikit-learn==0.24.2
pip install rdkit-pypi pandas tqdm matplotlib
```

#### Option 2: Manual Installation
```bash
# Clone repository
git clone https://github.com/yourusername/IntRo.git
cd IntRo

# Create conda environment
conda create -n intro python=3.7.10
conda activate intro

# Install all dependencies (except PyTorch Geometric extensions)
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install networkx==2.6.3 numpy==1.18.5 scikit-learn==0.24.2
pip install torch-geometric==2.0.0
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.7.1+cu110.html
pip install rdkit-pypi pandas tqdm matplotlib
```

**Note**: A `requirements.txt` file is provided, but PyTorch and PyTorch Geometric packages should be installed manually following the commands above due to CUDA-specific builds.

---

## 🚀 Quick Start

### Option 1: Use Precomputed Embeddings (Recommended)

```bash
# Directly train IntRo with provided molecular embeddings
cd DDIN/codes
python main.py
```

### Option 2: Full Pipeline from Scratch

#### Step 1: Generate Molecular Embeddings

```bash
cd MRL
python drug_mrl.py \
    --drug_file ../DDIN/Dengsdataset/drug_list.csv \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --output_model_file ./checkpoints/
```

**Outputs:**
- `checkpoints/Deng_drug_embeddings.npy`: 300-dim embeddings
- `checkpoints/TS_epoch100.pth`: Pretrained GNN model

#### Step 2: Train DDI Prediction Model

```bash
cd ../DDIN/codes
python main.py \
    --datasets Dengsdataset \
    --epochs 200 \
    --lr 5e-4 \
    --batch 256
```

**Outputs:**
- `checkpoints/IntRo_seed.pth`: Best model checkpoint
- `IntRo_results.txt`: Detailed metrics

---

## ⚙️ Key Hyperparameters

### MRL Pretraining (`drug_mrl.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 100 | Pretraining epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.001 | Learning rate |
| `--num_layer` | 5 | GNN layers |
| `--emb_dim` | 300 | Embedding dimension |
| `--weight_dis` | 0.30 | Weight for MADR detection loss |
| `--weight_ent` | 0.10 | Weight for MSR contrastive loss |
| `--aug_ratio1` | 0.15 | Augmentation ratio (view 1) |
| `--aug_ratio2` | 0.20 | Augmentation ratio (view 2) |

### IntRo Training (`parms_setting.py`)
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 200 | Training epochs |
| `--lr` | 5e-4 | Learning rate |
| `--batch` | 256 | Batch size |
| `--hidden1` | 150 | RGCN layer 1 hidden units |
| `--hidden2` | 75 | RGCN layer 2 hidden units |
| `--dropout` | 0.55 | Dropout rate |
| `--loss_ratio1` | 1.0 | Weight for DDI prediction loss |
| `--loss_ratio2` | 0.05 | Weight for MSC node corruption |
| `--loss_ratio3` | 0.1 | Weight for MSC edge corruption |
| `--loss_ratio_ipe` | 0.1 | Weight for IPE contrastive loss |
| `--ipe_margin` | 0.3 | Margin for IPE triplet loss |
| `--early_stop` | 50 | Early stopping patience |

---

## 📊 Datasets

### Deng Dataset
- **Drugs**: 572
- **DDI Types**: 65
- **Interactions**: ~37,000 training + validation + test

### Ryu Dataset
- **Drugs**: 548  
- **DDI Types**: 86
- **Interactions**: ~48,000 training + validation + test

**Data Format (CSV):**
```
d1,d2,type
DB00001,DB00002,45
DB00003,DB00004,12
...
```

---

## 🧪 Ablation Studies

To evaluate individual components, use precomputed embeddings without specific modules:

```bash
# Without MADR (Masked Atom-bond Detection)
cp MRL/checkpoints/Deng_drug_embeddings_WO_Det.npy DDIN/Dengsdataset/Deng_drug_embeddings.npy
python main.py

# Without GEN (Graph Encoder)
cp MRL/checkpoints/Deng_drug_embeddings_WO_GEN.npy DDIN/Dengsdataset/Deng_drug_embeddings.npy
python main.py

# Without MSR (Multi-scale Contrastive)
cp MRL/checkpoints/Deng_drug_embeddings_WO_MSR.npy DDIN/Dengsdataset/Deng_drug_embeddings.npy
python main.py
```