# THGC - Topology-aware Heterogeneous Graph for Circuits
### 4-Node Heterogeneous Graph Neural Network for Circuit Link Prediction

A Graph Neural Network (GNN) framework for electronic circuit analysis using 4-node heterogeneous graph representations with DRNL-HDE (Distance-based Ranking and Node Labeling with Heterogeneous Distance Encoding).

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Configuration](#configuration)
- [Output](#output)
- [Citation](#citation)

---

## 🔍 Overview

This project implements a heterogeneous graph neural network for circuit topology analysis and link prediction. The system represents electronic circuits as 4-node type heterogeneous graphs and uses advanced graph learning techniques to predict missing connections.

### Key Innovations

- **4-Node Heterogeneous Representation**: Categorizes circuit components into 4 types:
  - **Passive Components (P)**: Resistors, Capacitors, Inductors, etc.
  - **Active Components (A)**: Transistors (MOSFET, BJT), Diodes, Op-amps, ICs
  - **Source Components (S)**: Voltage sources, Current sources, Controlled sources
  - **Network Nodes (N)**: Circuit connection points (nets)

- **DRNL-HDE Encoding**: Combines Distance-based Ranking Node Labeling with Heterogeneous Distance Encoding for capturing both structural and semantic information

- **Multi-Loss Training**: Employs multiple loss functions for balanced learning across node types

---

## 📁 Project Structure

```
THGC/
├── heterogeneous_graph_processor_4_CCP.py   # Graph generation from circuit netlists
├── DRNL_HDE_4Loss_update.py                 # Link prediction model training
├── requirements.txt                          # Python dependencies
├── data/                                     # Dataset directory
│   └── [dataset]_4node_heterogeneous.pt     # Processed graph data
├── model-save-4node/                         # Trained model checkpoints
├── plot-4node/                               # Training curves and visualizations
└── README.md                                 # This file
```

### Main Components

1. **heterogeneous_graph_processor_4_CCP.py**: 
   - Parses JSON circuit netlists
   - Creates 4-node heterogeneous graphs
   - Generates PyTorch Geometric datasets
   - Supports 34 component types

2. **DRNL_HDE_4Loss_update.py**: 
   - Implements DRNL-HDE link prediction model
   - K-fold cross-validation training
   - Multi-loss optimization
   - Performance evaluation and visualization

---

## ✨ Features

- **Heterogeneous Graph Modeling**: Explicit modeling of different component types
- **DRNL Node Labeling**: Encodes structural distance information
- **HDE Feature Engineering**: Type-aware distance encoding with 4 node categories
- **Advanced GNN Architecture**: Multi-layer GCN with dropout and batch normalization
- **Comprehensive Training**:
  - 5-fold cross-validation
  - Early stopping with patience
  - Learning rate scheduling
  - Multiple loss functions (BCE + Type Balance + Encoding Consistency)
- **Rich Metrics**: AUC-ROC, Accuracy tracking for train/val/test sets
- **Visualization**: Automatic generation of training curves

---

## 📦 Requirements

### Core Dependencies

```
Python 3.12+
torch 2.8.0
torch-cluster 1.6.3+pt28
torch-geometric 2.6.1
torch-scatter 2.1.2+pt28cpu
torch-sparse 0.6.18+pt28cp
torch-spline-conv 1.2.2+pt28cpu
numpy 2.0.2
PyYAML 6.0.2
```

### Additional Dependencies

```
networkx
scipy
scikit-learn
matplotlib
GPUtil
tqdm
```

---

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/THGC.git
cd THGC
```

2. **Create virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify PyTorch CUDA** (for GPU acceleration)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 📊 Datasets

### Supported Datasets

The project supports multiple circuit netlist datasets from:

**Repository**: https://github.com/symbench/spice-datasets

Available datasets:
- **SpiceNetlist** (18 device types)
- **AnalogGenie** (18 device types)
- **Masala-CHAI** (18 device types)
- **KiCad_github** (34 device types)
- **LTspice_demos** (34 device types)
- **LTspice_examples** (34 device types)

### Data Format

Raw data should be in JSON format representing circuit netlists. Each component includes:
- `component_type`: Type of electronic component
- `port_connection`: Dictionary of port-to-net connections
- `instance_id`: Unique identifier (auto-generated)

### Data Preparation

1. **Download dataset** from the spice-datasets repository
2. **Place JSON files** in appropriate folder structure:
   ```
   ./[DatasetName]/JSON/  # For SpiceNetlist, Masala-CHAI, AnalogGenie
   ./[DatasetName]/json_files/  # For KiCad, LTspice
   ```

3. **Run graph processor**:
   ```bash
   python heterogeneous_graph_processor_4_CCP.py
   ```
   
   **Configuration**: Edit the script to select your dataset:
   ```python
   # In heterogeneous_graph_processor_4_CCP.py, line 15-21
   json_folder = "./KiCad_github/json_files"  # Change to your dataset
   ```

4. **Output**: Creates `./data/[Dataset]_4node_heterogeneous.pt`

---

## 🎯 Usage

### Step 1: Generate Heterogeneous Graph Dataset

```bash
python heterogeneous_graph_processor_4_CCP.py
```

This will:
- Read circuit JSON files
- Classify components into 4 node types
- Create heterogeneous graph structure
- Save as PyTorch Geometric dataset

### Step 2: Train Link Prediction Model

```bash
python DRNL_HDE_4Loss_update.py
```

**Important**: Before running, configure the dataset in the script:

```python
# In DRNL_HDE_4Loss_update.py, lines 35-42
DATASET = "KiCad_github"  # Choose your dataset
# Options: "SpiceNetlist", "Masala-CHAI", "AnalogGenie", 
#          "KiCad_github", "LTspice_demos", "LTspice_examples"

# Also set the device type count based on dataset:
# For SpiceNetlist/AnalogGenie/Masala-CHAI: 18 device types
# For KiCad/LTspice: 34 device types
```

### Step 3: Monitor Training

The training process will:
- Display real-time progress bars
- Print metrics for each epoch
- Save best models to `./model-save-4node/`
- Generate plots in `./plot-4node/`

### Example Output

```
[4-Node-Enhanced] Fold 1 Epoch 25 | Train: L=0.3245 AUC=0.9234 Acc=0.8756 | 
Val: AUC=0.9156 Acc=0.8623 | Test: AUC=0.9087 Acc=0.8534 | ██████████
```

---

## 🏗️ Model Architecture

### DRNL_HDE Model Components

1. **Subgraph Extraction**
   - K-hop neighborhood sampling around target edges
   - DRNL node labeling for structural encoding

2. **HDE Feature Extraction**
   - Computes type-aware distance distributions
   - Creates 4-type × (max_dist + 1) dimensional features
   - Captures heterogeneous graph semantics

3. **GNN Encoder**
   - Multi-layer GCN (Graph Convolutional Network)
   - Layer normalization and dropout for regularization
   - Hidden dimensions: 80 (configurable)
   - Number of layers: 4 (configurable)

4. **Global Pooling**
   - SortPooling for fixed-size graph representations
   - Maintains most informative nodes

5. **1D-CNN Classifier**
   - Conv1D layers for pattern recognition
   - MaxPooling for dimensionality reduction
   - MLP for final binary prediction

### Loss Functions

The model uses a composite loss function:

```python
Total_Loss = BCE_Loss + α × TypeBalance_Loss + β × EncodingConsistency_Loss
```

- **BCE Loss**: Binary cross-entropy for link prediction
- **Type Balance Loss**: Ensures balanced learning across node types
- **Encoding Consistency Loss**: Aligns DRNL and HDE representations

---

## ⚙️ Configuration

### Key Hyperparameters

Edit `DRNL_HDE_4Loss_update.py` to adjust:

```python
# Dataset Configuration
DATASET = "KiCad_github"           # Dataset name
NODE_TYPES = 4                     # Number of node types (fixed)
MAX_DIST = 3                       # Maximum distance for HDE

# Training Configuration
N_SPLITS = 5                       # K-fold cross-validation
MAX_NUM_EPOCHS = 60                # Maximum training epochs
MIN_NUM_EPOCHS = 8                 # Minimum epochs before early stopping
PATIENCE = 6                       # Early stopping patience
MIN_IMPROVEMENT = 0.001            # Minimum improvement threshold

# Model Configuration
HIDDEN_CHANNELS = 80               # Hidden layer dimensions
NUM_LAYERS = 4                     # Number of GCN layers
DROPOUT_RATE = 0.5                 # Dropout probability

# Optimization Configuration
LEARNING_RATE = 1e-4               # Adam learning rate
WEIGHT_DECAY = 1e-6                # L2 regularization
BATCH_SIZE = 6                     # Training batch size

# GPU Configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU device ID
```

### Node Type Mapping

```python
HDE_TYPE_MAPPING = {
    'P': 0,  # Passive Components
    'A': 1,  # Active Components
    'S': 2,  # Source Components
    'N': 3   # Network Nodes
}
```

### Component Categories (34 types)

Defined in `heterogeneous_graph_processor_4_CCP.py`:

- **Passive (11)**: Cap, Capacitor, Ind, Inductor, Res, Resistor, CoupledInd, TransLine, UniformRC, LossyTransLine, CoupledMultiLine
- **Active (14)**: NMOS, PMOS, MOSFET, NPN, PNP, BJT, Diode, Op_amp, IC, SubCircuit, JFET, MESFET, Behavioral, XSpice
- **Source (6)**: Voltage, Current, VCVS, VCCS, CCVS, CCCS
- **Network**: NET (auto-generated connection points)

---

## 📈 Output

### Saved Models

Location: `./model-save-4node/`

Each fold saves a checkpoint containing:
- Model state dictionary
- Best validation/test metrics
- Graph statistics (max_z, k-hop value)
- Configuration (use_hde, node_types)

Filename format: `4node_enhanced_model_fold{N}.pth`

### Visualizations

Location: `./plot-4node/`

Each fold generates a comprehensive plot with:
1. **Training Loss Curve**: BCE loss over epochs
2. **AUC Scores**: Train/Val/Test AUC-ROC curves
3. **Accuracy**: Train/Val/Test accuracy curves

Filename format: `4node_enhanced_results_fold{N}.png`

### Evaluation Metrics

Final results report includes:

- **Per-Fold Performance**: Individual fold results
- **Average Metrics**: Mean ± Standard Deviation
  - Validation AUC
  - Test AUC
  - Test Accuracy
- **Target Achievement**: Checks against thresholds
  - Val AUC ≥ 0.92
  - Test AUC ≥ 0.90
  - Test Acc ≥ 0.85

### Sample Results

```
=== FINAL 4-NODE ENHANCED RESULTS ===
Average Validation AUC: 0.9234 ± 0.0123
Average Test AUC:       0.9087 ± 0.0156
Average Test Accuracy:  0.8654 ± 0.0178

Target Achievement:
   ✅ Validation AUC ≥ 0.92
   ✅ Test AUC ≥ 0.90
   ✅ Test Accuracy ≥ 0.85

Per-Fold Results:
   Fold 1 ⭐: Val AUC=0.9256, Test AUC=0.9123, Test Acc=0.8712
   Fold 2 ⭐: Val AUC=0.9312, Test AUC=0.9145, Test Acc=0.8698
   ...
```

---

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `BATCH_SIZE` (default: 6)
   - Reduce `HIDDEN_CHANNELS` (default: 80)
   - Use CPU: `device = torch.device('cpu')`

2. **Dataset Not Found**
   - Verify file path in `DATASET_PT` variable
   - Ensure graph processor has been run
   - Check `./data/` directory exists

3. **Training Stuck at 0.5 Accuracy**
   - Script auto-restarts fold (max 3 attempts)
   - Check data balance and quality
   - Try different random seed

4. **PyTorch Geometric Version Issues**
   - Ensure compatible versions of torch-cluster, torch-scatter, torch-sparse
   - Match CUDA version with PyTorch installation

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{thgc2024,
  title={THGC: Topology-aware Heterogeneous Graph for Circuit Link Prediction},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/THGC}}
}
```

### Dataset Citation

```bibtex
@misc{spice-datasets,
  title={SPICE Datasets for Circuit Analysis},
  author={Symbench Team},
  year={2024},
  howpublished={\url{https://github.com/symbench/spice-datasets}}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]

---

## 🙏 Acknowledgments

- PyTorch Geometric team for the excellent graph learning library
- Symbench team for providing circuit datasets
- DRNL paper authors for the innovative node labeling approach

---

**Last Updated**: February 2026  
**Version**: 1.0.0
