"""
Configuration and hyperparameters for the Citeseer community-detection
pipeline described in:

Self-Supervised Graph Representation Learning with Community-Aware
Optimization for Attributed Networks Community Detection
"""

import torch

# ---------------------------
# Data
# ---------------------------
DATA_ROOT = "/tmp/Citeseer"
DATASET_NAME = "Citeseer"

# ---------------------------
# Experiment
# ---------------------------
NUM_RUNS = 5

# ---------------------------
# Loss weights
# ---------------------------
W_DGI = 0.0
W_KL = 0.05
W_MOD = 1.00
W_ENT = 0.01

# ---------------------------
# Training schedule
# ---------------------------
DGI_PRETRAIN_EPOCHS = 100
FINETUNE_EPOCHS = 200
PATIENCE = 30

# ---------------------------
# Model architecture
# ---------------------------
HIDDEN_DIM = 64
EMBEDDING_DIM = 32
NUM_HEADS = 8
NUM_LAYERS = 2
DROPOUT = 0.2

# ---------------------------
# Optimization
# ---------------------------
LR_PRETRAIN = 0.001
LR_FINETUNE = 0.005
WEIGHT_DECAY = 5e-4

# ---------------------------
# Device
# ---------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
