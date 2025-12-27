# Guiding Cellular Reprogramming with Reinforcement Learning

A reinforcement learning framework for predicting gene perturbation sequences to induce desired cell fate transitions in silico. This project uses CellOracle as the simulation environment and Proximal Policy Optimization (PPO) to learn optimal alteration sequences for cellular reprogramming.

## Overview

Cellular reprogramming is a critical challenge in regenerative medicine. While computational tools like CellOracle can simulate single-step gene perturbations, they cannot directly predict the multi-step alteration sequences required for complex cell fate transitions. This project addresses that gap by training an RL agent to autonomously discover optimal perturbation sequences through trial-and-error interaction with a modified CellOracle environment.

### Key Results
- **Success Rate**: 84.37% on unseen cell state transitions
- **Approach**: PPO agent trained on 2.5M+ timesteps with curriculum learning
- **Dataset**: Mouse embryonic stem cell (mESC) differentiation data with 30K cells and 3K genes
- **Performance**: Agent learns to navigate high-dimensional state spaces (~3K gene expression dimensions) and find valid paths with 216 possible perturbation actions (108 genes × 2 actions: knockout/overexpression)

---

## Quick Start: Running Everything

The **`RunEverythingNotebook.ipynb`** contains the complete end-to-end pipeline. Follow these steps to reproduce the full workflow:

### Step 1: Environment Setup

```python
# Install dependencies (GPU-optimized environment recommended)
conda create -n celloracle_rl python=3.10
conda activate celloracle_rl
pip install scanpy pandas numpy matplotlib torch stable-baselines3 gymnasium optuna wandb
```

### Step 2: Data Preparation

The notebook starts with raw scRNA-seq and scATAC-seq data from mouse embryonic stem cells:

```python
# Cell filtering: 30K cells, 3K highly variable genes
# CellOracle requirements: max 30K cells, max 3K genes for simulation efficiency
# Data split: 90% training, 5% validation, 5% test (unusual split to minimize smoothing leakage)
```

The preprocessing step includes:
- Log normalization of gene expression
- Selection of most variable genes via Pearson Residuals
- Z-score normalization with StandardScaler (fitted only on training data)
- UMAP embedding computation

### Step 3: CellOracle Setup

Initialize the cell state simulator:

```python
import CellOracleSetupWithTFDict as setup_module

# Create TF-to-target gene dictionary
tf_dict = tf_module.create_tf_target_dict(base_grn_file, adata)

# Initialize CellOracle
setup = setup_module.Setup(
    tf_dict=tf_dict,
    scRNA_dat=adata,
    cluster_name="cell_type",
    embedding_name="X_umap",
    output_dir="./celloracle_data",
    load_dir="./celloracle_data"
)
```

**Key CellOracle Parameters**:
- `n_simulation_iterations`: 3 (default - propagation steps)
- `knn_neighbors`: 200 (nearest neighbors for transition lookup)
- `min_coef_abs`: 0.01 (minimum edge weight threshold)
- `max_p_value`: 0.001 (edge significance threshold)

### Step 4: Transition Matrix Precomputation

Create a lookup table for fast simulation:

```python
import CreateTransitionMatrix as transition_module

transition_matrix = transition_module.create_transition_matrix(
    oracle=setup.celloracle,
    perturbable_genes=tf_dict.keys(),
    allow_activation=True
)
```

This matrix maps (current_state, action) → next_state, enabling 5-10x speedup during training.

### Step 5: Hyperparameter Tuning (Optional)

The notebook includes two phases of Optuna-based hyperparameter optimization:

```python
# Phase 1: Broad search over 30K timesteps
trained_model_phase_1 = tuning_module1.run_optuna_hpo_phase_1(config)

# Phase 2: Fine-tuning top 5 params from Phase 1
trained_model_phase_2 = tuning_module2.run_optuna_hpo_phase_2(config_phase_2)
```

**Tuned Hyperparameters** (from thesis results):
- Learning Rate: 0.000107
- Gamma (discount): 0.9831
- GAE Lambda: 0.9013
- Entropy Coefficient: 0.0054
- Value Function Coefficient: 0.4075

### Step 6: Training Configuration

Set up the main RL training pipeline:

```python
import SingleEnvRunBuildAI as single_ai_module

notebook_config = {
    # Paths
    "ORACLE_PATH": "./celloracle_data/ready_oracle.pkl",
    "TRANSITION_MATRIX": "./celloracle_data/transition_matrix.pkl",
    "MODEL_SAVE_PATH": "./models/final_model",
    
    # Training parameters
    "TOTAL_TIMESTEPS": 2500000,
    "N_ENVS": 8,  # Parallel environments
    "BATCH_SIZE": 256,
    "PPO_N_STEPS": 1024,
    "PPO_N_EPOCHS": 11,
    "PPO_BATCH_SIZE": 64,
    
    # Environment setup
    "MAX_STEPS_PER_EPISODE": 20,  # Initial episode length
    "ALLOW_GENE_ACTIVATION": True,
    "STEP_PENALTY": -1,
    "GOAL_BONUS": 0,
    "DISTANCE_REWARD_SCALE": 5,
    
    # Curriculum learning
    "TARGET_CELLS_PER_PHASE": 6,
    "MAX_STEPS_FIRST_PHASE": 100000,
    "MAX_STEP_INCREASE_PER_PHASE": 4,
    
    # Neural network
    "NET_WIDTH_FIRST_P": 256,
    "NET_WIDTH_SECOND_P": 64,
    "NET_WIDTH_FIRST_V": 128,
    "NET_WIDTH_SECOND_V": 64,
    "ACTIVATION_FN": "leaky_relu",
    
    # Hardware
    "DEVICE": "auto",  # GPU if available
    "RANDOM_SEED": 77
}

trained_model = single_ai_module.run_training(notebook_config)
```

### Step 7: Training Execution

```python
trained_model_main = single_ai_module.run_training(notebook_config)
# Model saves checkpoints every 500K timesteps
# Weights & Biases tracks all metrics in real-time
```

### Step 8: Evaluation & Visualization

The framework includes comprehensive evaluation tools:

```python
# Success rate on test set: measure % of episodes reaching target
# Path efficiency: compare agent paths vs. Breadth-First Search optimal
# Activation fraction: track agent's use of overexpression vs. knockout
```

---

## Architecture Overview

### 1. **Environment** (`FinalEnvSingleInstance.py`)

A Gymnasium-compatible environment that simulates cellular state transitions:

```
State:   {current_expression: [3000-dim vector], 
          target_expression: [3000-dim vector]}

Actions: 216 discrete actions
         - 0-107:   Knockout gene_0 to gene_107
         - 108-215: Overexpress gene_0 to gene_107

Transition: CellOracle simulation (deterministic)
Reward:     Sparse reward (goal reached) + Dense reward (distance to target)
```

**Curriculum Learning**: Agent starts with 6 target cells per phase, increasing complexity as success rate improves. Episode length grows from 20 to 100+ steps.

### 2. **Reward Function**

```
reward = distance_reward + goal_bonus + step_penalty

distance_reward = -euclidean_distance(current_expr, target_expr) * DISTANCE_REWARD_SCALE
goal_bonus = +1.0 if successfully reached target else 0.0
step_penalty = -1 per step (encourages efficient paths)
```

### 3. **Policy Network**

Custom neural network with embedding layer:

```
Input: Concatenated [current_state, target_state] (6000-dim)
       ↓
Embedding Layer: Custom dimension reduction
       ↓
Actor Head:   Dense(256) → LeakyReLU → Dense(64) → softmax (action logits)
Value Head:   Dense(128) → LeakyReLU → Dense(64) → scalar (value estimate)
```

Action masking ensures invalid perturbations are masked out at each step.

### 4. **CellOracle Integration**

Modified CellOracle with optimizations:
- GPU acceleration for batch simulations
- NumPy/CuPy matrix operations (5-10x speedup)
- Pre-computed transition lookup table
- Batch processing for multi-cell perturbations

---

## File Structure

```
project/
├── RunEverythingNotebook.ipynb          # Main execution notebook
├── FinalEnvSingleInstance.py            # Gymnasium environment definition
├── SingleEnvRunBuildAI.py               # PPO training loop & callbacks
├── CustomNeuralNetwork.py               # Actor-Critic policy architecture
├── CellOracleSetupWithTFDict.py         # CellOracle initialization
├── CreateTransitionMatrix.py            # Transition lookup table
├── CreateNewTFToTargetGeneList.py       # TF regulatory network parsing
├── HPO_AULC_phase_1.py                  # Phase 1 hyperparameter tuning
├── HPO_AULC_phase_2.py                  # Phase 2 hyperparameter optimization
├── ShortestPathTransitionMatrix.ipynb   # BFS evaluation & benchmarking
├── Filter_Genes.ipynb                   # Gene filtering & selection
├── inference.ipynb                      # Inference on trained models
├── GenerateNeededGraphs.ipynb           # Visualization & reporting
└── README.md                            # This file
```

---

## Key Design Decisions

### Why Reinforcement Learning?

1. **No labeled data**: There are no pre-defined optimal sequences for arbitrary cell transitions
2. **Vast search space**: 216^n possible action sequences (computationally infeasible to brute-force)
3. **Generalization**: RL agents learn dense state representations that transfer to unseen cell states
4. **Adaptability**: Policy easily updates when simulator improves (just retrain on new simulator outputs)

### Why Modified CellOracle?

- **Accuracy**: Single-cell resolution with chromatin accessibility
- **Efficiency**: Handles only 30K cells × 3K genes (trade-off for speed)
- **Compatibility**: Deterministic transitions suitable for RL Markov property
- **Future-proof**: Can swap CellOracle for SCENIC+ when computational resources allow

### Curriculum Learning Strategy

Episodes progressively increase in difficulty:
- **Phase 1** (0-100K steps): 6 target cells, 20 steps max → Success on easy transitions
- **Phase 2** (100K-200K steps): 6 target cells, 24 steps max
- **Phase 3+**: Increasing cell diversity and episode length as policy improves

This prevents early training collapse and improves sample efficiency.

---

## Expected Outcomes

After 2.5M timesteps of training (12-24 hours on GPU):

| Metric | Value | Notes |
|--------|-------|-------|
| **Test Success Rate** | 84.37% | % of test episodes reaching target |
| **Avg Path Length** | ~15-20 steps | Agent solution length |
| **BFS Optimal Length** | ~8-12 steps | Theoretical optimal (via exhaustive search) |
| **Efficiency Gap** | 40-50% | Suboptimal vs. BFS (conservative exploration) |

The agent consistently finds valid solutions but favors safe, repetitive actions over efficient ones.

---

## Troubleshooting

### Common Issues

**ModuleNotFoundError: velocyto**
- Scanpy imports velocyto even if not used
- Solution: `pip install velocyto`

**GPU Out of Memory**
- Reduce `N_ENVS` from 8 to 4
- Reduce `BATCH_SIZE` from 256 to 128
- Increase `PPO_N_EPOCHS` slightly to compensate

**Slow simulation (>1 second per step)**
- Ensure transition matrix is precomputed
- Check GPU is being utilized: `nvidia-smi`
- Verify CuPy installation: `python -c "import cupy; print(cupy.__version__)"`

**Poor convergence (<50% success rate at 1M steps)**
- Check curriculum callback is advancing phases
- Increase `DISTANCE_REWARD_SCALE` from 5 to 10
- Verify transition matrix correctness with `CheckSelfMadeBaseGRNAndCompare.py`

---

## Extensions & Future Work

1. **Multi-environment transfer**: Train on mouse data, evaluate on human cells
2. **SCENIC+ integration**: Replace CellOracle for higher accuracy
3. **Imitation learning warmstart**: Pre-train with BFS-generated trajectories
4. **Graph neural networks**: Learn gene network structure end-to-end
5. **Real-world validation**: In vitro testing of predicted perturbations

---

## References

**Primary Citation**: 
Bannink, C. "Guiding Cellular Reprogramming: A Reinforcement Learning Approach with In Silico Perturbation Models." Master's thesis, Utrecht University, 2025.

**Key Frameworks**:
- CellOracle: Kamimoto et al. (2020) - Gene regulatory network simulation
- Stable Baselines 3: Raffin et al. (2021) - RL implementations (PPO, etc.)
- Gymnasium: OpenAI - Environment interface standard
- Weights & Biases: Experiment tracking & hyperparameter visualization

---

## Contact & Attribution

**Author**: Caspar Bannink  
**Supervisor**: Dr. V. Bhardwaj  
**Institution**: Utrecht University, Dept. of Artificial Intelligence  
**Thesis**: "Guiding Cellular Reprogramming: A Reinforcement Learning Approach with In Silico Perturbation Models" (November 2025)

---

## License

This project is provided for research and educational purposes. Please refer to individual package licenses (CellOracle, Stable Baselines 3, etc.) for restrictions.
