HCASFE-Net
Hierarchical Causal Associative Sparse Free Energy Network — a pure-NumPy online learning architecture for temporal sequence understanding.
HCASFE-Net processes sequential data one step at a time without backpropagation through time. It combines hierarchical temporal processing, online causal structure learning, associative memory, and variational free energy minimisation into a single coherent architecture that runs on constrained hardware including Android phones.
Architecture
The core architecture has seven components that interact at every timestep:
Hierarchical Temporal Network (HTN) processes input across multiple timescales simultaneously. Version 8.1 introduced LeakyTemporalLevel — a learned gated continuous update that improves adaptation to nonlinear dynamics. Default configuration uses 6 levels with ratios (3, 7, 10, 30, 60, 120), spanning fast phoneme-level changes to slow discourse-level drift.
SparseRouter directs information through a mixture of expert modules, allowing the architecture to specialise different modules for different regime types without explicit supervision.
CausalGraph implements online causal structure learning inspired by NOTEARS. At each step it learns which HTN timescale levels causally drive which others, producing a human-readable adjacency matrix A. This matrix serves as a regime fingerprint — distinct regimes produce distinct causal structures, enabling 97-98% regime identification accuracy via 1-nearest-neighbour classification.
AssociativeMemory is a Hopfield network with importance-weighted writes, episodic retrieval, and a forgetting mechanism. It achieves 100% delayed recall at delays of 5 to 100 steps on tasks where tuned LSTMs score near chance. Capacity 2048 slots at default configuration.
FreeEnergyCore is a variational autoencoder producing a surprise signal normalised against a running baseline. Version 8.2 introduced dual cores — a fast core tracking step-by-step surprise and a slow core tracking regime-level drift via an exponential moving average of the fused state.
GlobalWorkspace integrates all component representations via learned attention and produces a broadcast vector fed back as top-down context into all other components, implementing a form of global information sharing across the architecture.
Prediction head projects the broadcast to the output space via predictive coding. Current work focuses on a StrongPredictionHead — a two-layer MLP receiving both fast HTN states and broadcast, trained on 1-step and 3-step prediction horizons simultaneously.
Key Results
Memory — 100% delayed recall at delays 5, 20, 50, and 100 steps. Tuned LSTM H=96 scores near or below chance at all delays. MLP scores 34-38% (chance level). This result is stable across architecture versions and random seeds.
Causal fingerprinting — 97-98% 1-NN regime identification accuracy from the causal A matrix across 6 overlapping regimes with signal leakage. Interpretable edges describe what the model learned about the causal structure of the data.
ECG anomaly detection — 73% arrhythmia surprise detection on 10 MIT-BIH recordings versus 19% for a lag predictor baseline. Evaluated online without any offline training phase.
EEG sleep staging — 100% detection of WAKE to N1 and REM transitions on Sleep-EDF data. Balancing fix applied to address 67.9% WAKE class imbalance.
LeakyHTN — +38.9% improvement on Van der Pol prediction and +13.8% on coupled oscillators versus hard HCASFE. The learned gate enables faster adaptation to nonlinear regime changes.
Prediction — HCASFE currently wins 2 of 10 synthetic environments against a tuned LSTM. This is an honest limitation under active development. The architecture is not a general prediction improvement over LSTMs — its advantages are in long-range memory, interpretable causal structure, and online anomaly detection.
Active inference — Agent trained with REINFORCE achieves 21.8% FE surprise reduction over passive baseline. Imagination module with 17 candidate policies adds a further 6.6% on early regime identification at a 50-step delay.
Financial domain — +14.4% over MLP and 91% surprise detection versus 82% lag baseline on 379 windows of real market data across 4 regimes.
Current Version
Version 8.2 introduces H=96 as default (was 48), 6 HTN levels with ratio 120 for discourse-scale dynamics (was 5 levels), memory capacity 2048 (was 512), top-k routing of 6 (was 5), and dual FE cores. Fully backward compatible — pass H=24, n_levels=5, dual_fe=False for v8.1 behaviour.
Project Structure
Active development is restructuring the codebase from a single file into a clean module layout:
Code
Roadmap
Immediate priorities are improving prediction performance via StrongPredictionHead, completing the module restructure, extending the memory benchmark systematically across delays of 1 to 1000 steps and 2 to 20 keys, running against additional baselines including reservoir computing and Continual Backprop, and validating on all 48 MIT-BIH ECG recordings.
Paused pending core validation: multimodal architecture, active inference agent, imagination module, binary action system, transformer comparison.
Future architectural directions under consideration: stacked HCASFE-MLP blocks analogous to transformer depth, pretrained frozen encoders feeding the temporal core for the 30-50M parameter target, and an internal voice module for real-time interpretability of model states.
Applications
Target applications are Neo Health (cardiac and sleep monitoring with interpretable anomaly detection), Neo Assistant (multimodal online learning agent), and Neo Climate (environmental time-series regime detection). Target deployment timeline 2027.
Hardware
Runs on Pydroid 3 (Android). Evaluated on Kaggle free tier (T4 GPU, 16GB RAM). No GPU required for inference or online learning at H=96. PyTorch transition planned for encoder-stage training at scale.
License
Proprietary — Neo Corporation. All rights reserved.
