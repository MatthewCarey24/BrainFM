# BrainFM

Prediction of cellular perturbations (gene knockouts, drug treatments) from voltage imaging data using transformer models.

## Overview

This repository processes voltage imaging recordings of neurons and trains transformer models to predict which perturbation was applied based on neural activity patterns. The pipeline extracts individual neuron traces from imaging data, then uses a transformer architecture with patch-based temporal encoding to map activity patterns to perturbation classes.

## Files

- **`extract_traces.py`** - Main data processing pipeline. Loads voltage imaging `.mat` files, identifies neuron ROIs using correlation-based segmentation and spike variance filtering, extracts time-series traces, and applies ΔF/F normalization.

- **`toy_perturbation_prediction.py`** - Transformer model implementation. Defines `NeuralPerturbationTransformer` with patch-based temporal encoding, supports both masked self-supervised pretraining and supervised perturbation classification. Includes dataset loader and training loops.

- **`visualize.py`** - Visualization utilities for correlation masks, binary ROI masks, and neural activity heatmaps.

- **`config.py`** - Hyperparameters for trace extraction (correlation thresholds, region size limits, normalization parameters).

- **`metadata.json`** - Experimental metadata and recording parameters.

## Future Directions

- **Spatial Encoding**: Current model treats patches as temporal segments. Could transpose architecture so each token represents a neuron (spatial) with temporal values, potentially better capturing inter-neuron dynamics.

- **Multi-Modal Embedding**: Embed perturbation data directly in pretraining, rather than just as a finetuning objective. 

- **Continuous Perturbation Space**: Rather than a perturbation vocab, explore methods of mapping between drug structure and latent space to enable de novo drug discovery
