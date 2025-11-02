# NEURON & pytorch tutorial

This repo will contain all of the code to support:  
(1) Simulating Ball & Stick neurons using NEURON   
(2) Training DNN twins of single neuron models using pytorch  

An accompaying dataset can be found on kaggle at [ball-and-stick-neuron-data](https://www.kaggle.com/datasets/selfishgene/ball-and-stick-neuron-data) with two pre-prepared datasets with very different input-output complexities:
- BallAndStickNeuron_Soma_Hay2011_Dend_Lxd_2048um_x_1_0um_8segs - large dendrite (~3.5 times lambda) - with NMDA synapses  
- BallAndStickNeuron_Soma_Hay2011_Dend_Lxd_128um_x_4_0um_8segs - small dendrite (~0.1 times lambda) - with AMPA synapses (basically a point neuron)  

## File breakdown of the repo 
In order of how a new user would want to use the repo

## Notebooks
- `simulate_ball_and_stick_neuron.ipynb` - Tutorial for building and simulating ball-and-stick neuron models using NEURON simulator with Hay2011 active conductances, demonstrating the effect of dendritic length on spatial integration.
- `create_dataset_BS_neuron.ipynb` - Generate training/validation/test datasets by simulating Ball & Stick neurons with randomized input patterns, post-processing filtering, and visualizing dataset statistics.
- `create_dataset_BS_neuron_AMPA_point.ipynb` - Specialized dataset generation for the small dendrite configuration (128um x 4.0um, ~0.1 lambda) with AMPA synapses, creating the point neuron-like behavior dataset.
- `create_dataset_BS_neuron_NMDA_cable.ipynb` - Specialized dataset generation for the large dendrite configuration (2048um x 1.0um, ~3.5 lambda) with NMDA synapses, creating the spatially extended cable neuron dataset.
- `train_DNN_twin_of_BS_neuron.ipynb` - Train deep neural networks (TCN, ResNetTCN, ELM, or Transformer) to act as fast approximations of biophysical neuron models, with multi-task learning for spikes, voltages, and firing rates.
- `train_DNN_twin_of_BS_neuron_AMPA_point.ipynb` - Training script specifically configured for the AMPA point neuron dataset, optimized for the simpler input-output mapping of the small dendrite configuration.
- `train_DNN_twin_of_BS_neuron_NMDA_cable.ipynb` - Training script specifically configured for the NMDA cable neuron dataset, optimized for the complex spatiotemporal integration of the large dendrite configuration.

### Single Neuron Model
- `neuron_model_ball_and_stick.py` - Implementation of the Ball & Stick neuron model with active soma (Hay2011) and passive dendrite (passive biophysics) with various synaptic inputs (AMPA, NMDA, NMDA wth human parameters)

- `create_dataset_BS_neuron.py` - Dataset generation script that creates training/validation/test data by simulating Ball & Stick neurons with various input patterns and storing the output in an orderly fashion for easy loading and access
  - uses the `neuron_model_ball_and_stick.py` for simulating the Ball & Stick neuron
  - uses the `inspect_and_prune_neuron_dataset.py` for dataset sample filtering functions

### DNN Twin Model
- `train_DNN_twin_of_BS_neuron.py` - Main training script for DNN twin models
  - uses the `dataloader_BS_neuron.py` for loading a previously created dataset
  - uses the `twin_model_definitions.py` for creating the DNN twin model
  - uses the `evaluate_twin_model.py` for functions to evaluate the trained model

- `dataloader_BS_neuron.py` - PyTorch DataLoader for efficiently loading neuron simulation data for training/evaluation

- `evaluate_twin_model.py` - script that loads a previously trained twin model and evaluates it on a given neuron simulation dataset

- `twin_model_definitions.py` - contains the class definitions for various DNN twin architectures.
  - uses the `torch_module_definitions.py` for the torch modules used in the twin models

- `torch_module_definitions.py` - contains the class definitions for various torch modules used in the repo.

### Configuration and Utilities
- `config.py` - configuration file containing data and model paths, and other global settings
