# NEURON & pytorch tutorial

This repo will contain all of the code to support:  
(1) Simulating Ball & Stick neurons using NEURON   
(2) Training DNN twins of Ball & Stick neurons using pytorch  

## File breakdown of the repo 
In order of how a new user would want to use the repo

## Notebooks
- simulate a single Ball & Stick neuron and plot the simulation results
- create a dataset of Ball & Stick neurons and plot the dataset statistics
- train a DNN twin of a Ball & Stick neuron and plot the training results

### Single Neuron Model
- `neuron_model_ball_and_stick.py` - Implementation of the Ball & Stick neuron model with synaptic filtering, dendritic branching, and various nonlinearities (NMDA-like, AMPA-like, two sided saturation)

- `create_dataset_BS_neuron.py` - Dataset generation script that creates training/validation/test data by simulating Ball & Stick neurons with various input patterns and storing the output in an orderly fashion for easy loading and access
  - uses the `neuron_model_ball_and_stick.py` for simulating the Ball & Stick neuron
  - uses the `inspect_and_prune_neuron_dataset.py` for dataset sample filtering functions

### DNN Twin Model
- `train_DNN_twin_of_BS_neuron.py` - Main training script for DNN twin models
  - uses the `dataloader_BS_neuron.py` for loading a previously created dataset
  - uses the `twin_model_definitions.py` for creating the DNN twin model
  - uses the `evaluate_twin_model.py` for functions to evaluate the trained model

- `train_DNN_twin_of_FF_neuron_noTC.py` - identical to `train_DNN_twin_of_FF_neuron.py` but without talyor consistency loss

- `dataloader_BS_neuron.py` - PyTorch DataLoader for efficiently loading Ball & Stick neuron simulation for training/evaluation

- `evaluate_twin_model.py` - script that loads a previously trained twin model and evaluates it on a given neuron simulation dataset - evaluates f(x) accuracy

- `twin_model_definitions.py` - contains the class definitions for various DNN twin architectures. any new twin architectures should be added here (ELM, transformer, etc.)
  - uses the `torch_module_definitions.py` for the torch modules used in the twin models

- `torch_module_definitions.py` - contains the class definitions for various torch modules used in the repo. any new torch modules that is not a full model should be added here

### Configuration and Utilities
- `config.py` - configuration file containing data and model paths, and other global settings can be added here
