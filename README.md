# NEURON & pytorch tutorial

This repo will contain all of the code to support:  
(1) Simulating Ball & Stick neurons using NEURON   
(2) Training DNN twins of single neuron models using pytorch  

An accompaying dataset can be found on kaggle at [ball-and-stick-neuron-data](https://www.kaggle.com/datasets/selfishgene/ball-and-stick-neuron-data) with two pre-prepared datasets with very different input-output complexities:
- BallAndStickNeuron_Soma_Hay2011_Dend_Lxd_2048um_x_1_0um_8segs - large dendrite (~3.5 times lambda) - with NMDA synapses  
- BallAndStickNeuron_Soma_Hay2011_Dend_Lxd_128um_x_4_0um_8segs - small dendrite (~0.1 times lambda) - with AMPA synapses (basically a point neuron)  

An accompanying possible assignment for potential students to get started with the code base can be found at [Example Assignment](https://docs.google.com/document/d/1K7-6n9SaA-GnQn_CTJZDyIy44OjsenM8ukK8VC0S-_o/edit?tab=t.0) and also under the `pdf` folder.  

## File breakdown of the repo 
In order of how a new user would want to use the repo

### Notebooks (good entry point into code base)
- `simulate_ball_and_stick_neuron.ipynb` - Tutorial for building and simulating ball-and-stick neuron models using NEURON simulator with Hay2011 active conductances, demonstrating the effect of dendritic length on spatial integration.
- `create_dataset_BS_neuron.ipynb` - Generate training/validation/test datasets by simulating Ball & Stick neurons with randomized input patterns, post-processing filtering, and visualizing dataset statistics.
- `create_dataset_BS_neuron_AMPA_point.ipynb` - Specialized dataset generation for the small dendrite cable configuration (128um x 4.0um, ~0.1 lambda) with AMPA synapses, creating the point neuron-like behavior dataset.
- `create_dataset_BS_neuron_NMDA_cable.ipynb` - Specialized dataset generation for the large dendrite cable configuration (2048um x 1.0um, ~3.5 lambda) with NMDA synapses, creating the spatially extended cable neuron dataset.
- `train_DNN_twin_of_BS_neuron.ipynb` - Train deep neural networks (TCN, ResNetTCN, ELM, or Transformer) to act as fast approximations of biophysical neuron models, with multi-task learning for spikes, somatic and dendritic voltages, and instantaneous firing rates.
- `train_DNN_twin_of_BS_neuron_AMPA_point.ipynb` - Training script specifically configured for the AMPA point neuron dataset.
- `train_DNN_twin_of_BS_neuron_NMDA_cable.ipynb` - Training script specifically configured for the NMDA long and wide cable neuron dataset.

### Colab notebook:
- example colab notebook that properly installs neuron and compiles the mod files [tutorial_NEURON_starter_code](https://colab.research.google.com/drive/1wgmJzoowipmzI5aUEvWtPohrGMSUJ93k?usp=sharing)

### Single Neuron Model Simulation
- `neuron_model_ball_and_stick.py` - Implementation of the Ball & Stick neuron model with active soma (Hay2011) and passive dendrite (passive biophysics) with various synaptic inputs (AMPA, NMDA, NMDA wth human parameters)

- `create_dataset_BS_neuron.py` - Dataset generation script that creates training/validation/test data by simulating Ball & Stick neurons with various input patterns and storing the output in an orderly fashion for easy loading and access
  - uses the `neuron_model_ball_and_stick.py` for simulating the Ball & Stick neuron
  - uses the `inspect_and_prune_neuron_dataset.py` for dataset sample filtering functions

### DNN Twin Model Training
- `train_DNN_twin_of_BS_neuron.py` - Main training script for DNN twin models
  - uses the `dataloader_BS_neuron.py` for loading a previously created dataset
  - uses the `twin_model_definitions.py` for creating the DNN twin model
  - uses the `evaluate_twin_model.py` for functions to evaluate the trained model

- `dataloader_BS_neuron.py` - PyTorch DataLoader for efficiently loading neuron simulation data for training/evaluation

- `evaluate_twin_model.py` - script that loads a previously trained twin model and evaluates it on a given neuron simulation dataset

- `twin_model_definitions.py` - contains the class definitions for various DNN twin architectures (TCN, ResNetTCN, ELM, Transformer).
  - uses the `torch_module_definitions.py` for the torch modules used in the twin models

- `torch_module_definitions.py` - contains the class definitions for various torch modules used in the repo.

### Configuration and Utilities
- `config.py` - configuration file containing data and model paths, and other global settings

## Acknowledgments and References

This work builds upon and inspired by:


1. **Beniaguev, D., Segev, I., & London, M. (2021).** Single Cortical Neurons as Deep Artificial Neural Networks. *Neuron*, 109(17), 2727–2739.e3. https://doi.org/10.1016/j.neuron.2021.07.002
  - GitHub: https://github.com/SelfishGene/neuron_as_deep_net

2. **Aizenbud, I., Yoeli, D., Beniaguev, D., de Kock, C. P. J., London, M., & Segev, I. (2024).** What Makes Human Cortical Pyramidal Neurons Functionally Complex. Available via PubMed Central (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC11702691/

3. **Hay, E., Hill, S., Schürmann, F., Markram, H., & Segev, I. (2011).** Models of Neocortical Layer 5b Pyramidal Cells Capturing a Wide Range of Dendritic and Perisomatic Active Properties. *PLoS Computational Biology*, 7(7), e1002107. https://doi.org/10.1371/journal.pcbi.1002107

4. **Spieler, A., Tetenov, A., Prabhakaran, S., & Martius, G. (2023).** The Expressive Leaky Memory Neuron: an Efficient and Expressive Phenomenological Neuron Model Can Solve Long-Horizon Tasks. *Advances in Neural Information Processing Systems (NeurIPS) 36*. https://arxiv.org/abs/2306.16922
  - GitHub: https://github.com/AaronSpieler/elmneuron

