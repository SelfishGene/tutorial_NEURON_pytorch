#%% Imports

import os
import glob
import time
import torch
import copy
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F

from neuron_model_ball_and_stick import BallAndStickNeuron
from dataloader_BS_neuron import BSNeuronDataset
from twin_model_definitions import SingleNeuronTwinModel_ResNetTCN, SingleNeuronTwinModel_TCN
from twin_model_definitions import SingleNeuronTwinModel_ELM, SingleNeuronTwinModel_Transformer
import config

# Import evaluation functions
from evaluate_twin_model import predict_on_all_simulations, evaluate_model_on_dataset
from evaluate_twin_model import plot_evaluation_figures
from evaluate_twin_model import display_sample_predictions_minimal, display_sample_predictions_full
from evaluate_twin_model import calculate_calibration_metrics, display_calibration_figure

#%% Main training script

if __name__ == "__main__":

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('----------------------------')
    print(f'Using device: {device}')
    print('----------------------------')
    torch.set_float32_matmul_precision('high')

    # Dataset parameters
    data_root = config.NEURON_DATA_ROOT

    all_data_folder_names = glob.glob(os.path.join(data_root, '*'))

    print('----------------------------')
    print('all_data_folder_names:')
    for data_folder_name in all_data_folder_names:
        print(data_folder_name)
    print('----------------------------')

    # select the data folder
    # BallAndStickNeuron: large dendrite (~3.5 times lambda) - with NMDA synapses
    data_folder_name = r"BallAndStickNeuron_Soma_Hay2011_Dend_Lxd_2048um_x_1_0um_8segs"

    # BallAndStickNeuron: small dendrite (~0.1 times lambda) - with AMPA synapses (basically a point neuron)
    # data_folder_name = r"BallAndStickNeuron_Soma_Hay2011_Dend_Lxd_128um_x_4_0um_8segs"

    data_folder = os.path.join(data_root, data_folder_name)
    train_data_folder = os.path.join(data_folder, 'train')
    valid_data_folder = os.path.join(data_folder, 'valid')
    test_data_folder = os.path.join(data_folder, 'test')

    # load the GT neuron model
    neuron_file = glob.glob(os.path.join(data_folder, '*.pkl'))[0]
    bs_neuron = BallAndStickNeuron.load(neuron_file)

    original_neuron_model_metadata = {
        'original_neuron_model_name': bs_neuron.short_name,
        'original_neuron_model_file': neuron_file,
        'original_neuron_model_folder': data_folder_name,
    }

    # set the models folder
    models_root = config.MODELS_ROOT
    models_folder = os.path.join(models_root, bs_neuron.short_name)
    os.makedirs(models_folder, exist_ok=True)

    # Set the twin model type
    twin_model_type = 'TCN'
    # twin_model_type = 'ResNetTCN'
    # twin_model_type = 'ELM'
    # twin_model_type = 'Transformer'

    if twin_model_type in ['ELM']:
        compile_model = False
    else:
        compile_model = True
    # compile_model = True
    # compile_model = False

    # set some of the training data parameters
    if twin_model_type in ['ELM']:
        # train_time_window_size = 768
        train_time_window_size = 1024
        valid_time_window_size = 2048
        # valid_time_window_size = 3072
        # valid_time_window_size = 4096
        # valid_time_window_size = 7168 + 768
        test_time_window_size = 8192 - 64

    elif twin_model_type in ['Transformer']:
        # train_time_window_size = 768
        # train_time_window_size = 1024
        train_time_window_size = 1280
        valid_time_window_size = 2048
        # valid_time_window_size = 3072
        # valid_time_window_size = 4096
        # valid_time_window_size = 7168 + 768
        test_time_window_size = 8192 - 64
    else:
        # train_time_window_size = 1024
        # train_time_window_size = 1536
        train_time_window_size = 2048
        # valid_time_window_size = 3072
        valid_time_window_size = 4096
        # valid_time_window_size = 7168 + 768
        test_time_window_size = 8192 - 64

    # set some of the training data parameters
    if twin_model_type in ['ELM']:
        # train_batch_size = 512
        # train_batch_size = 384
        # train_batch_size = 256
        # train_batch_size = 192
        # train_batch_size = 128
        # train_batch_size = 64
        # train_batch_size = 32
        train_batch_size = 16

        # valid_batch_size = 384
        # valid_batch_size = 320
        # valid_batch_size = 256
        # valid_batch_size = 192
        # valid_batch_size = 128
        # valid_batch_size = 64
        valid_batch_size = 32

        # test_batch_size = 128
        test_batch_size = 32

    elif twin_model_type in ['Transformer']:
        # train_batch_size = 384
        # train_batch_size = 256
        # train_batch_size = 192
        # train_batch_size = 128
        # train_batch_size = 64
        # train_batch_size = 32
        train_batch_size = 16

        valid_batch_size = 32

        test_batch_size = 32
    else:
        # train_batch_size = 384
        # train_batch_size = 256
        # train_batch_size = 192
        # train_batch_size = 128
        # train_batch_size = 64
        # train_batch_size = 32
        train_batch_size = 16

        valid_batch_size = 32

        test_batch_size = 32

    # set whether to preload the data or not
    preload_data = True
    preload_data = False

    # Create datasets and dataloaders
    train_dataset = BSNeuronDataset(train_data_folder, train_time_window_size, preload_data=preload_data)
    valid_dataset = BSNeuronDataset(valid_data_folder, valid_time_window_size, preload_data=preload_data)
    test_dataset = BSNeuronDataset(test_data_folder, test_time_window_size, preload_data=preload_data)
    
    # num_workers = 8
    num_workers = 0
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, num_workers=num_workers)
    
    # sample a batch to make sure dataloader works well
    print('----------------------------------------------------')
    print('fetch a batch')
    print('-------------')
    batch_data = next(iter(train_dataloader))
    X_spikes = batch_data['X_spikes']
    y_spikes_GT = batch_data['y_spikes']
    y_soma_GT = batch_data['y_soma']
    y_near_spike_GT = batch_data['y_near_spike']
    y_inst_rate_GT = batch_data['y_inst_rate']
    y_dend_v_GT = batch_data['y_dend_v']
    print(f'X_spikes.shape: {X_spikes.shape}')
    print(f'y_spikes_GT.shape: {y_spikes_GT.shape}')
    print(f'y_soma_GT.shape: {y_soma_GT.shape}')
    print(f'y_near_spike_GT.shape: {y_near_spike_GT.shape}')
    print(f'y_inst_rate_GT.shape: {y_inst_rate_GT.shape}')
    print(f'y_dend_v_GT.shape: {y_dend_v_GT.shape}')
    print('----------------------------------------------------')

    # calclulate the basic statistics of X_spikes, y_soma_GT and y_inst_rate_GT (mean, std, 5% and 95% percentiles)
    X_spikes_np = X_spikes.numpy()
    x_mean = X_spikes_np.mean(); x_std = X_spikes_np.std()
    x_80p = np.percentile(X_spikes_np, 80); x_99p = np.percentile(X_spikes_np, 99)
    y_soma_GT_np = y_soma_GT.numpy()
    y_mean = y_soma_GT_np.mean(); y_std = y_soma_GT_np.std()
    y_2p = np.percentile(y_soma_GT_np, 2); y_98p = np.percentile(y_soma_GT_np, 98)
    y_inst_rate_GT_np = y_inst_rate_GT.numpy()
    y_inst_rate_mean = y_inst_rate_GT_np.mean(); y_inst_rate_std = y_inst_rate_GT_np.std()
    y_inst_rate_2p = np.percentile(y_inst_rate_GT_np, 2); y_inst_rate_98p = np.percentile(y_inst_rate_GT_np, 98)
    y_branch_v_GT_np = y_dend_v_GT.numpy()
    y_branch_v_mean = y_branch_v_GT_np.mean(); y_branch_v_std = y_branch_v_GT_np.std()
    y_branch_v_2p = np.percentile(y_branch_v_GT_np, 2); y_branch_v_98p = np.percentile(y_branch_v_GT_np, 98)

    print('--------------------------------------------------------------------------------------------')
    print('some statistics of the data')
    print('------------------------------------------------------------')
    print(f'X_spikes (mean, std), [80%, 99%] percentiles = ({x_mean:.4f}, {x_std:.4f}), [{x_80p:.4f}, {x_99p:.4f}]')
    print(f'X_spikes (min, median, max) = ({X_spikes_np.min():.4f}, {np.median(X_spikes_np):.4f}, {X_spikes_np.max():.4f})')
    print('------------------------------------------------------------')
    print(f'y_soma_GT (mean, std), [2%, 98%] percentiles = ({y_mean:.4f}, {y_std:.4f}), [{y_2p:.4f}, {y_98p:.4f}]')
    print(f'y_soma_GT (min, median, max) = ({y_soma_GT_np.min():.4f}, {np.median(y_soma_GT_np):.4f}, {y_soma_GT_np.max():.4f})')
    print('------------------------------------------------------------')
    print(f'y_inst_rate_GT (mean, std), [2%, 98%] percentiles = ({y_inst_rate_mean:.4f}, {y_inst_rate_std:.4f}), [{y_inst_rate_2p:.4f}, {y_inst_rate_98p:.4f}]')
    print(f'y_inst_rate_GT (min, median, max) = ({y_inst_rate_GT_np.min():.4f}, {np.median(y_inst_rate_GT_np):.4f}, {y_inst_rate_GT_np.max():.4f})')
    print('------------------------------------------------------------')
    print(f'y_dend_v_GT (mean, std), [2%, 98%] percentiles = ({y_branch_v_mean:.4f}, {y_branch_v_std:.4f}), [{y_branch_v_2p:.4f}, {y_branch_v_98p:.4f}]')
    print(f'y_dend_v_GT (min, median, max) = ({y_branch_v_GT_np.min():.4f}, {np.median(y_branch_v_GT_np):.4f}, {y_branch_v_GT_np.max():.4f})')
    print('--------------------------------------------------------------------------------------------')

    # set the data scaling parameters that will be used for training and evaluation
    # (to make the inputs and outputs of the DNN be around zero mean and unit variance)
    X_scale = 10.0
    V_bias_soma = bs_neuron.epas_mV
    V_scale_soma = (bs_neuron.soma_voltage_cap_mV - bs_neuron.epas_mV) / 2
    V_clip_soma_min = bs_neuron.epas_mV - 5.0
    V_clip_soma_max = bs_neuron.soma_voltage_cap_mV + 5.0
    V_bias_dend = bs_neuron.epas_mV
    V_scale_dend = (bs_neuron.dend_voltage_cap_mV - bs_neuron.epas_mV) / 2
    V_clip_dend_min = bs_neuron.epas_mV - 5.0
    V_clip_dend_max = bs_neuron.dend_voltage_cap_mV + 5.0
    y_inst_rate_multiplier = 10.0
    print(f'V_bias_soma = {V_bias_soma:.4f}, V_scale_soma = {V_scale_soma:.4f}')
    print(f'V_clip_soma_min = {V_clip_soma_min:.4f}, V_clip_soma_max = {V_clip_soma_max:.4f}')
    print(f'V_bias_dend = {V_bias_dend:.4f}, V_scale_dend = {V_scale_dend:.4f}')
    print(f'V_clip_dend_min = {V_clip_dend_min:.4f}, V_clip_dend_max = {V_clip_dend_max:.4f}')

    # Set all the remaining model parameters

    # backbone parameters
    in_channels = 2 # exc & inh
    in_spatial_dim = bs_neuron.num_segments

    # head parameters
    head_prefix_names = ['spikes', 'soma', 'near_spike', 'inst_rate', 'dend_v']
    head_out_channels = [1, 1, 1, 1, train_dataset.num_dend_channels]
    head_convert_out_ch_to_sp = [False, False, False, False, False]


    if twin_model_type == 'TCN':

        # first_layer_temporal_kernel_size = 5
        # first_layer_temporal_kernel_size = 7
        # first_layer_temporal_kernel_size = 9
        # first_layer_temporal_kernel_size = 11
        # first_layer_temporal_kernel_size = 13
        # first_layer_temporal_kernel_size = 15
        # first_layer_temporal_kernel_size = 17
        # first_layer_temporal_kernel_size = 21
        first_layer_temporal_kernel_size = 31
        # first_layer_temporal_kernel_size = 41
        # first_layer_temporal_kernel_size = 51
        # first_layer_temporal_kernel_size = 61
        # first_layer_temporal_kernel_size = 81

        # 1 block
        # num_layers_per_block_list = [3]
        # num_layers_per_block_list = [4]
        num_layers_per_block_list = [5]
        # num_layers_per_block_list = [6]
        # num_layers_per_block_list = [7]
        # num_layers_per_block_list = [8]
        # num_layers_per_block_list = [9]
        # num_layers_per_block_list = [10]
        # num_layers_per_block_list = [11]
        # num_layers_per_block_list = [15]

        # num_features_per_block_list = [8]
        num_features_per_block_list = [16]
        # num_features_per_block_list = [24]
        # num_features_per_block_list = [32]
        # num_features_per_block_list = [40]
        # num_features_per_block_list = [48]
        # num_features_per_block_list = [64]
        # num_features_per_block_list = [96]
        # num_features_per_block_list = [128]

        # temporal_kernel_size_per_block_list = [5]
        # temporal_kernel_size_per_block_list = [9]
        # temporal_kernel_size_per_block_list = [11]
        # temporal_kernel_size_per_block_list = [13]
        # temporal_kernel_size_per_block_list = [15]
        # temporal_kernel_size_per_block_list = [23]
        # temporal_kernel_size_per_block_list = [25]
        # temporal_kernel_size_per_block_list = [27]
        # temporal_kernel_size_per_block_list = [29]
        # temporal_kernel_size_per_block_list = [31]
        # temporal_kernel_size_per_block_list = [33]
        # temporal_kernel_size_per_block_list = [35]
        # temporal_kernel_size_per_block_list = [37]
        temporal_kernel_size_per_block_list = [39]
        # temporal_kernel_size_per_block_list = [41]
        # temporal_kernel_size_per_block_list = [47]
        # temporal_kernel_size_per_block_list = [51]
        # temporal_kernel_size_per_block_list = [61]

        temporal_dilation_per_block_list = [1]

        # 2 blocks
        # num_layers_per_block_list = [2, 2]
        # num_layers_per_block_list = [3, 3]
        # num_layers_per_block_list = [4, 4]
        # num_layers_per_block_list = [6, 6]

        # num_features_per_block_list = [16, 16]
        # num_features_per_block_list = [32, 32]
        # num_features_per_block_list = [48, 48]
        # num_features_per_block_list = [64, 64]
        # num_features_per_block_list = [128, 64]
        # num_features_per_block_list = [128, 128]

        # temporal_kernel_size_per_block_list = [13, 31]
        # temporal_kernel_size_per_block_list = [17, 41]

        # temporal_dilation_per_block_list = [1, 1]

        # norm_type = 'BatchNorm_B_only'
        # norm_type = 'BatchNorm'
        # norm_type = 'LayerNorm'
        norm_type = 'RMSNorm'

        # nonlinearity_str = 'silu'
        nonlinearity_str = 'leaky_gelu'
        # nonlinearity_str = 'leaky_silu'
        # nonlinearity_str = 'two_sided_leaky_silu'
        # nonlinearity_str = 'two_sided_leaky_gelu'
        # nonlinearity_str = 'leaky_tanh'
        # nonlinearity_str = 'leaky_sigmoid'

        # leaky_slope = 0.5
        # leaky_slope = 0.4
        # leaky_slope = 0.3
        # leaky_slope = 0.25
        # leaky_slope = 0.2
        leaky_slope = 0.15
        # leaky_slope = 0.1

        twin_model = SingleNeuronTwinModel_TCN(
            in_channels=in_channels,
            in_spatial_dim=in_spatial_dim,
            first_layer_temporal_kernel_size=first_layer_temporal_kernel_size,
            num_layers_per_block_list=num_layers_per_block_list,
            num_features_per_block_list=num_features_per_block_list,
            temporal_kernel_size_per_block_list=temporal_kernel_size_per_block_list,
            temporal_dilation_per_block_list=temporal_dilation_per_block_list,
            nonlinearity_str=nonlinearity_str,
            leaky_slope=leaky_slope,
            norm_type=norm_type,
            head_prefix_names=head_prefix_names,
            head_out_channels=head_out_channels,
            head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
            X_scale=X_scale,
            V_bias_soma=V_bias_soma,
            V_scale_soma=V_scale_soma,
            V_clip_soma_min=V_clip_soma_min,
            V_clip_soma_max=V_clip_soma_max,
            V_bias_dend=V_bias_dend,
            V_scale_dend=V_scale_dend,
            V_clip_dend_min=V_clip_dend_min,
            V_clip_dend_max=V_clip_dend_max,
            y_inst_rate_multiplier=y_inst_rate_multiplier,
        ).to(device)


    if twin_model_type == 'ResNetTCN':
        # first_layer_temporal_kernel_size = 5
        # first_layer_temporal_kernel_size = 7
        first_layer_temporal_kernel_size = 9
        # first_layer_temporal_kernel_size = 11
        # first_layer_temporal_kernel_size = 13
        # first_layer_temporal_kernel_size = 15
        # first_layer_temporal_kernel_size = 17
        # first_layer_temporal_kernel_size = 21
        # first_layer_temporal_kernel_size = 31
        # first_layer_temporal_kernel_size = 41
        # first_layer_temporal_kernel_size = 51
        # first_layer_temporal_kernel_size = 61
        # first_layer_temporal_kernel_size = 81

        # 1 block
        # num_miniblocks_per_block_list = [2]
        # num_miniblocks_per_block_list = [3]
        # num_miniblocks_per_block_list = [4]
        # num_miniblocks_per_block_list = [7]
        # num_miniblocks_per_block_list = [9]
        # num_miniblocks_per_block_list = [11]
        # num_miniblocks_per_block_list = [13]
        # num_miniblocks_per_block_list = [14]
        num_miniblocks_per_block_list = [15]
        # num_miniblocks_per_block_list = [17]
        # num_miniblocks_per_block_list = [19]
        # num_miniblocks_per_block_list = [20]
        # num_miniblocks_per_block_list = [23]
        # num_miniblocks_per_block_list = [31]

        # num_features_per_block_list = [6]
        # num_features_per_block_list = [8]
        # num_features_per_block_list = [12]
        # num_features_per_block_list = [16]
        # num_features_per_block_list = [24]
        # num_features_per_block_list = [32]
        # num_features_per_block_list = [40]
        num_features_per_block_list = [48]
        # num_features_per_block_list = [64]
        # num_features_per_block_list = [80]
        # num_features_per_block_list = [96]
        # num_features_per_block_list = [128]

        # temporal_kernel_size_per_block_list = [5]
        # temporal_kernel_size_per_block_list = [7]
        temporal_kernel_size_per_block_list = [9]
        # temporal_kernel_size_per_block_list = [11]
        # temporal_kernel_size_per_block_list = [13]
        # temporal_kernel_size_per_block_list = [15]
        # temporal_kernel_size_per_block_list = [23]
        # temporal_kernel_size_per_block_list = [27]
        # temporal_kernel_size_per_block_list = [29]
        # temporal_kernel_size_per_block_list = [31]
        # temporal_kernel_size_per_block_list = [35]
        # temporal_kernel_size_per_block_list = [37]
        # temporal_kernel_size_per_block_list = [41]
        # temporal_kernel_size_per_block_list = [47]
        # temporal_kernel_size_per_block_list = [51]
        # temporal_kernel_size_per_block_list = [61]

        temporal_dilation_per_block_list = [1]

        # 2 blocks
        # num_miniblocks_per_block_list = [1, 1]
        # num_miniblocks_per_block_list = [2, 1]
        # num_miniblocks_per_block_list = [3, 1]
        # num_miniblocks_per_block_list = [2, 3]
        # num_miniblocks_per_block_list = [2, 2]
        # num_features_per_block_list = [4, 4]
        # num_features_per_block_list = [6, 6]
        # num_features_per_block_list = [8, 8]
        # num_features_per_block_list = [12, 12]
        # num_features_per_block_list = [16, 16]
        # num_features_per_block_list = [24, 24]
        # num_features_per_block_list = [32, 32]
        # num_features_per_block_list = [48, 48]
        # num_features_per_block_list = [64, 64]
        # num_features_per_block_list = [128, 128]
        # temporal_kernel_size_per_block_list = [61, 61]
        # temporal_kernel_size_per_block_list = [31, 41]
        # temporal_kernel_size_per_block_list = [31, 31]
        # temporal_kernel_size_per_block_list = [41, 41]
        # temporal_kernel_size_per_block_list = [41, 61]
        # temporal_kernel_size_per_block_list = [31, 51]
        # temporal_kernel_size_per_block_list = [31, 41]
        # temporal_kernel_size_per_block_list = [21, 21]
        # temporal_kernel_size_per_block_list = [23, 23]
        # temporal_kernel_size_per_block_list = [21, 31]
        # temporal_kernel_size_per_block_list = [21, 41]
        # temporal_kernel_size_per_block_list = [17, 29]
        # temporal_kernel_size_per_block_list = [15, 25]
        # temporal_kernel_size_per_block_list = [15, 43]
        # temporal_dilation_per_block_list = [1, 1]

        # 3 blocks
        # num_miniblocks_per_block_list = [2, 2, 2]
        # num_features_per_block_list = [8, 8, 8]
        # num_features_per_block_list = [16, 16, 16]
        # num_features_per_block_list = [24, 24, 24]
        # num_features_per_block_list = [32, 32, 32]
        # num_features_per_block_list = [40, 40, 40]
        # num_features_per_block_list = [48, 48, 48]
        # num_features_per_block_list = [64, 64, 64]
        # num_features_per_block_list = [80, 80, 80]
        # num_features_per_block_list = [96, 96, 96]
        # temporal_kernel_size_per_block_list = [9, 19, 29]
        # temporal_kernel_size_per_block_list = [9, 17, 25]
        # temporal_dilation_per_block_list = [1, 1, 1]

        # norm_type = 'BatchNorm_B_only'
        # norm_type = 'BatchNorm'
        # norm_type = 'LayerNorm'
        norm_type = 'RMSNorm'

        # nonlinearity_str = 'silu'
        nonlinearity_str = 'leaky_gelu'
        # nonlinearity_str = 'leaky_silu'
        # nonlinearity_str = 'two_sided_leaky_silu'
        # nonlinearity_str = 'two_sided_leaky_gelu'
        # nonlinearity_str = 'leaky_tanh'
        # nonlinearity_str = 'leaky_sigmoid'

        # leaky_slope = 0.5
        # leaky_slope = 0.4
        # leaky_slope = 0.3
        # leaky_slope = 0.2
        leaky_slope = 0.15
        # leaky_slope = 0.1

        twin_model = SingleNeuronTwinModel_ResNetTCN(
            in_channels=in_channels,
            in_spatial_dim=in_spatial_dim,
            first_layer_temporal_kernel_size=first_layer_temporal_kernel_size,
            num_miniblocks_per_block_list=num_miniblocks_per_block_list,
            num_features_per_block_list=num_features_per_block_list,
            temporal_kernel_size_per_block_list=temporal_kernel_size_per_block_list,
            temporal_dilation_per_block_list=temporal_dilation_per_block_list,
            nonlinearity_str=nonlinearity_str,
            leaky_slope=leaky_slope,
            norm_type=norm_type,
            head_prefix_names=head_prefix_names,
            head_out_channels=head_out_channels,
            head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
            X_scale=X_scale,
            V_bias_soma=V_bias_soma,
            V_scale_soma=V_scale_soma,
            V_clip_soma_min=V_clip_soma_min,
            V_clip_soma_max=V_clip_soma_max,
            V_bias_dend=V_bias_dend,
            V_scale_dend=V_scale_dend,
            V_clip_dend_min=V_clip_dend_min,
            V_clip_dend_max=V_clip_dend_max,
            y_inst_rate_multiplier=y_inst_rate_multiplier,
        ).to(device)


    if twin_model_type == 'ELM':

        # ELM-specific parameters
        # memory_dim = 32
        # memory_dim = 64
        # memory_dim = 96
        memory_dim = 128
        # memory_dim = 256

        mlp_num_hidden_layers = 1
        # mlp_num_hidden_layers = 2

        mlp_hidden_dim = None  # Will default to 2 * memory_dim
        # mlp_hidden_dim = 128
        # mlp_hidden_dim = 256
        # mlp_hidden_dim = 384
        # mlp_hidden_dim = 512

        # mlp_nonlinearity_str = 'silu'
        # mlp_nonlinearity_str = 'leaky_silu'
        mlp_nonlinearity_str = 'leaky_gelu'

        # mlp_leaky_slope = 0.0
        # mlp_leaky_slope = 0.1
        mlp_leaky_slope = 0.2
        # mlp_leaky_slope = 0.3
        
        # mlp_pre_norm_type = 'BatchNorm'
        # mlp_pre_norm_type = 'LayerNorm'
        mlp_pre_norm_type = 'RMSNorm'
        # mlp_pre_norm_type = 'none'

        post_mlp_nonlinearity_str = 'leaky_tanh'
        # post_mlp_nonlinearity_str = 'tanh'

        # post_mlp_leaky_slope = 0.0
        # post_mlp_leaky_slope = 0.1
        post_mlp_leaky_slope = 0.2
        # post_mlp_leaky_slope = 0.3

        # lambda_value = 7.0
        # lambda_value = 6.0
        # lambda_value = 5.0
        # lambda_value = 4.0
        # lambda_value = 3.0
        lambda_value = 2.0

        # synapse_tau_value = 2.5
        synapse_tau_value = 3.0
        # synapse_tau_value = 5.0
        # synapse_tau_value = 10.0

        # memory_tau_min = 1.0
        memory_tau_min = 2.0
        # memory_tau_min = 3.0
        # memory_tau_min = 4.0

        # memory_tau_max = 32.0
        memory_tau_max = 40.0
        # memory_tau_max = 48.0
        # memory_tau_max = 64.0
        # memory_tau_max = 64.0
        # memory_tau_max = 300.0
        # memory_tau_max = 1000.0
        # memory_tau_max = 2000.0

        learn_memory_tau = True
        # learn_memory_tau = False

        w_s_value = 0.5
        # w_s_value = 1.0

        delta_t = 1.0

        # compile_recurrent_step = True
        compile_recurrent_step = False

        twin_model = SingleNeuronTwinModel_ELM(
            in_channels=in_channels,
            in_spatial_dim=in_spatial_dim,
            memory_dim=memory_dim,
            mlp_num_hidden_layers=mlp_num_hidden_layers,
            mlp_hidden_dim=mlp_hidden_dim,
            mlp_nonlinearity_str=mlp_nonlinearity_str,
            mlp_leaky_slope=mlp_leaky_slope,
            mlp_pre_norm_type=mlp_pre_norm_type,
            post_mlp_nonlinearity_str=post_mlp_nonlinearity_str,
            post_mlp_leaky_slope=post_mlp_leaky_slope,
            lambda_value=lambda_value,
            synapse_tau_value=synapse_tau_value,
            memory_tau_min=memory_tau_min,
            memory_tau_max=memory_tau_max,
            learn_memory_tau=learn_memory_tau,
            w_s_value=w_s_value,
            delta_t=delta_t,
            compile_recurrent_step=compile_recurrent_step,
            head_prefix_names=head_prefix_names,
            head_out_channels=head_out_channels,
            head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
            X_scale=X_scale,
            V_bias_soma=V_bias_soma,
            V_scale_soma=V_scale_soma,
            V_clip_soma_min=V_clip_soma_min,
            V_clip_soma_max=V_clip_soma_max,
            V_bias_dend=V_bias_dend,
            V_scale_dend=V_scale_dend,
            V_clip_dend_min=V_clip_dend_min,
            V_clip_dend_max=V_clip_dend_max,
            y_inst_rate_multiplier=y_inst_rate_multiplier,
        ).to(device)


    if twin_model_type == 'Transformer':

        # First layer parameters
        first_layer_temporal_kernel_size = 1
        # first_layer_temporal_kernel_size = 3
        # first_layer_temporal_kernel_size = 5
        # first_layer_temporal_kernel_size = 7
        # first_layer_temporal_kernel_size = 9
        # first_layer_temporal_kernel_size = 11

        # first_norm_type = 'BatchNorm'
        # first_norm_type = 'LayerNorm'
        first_norm_type = 'RMSNorm'

        # first_nonlinearity_str = 'silu'
        # first_nonlinearity_str = 'leaky_silu'
        first_nonlinearity_str = 'leaky_gelu'
        # first_nonlinearity_str = 'leaky_tanh'

        # first_leaky_slope = 0.1
        # first_leaky_slope = 0.15
        first_leaky_slope = 0.2
        # first_leaky_slope = 0.3

        # Transformer architecture parameters
        # d_model = 12
        # d_model = 16
        # d_model = 24
        # d_model = 32
        # d_model = 48
        d_model = 64
        # d_model = 96
        # d_model = 128
        # d_model = 192
        # d_model = 256

        # n_heads = 2
        # n_heads = 4
        # n_heads = 6
        n_heads = 8
        # n_heads = 16

        # num_layers = 2
        # num_layers = 3
        # num_layers = 4
        # num_layers = 5
        num_layers = 6
        # num_layers = 6
        # num_layers = 8

        # window_size = 16
        # window_size = 24
        # window_size = 32
        # window_size = 40
        window_size = 48
        # window_size = 64
        # window_size = 96
        # window_size = 128

        max_seq_len = 8192
        # max_seq_len = 16384

        dropout_rate = 0.0
        # dropout_rate = 0.1
        # dropout_rate = 0.2

        # ffn_type = 'gelu'
        ffn_type = 'swiglu'

        # ffn_hidden_dim = None  # Will default to 4 * d_model for gelu, or 8/3 * d_model for swiglu
        # ffn_hidden_dim = 2 * d_model
        # ffn_hidden_dim = int(2.5 * d_model)
        ffn_hidden_dim = 3 * d_model
        # ffn_hidden_dim = int(3.5 * d_model)
        # ffn_hidden_dim = 4 * d_model

        SWA_type = 'blocked'
        # SWA_type = 'full'

        # SWA_block_size = None
        SWA_block_size = 64
        # SWA_block_size = int(1.25 * window_size)
        # SWA_block_size = int(1.5 * window_size)
        # SWA_block_size = 2 * window_size
        # SWA_block_size = 4 * window_size
        # SWA_block_size = 6 * window_size
        # SWA_block_size = 8 * window_size

        use_complex_number_rope = False
        # use_complex_number_rope = True

        # norm_type = 'layer_norm'
        norm_type = 'rms_norm'

        twin_model = SingleNeuronTwinModel_Transformer(
            in_channels=in_channels,
            in_spatial_dim=in_spatial_dim,
            first_layer_temporal_kernel_size=first_layer_temporal_kernel_size,
            first_norm_type=first_norm_type,
            first_nonlinearity_str=first_nonlinearity_str,
            first_leaky_slope=first_leaky_slope,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
            window_size=window_size,
            max_seq_len=max_seq_len,
            dropout_rate=dropout_rate,
            ffn_type=ffn_type,
            ffn_hidden_dim=ffn_hidden_dim,
            SWA_type=SWA_type,
            SWA_block_size=SWA_block_size,
            use_complex_number_rope=use_complex_number_rope,
            norm_type=norm_type,
            head_prefix_names=head_prefix_names,
            head_out_channels=head_out_channels,
            head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
            X_scale=X_scale,
            V_bias_soma=V_bias_soma,
            V_scale_soma=V_scale_soma,
            V_clip_soma_min=V_clip_soma_min,
            V_clip_soma_max=V_clip_soma_max,
            V_bias_dend=V_bias_dend,
            V_scale_dend=V_scale_dend,
            V_clip_dend_min=V_clip_dend_min,
            V_clip_dend_max=V_clip_dend_max,
            y_inst_rate_multiplier=y_inst_rate_multiplier,
        ).to(device)

    print('------------------------------------------------------------------------------------------')
    print(f'full model: "{twin_model.short_name}"')
    print('------------------------------------------------------------------------------------------')
    y_spikes_hat, y_soma_hat, y_near_spike_hat, y_inst_rate_hat, y_dend_v_hat = twin_model.forward_debug(X_spikes.to(device))
    print('------------------------------------------------------------------------------------------')

    # store the original neuron model metadata in the model metadata
    twin_model.set_metadata_original_neuron(original_neuron_model_metadata)
    twin_model.print_main_metadata()

    # Compile the model for better performance
    if compile_model:
        print('Compiling model with torch.compile...')
        twin_model = torch.compile(twin_model)

    #%% Training loop

    # Set all the remaining training parameters
    # num_epochs = 800
    # num_epochs = 600
    # num_epochs = 480
    # num_epochs = 360
    # num_epochs = 240
    # num_epochs = 200
    # num_epochs = 180
    num_epochs = 150
    # num_epochs = 120
    # num_epochs = 100
    # num_epochs = 80
    # num_epochs = 70
    # num_epochs = 60
    # num_epochs = 50
    # num_epochs = 40
    # num_epochs = 30
    # num_epochs = 20
    # num_epochs = 15
    # num_epochs = 10

    # learning_rate = 0.00300
    # learning_rate = 0.00200
    # learning_rate = 0.00100
    # learning_rate = 0.00080
    # learning_rate = 0.00070
    # learning_rate = 0.00060
    # learning_rate = 0.00050
    # learning_rate = 0.00040
    learning_rate = 0.00030
    # learning_rate = 0.00020
    # learning_rate = 0.00010
    # learning_rate = 0.00005
    # learning_rate = 0.00001

    # Gradient clipping
    # max_grad_norm = 40.0
    # max_grad_norm = 35.0
    # max_grad_norm = 30.0
    max_grad_norm = 25.0
    # max_grad_norm = 20.0
    # max_grad_norm = 15.0
    # max_grad_norm = 10.0
    # max_grad_norm = 5.0
    # max_grad_norm = 2.0
    # max_grad_norm = 1.0
    # max_grad_norm = 0.5
    # max_grad_norm = 0.1

    # weight_decay = 0.5000
    # weight_decay = 0.3000
    # weight_decay = 0.2000
    # weight_decay = 0.1500
    # weight_decay = 0.1000
    # weight_decay = 0.0500
    # weight_decay = 0.0300
    # weight_decay = 0.0200
    # weight_decay = 0.0100
    # weight_decay = 0.0050
    weight_decay = 0.0030
    # weight_decay = 0.0010
    # weight_decay = 0.0001
    # weight_decay = 0.0000

    # learning rate scheduler parameters
    num_warmup_epochs = min(int(0.10 * num_epochs), 40)
    num_cooldown_epochs = int(0.25 * num_epochs)
    warmup_lr_start_end_factors = (0.001, 0.6)
    cooldown_lr_start_end_factors = (0.5, 0.0005)

    num_warmup_epochs = max(min(num_warmup_epochs, num_epochs // 4), 2)
    num_cooldown_epochs = max(min(num_cooldown_epochs, num_epochs // 3), 3)

    # initial loss weights for the different heads
    spikes_loss_weight_init = 2.0
    soma_loss_weight_init = 0.700
    near_spike_loss_weight_init = 1.800
    inst_rate_loss_weight_init = 3.0
    dend_v_loss_weight_init = 6.0

    # final loss weights for the different heads
    if weight_decay < 0.1:
        spikes_loss_weight_final = 24.0
        soma_loss_weight_final = 0.150
        near_spike_loss_weight_final = 0.250
        inst_rate_loss_weight_final = 3.0
        dend_v_loss_weight_final = 0.6
    else:
        spikes_loss_weight_final = 24.0
        soma_loss_weight_final = 0.165
        near_spike_loss_weight_final = 0.300
        inst_rate_loss_weight_final = 3.5
        dend_v_loss_weight_final = 0.8

    loss_weight_transition_start_epoch = 2.5 * num_warmup_epochs
    loss_weight_transition_end_epoch = 4.0 * num_warmup_epochs

    assert num_warmup_epochs + num_cooldown_epochs < num_epochs, 'num_warmup_epochs + num_cooldown_epochs must be less than num_epochs'

    # loss weight "scheduler" (determines the final loss weight fraction as a function of the epoch number)
    def get_final_loss_weight_fraction(epoch, transition_start_epoch, transition_end_epoch):
        if epoch < transition_start_epoch:
            return 0.0
        elif epoch >= transition_end_epoch:
            return 1.0
        else:
            return (epoch - transition_start_epoch) / (transition_end_epoch - transition_start_epoch)

    # learning rate scheduler (determines the multiplicative factor for the learning rate as a function of the epoch number)
    def get_lr_factor(epoch, num_warmup_epochs, num_cooldown_epochs, warmup_lr_start_end_factors, cooldown_lr_start_end_factors):
        if epoch < num_warmup_epochs:
            progress = epoch / (num_warmup_epochs - 1)
            return warmup_lr_start_end_factors[0] + (warmup_lr_start_end_factors[1] - warmup_lr_start_end_factors[0]) * progress
        elif epoch >= num_epochs - num_cooldown_epochs:
            progress = (epoch - (num_epochs - num_cooldown_epochs)) / (num_cooldown_epochs - 1)
            return cooldown_lr_start_end_factors[0] + (cooldown_lr_start_end_factors[1] - cooldown_lr_start_end_factors[0]) * progress
        else:
            return 1.0

    num_batches_per_valid_eval = int(0.95 * len(train_dataloader))

    training_params_metadata = {
        'train_time_window_size': train_time_window_size,
        'valid_time_window_size': valid_time_window_size,
        'train_batch_size': train_batch_size,
        'valid_batch_size': valid_batch_size,

        'spikes_loss_weight_init': spikes_loss_weight_init,
        'soma_loss_weight_init': soma_loss_weight_init,
        'near_spike_loss_weight_init': near_spike_loss_weight_init,
        'inst_rate_loss_weight_init': inst_rate_loss_weight_init,
        'dend_v_loss_weight_init': dend_v_loss_weight_init,

        'spikes_loss_weight_final': spikes_loss_weight_final,
        'soma_loss_weight_final': soma_loss_weight_final,
        'near_spike_loss_weight_final': near_spike_loss_weight_final,
        'inst_rate_loss_weight_final': inst_rate_loss_weight_final,
        'dend_v_loss_weight_final': dend_v_loss_weight_final,

        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'max_grad_norm': max_grad_norm,
        'num_warmup_epochs': num_warmup_epochs,
        'num_cooldown_epochs': num_cooldown_epochs,
        'warmup_lr_start_end_factors': warmup_lr_start_end_factors,
        'cooldown_lr_start_end_factors': cooldown_lr_start_end_factors,
        'num_batches_per_valid_eval': num_batches_per_valid_eval,
        'loss_weight_transition_start_epoch': loss_weight_transition_start_epoch,
        'loss_weight_transition_end_epoch': loss_weight_transition_end_epoch,
    }

    # store the training parameters in the model metadata
    twin_model.set_metadata_training_params(training_params_metadata)
    twin_model.print_main_metadata()

    # Lists to store metrics
    # check if variable train_iter_list is already defined
    if 'train_iter_list' not in locals():
        learning_rate_list = []

        train_iter_list = []
        train_losses_spikes = []
        train_losses_soma = []
        train_losses_near_spike = []
        train_losses_inst_rate = []
        train_losses_dend_v = []
        train_losses_total = []

        train_losses_weights_spikes = []
        train_losses_weights_soma = []
        train_losses_weights_near_spike = []
        train_losses_weights_inst_rate = []
        train_losses_weights_dend_v = []

        train_grad_norms = []  # Track gradient norms

        valid_iter_list = []
        valid_losses_spikes = []
        valid_losses_soma = []
        valid_losses_near_spike = []
        valid_losses_inst_rate = []
        valid_losses_dend_v = []
        valid_losses_total = []

        valid_losses_weights_spikes = []
        valid_losses_weights_soma = []
        valid_losses_weights_near_spike = []
        valid_losses_weights_inst_rate = []
        valid_losses_weights_dend_v = []

        iter_num = 0

    # Loss functions and optimizer
    criterion_spikes = nn.BCEWithLogitsLoss()
    criterion_soma = nn.L1Loss()
    criterion_near_spike = nn.BCEWithLogitsLoss()
    criterion_inst_rate = nn.MSELoss()
    criterion_dend_v = nn.L1Loss()
    optimizer = optim.AdamW(twin_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Best checkpoint tracking
    twin_model_best_checkpoint = None
    best_valid_loss_spikes = float('inf')
    best_valid_iter = -1
    best_checkpoint_losses = {}

    training_start_time = time.time()

    for epoch in range(num_epochs):
        twin_model.train()

        # Set learning rate for the epoch
        current_learning_rate = learning_rate * get_lr_factor(epoch, num_warmup_epochs, num_cooldown_epochs, 
                                                              warmup_lr_start_end_factors, cooldown_lr_start_end_factors)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_learning_rate
        learning_rate_list.append(current_learning_rate)

        final_loss_weight_fraction = get_final_loss_weight_fraction(epoch, loss_weight_transition_start_epoch, loss_weight_transition_end_epoch)
        curr_spikes_loss_weight = spikes_loss_weight_init + (spikes_loss_weight_final - spikes_loss_weight_init) * final_loss_weight_fraction
        curr_soma_loss_weight = soma_loss_weight_init + (soma_loss_weight_final - soma_loss_weight_init) * final_loss_weight_fraction
        curr_near_spike_loss_weight = near_spike_loss_weight_init + (near_spike_loss_weight_final - near_spike_loss_weight_init) * final_loss_weight_fraction
        curr_inst_rate_loss_weight = inst_rate_loss_weight_init + (inst_rate_loss_weight_final - inst_rate_loss_weight_init) * final_loss_weight_fraction
        curr_dend_v_loss_weight = dend_v_loss_weight_init + (dend_v_loss_weight_final - dend_v_loss_weight_init) * final_loss_weight_fraction

        print(f'Epoch {epoch + 1}/{num_epochs}, learning rate ={round(current_learning_rate, 7)}, loss w Frac ={final_loss_weight_fraction:.2f}  | Losses (spikes, soma, near_sp, inst_rate, dend_v)')

        progress_bar = tqdm(train_dataloader, desc=f'Training')
        for batch_idx, batch_data in enumerate(progress_bar):
            # Unpack dictionary
            X_spikes = batch_data['X_spikes'].to(device)
            y_spikes_GT = batch_data['y_spikes'].to(device)
            y_soma_GT = batch_data['y_soma'].to(device)
            y_near_spike_GT = batch_data['y_near_spike'].to(device)
            y_inst_rate_GT = batch_data['y_inst_rate'].to(device)
            y_dend_v_GT = batch_data['y_dend_v'].to(device)

            # apply the various scaling and clipping operations to the GT data that are needed for the model for proper training
            X_spikes = X_spikes / twin_model.X_scale
            y_soma_GT = torch.clamp(y_soma_GT, twin_model.V_clip_soma_min, twin_model.V_clip_soma_max)
            y_soma_GT = (y_soma_GT - twin_model.V_bias_soma) / twin_model.V_scale_soma
            y_inst_rate_GT = y_inst_rate_GT * twin_model.y_inst_rate_multiplier
            y_dend_v_GT = torch.clamp(y_dend_v_GT, twin_model.V_clip_dend_min, twin_model.V_clip_dend_max)
            y_dend_v_GT = (y_dend_v_GT - twin_model.V_bias_dend) / twin_model.V_scale_dend

            # Handle dendritic voltage GT dimensions to match predictions
            if y_dend_v_GT.dim() == 3 and y_dend_v_GT.shape[1] == 1:  # [batch, 1, temporal]
                y_dend_v_GT = y_dend_v_GT.squeeze(1)  # Remove channel dimension: [batch, temporal]

            # Forward pass
            y_spikes_pred, y_soma_pred, y_near_spike_pred, y_inst_rate_pred, y_dend_v_pred = twin_model(X_spikes)
            
            # Reshape predictions - use targeted squeezing to preserve batch dimension
            y_spikes_pred = y_spikes_pred.squeeze(1).squeeze(1)          # Remove channel and spatial dims but keep batch
            y_soma_pred = y_soma_pred.squeeze(1).squeeze(1)              # Remove channel and spatial dims but keep batch
            y_near_spike_pred = y_near_spike_pred.squeeze(1).squeeze(1)  # Remove channel and spatial dims but keep batch
            y_inst_rate_pred = y_inst_rate_pred.squeeze(1).squeeze(1)    # Remove channel and spatial dims but keep batch

            # Handle dendritic voltage dimensions carefully
            if y_dend_v_pred.dim() == 4:  # [batch, channels, spatial, temporal]
                y_dend_v_pred = y_dend_v_pred.squeeze(2)  # Remove spatial dimension: [batch, channels, temporal]
                if y_dend_v_pred.shape[1] == 1:  # [batch, 1, temporal]
                    y_dend_v_pred = y_dend_v_pred.squeeze(1)  # Remove channel dimension: [batch, temporal]
            elif y_dend_v_pred.dim() == 3 and y_dend_v_pred.shape[1] == 1:  # [batch, 1, temporal]
                y_dend_v_pred = y_dend_v_pred.squeeze(1)  # Remove channel dimension: [batch, temporal]

            # Compute losses
            loss_spikes = criterion_spikes(y_spikes_pred, y_spikes_GT)
            loss_soma = criterion_soma(y_soma_pred, y_soma_GT)
            loss_near_spike = criterion_near_spike(y_near_spike_pred, y_near_spike_GT)
            loss_inst_rate = criterion_inst_rate(y_inst_rate_pred, y_inst_rate_GT)
            loss_dend_v = criterion_dend_v(y_dend_v_pred, y_dend_v_GT)

            loss = curr_spikes_loss_weight * loss_spikes + curr_soma_loss_weight * loss_soma
            loss += curr_near_spike_loss_weight * loss_near_spike + curr_inst_rate_loss_weight * loss_inst_rate
            loss += curr_dend_v_loss_weight * loss_dend_v
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            pre_clip_grad_norm = torch.nn.utils.clip_grad_norm_(twin_model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            
            # Store training metrics
            train_iter_list.append(iter_num)
            train_losses_spikes.append(loss_spikes.item())
            train_losses_soma.append(loss_soma.item())
            train_losses_near_spike.append(loss_near_spike.item())
            train_losses_inst_rate.append(loss_inst_rate.item())
            train_losses_dend_v.append(loss_dend_v.item())
            train_losses_total.append(loss.item())
            train_grad_norms.append(pre_clip_grad_norm.item())

            train_losses_weights_spikes.append(curr_spikes_loss_weight)
            train_losses_weights_soma.append(curr_soma_loss_weight)
            train_losses_weights_near_spike.append(curr_near_spike_loss_weight)
            train_losses_weights_inst_rate.append(curr_inst_rate_loss_weight)
            train_losses_weights_dend_v.append(curr_dend_v_loss_weight)
            
            # Update progress bar
            train_progress_bar_str = f'({loss_spikes.item():.4f}, {loss_soma.item():.4f}, {loss_near_spike.item():.4f}, {loss_inst_rate.item():.4f}, {loss_dend_v.item():.4f}) | grad_norm: {pre_clip_grad_norm:.3f}'
            try:
                valid_progress_bar_str = f'({valid_losses_spikes[-1]:.4f}, {valid_losses_soma[-1]:.4f}, {valid_losses_near_spike[-1]:.4f}, {valid_losses_inst_rate[-1]:.4f}, {valid_losses_dend_v[-1]:.4f})'
            except:
                pass
            progress_bar.set_postfix({
                'train': train_progress_bar_str,
                'val': 'N/A' if not valid_losses_total else valid_progress_bar_str,
            })

            # Periodic validation
            if (iter_num + 1) % num_batches_per_valid_eval == 0:
                twin_model.eval()
                valid_losses_spikes_batch = []
                valid_losses_soma_batch = []
                valid_losses_near_spike_batch = []
                valid_losses_inst_rate_batch = []
                valid_losses_dend_v_batch = []
                valid_losses_total_batch = []
                
                with torch.no_grad():
                    for batch_data in valid_dataloader:
                        # Unpack dictionary
                        X_spikes = batch_data['X_spikes'].to(device)
                        y_spikes_GT = batch_data['y_spikes'].to(device)
                        y_soma_GT = batch_data['y_soma'].to(device)
                        y_near_spike_GT = batch_data['y_near_spike'].to(device)
                        y_inst_rate_GT = batch_data['y_inst_rate'].to(device)
                        y_dend_v_GT = batch_data['y_dend_v'].to(device)

                        # apply the various scaling and clipping operations to the GT data that are needed for the model for proper evaluation
                        X_spikes = X_spikes / twin_model.X_scale
                        y_soma_GT = torch.clamp(y_soma_GT, twin_model.V_clip_soma_min, twin_model.V_clip_soma_max)
                        y_soma_GT = (y_soma_GT - twin_model.V_bias_soma) / twin_model.V_scale_soma
                        y_inst_rate_GT = y_inst_rate_GT * twin_model.y_inst_rate_multiplier
                        y_dend_v_GT = torch.clamp(y_dend_v_GT, twin_model.V_clip_dend_min, twin_model.V_clip_dend_max)
                        y_dend_v_GT = (y_dend_v_GT - twin_model.V_bias_dend) / twin_model.V_scale_dend

                        # Handle dendritic voltage GT dimensions to match predictions
                        if y_dend_v_GT.dim() == 3 and y_dend_v_GT.shape[1] == 1:  # [batch, 1, temporal]
                            y_dend_v_GT = y_dend_v_GT.squeeze(1)  # Remove channel dimension: [batch, temporal]
                                                
                        y_spikes_pred, y_soma_pred, y_near_spike_pred, y_inst_rate_pred, y_dend_v_pred = twin_model(X_spikes)
                        
                        # Use targeted squeezing to preserve batch dimension
                        y_spikes_pred = y_spikes_pred.squeeze(1).squeeze(1)          # Remove channel and spatial dims but keep batch
                        y_soma_pred = y_soma_pred.squeeze(1).squeeze(1)              # Remove channel and spatial dims but keep batch
                        y_near_spike_pred = y_near_spike_pred.squeeze(1).squeeze(1)  # Remove channel and spatial dims but keep batch
                        y_inst_rate_pred = y_inst_rate_pred.squeeze(1).squeeze(1)    # Remove channel and spatial dims but keep batch

                        # Handle dendritic voltage dimensions carefully
                        if y_dend_v_pred.dim() == 4:  # [batch, channels, spatial, temporal]
                            y_dend_v_pred = y_dend_v_pred.squeeze(2)  # Remove spatial dimension: [batch, channels, temporal]
                            if y_dend_v_pred.shape[1] == 1:  # [batch, 1, temporal]
                                y_dend_v_pred = y_dend_v_pred.squeeze(1)  # Remove channel dimension: [batch, temporal]
                        elif y_dend_v_pred.dim() == 3 and y_dend_v_pred.shape[1] == 1:  # [batch, 1, temporal]
                            y_dend_v_pred = y_dend_v_pred.squeeze(1)  # Remove channel dimension: [batch, temporal]
                        
                        loss_spikes = criterion_spikes(y_spikes_pred, y_spikes_GT)
                        loss_soma = criterion_soma(y_soma_pred, y_soma_GT)
                        loss_near_spike = criterion_near_spike(y_near_spike_pred, y_near_spike_GT)
                        loss_inst_rate = criterion_inst_rate(y_inst_rate_pred, y_inst_rate_GT)
                        loss_dend_v = criterion_dend_v(y_dend_v_pred, y_dend_v_GT)
                        
                        loss = curr_spikes_loss_weight * loss_spikes + curr_soma_loss_weight * loss_soma
                        loss += curr_near_spike_loss_weight * loss_near_spike + curr_inst_rate_loss_weight * loss_inst_rate
                        loss += curr_dend_v_loss_weight * loss_dend_v
                        
                        valid_losses_spikes_batch.append(loss_spikes.item())
                        valid_losses_soma_batch.append(loss_soma.item())
                        valid_losses_near_spike_batch.append(loss_near_spike.item())
                        valid_losses_inst_rate_batch.append(loss_inst_rate.item())
                        valid_losses_dend_v_batch.append(loss_dend_v.item())
                        valid_losses_total_batch.append(loss.item())
                
                # Store validation metrics
                valid_iter_list.append(iter_num)
                valid_losses_spikes.append(np.mean(valid_losses_spikes_batch))
                valid_losses_soma.append(np.mean(valid_losses_soma_batch))
                valid_losses_near_spike.append(np.mean(valid_losses_near_spike_batch))
                valid_losses_inst_rate.append(np.mean(valid_losses_inst_rate_batch))
                valid_losses_dend_v.append(np.mean(valid_losses_dend_v_batch))
                valid_losses_total.append(np.mean(valid_losses_total_batch))

                valid_losses_weights_spikes.append(curr_spikes_loss_weight)
                valid_losses_weights_soma.append(curr_soma_loss_weight)
                valid_losses_weights_near_spike.append(curr_near_spike_loss_weight)
                valid_losses_weights_inst_rate.append(curr_inst_rate_loss_weight)
                valid_losses_weights_dend_v.append(curr_dend_v_loss_weight)

                # Check if this is the best checkpoint based on validation spike loss
                current_valid_loss_spikes = valid_losses_spikes[-1]
                if current_valid_loss_spikes < best_valid_loss_spikes:
                    best_valid_loss_spikes = current_valid_loss_spikes + 0.0001 # slight preference for the longer trained model
                    best_valid_iter = iter_num
                    
                    # Store best checkpoint losses
                    best_checkpoint_losses = {
                        'valid_loss_spikes': current_valid_loss_spikes,
                        'valid_loss_soma': valid_losses_soma[-1],
                        'valid_loss_near_spike': valid_losses_near_spike[-1],
                        'valid_loss_inst_rate': valid_losses_inst_rate[-1],
                        'valid_loss_dend_v': valid_losses_dend_v[-1],
                        'valid_loss_total': valid_losses_total[-1],
                        'train_loss_spikes': train_losses_spikes[-1],
                        'train_loss_soma': train_losses_soma[-1],
                        'train_loss_near_spike': train_losses_near_spike[-1],
                        'train_loss_inst_rate': train_losses_inst_rate[-1],
                        'train_loss_dend_v': train_losses_dend_v[-1],
                        'train_loss_total': train_losses_total[-1],
                    }
                    
                    # Save deep copy of the best model state
                    twin_model_best_checkpoint = copy.deepcopy(twin_model)

                twin_model.train()
            
            iter_num += 1

    training_duration_minutes = (time.time() - training_start_time) / 60
    
    # store last checkpoint in a seperate variable
    twin_model_last_checkpoint = copy.deepcopy(twin_model)

    print('---------------------------------------------------------------')
    # Restore best checkpoint if available
    if twin_model_best_checkpoint is not None:
        print('---------------------------------------------------------------')
        print(f'Restoring best checkpoint from iteration {best_valid_iter} with validation spike loss: {best_valid_loss_spikes:.5f}')
        twin_model.load_state_dict(twin_model_best_checkpoint.state_dict())
        print('---------------------------------------------------------------')
    
    print('---------------------------------------------------------------')
    print(f'Training finished! training duration: {training_duration_minutes:.2f} minutes')
    print('---------------------------------------------------------------')
    print(f'Final Checkpoint spikes loss (train, valid)     : ({np.mean(train_losses_spikes[-50:]):.5f}, {valid_losses_spikes[-1]:.5f})')
    print(f'Final Checkpoint soma V loss (train, valid)     : ({np.mean(train_losses_soma[-50:]):.5f}, {valid_losses_soma[-1]:.5f})')
    print(f'Final Checkpoint inst rate loss (train, valid)  : ({np.mean(train_losses_inst_rate[-50:]):.5f}, {valid_losses_inst_rate[-1]:.5f})')
    print(f'Final Checkpoint near spike loss (train, valid) : ({np.mean(train_losses_near_spike[-50:]):.5f}, {valid_losses_near_spike[-1]:.5f})')
    print(f'Final Checkpoint dend V loss (train, valid)     : ({np.mean(train_losses_dend_v[-50:]):.5f}, {valid_losses_dend_v[-1]:.5f})')
    print('---------------------------------------------------------------')
    print(f'Final gradient norm (train)           : {np.mean(train_grad_norms[-50:]):.5f} (last 50 iters), max: {np.max(train_grad_norms):.3f}, clip: {max_grad_norm}')
    print('---------------------------------------------------------------')
    print(f'Final Checkpoint total loss (train, valid)      : ({np.mean(train_losses_total[-50:]):.5f}, {valid_losses_total[-1]:.5f})')
    print('---------------------------------------------------------------')
    
    # Print best checkpoint losses if available
    if best_checkpoint_losses:
        print('Best checkpoint losses:')
        print('---------------------------------------------------------------')
        print(f'Best Checkpoint spikes loss (train, valid)      : ({best_checkpoint_losses["train_loss_spikes"]:.5f}, {best_checkpoint_losses["valid_loss_spikes"]:.5f}) at iter {best_valid_iter}')
        print(f'Best Checkpoint soma V loss (train, valid)      : ({best_checkpoint_losses["train_loss_soma"]:.5f}, {best_checkpoint_losses["valid_loss_soma"]:.5f}) at iter {best_valid_iter}')
        print(f'Best Checkpoint inst rate loss (train, valid)   : ({best_checkpoint_losses["train_loss_inst_rate"]:.5f}, {best_checkpoint_losses["valid_loss_inst_rate"]:.5f}) at iter {best_valid_iter}')
        print(f'Best Checkpoint near spike loss (train, valid)  : ({best_checkpoint_losses["train_loss_near_spike"]:.5f}, {best_checkpoint_losses["valid_loss_near_spike"]:.5f}) at iter {best_valid_iter}')
        print(f'Best Checkpoint dend V loss (train, valid)      : ({best_checkpoint_losses["train_loss_dend_v"]:.5f}, {best_checkpoint_losses["valid_loss_dend_v"]:.5f}) at iter {best_valid_iter}')
        print('---------------------------------------------------------------')
        print(f'Best Checkpoint total loss (train, valid)       : ({best_checkpoint_losses["train_loss_total"]:.5f}, {best_checkpoint_losses["valid_loss_total"]:.5f}) at iter {best_valid_iter}')
        print('---------------------------------------------------------------')

    #%% manual intervention point if I want to use a specific checkpoint

    # checkpoint_to_use = 'last'
    # checkpoint_to_use = 'best'
    checkpoint_to_use = 'skip'

    if checkpoint_to_use == 'last':
        twin_model.load_state_dict(twin_model_last_checkpoint.state_dict())
        print(f'Loaded last checkpoint from iteration {iter_num}')

    if checkpoint_to_use == 'best':
        twin_model.load_state_dict(twin_model_best_checkpoint.state_dict())
        print(f'Loaded best checkpoint from iteration {best_valid_iter}')

    #%% Plot learning curves

    # y_scale_list = ['linear', 'log', 'log']
    # x_scale_list = ['linear', 'linear', 'log']
    y_scale_list = ['log', 'log']
    x_scale_list = ['linear', 'log']

    for y_scale, x_scale in zip(y_scale_list, x_scale_list):
        
        plt.figure(figsize=(10, 16))  # Figure for 7 subplots
        num_subplots = 7

        # Spikes loss
        plt.subplot(num_subplots, 1, 1)
        plt.plot(train_iter_list, train_losses_spikes, label='Train')
        plt.plot(valid_iter_list, valid_losses_spikes, label='Valid')
        if best_valid_iter >= 0 and best_checkpoint_losses:
            plt.scatter(best_valid_iter, best_checkpoint_losses['valid_loss_spikes'], 
                       color='red', s=50, marker='o', zorder=5, label='Best Checkpoint')
        plt.title('Spikes Loss')
        plt.legend(fontsize=12)
        plt.yscale(y_scale)
        plt.xscale(x_scale)
        plt.grid(True)

        if x_scale == 'log':
            plt.xlim([30, max(train_iter_list)])
        else:
            plt.xlim([0, max(train_iter_list)])

        if y_scale == 'log':
            avg_loss = np.mean(train_losses_spikes[30:50])
            min_loss = 0.8 * np.min(train_losses_spikes + valid_losses_spikes)
            plt.ylim([min_loss, 1.2 * avg_loss])

        # Soma loss
        plt.subplot(num_subplots, 1, 2)
        plt.plot(train_iter_list, train_losses_soma, label='Train')
        plt.plot(valid_iter_list, valid_losses_soma, label='Valid')
        if best_valid_iter >= 0 and best_checkpoint_losses:
            plt.scatter(best_valid_iter, best_checkpoint_losses['valid_loss_soma'], 
                       color='red', s=50, marker='o', zorder=5, label='Best Checkpoint')
        plt.title('Soma Loss')
        plt.legend()
        plt.yscale(y_scale)
        plt.xscale(x_scale)
        plt.grid(True)

        if x_scale == 'log':
            plt.xlim([30, max(train_iter_list)])
        else:
            plt.xlim([0, max(train_iter_list)])

        if y_scale == 'log':
            avg_loss = np.mean(train_losses_soma[30:50])
            min_loss = 0.8 * np.min(train_losses_soma + valid_losses_soma)
            plt.ylim([min_loss, 1.2 * avg_loss])

        # Near spike loss
        plt.subplot(num_subplots, 1, 3)
        plt.plot(train_iter_list, train_losses_near_spike, label='Train')
        plt.plot(valid_iter_list, valid_losses_near_spike, label='Valid')
        if best_valid_iter >= 0 and best_checkpoint_losses:
            plt.scatter(best_valid_iter, best_checkpoint_losses['valid_loss_near_spike'], 
                       color='red', s=50, marker='o', zorder=5, label='Best Checkpoint')
        plt.title('Near Spike Loss')
        plt.legend()
        plt.yscale(y_scale)
        plt.xscale(x_scale)
        plt.grid(True)

        if x_scale == 'log':
            plt.xlim([30, max(train_iter_list)])
        else:
            plt.xlim([0, max(train_iter_list)])

        if y_scale == 'log':
            avg_loss = np.mean(train_losses_near_spike[30:50])
            min_loss = 0.8 * np.min(train_losses_near_spike + valid_losses_near_spike)
            plt.ylim([min_loss, 1.2 * avg_loss])

        # Instantaneous rate loss
        plt.subplot(num_subplots, 1, 4)
        plt.plot(train_iter_list, train_losses_inst_rate, label='Train')
        plt.plot(valid_iter_list, valid_losses_inst_rate, label='Valid')
        if best_valid_iter >= 0 and best_checkpoint_losses:
            plt.scatter(best_valid_iter, best_checkpoint_losses['valid_loss_inst_rate'], 
                       color='red', s=50, marker='o', zorder=5, label='Best Checkpoint')
        plt.title('Inst Rate Loss')
        plt.legend()
        plt.yscale(y_scale)
        plt.xscale(x_scale)
        plt.grid(True)

        if x_scale == 'log':
            plt.xlim([30, max(train_iter_list)])
        else:
            plt.xlim([0, max(train_iter_list)])

        if y_scale == 'log':
            avg_loss = np.mean(train_losses_inst_rate[30:50])
            min_loss = 0.8 * np.min(train_losses_inst_rate + valid_losses_inst_rate)
            plt.ylim([min_loss, 1.2 * avg_loss])

        # Dendritic voltage loss
        plt.subplot(num_subplots, 1, 5)
        plt.plot(train_iter_list, train_losses_dend_v, label='Train')
        plt.plot(valid_iter_list, valid_losses_dend_v, label='Valid')
        if best_valid_iter >= 0 and best_checkpoint_losses:
            plt.scatter(best_valid_iter, best_checkpoint_losses['valid_loss_dend_v'], 
                       color='red', s=50, marker='o', zorder=5, label='Best Checkpoint')
        plt.title('Dend V Loss')
        plt.legend()
        plt.yscale(y_scale)
        plt.xscale(x_scale)
        plt.grid(True)

        if x_scale == 'log':
            plt.xlim([30, max(train_iter_list)])
        else:
            plt.xlim([0, max(train_iter_list)])

        if y_scale == 'log':
            avg_loss = np.mean(train_losses_dend_v[30:50])
            min_loss = 0.8 * np.min(train_losses_dend_v + valid_losses_dend_v)
            plt.ylim([min_loss, 1.2 * avg_loss])

        # Gradient norm
        plt.subplot(num_subplots, 1, 6)
        plt.plot(train_iter_list, train_grad_norms, label='Train Grad Norm')
        if max_grad_norm is not None:
            plt.axhline(y=max_grad_norm, color='red', linestyle='--', alpha=0.7, 
                       label=f'Clip Threshold ({max_grad_norm})')
        plt.title('Gradient Norm')
        plt.legend()
        plt.yscale(y_scale)
        plt.xscale(x_scale)
        plt.grid(True)

        if x_scale == 'log':
            plt.xlim([30, max(train_iter_list)])
        else:
            plt.xlim([0, max(train_iter_list)])

        if y_scale == 'log':
            min_norm = max(0.01, 0.1 * np.min([n for n in train_grad_norms if n > 0]))
            max_norm = 2.0 * np.max(train_grad_norms)
            plt.ylim([min_norm, max_norm])

        # Total loss
        plt.subplot(num_subplots, 1, 7)
        plt.plot(train_iter_list, train_losses_total, label='Train')
        plt.plot(valid_iter_list, valid_losses_total, label='Valid')
        if best_valid_iter >= 0 and best_checkpoint_losses:
            plt.scatter(best_valid_iter, best_checkpoint_losses['valid_loss_total'], 
                       color='red', s=50, marker='o', zorder=5, label='Best Checkpoint')
        plt.title('Total Loss')
        plt.xlabel('Train Iteration')
        plt.legend()
        plt.yscale(y_scale)
        plt.xscale(x_scale)
        plt.grid(True)

        if x_scale == 'log':
            plt.xlim([30, max(train_iter_list)])
        else:
            plt.xlim([0, max(train_iter_list)])

        plt.tight_layout()
        plt.show()

    print('---------------------------------------------------------------')
    print(f'Training finished! training duration: {training_duration_minutes:.2f} minutes')
    print('---------------------------------------------------------------')
    print(f'Final Checkpoint spikes loss (train, valid)     : ({np.mean(train_losses_spikes[-50:]):.5f}, {valid_losses_spikes[-1]:.5f})')
    print(f'Final Checkpoint soma V loss (train, valid)     : ({np.mean(train_losses_soma[-50:]):.5f}, {valid_losses_soma[-1]:.5f})')
    print(f'Final Checkpoint near spike loss (train, valid) : ({np.mean(train_losses_near_spike[-50:]):.5f}, {valid_losses_near_spike[-1]:.5f})')
    print(f'Final Checkpoint inst rate loss (train, valid)  : ({np.mean(train_losses_inst_rate[-50:]):.5f}, {valid_losses_inst_rate[-1]:.5f})')
    print(f'Final Checkpoint dend V loss (train, valid)     : ({np.mean(train_losses_dend_v[-50:]):.5f}, {valid_losses_dend_v[-1]:.5f})')
    print('---------------------------------------------------------------')
    print(f'Final Checkpoint total loss (train, valid)      : ({np.mean(train_losses_total[-50:]):.5f}, {valid_losses_total[-1]:.5f})')
    print('---------------------------------------------------------------')

    # Print best checkpoint losses if available
    if best_checkpoint_losses:
        print('Best checkpoint losses:')
        print('---------------------------------------------------------------')
        print(f'Best Checkpoint spikes loss (train, valid)      : ({best_checkpoint_losses["train_loss_spikes"]:.5f}, {best_checkpoint_losses["valid_loss_spikes"]:.5f}) at iter {best_valid_iter}')
        print(f'Best Checkpoint soma V loss (train, valid)      : ({best_checkpoint_losses["train_loss_soma"]:.5f}, {best_checkpoint_losses["valid_loss_soma"]:.5f}) at iter {best_valid_iter}')
        print(f'Best Checkpoint inst rate loss (train, valid)   : ({best_checkpoint_losses["train_loss_inst_rate"]:.5f}, {best_checkpoint_losses["valid_loss_inst_rate"]:.5f}) at iter {best_valid_iter}')
        print(f'Best Checkpoint near spike loss (train, valid)  : ({best_checkpoint_losses["train_loss_near_spike"]:.5f}, {best_checkpoint_losses["valid_loss_near_spike"]:.5f}) at iter {best_valid_iter}')
        print(f'Best Checkpoint dend V loss (train, valid)      : ({best_checkpoint_losses["train_loss_dend_v"]:.5f}, {best_checkpoint_losses["valid_loss_dend_v"]:.5f}) at iter {best_valid_iter}')
        print('---------------------------------------------------------------')
        print(f'Best Checkpoint total loss (train, valid)       : ({best_checkpoint_losses["train_loss_total"]:.5f}, {best_checkpoint_losses["valid_loss_total"]:.5f}) at iter {best_valid_iter}')
        print('---------------------------------------------------------------')

    # print the best ever validation losses throughout the training
    print('---------------------------------------------------------------')
    best_valid_loss_spikes = np.min(valid_losses_spikes)
    best_valid_iter_spikes = valid_iter_list[np.argmin(valid_losses_spikes)]
    print(f'Best ever valid spikes      loss:  {best_valid_loss_spikes:.5f}   at iteration {best_valid_iter_spikes}')
    best_valid_loss_soma = np.min(valid_losses_soma)
    best_valid_iter_soma = valid_iter_list[np.argmin(valid_losses_soma)]
    print(f'Best ever valid soma        loss:  {best_valid_loss_soma:.5f}   at iteration {best_valid_iter_soma}')
    best_valid_loss_near_spike = np.min(valid_losses_near_spike)
    best_valid_iter_near_spike = valid_iter_list[np.argmin(valid_losses_near_spike)]
    print(f'Best ever valid near spike  loss:  {best_valid_loss_near_spike:.5f}   at iteration {best_valid_iter_near_spike}')
    best_valid_loss_inst_rate = np.min(valid_losses_inst_rate)
    best_valid_iter_inst_rate = valid_iter_list[np.argmin(valid_losses_inst_rate)]
    print(f'Best ever valid inst rate   loss:  {best_valid_loss_inst_rate:.5f}   at iteration {best_valid_iter_inst_rate}')
    best_valid_loss_dend_v = np.min(valid_losses_dend_v)
    best_valid_iter_dend_v = valid_iter_list[np.argmin(valid_losses_dend_v)]
    print(f'Best ever valid dend V      loss:  {best_valid_loss_dend_v:.5f}   at iteration {best_valid_iter_dend_v}')
    print('---------------------------------------------------------------')

    learning_curves_dict = {
        'learning_rate_list': learning_rate_list,

        'train_iter_list': train_iter_list,
        'train_losses_spikes': train_losses_spikes,
        'train_losses_soma': train_losses_soma,
        'train_losses_dend_v': train_losses_dend_v,
        'train_losses_inst_rate': train_losses_inst_rate,
        'train_losses_near_spike': train_losses_near_spike,
        'train_losses_total': train_losses_total,
        'train_losses_weights_spikes': train_losses_weights_spikes,
        'train_losses_weights_soma': train_losses_weights_soma,
        'train_losses_weights_dend_v': train_losses_weights_dend_v,
        'train_losses_weights_inst_rate': train_losses_weights_inst_rate,
        'train_losses_weights_near_spike': train_losses_weights_near_spike,
        
        'train_grad_norms': train_grad_norms,

        'valid_iter_list': valid_iter_list,
        'valid_losses_spikes': valid_losses_spikes,
        'valid_losses_soma': valid_losses_soma,
        'valid_losses_dend_v': valid_losses_dend_v,
        'valid_losses_inst_rate': valid_losses_inst_rate,
        'valid_losses_near_spike': valid_losses_near_spike,
        'valid_losses_total': valid_losses_total,
        'valid_losses_weights_spikes': valid_losses_weights_spikes,
        'valid_losses_weights_soma': valid_losses_weights_soma,
        'valid_losses_weights_dend_v': valid_losses_weights_dend_v,
        'valid_losses_weights_inst_rate': valid_losses_weights_inst_rate,
        'valid_losses_weights_near_spike': valid_losses_weights_near_spike,
    }

    # store the learning curves in the model metadata
    twin_model.set_metadata_learning_curves(learning_curves_dict)

    #%% plot the relative fractions of the different losses compared to the total loss during training

    weighted_train_losses_spikes = [train_losses_spikes[i] * train_losses_weights_spikes[i] for i in range(len(train_losses_spikes))]
    weighted_train_losses_soma = [train_losses_soma[i] * train_losses_weights_soma[i] for i in range(len(train_losses_soma))]
    weighted_train_losses_near_spike = [train_losses_near_spike[i] * train_losses_weights_near_spike[i] for i in range(len(train_losses_near_spike))]
    weighted_train_losses_inst_rate = [train_losses_inst_rate[i] * train_losses_weights_inst_rate[i] for i in range(len(train_losses_inst_rate))]
    weighted_train_losses_dend_v = [train_losses_dend_v[i] * train_losses_weights_dend_v[i] for i in range(len(train_losses_dend_v))]
    
    weighted_valid_losses_spikes = [valid_losses_spikes[i] * valid_losses_weights_spikes[i] for i in range(len(valid_losses_spikes))]
    weighted_valid_losses_soma = [valid_losses_soma[i] * valid_losses_weights_soma[i] for i in range(len(valid_losses_soma))]
    weighted_valid_losses_near_spike = [valid_losses_near_spike[i] * valid_losses_weights_near_spike[i] for i in range(len(valid_losses_near_spike))]
    weighted_valid_losses_inst_rate = [valid_losses_inst_rate[i] * valid_losses_weights_inst_rate[i] for i in range(len(valid_losses_inst_rate))]
    weighted_valid_losses_dend_v = [valid_losses_dend_v[i] * valid_losses_weights_dend_v[i] for i in range(len(valid_losses_dend_v))]

    # apply reweighting to the total loss
    def calculate_total_weighted_loss(weighted_losses_spikes, weighted_losses_soma, weighted_losses_near_spike,
                                      weighted_losses_inst_rate, weighted_losses_dend_v):

        total_weights_loss_list = []
        for i in range(len(weighted_losses_spikes)):
            total_weighted_loss = weighted_losses_spikes[i]
            total_weighted_loss += weighted_losses_soma[i]
            total_weighted_loss += weighted_losses_near_spike[i]
            total_weighted_loss += weighted_losses_inst_rate[i]
            total_weighted_loss += weighted_losses_dend_v[i]
            
            total_weights_loss_list.append(total_weighted_loss)

        return total_weights_loss_list

    train_losses_total_reweighted = calculate_total_weighted_loss(weighted_train_losses_spikes, weighted_train_losses_soma, weighted_train_losses_near_spike,
                                                                  weighted_train_losses_inst_rate, weighted_train_losses_dend_v)
    valid_losses_total_reweighted = calculate_total_weighted_loss(weighted_valid_losses_spikes, weighted_valid_losses_soma, weighted_valid_losses_near_spike,
                                                                  weighted_valid_losses_inst_rate, weighted_valid_losses_dend_v)

    fraction_of_total_loss_spikes_train = [weighted_train_losses_spikes[i] / train_losses_total_reweighted[i] for i in range(len(train_losses_total_reweighted))]
    fraction_of_total_loss_soma_train = [weighted_train_losses_soma[i] / train_losses_total_reweighted[i] for i in range(len(train_losses_total_reweighted))]
    fraction_of_total_loss_near_spike_train = [weighted_train_losses_near_spike[i] / train_losses_total_reweighted[i] for i in range(len(train_losses_total_reweighted))]
    fraction_of_total_loss_inst_rate_train = [weighted_train_losses_inst_rate[i] / train_losses_total_reweighted[i] for i in range(len(train_losses_total_reweighted))]
    fraction_of_total_loss_dend_v_train = [weighted_train_losses_dend_v[i] / train_losses_total_reweighted[i] for i in range(len(train_losses_total_reweighted))]

    fraction_of_total_loss_spikes_valid = [weighted_valid_losses_spikes[i] / valid_losses_total_reweighted[i] for i in range(len(valid_losses_total_reweighted))]
    fraction_of_total_loss_soma_valid = [weighted_valid_losses_soma[i] / valid_losses_total_reweighted[i] for i in range(len(valid_losses_total_reweighted))]
    fraction_of_total_loss_near_spike_valid = [weighted_valid_losses_near_spike[i] / valid_losses_total_reweighted[i] for i in range(len(valid_losses_total_reweighted))]
    fraction_of_total_loss_inst_rate_valid = [weighted_valid_losses_inst_rate[i] / valid_losses_total_reweighted[i] for i in range(len(valid_losses_total_reweighted))]
    fraction_of_total_loss_dend_v_valid = [weighted_valid_losses_dend_v[i] / valid_losses_total_reweighted[i] for i in range(len(valid_losses_total_reweighted))]

    loss_colors = ['blue', 'orange', 'purple', 'green', 'red', 'brown']

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(train_iter_list, weighted_train_losses_spikes, label='Spikes (train)', alpha=0.5, linewidth=1, color=loss_colors[0])
    plt.plot(train_iter_list, weighted_train_losses_soma, label='Soma (train)', alpha=0.5, linewidth=1, color=loss_colors[1])
    plt.plot(train_iter_list, weighted_train_losses_near_spike, label='Near Spike (train)', alpha=0.5, linewidth=1, color=loss_colors[2])
    plt.plot(train_iter_list, weighted_train_losses_inst_rate, label='Inst Rate (train)', alpha=0.5, linewidth=1, color=loss_colors[3])
    plt.plot(train_iter_list, weighted_train_losses_dend_v, label='Dend V (train)', alpha=0.5, linewidth=1, color=loss_colors[4])

    plt.plot(valid_iter_list, weighted_valid_losses_spikes, label='Spikes (valid)', linewidth=3, color=loss_colors[0])
    plt.plot(valid_iter_list, weighted_valid_losses_soma, label='Soma (valid)', linewidth=3, color=loss_colors[1])
    plt.plot(valid_iter_list, weighted_valid_losses_near_spike, label='Near Spike (valid)', linewidth=3, color=loss_colors[2])
    plt.plot(valid_iter_list, weighted_valid_losses_inst_rate, label='Inst Rate (valid)', linewidth=3, color=loss_colors[3])
    plt.plot(valid_iter_list, weighted_valid_losses_dend_v, label='Dend V (valid)', linewidth=3, color=loss_colors[4])

    max_value = max(weighted_train_losses_spikes)
    max_value = max(max_value, max(weighted_train_losses_soma))
    max_value = max(max_value, max(weighted_train_losses_near_spike))
    max_value = max(max_value, max(weighted_train_losses_inst_rate))
    max_value = max(max_value, max(weighted_train_losses_dend_v))

    plt.ylim([0.005, 1.2 * max_value])

    plt.xscale('log')
    plt.yscale('log')
    plt.legend(ncol=2, loc='lower left')
    plt.xlabel('Train Iteration')
    plt.ylabel('Weighted Loss')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(train_iter_list, fraction_of_total_loss_spikes_train, label='Spikes (train)', alpha=0.5, linewidth=1, color=loss_colors[0])
    plt.plot(train_iter_list, fraction_of_total_loss_soma_train, label='Soma (train)', alpha=0.5, linewidth=1, color=loss_colors[1])
    plt.plot(train_iter_list, fraction_of_total_loss_near_spike_train, label='Near Spike (train)', alpha=0.5, linewidth=1, color=loss_colors[2])
    plt.plot(train_iter_list, fraction_of_total_loss_inst_rate_train, label='Inst Rate (train)', alpha=0.5, linewidth=1, color=loss_colors[3])
    plt.plot(train_iter_list, fraction_of_total_loss_dend_v_train, label='Dend V (train)', alpha=0.5, linewidth=1, color=loss_colors[4])

    plt.plot(valid_iter_list, fraction_of_total_loss_spikes_valid, label='Spikes (valid)', linewidth=3, color=loss_colors[0])
    plt.plot(valid_iter_list, fraction_of_total_loss_soma_valid, label='Soma (valid)', linewidth=3, color=loss_colors[1])
    plt.plot(valid_iter_list, fraction_of_total_loss_near_spike_valid, label='Near Spike (valid)', linewidth=3, color=loss_colors[2])
    plt.plot(valid_iter_list, fraction_of_total_loss_inst_rate_valid, label='Inst Rate (valid)', linewidth=3, color=loss_colors[3])
    plt.plot(valid_iter_list, fraction_of_total_loss_dend_v_valid, label='Dend V (valid)', linewidth=3, color=loss_colors[4])

    plt.xscale('log')
    plt.xlabel('Train Iteration')
    plt.ylabel('Fraction of Total Loss')
    plt.grid(True)
    plt.ylim([-0.01, 1.01])
    
    plt.tight_layout()
    plt.show()

    num_iter_per_epoch = int(len(train_losses_spikes) / num_epochs)
    warmup_num_iter = num_warmup_epochs * num_iter_per_epoch

    # calculate the average fraction of the total loss for all individual losses of first 100 and last 100 train iterations
    first_warmup_epochs_train_iter_losses_spikes = [fraction_of_total_loss_spikes_train[i] for i in range(warmup_num_iter)]
    last_epoch_train_iter_losses_spikes = [fraction_of_total_loss_spikes_train[i] for i in range(len(train_losses_total) - num_iter_per_epoch, len(train_losses_total))]
    first_warmup_epochs_train_iter_losses_soma = [fraction_of_total_loss_soma_train[i] for i in range(warmup_num_iter)]
    last_epoch_train_iter_losses_soma = [fraction_of_total_loss_soma_train[i] for i in range(len(train_losses_total) - num_iter_per_epoch, len(train_losses_total))]
    first_warmup_epochs_train_iter_losses_near_spike = [fraction_of_total_loss_near_spike_train[i] for i in range(warmup_num_iter)]
    last_epoch_train_iter_losses_near_spike = [fraction_of_total_loss_near_spike_train[i] for i in range(len(train_losses_total) - num_iter_per_epoch, len(train_losses_total))]
    first_warmup_epochs_train_iter_losses_inst_rate = [fraction_of_total_loss_inst_rate_train[i] for i in range(warmup_num_iter)]
    last_epoch_train_iter_losses_inst_rate = [fraction_of_total_loss_inst_rate_train[i] for i in range(len(train_losses_total) - num_iter_per_epoch, len(train_losses_total))]
    first_warmup_epochs_train_iter_losses_dend_v = [fraction_of_total_loss_dend_v_train[i] for i in range(warmup_num_iter)]
    last_epoch_train_iter_losses_dend_v = [fraction_of_total_loss_dend_v_train[i] for i in range(len(train_losses_total) - num_iter_per_epoch, len(train_losses_total))]

    first_warmup_epochs_train_iter_losses_spikes_avg = np.mean(first_warmup_epochs_train_iter_losses_spikes)
    last_epoch_train_iter_losses_spikes_avg = np.mean(last_epoch_train_iter_losses_spikes)
    first_warmup_epochs_train_iter_losses_soma_avg = np.mean(first_warmup_epochs_train_iter_losses_soma)
    last_epoch_train_iter_losses_soma_avg = np.mean(last_epoch_train_iter_losses_soma)
    first_warmup_epochs_train_iter_losses_near_spike_avg = np.mean(first_warmup_epochs_train_iter_losses_near_spike)
    last_epoch_train_iter_losses_near_spike_avg = np.mean(last_epoch_train_iter_losses_near_spike)
    first_warmup_epochs_train_iter_losses_inst_rate_avg = np.mean(first_warmup_epochs_train_iter_losses_inst_rate)
    last_epoch_train_iter_losses_inst_rate_avg = np.mean(last_epoch_train_iter_losses_inst_rate)
    first_warmup_epochs_train_iter_losses_dend_v_avg = np.mean(first_warmup_epochs_train_iter_losses_dend_v)
    last_epoch_train_iter_losses_dend_v_avg = np.mean(last_epoch_train_iter_losses_dend_v)

    print(f'spikes     frac of total loss for (first warmup epochs, last epoch) iter: {first_warmup_epochs_train_iter_losses_spikes_avg:.3f} -> {last_epoch_train_iter_losses_spikes_avg:.3f}')
    print(f'soma       frac of total loss for (first warmup epochs, last epoch) iter: {first_warmup_epochs_train_iter_losses_soma_avg:.3f} -> {last_epoch_train_iter_losses_soma_avg:.3f}')
    print(f'near spike frac of total loss for (first warmup epochs, last epoch) iter: {first_warmup_epochs_train_iter_losses_near_spike_avg:.3f} -> {last_epoch_train_iter_losses_near_spike_avg:.3f}')
    print(f'inst rate  frac of total loss for (first warmup epochs, last epoch) iter: {first_warmup_epochs_train_iter_losses_inst_rate_avg:.3f} -> {last_epoch_train_iter_losses_inst_rate_avg:.3f}')
    print(f'dend v     frac of total loss for (first warmup epochs, last epoch) iter: {first_warmup_epochs_train_iter_losses_dend_v_avg:.3f} -> {last_epoch_train_iter_losses_dend_v_avg:.3f}')

    #%% Evaluation

    # Evaluate model on validation set
    # valid_metrics_dict = evaluate_model_on_dataset(twin_model, valid_dataset, batch_size=valid_batch_size, verbose=1)
    valid_metrics_dict = evaluate_model_on_dataset(twin_model, test_dataset, batch_size=test_batch_size, verbose=1)

    # Print metrics summary
    print('--------------------------------------------------')
    print('valid_metrics_dict.keys():')
    print('--------------------------')
    for key in valid_metrics_dict.keys():
        print(f"  '{key}'")
    print('--------------------------------------------------')
    
    interesting_keys = [
        'requested_false_positive_rate', 'true_positive_at_FP', 'AUC_score',
        'near_spike_true_positive_at_FP', 'near_spike_AUC_score',
        'soma_explained_variance_percent', 'soma_RMSE', 'soma_MAE',
        'inst_rate_explained_variance_percent', 'inst_rate_RMSE', 'inst_rate_MAE',
        'dend_v_explained_variance_percent', 'dend_v_RMSE', 'dend_v_MAE']
    
    for key in interesting_keys:
        start_string = f"valid_metrics_dict['{key}']"
        filler_string = ' ' * (59 - len(start_string))
        print(f"{start_string} {filler_string} = {valid_metrics_dict[key]:.5f}")
    print('--------------------------------------------------')

    # Plot evaluation figures
    fig = plot_evaluation_figures(valid_metrics_dict)
    plt.show()

    # store the evaluation metrics in the model metadata
    twin_model.set_metadata_eval_metrics(valid_metrics_dict)
    twin_model.print_main_metadata()

    #%% load a random sample from test dataset and run the model on it to show example traces and predictions
    
    # load a random sample from test dataset and run the model on it
    sample_filename = np.random.choice(glob.glob(os.path.join(test_data_folder, 'output_spikes', '*.npy')))
    sample_basename = os.path.basename(sample_filename)

    # display the sample predictions (minimal form - just input, spikes, soma voltage)
    fig = display_sample_predictions_minimal(twin_model, test_data_folder, bs_neuron, sample_basename)
    plt.show()

    # display the sample predictions (full form - all 5 outputs: spikes, soma, near_spike, inst_rate, dend_v)
    fig = display_sample_predictions_full(twin_model, test_data_folder, bs_neuron, sample_basename)
    plt.show()

    #%% check calibration of the model
    
    # fetch validation data (use the soma voltage predictions are as they are from model output)
    # output_dict = predict_on_all_simulations(twin_model, valid_dataset, batch_size=valid_batch_size)
    output_dict = predict_on_all_simulations(twin_model, test_dataset, batch_size=test_batch_size)

    y_spikes_pred = output_dict['y_spikes_pred']
    y_spikes_gt = output_dict['y_spikes_gt']

    calib_corr, calib_explained_var = calculate_calibration_metrics(y_spikes_pred, y_spikes_gt, num_bins_per_simulation=8)
    print(f'Calibration Correlation: {calib_corr:.4f}')
    print(f'Calibration Explained Variance Percent: {100 * calib_explained_var:.2f}%')

    fig = display_calibration_figure(y_spikes_pred, y_spikes_gt)
    plt.show()

    #%% Save model

    save_model_to_disk = True
    # save_model_to_disk = False

    if save_model_to_disk:
        valid_acc_str = f'_AUC_0_{10000 * valid_metrics_dict["AUC_score"]:.0f}'
        valid_acc_str += f'_somaR2_{10 * valid_metrics_dict["soma_explained_variance_percent"]:.0f}'
        calib_str = f'_calibR2_{10000 * calib_explained_var:.0f}'
        checkpoint_model_name_pt = bs_neuron.short_name + '_' + twin_model.short_name + f"{valid_acc_str}_{calib_str}.pt"
        twin_model.save_model(os.path.join(models_folder, checkpoint_model_name_pt))

        print(f'Model saved to folder "{models_folder}"')
        print(f'checkpoint name = "{checkpoint_model_name_pt}"')

# %%
