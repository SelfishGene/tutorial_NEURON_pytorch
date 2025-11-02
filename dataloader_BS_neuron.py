#%% Imports

import os
import glob
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import config

#%% BSNeuronDataset class

class BSNeuronDataset(Dataset):
    """Dataset class for Ball and Stick neuron simulation data"""

    def __init__(self, data_folder, time_window_size=1024, safety_margin=16, preload_data=True):
        self.inputs_folder = os.path.join(data_folder, 'inputs')
        self.output_spikes_folder = os.path.join(data_folder, 'output_spikes')
        self.output_soma_v_folder = os.path.join(data_folder, 'output_soma_v')
        self.output_near_spikes_folder = os.path.join(data_folder, 'output_near_spikes')
        self.output_inst_rate_folder = os.path.join(data_folder, 'output_inst_rate')
        self.intermidiate_branch_v_folder = os.path.join(data_folder, 'intermidiate_branch_v')
        
        self.time_window_size = time_window_size
        self.safety_margin = safety_margin
        self.preload_data = preload_data

        # Get list of all files
        self.file_list = sorted(glob.glob(os.path.join(self.inputs_folder, '*.npy')))
        self.num_samples = len(self.file_list)

        # Load first file to get dimensions
        sample_input = np.load(self.file_list[0])
        self.num_segments = sample_input.shape[0]
        self.sim_duration = sample_input.shape[1]
        
        # Load sample dendritic voltage to determine structure
        # Ball and Stick neuron always has 2D branch voltages with shape (num_segments, time_steps)
        sample_dend_v = np.load(os.path.join(self.intermidiate_branch_v_folder, 
                                           os.path.basename(self.file_list[0])))
        self.dend_v_shape = sample_dend_v.shape
        self.num_dend_channels = sample_dend_v.shape[0]

        assert self.sim_duration > (self.time_window_size + 2 * self.safety_margin), 'window size is too big for sim duration'
        
        print('----------------------------------------------------')
        # Preload data if requested
        if self.preload_data:
            print('Preloading all data into memory...')
            self.X_spikes_data = np.empty((self.num_samples, self.num_segments, self.sim_duration, 2))
            self.y_spikes_data = np.empty((self.num_samples, self.sim_duration))
            self.y_soma_data = np.empty((self.num_samples, self.sim_duration))
            self.y_near_spike = np.empty((self.num_samples, self.sim_duration))
            self.y_inst_rate = np.empty((self.num_samples, self.sim_duration))
            self.y_dend_v_data = np.empty((self.num_samples, self.num_dend_channels, self.sim_duration))
            
            for i, file_path in enumerate(tqdm(self.file_list)):
                basename = os.path.basename(file_path)
                self.X_spikes_data[i] = np.load(file_path)
                self.y_spikes_data[i] = np.load(os.path.join(self.output_spikes_folder, basename))
                self.y_soma_data[i] = np.load(os.path.join(self.output_soma_v_folder, basename))
                self.y_near_spike[i] = np.load(os.path.join(self.output_near_spikes_folder, basename))
                self.y_inst_rate[i] = np.load(os.path.join(self.output_inst_rate_folder, basename))
                self.y_dend_v_data[i] = np.load(os.path.join(self.intermidiate_branch_v_folder, basename))
        
        print(f'Dataset initialized with:')
        print(f'- {len(self.file_list)} simulation files')
        print(f'- {self.num_segments} segments')
        print(f'- {self.sim_duration} timesteps per simulation')
        print(f'- {time_window_size} timesteps per training sample')
        print(f'- Dendritic voltage channels: {self.num_dend_channels}')
        print(f'- Data preloaded: {self.preload_data}')
        print('----------------------------------------------------')
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        file_idx = idx

        # randomly select a start time with safety margin from both ends of the simulation
        start_time = np.random.randint(self.safety_margin, self.sim_duration - self.time_window_size - self.safety_margin)
        end_time = start_time + self.time_window_size
        
        if self.preload_data:
            # Sample directly from memory
            X_spikes = self.X_spikes_data[file_idx, :, start_time:end_time, :]
            y_spikes = self.y_spikes_data[file_idx, start_time:end_time]
            y_soma = self.y_soma_data[file_idx, start_time:end_time]
            y_near_spike = self.y_near_spike[file_idx, start_time:end_time]
            y_inst_rate = self.y_inst_rate[file_idx, start_time:end_time]
            y_dend_v = self.y_dend_v_data[file_idx, :, start_time:end_time]
        else:
            # Load from disk
            input_file = self.file_list[file_idx]
            simulation_basename = os.path.basename(input_file)
            X_spikes = np.load(input_file)  # shape: (num_segments, sim_duration, 2)
            y_spikes = np.load(os.path.join(self.output_spikes_folder, simulation_basename))
            y_soma = np.load(os.path.join(self.output_soma_v_folder, simulation_basename))
            y_near_spike = np.load(os.path.join(self.output_near_spikes_folder, simulation_basename))
            y_inst_rate = np.load(os.path.join(self.output_inst_rate_folder, simulation_basename))
            y_dend_v = np.load(os.path.join(self.intermidiate_branch_v_folder, simulation_basename))
            
            # Extract time window
            X_spikes = X_spikes[:, start_time:end_time, :]
            y_spikes = y_spikes[start_time:end_time]
            y_soma = y_soma[start_time:end_time]
            y_near_spike = y_near_spike[start_time:end_time]
            y_inst_rate = y_inst_rate[start_time:end_time]
            y_dend_v = y_dend_v[:, start_time:end_time]
        
        # Reshape X_spikes to match expected format (B, 2, S, T)
        X_spikes = np.transpose(X_spikes, (2, 0, 1))  # -> (2, num_segments, time_window)
        
        # Convert to tensors
        X_spikes = torch.FloatTensor(X_spikes)          # shape: (2, num_segments, time_window)
        y_spikes = torch.FloatTensor(y_spikes)          # shape: (time_window)
        y_soma = torch.FloatTensor(y_soma)              # shape: (time_window)
        y_near_spike = torch.FloatTensor(y_near_spike)  # shape: (time_window)
        y_inst_rate = torch.FloatTensor(y_inst_rate)    # shape: (time_window)
        y_dend_v = torch.FloatTensor(y_dend_v)          # shape: (num_dend_channels, time_window)
        
        # Return as dictionary
        return {
            'X_spikes': X_spikes,
            'y_spikes': y_spikes,
            'y_soma': y_soma,
            'y_near_spike': y_near_spike,
            'y_inst_rate': y_inst_rate,
            'y_dend_v': y_dend_v
        }

#%% Example usage
if __name__ == "__main__":

    print('----------------------------------------------------')
    print("Example usage of BSNeuronDataset dataloader class")
    print('----------------------------------------------------')
    
    # Dataset parameters - modify these paths as needed
    data_root = config.NEURON_DATA_ROOT
    data_folder_name = r'BallAndStickNeuron_Soma_Hay2011_Dend_Lxd_2048um_x_1_0um_8segs'
    train_data_folder = os.path.join(data_root, data_folder_name, 'train')
    
    # Check if data folder exists
    if not os.path.exists(train_data_folder):
        print(f"Warning: Data folder does not exist: {train_data_folder}")
        print("Please update the data_root and data_folder_name variables to point to your data.")
        exit()
    
    # Training parameters
    train_time_window_size = 1024
    train_batch_size = 8
    preload_data = False  # Set to False for example to avoid long loading times
    
    # Create dataset and dataloader
    print(f"Creating dataset from: {train_data_folder}")
    train_dataset = BSNeuronDataset(train_data_folder, train_time_window_size, preload_data=preload_data)
    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    
    # Fetch a batch to demonstrate usage
    print('----------------------------------------------------')
    print('Fetching a batch...')
    print('-------------------')
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

    # Calculate basic statistics similar to the train script
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

    print('----------------------------------------------------')
    print('Batch Statistics:')
    print('-----------------')
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
    print('----------------------------------------------------')
    
    print("Dataset example completed successfully!") 



