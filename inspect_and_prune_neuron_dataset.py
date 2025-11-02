#%% Imports

import os
import time
import glob
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.gridspec as gridspec
from neuron_model_ball_and_stick import BallAndStickNeuron
import config

#%% Helper functions

def collect_dataset_statistics(data_folder):
    """Collect comprehensive statistics from a dataset folder."""
    
    # Get all input files
    all_input_files = glob.glob(os.path.join(data_folder, 'inputs', '*.npy'))
    
    # Initialize statistics lists
    statistics = {
        'sample_names': [],
        'input_exc_mean': [],
        'input_inh_mean': [],
        'input_exc_std': [],
        'input_inh_std': [],
        'output_firing_rates': [],
        'output_spike_ISIs': [],
        'output_soma_v_means': [],
        'output_soma_v_stds': [],
        'output_near_spike_means': [],
        'output_inst_rate_means': [],
        'output_inst_rate_stds': [],
        'branch_voltage_means': [],
        'branch_voltage_stds': []
    }
    
    print(f'Processing {len(all_input_files)} files from {os.path.basename(data_folder)}...')
    dataset_processing_start_time = time.time()

    for input_file in all_input_files:
        
        # Get the file name from the full path of input file
        curr_samplename = os.path.basename(input_file)
        output_spikes_file = os.path.join(data_folder, 'output_spikes', curr_samplename)
        output_soma_v_file = os.path.join(data_folder, 'output_soma_v', curr_samplename)
        output_near_spikes_file = os.path.join(data_folder, 'output_near_spikes', curr_samplename)
        output_inst_rate_file = os.path.join(data_folder, 'output_inst_rate', curr_samplename)
        branch_voltage_file = os.path.join(data_folder, 'intermidiate_branch_v', curr_samplename)
        
        # Load all data and convert to float32 immediately
        X_inputs = np.load(input_file).astype(np.float32)
        y_spike = np.load(output_spikes_file).astype(np.float32)
        y_soma = np.load(output_soma_v_file).astype(np.float32)
        y_near_spike = np.load(output_near_spikes_file).astype(np.float32)
        y_inst_rate = np.load(output_inst_rate_file).astype(np.float32)
        branch_voltage = np.load(branch_voltage_file).astype(np.float32)

        X_exc, X_inh = X_inputs[..., 0], X_inputs[..., 1]

        # Calculate input statistics
        input_exc_mean = np.mean(X_exc)
        input_inh_mean = np.mean(X_inh)
        input_exc_std = np.std(X_exc)
        input_inh_std = np.std(X_inh)
        
        # Calculate output statistics
        output_firing_rate_Hz = np.mean(y_spike) * 1000
        output_spiking_ISIs = np.diff(np.where(y_spike)[0])
        output_soma_v_mean = np.mean(y_soma)
        output_soma_v_std = np.std(y_soma)
        output_near_spike_mean = np.mean(y_near_spike)
        output_inst_rate_mean = np.mean(y_inst_rate) * 1000  # Convert to Hz
        output_inst_rate_std = np.std(y_inst_rate) * 1000    # Convert to Hz
        
        # Calculate branch voltage statistics
        branch_voltage_mean = np.mean(branch_voltage)
        branch_voltage_std = np.std(branch_voltage)
        
        # Append to statistics lists
        statistics['sample_names'].append(curr_samplename)
        statistics['input_exc_mean'].append(input_exc_mean)
        statistics['input_inh_mean'].append(input_inh_mean)
        statistics['input_exc_std'].append(input_exc_std)
        statistics['input_inh_std'].append(input_inh_std)
        statistics['output_firing_rates'].append(output_firing_rate_Hz)
        statistics['output_spike_ISIs'].extend(output_spiking_ISIs)
        statistics['output_soma_v_means'].append(output_soma_v_mean)
        statistics['output_soma_v_stds'].append(output_soma_v_std)
        statistics['output_near_spike_means'].append(output_near_spike_mean)
        statistics['output_inst_rate_means'].append(output_inst_rate_mean)
        statistics['output_inst_rate_stds'].append(output_inst_rate_std)
        statistics['branch_voltage_means'].append(branch_voltage_mean)
        statistics['branch_voltage_stds'].append(branch_voltage_std)

        if len(statistics['input_exc_mean']) % 5000 == 0:
            print(f'Processed {len(statistics["input_exc_mean"])}/{len(all_input_files)} samples')

    dataset_processing_time_minutes = (time.time() - dataset_processing_start_time) / 60
    print(f'Dataset processing time: {dataset_processing_time_minutes:.2f} minutes')
    
    return statistics

def plot_dataset_statistics(stats_dict, figure_title="Dataset Statistics"):
    """Plot comprehensive dataset statistics including branch voltage distributions."""
    
    # Create figure with extended grid to accommodate new plots
    plt.figure(figsize=(14, 16))
    gs = gridspec.GridSpec(6, 2, height_ratios=[2, 1, 1, 1, 1, 1])

    # Plot the input mean distribution
    ax1 = plt.subplot(gs[0])
    ax1.hist(stats_dict['input_exc_mean'], bins=100, color='red', alpha=0.6, label='Excitatory')
    ax1.hist(stats_dict['input_inh_mean'], bins=100, color='blue', alpha=0.6, label='Inhibitory')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.set_title(f'{figure_title}\nInput Spike mean Distribution')
    ax1.set_xlabel('Num spikes per ms')

    # Plot the input std distribution
    ax2 = plt.subplot(gs[1])
    ax2.hist(stats_dict['input_exc_std'], bins=100, color='red', alpha=0.6, label='Excitatory')
    ax2.hist(stats_dict['input_inh_std'], bins=100, color='blue', alpha=0.6, label='Inhibitory')
    ax2.set_yscale('log')
    ax2.legend()
    ax2.set_title('Input Spike stdev Distribution')
    ax2.set_xlabel('Num spikes per ms')

    # Plot the output firing rate distribution
    ax3 = plt.subplot(gs[2])
    ax3.hist(stats_dict['output_firing_rates'], bins=100, color='blue')
    ax3.set_yscale('log')
    ax3.set_title('Output Firing Rate Distribution')
    ax3.set_xlabel('Firing Rate (Hz)')
    ax3.set_ylabel('Count')

    # Plot the output spike ISI distribution
    ax4 = plt.subplot(gs[3])
    ax4.hist(stats_dict['output_spike_ISIs'], bins=256, color='blue', label='ISI')
    ax4.set_yscale('log')
    ax4.legend()
    ax4.set_title('Output Spike ISI Distribution')
    ax4.set_xlabel('ISI (ms)')
    ax4.set_ylabel('Count')

    # Plot the output soma voltage mean distribution
    ax5 = plt.subplot(gs[4])
    ax5.hist(stats_dict['output_soma_v_means'], bins=100, color='green', alpha=0.9)
    ax5.set_yscale('log')
    ax5.set_title('Output Soma Voltage mean Distribution')
    ax5.set_xlabel('Voltage (mV)')
    ax5.set_ylabel('Count')

    # Plot the output soma voltage std distribution
    ax6 = plt.subplot(gs[5])
    ax6.hist(stats_dict['output_soma_v_stds'], bins=100, color='green', alpha=0.9)
    ax6.set_yscale('log')
    ax6.set_title('Output Soma Voltage stdev Distribution')
    ax6.set_xlabel('Voltage (mV)')
    ax6.set_ylabel('Count')

    # Plot the output near spike mean distribution
    ax7 = plt.subplot(gs[6])
    ax7.hist(stats_dict['output_near_spike_means'], bins=100, color='green', alpha=0.9)
    ax7.set_yscale('log')
    ax7.set_title('Output Near Spike Probability Distribution')
    ax7.set_xlabel('Binary')
    ax7.set_ylabel('Count')

    # Plot the output spike mean distribution
    all_output_spike_means = np.array(stats_dict['output_firing_rates']) / 1000
    ax8 = plt.subplot(gs[7])
    ax8.hist(all_output_spike_means, bins=100, color='green')
    ax8.set_yscale('log')
    ax8.set_title('Output Spike Probability Distribution')
    ax8.set_xlabel('Binary')
    ax8.set_ylabel('Count')

    # Plot the output inst rate mean distribution
    ax9 = plt.subplot(gs[8])
    ax9.hist(stats_dict['output_inst_rate_means'], bins=100, color='green')
    ax9.set_yscale('log')
    ax9.set_title('Output Instantaneous Rate mean Distribution')
    ax9.set_xlabel('Rate (Hz)')
    ax9.set_ylabel('Count')

    # Plot the output inst rate std distribution
    ax10 = plt.subplot(gs[9])
    ax10.hist(stats_dict['output_inst_rate_stds'], bins=100, color='green')
    ax10.set_yscale('log')
    ax10.set_title('Output Instantaneous Rate stdev Distribution')
    ax10.set_xlabel('Rate (Hz)')
    ax10.set_ylabel('Count')

    # NEW: Plot the branch voltage mean distribution
    ax11 = plt.subplot(gs[10])
    ax11.hist(stats_dict['branch_voltage_means'], bins=100, color='purple')
    ax11.set_yscale('log')
    ax11.set_title('Branch Voltage mean Distribution')
    ax11.set_xlabel('Voltage (mV)')
    ax11.set_ylabel('Count')

    # NEW: Plot the branch voltage std distribution
    ax12 = plt.subplot(gs[11])
    ax12.hist(stats_dict['branch_voltage_stds'], bins=100, color='purple')
    ax12.set_yscale('log')
    ax12.set_title('Branch Voltage stdev Distribution')
    ax12.set_xlabel('Voltage (mV)')
    ax12.set_ylabel('Count')

    plt.tight_layout()
    return plt.gcf()

def plot_firing_rate_scatter_plots(stats_dict, figure_title="Firing Rate vs Other Variables"):
    """Create scatter plots with firing rate on y-axis and other variables on x-axis."""
    
    # Create figure with 6x2 subplots
    fig, axes = plt.subplots(5, 2, figsize=(10, 12))
    fig.suptitle(figure_title, fontsize=16)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Define the x variables and their labels
    scatter_vars = [
        ('input_exc_mean', 'Excitatory Input Mean (spikes/ms)'),
        ('input_exc_std', 'Excitatory Input Std (spikes/ms)'),
        ('input_inh_mean', 'Inhibitory Input Mean (spikes/ms)'), 
        ('input_inh_std', 'Inhibitory Input Std (spikes/ms)'),
        ('output_soma_v_means', 'Soma Voltage Mean (mV)'),
        ('output_soma_v_stds', 'Soma Voltage Std (mV)'),
        ('branch_voltage_means', 'Branch Voltage Mean (mV)'),
        ('branch_voltage_stds', 'Branch Voltage Std (mV)'),
        ('output_near_spike_means', 'Near Spike Probability'),
        ('output_inst_rate_stds', 'Instantaneous Rate Std (Hz)'),
    ]
    
    # Create scatter plots
    for i, (var_key, xlabel) in enumerate(scatter_vars):
        ax = axes[i]
        
        # Create scatter plot
        ax.scatter(stats_dict[var_key], stats_dict['output_firing_rates'], s=1, color='blue')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Output Firing Rate (Hz)')
        ax.grid(True, alpha=0.3)
        
        # Set title
        ax.set_title(f'Firing Rate vs {xlabel.split(" (")[0]}')
    
    # Hide the last unused subplot
    if len(scatter_vars) < len(axes):
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    return fig

def filter_samples_by_criteria(stats_dict, criteria_ranges):
    """Filter samples based on statistical criteria and return matching sample names.
    
    Args:
        stats_dict: Dictionary from collect_dataset_statistics() containing sample statistics
        criteria_ranges: Dictionary with criteria ranges. Keys can include:
            - 'firing_rate_hz': (min, max) range for firing rate in Hz
            - 'input_exc_mean': (min, max) range for excitatory input mean
            - 'input_inh_mean': (min, max) range for inhibitory input mean  
            - 'input_exc_std': (min, max) range for excitatory input std
            - 'input_inh_std': (min, max) range for inhibitory input std
            - 'soma_v_mean': (min, max) range for soma voltage mean
            - 'soma_v_std': (min, max) range for soma voltage std
            - 'near_spike_prob': (min, max) range for near spike probability
            - 'inst_rate_mean_hz': (min, max) range for instantaneous rate mean in Hz
            - 'inst_rate_std_hz': (min, max) range for instantaneous rate std in Hz
            - 'branch_v_mean': (min, max) range for branch voltage mean
            - 'branch_v_std': (min, max) range for branch voltage std
    
    Returns:
        List of sample names that meet all specified criteria
    """
    
    matching_samples = []
    num_samples = len(stats_dict['sample_names'])
    
    for i in range(num_samples):
        sample_name = stats_dict['sample_names'][i]
        meets_criteria = True
        
        # Check firing rate
        if 'firing_rate_hz' in criteria_ranges:
            firing_rate = stats_dict['output_firing_rates'][i]
            min_rate, max_rate = criteria_ranges['firing_rate_hz']
            if firing_rate < min_rate or firing_rate > max_rate:
                meets_criteria = False
        
        # Check input excitatory mean
        if 'input_exc_mean' in criteria_ranges and meets_criteria:
            exc_mean = stats_dict['input_exc_mean'][i]
            min_val, max_val = criteria_ranges['input_exc_mean']
            if exc_mean < min_val or exc_mean > max_val:
                meets_criteria = False
        
        # Check input inhibitory mean  
        if 'input_inh_mean' in criteria_ranges and meets_criteria:
            inh_mean = stats_dict['input_inh_mean'][i]
            min_val, max_val = criteria_ranges['input_inh_mean']
            if inh_mean < min_val or inh_mean > max_val:
                meets_criteria = False
        
        # Check input excitatory std
        if 'input_exc_std' in criteria_ranges and meets_criteria:
            exc_std = stats_dict['input_exc_std'][i]
            min_val, max_val = criteria_ranges['input_exc_std']
            if exc_std < min_val or exc_std > max_val:
                meets_criteria = False
        
        # Check input inhibitory std
        if 'input_inh_std' in criteria_ranges and meets_criteria:
            inh_std = stats_dict['input_inh_std'][i]
            min_val, max_val = criteria_ranges['input_inh_std']
            if inh_std < min_val or inh_std > max_val:
                meets_criteria = False
        
        # Check soma voltage mean
        if 'soma_v_mean' in criteria_ranges and meets_criteria:
            soma_mean = stats_dict['output_soma_v_means'][i]
            min_val, max_val = criteria_ranges['soma_v_mean']
            if soma_mean < min_val or soma_mean > max_val:
                meets_criteria = False
        
        # Check soma voltage std
        if 'soma_v_std' in criteria_ranges and meets_criteria:
            soma_std = stats_dict['output_soma_v_stds'][i]
            min_val, max_val = criteria_ranges['soma_v_std']
            if soma_std < min_val or soma_std > max_val:
                meets_criteria = False
        
        # Check near spike probability
        if 'near_spike_prob' in criteria_ranges and meets_criteria:
            near_spike_prob = stats_dict['output_near_spike_means'][i]
            min_val, max_val = criteria_ranges['near_spike_prob']
            if near_spike_prob < min_val or near_spike_prob > max_val:
                meets_criteria = False
        
        # Check instantaneous rate mean
        if 'inst_rate_mean_hz' in criteria_ranges and meets_criteria:
            inst_rate_mean = stats_dict['output_inst_rate_means'][i]
            min_val, max_val = criteria_ranges['inst_rate_mean_hz']
            if inst_rate_mean < min_val or inst_rate_mean > max_val:
                meets_criteria = False
        
        # Check instantaneous rate std
        if 'inst_rate_std_hz' in criteria_ranges and meets_criteria:
            inst_rate_std = stats_dict['output_inst_rate_stds'][i]
            min_val, max_val = criteria_ranges['inst_rate_std_hz']
            if inst_rate_std < min_val or inst_rate_std > max_val:
                meets_criteria = False
        
        # Check branch voltage mean
        if 'branch_v_mean' in criteria_ranges and meets_criteria:
            branch_mean = stats_dict['branch_voltage_means'][i]
            min_val, max_val = criteria_ranges['branch_v_mean']
            if branch_mean < min_val or branch_mean > max_val:
                meets_criteria = False
        
        # Check branch voltage std
        if 'branch_v_std' in criteria_ranges and meets_criteria:
            branch_std = stats_dict['branch_voltage_stds'][i]
            min_val, max_val = criteria_ranges['branch_v_std']
            if branch_std < min_val or branch_std > max_val:
                meets_criteria = False
        
        # If sample meets all criteria, add to list
        if meets_criteria:
            matching_samples.append(sample_name)
    
    return matching_samples

def filter_dataset_folder(folder_path,
                          firing_rate_hz_range=(0.8, 120), avg_near_spike_prob_range=(0.003, 0.95),
                          soma_v_mV_mean_range=(-90, -65), soma_v_mV_std_range=(4, 20),
                          inst_rate_hz_mean_range=(0.8, 120), inst_rate_hz_std_range=(2, 50),
                          branch_v_mV_mean_range=(-20, 30), branch_v_mV_std_range=(1, 20),
                          drop_prob_for_firing_rate=0.95, drop_prob_for_near_spike_prob=0.95,
                          drop_prob_for_soma_v=0.95, drop_prob_for_inst_rate=0.95, drop_prob_for_branch_v=0.95):
    
    # Get all files in the folder
    inputs_folder = os.path.join(folder_path, 'inputs')
    spikes_folder = os.path.join(folder_path, 'output_spikes')
    soma_folder = os.path.join(folder_path, 'output_soma_v')
    near_spikes_folder = os.path.join(folder_path, 'output_near_spikes')
    inst_rate_folder = os.path.join(folder_path, 'output_inst_rate')
    branch_voltage_folder = os.path.join(folder_path, 'intermidiate_branch_v')
    
    all_files = [f for f in os.listdir(spikes_folder) if f.endswith('.npy')]
            
    deleted_files = []
    deleted_stats = defaultdict(list)
    
    for filename in all_files:
        input_path = os.path.join(inputs_folder, filename)
        soma_path = os.path.join(soma_folder, filename)
        spikes_path = os.path.join(spikes_folder, filename)
        near_spikes_path = os.path.join(near_spikes_folder, filename)
        inst_rate_path = os.path.join(inst_rate_folder, filename)
        branch_voltage_path = os.path.join(branch_voltage_folder, filename)
        
        # Load data and cast to float32 for numerical stability in std calculation
        y_soma = np.load(soma_path).astype(np.float32)
        y_spike = np.load(spikes_path).astype(np.float32)
        y_near_spike = np.load(near_spikes_path).astype(np.float32)
        y_inst_rate = np.load(inst_rate_path).astype(np.float32)
        branch_voltage = np.load(branch_voltage_path).astype(np.float32)
        
        # Calculate statistics
        soma_mean = np.mean(y_soma)
        soma_std = np.std(y_soma)
        firing_rate_hz = np.mean(y_spike) * 1000
        near_spike_mean = np.mean(y_near_spike)
        inst_rate_mean_hz = np.mean(y_inst_rate) * 1000
        inst_rate_std_hz = np.std(y_inst_rate) * 1000
        branch_voltage_mean = np.mean(branch_voltage)
        branch_voltage_std = np.std(branch_voltage)
        
        # firing rate conditions
        if firing_rate_hz < firing_rate_hz_range[0] or firing_rate_hz > firing_rate_hz_range[1]:
            firing_rate_not_in_range = np.random.rand() < drop_prob_for_firing_rate
        else:
            firing_rate_not_in_range = False

        # near spike probability conditions
        if near_spike_mean < avg_near_spike_prob_range[0] or near_spike_mean > avg_near_spike_prob_range[1]:
            near_spike_prob_not_in_range = np.random.rand() < drop_prob_for_near_spike_prob
        else:
            near_spike_prob_not_in_range = False

        # soma voltage conditions
        if soma_mean < soma_v_mV_mean_range[0] or soma_mean > soma_v_mV_mean_range[1]:
            soma_mean_not_in_range = np.random.rand() < drop_prob_for_soma_v
        else:
            soma_mean_not_in_range = False

        if soma_std < soma_v_mV_std_range[0] or soma_std > soma_v_mV_std_range[1]:
            soma_std_not_in_range = np.random.rand() < drop_prob_for_soma_v
        else:
            soma_std_not_in_range = False

        # instantaneous rate conditions
        if inst_rate_mean_hz < inst_rate_hz_mean_range[0] or inst_rate_mean_hz > inst_rate_hz_mean_range[1]:
            inst_rate_mean_not_in_range = np.random.rand() < drop_prob_for_inst_rate
        else:
            inst_rate_mean_not_in_range = False
        
        if inst_rate_std_hz < inst_rate_hz_std_range[0] or inst_rate_std_hz > inst_rate_hz_std_range[1]:
            inst_rate_std_not_in_range = np.random.rand() < drop_prob_for_inst_rate
        else:
            inst_rate_std_not_in_range = False
        
        # branch voltage conditions
        if branch_voltage_mean < branch_v_mV_mean_range[0] or branch_voltage_mean > branch_v_mV_mean_range[1]:
            branch_voltage_mean_not_in_range = np.random.rand() < drop_prob_for_branch_v
        else:
            branch_voltage_mean_not_in_range = False

        if branch_voltage_std < branch_v_mV_std_range[0] or branch_voltage_std > branch_v_mV_std_range[1]:
            branch_voltage_std_not_in_range = np.random.rand() < drop_prob_for_branch_v
        else:
            branch_voltage_std_not_in_range = False

        rate_conditions = firing_rate_not_in_range
        near_spike_conditions = near_spike_prob_not_in_range
        soma_conditions = soma_mean_not_in_range or soma_std_not_in_range
        inst_rate_conditions = inst_rate_mean_not_in_range or inst_rate_std_not_in_range
        branch_v_conditions = branch_voltage_mean_not_in_range or branch_voltage_std_not_in_range

        if rate_conditions or near_spike_conditions or soma_conditions or inst_rate_conditions or branch_v_conditions:
            deleted_files.append(filename)
            deleted_stats['soma_mean'].append(soma_mean)
            deleted_stats['soma_std'].append(soma_std)
            deleted_stats['firing_rate'].append(firing_rate_hz)
            deleted_stats['near_spike_prob'].append(near_spike_mean)
            deleted_stats['inst_rate'].append(inst_rate_mean_hz)
            deleted_stats['branch_v'].append(branch_voltage_mean)
            
            # Delete files
            os.remove(input_path)
            os.remove(soma_path)
            os.remove(spikes_path)
            os.remove(near_spikes_path)
            os.remove(inst_rate_path)
            os.remove(branch_voltage_path)
    
    # Calculate summary statistics
    deleted_stats_summary = {
        'num_deleted': len(deleted_files),
        'mean_soma_v': np.mean(deleted_stats['soma_mean']) if deleted_stats['soma_mean'] else 0,
        'std_soma_v': np.mean(deleted_stats['soma_std']) if deleted_stats['soma_std'] else 0,
        'mean_near_spike_prob': np.mean(deleted_stats['near_spike_prob']) if deleted_stats['near_spike_prob'] else 0,
        'mean_firing_rate': np.mean(deleted_stats['firing_rate']) if deleted_stats['firing_rate'] else 0,
        'mean_inst_rate': np.mean(deleted_stats['inst_rate']) if deleted_stats['inst_rate'] else 0,
        'mean_branch_v': np.mean(deleted_stats['branch_v']) if deleted_stats['branch_v'] else 0
    }
    
    return deleted_stats_summary


#%% Main script

if __name__ == "__main__":

    from create_dataset_BS_neuron import load_and_display_sample

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

    #%% load random sample from the dataset and plot it

    sample_filename = np.random.choice(glob.glob(os.path.join(train_data_folder, 'output_spikes', '*.npy')))
    sample_name = os.path.basename(sample_filename)
    
    # Use the helper function to load and display the sample
    fig = load_and_display_sample(sample_name, train_data_folder, bs_neuron)
    plt.show()

    #%% collect statistics of the entire train dataset and plot it

    train_stats = collect_dataset_statistics(train_data_folder)

    #%% plot statistics

    figure_title_str = "Train Dataset Statistics"
    stats_fig = plot_dataset_statistics(train_stats, figure_title_str)
    plt.show()

    # Create scatter plots showing relationships with firing rate
    scatter_fig = plot_firing_rate_scatter_plots(train_stats, "Train Dataset: Firing Rate vs Other Variables")
    plt.show()

    #%% filter samples by specific criteria
        
    arbitrary_criteria = {
        'branch_v_std': (30, 100),
        'inst_rate_std_hz': (30, 100),
        'soma_v_std': (5, 30),

        'soma_v_mean': (-90, -50),
        'branch_v_mean': (-20, 100),

        'inst_rate_mean_hz': (1, 150),
        'firing_rate_hz': (1, 150),
        'near_spike_prob': (0.001, 1.0),
    }
    
    arbitrary_samples = filter_samples_by_criteria(train_stats, arbitrary_criteria)
    print(f"Found {len(arbitrary_samples)} samples") 
    
    if len(arbitrary_samples) > 0:        
        sample_to_inspect = np.random.choice(arbitrary_samples)
        print(f"\nInspecting sample: {sample_to_inspect}")
        fig = load_and_display_sample(sample_to_inspect, train_data_folder, bs_neuron)
        plt.show()

    #%% Filter datasets based on various criteria

    apply_pruning = True
    apply_pruning = False

    firing_rate_hz_range = (0.3, 120)
    avg_near_spike_prob_range = (0.0, 0.98)
    soma_v_mV_mean_range = (-98, -64)
    soma_v_mV_std_range = (3, 22)
    inst_rate_hz_mean_range = (1, 120)
    inst_rate_hz_std_range = (3, 42)
    branch_v_mV_mean_range = (-35, 65)
    branch_v_mV_std_range = (3, 42)

    drop_prob_for_firing_rate = 1.0
    drop_prob_for_near_spike_prob = 1.0
    drop_prob_for_soma_v = 1.0
    drop_prob_for_inst_rate = 1.0
    drop_prob_for_branch_v = 1.0

    if apply_pruning:
        print("Filtering datasets...")
        print("---------------------")

        folders = {
            'Train': train_data_folder,
            'Valid': valid_data_folder,
            'Test': test_data_folder
        }

        for name, folder in folders.items():
            print(f"Filtering {name} dataset...")
            stats = filter_dataset_folder(folder, 
                                          firing_rate_hz_range=firing_rate_hz_range, 
                                          avg_near_spike_prob_range=avg_near_spike_prob_range,
                                          soma_v_mV_mean_range=soma_v_mV_mean_range, 
                                          soma_v_mV_std_range=soma_v_mV_std_range, 
                                          inst_rate_hz_mean_range=inst_rate_hz_mean_range,
                                          inst_rate_hz_std_range=inst_rate_hz_std_range,
                                          branch_v_mV_mean_range=branch_v_mV_mean_range,
                                          branch_v_mV_std_range=branch_v_mV_std_range,
                                          drop_prob_for_firing_rate=drop_prob_for_firing_rate,
                                          drop_prob_for_near_spike_prob=drop_prob_for_near_spike_prob,
                                          drop_prob_for_soma_v=drop_prob_for_soma_v,
                                          drop_prob_for_inst_rate=drop_prob_for_inst_rate,
                                          drop_prob_for_branch_v=drop_prob_for_branch_v)
            print('-----------------------------------------------------------------------')
            print(f"Summary for {name} dataset:")
            print('----------------------------')
            print(f"Files deleted: {stats['num_deleted']}")
            if stats['num_deleted'] > 0:
                print(f"Mean soma voltage of deleted files: {stats['mean_soma_v']:.2f} mV")
                print(f"Mean soma voltage std of deleted files: {stats['std_soma_v']:.2f} mV")
                print(f"Mean firing rate of deleted files: {stats['mean_firing_rate']:.2f} Hz")
                print(f"Mean near spike probability of deleted files: {stats['mean_near_spike_prob']:.2f}")
                print(f"Mean instantaneous rate of deleted files: {stats['mean_inst_rate']:.2f} Hz")
                print(f"Mean branch voltage of deleted files: {stats['mean_branch_v']:.2f} mV")
            print('-----------------------------------------------------------------------')
            
        print("Filtering complete!")

        # collect statistics of the entire train dataset post-filtering and plot it
        train_stats = collect_dataset_statistics(train_data_folder)

        figure_title_str = "Train Dataset Statistics Post-Filtering"
        stats_fig = plot_dataset_statistics(train_stats, figure_title_str)
        plt.show()

    else:
        print("Post-filtering is disabled")



