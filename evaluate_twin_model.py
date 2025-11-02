#%% Imports

import os
import glob
import time
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import pearsonr
from torch.utils.data import Dataset, DataLoader
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_curve, auc, roc_auc_score, explained_variance_score

from neuron_model_multi_branch_filter_and_fire import MultiBranchFilterAndFireNeuron
from dataloader_FF_neuron import FFNeuronDataset
from twin_model_definitions import load_twin_model
import config

#%% Evaluation Functions

def predict_on_all_simulations(model, dataset, batch_size=8):
    """Run predictions on entire dataset."""
    model.eval()
    device = next(model.parameters()).device
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_y_spikes_pred = []
    all_y_soma_pred = []
    all_y_near_spike_pred = []
    all_y_inst_rate_pred = []
    all_y_dend_v_pred = []
    all_y_spikes_gt = []
    all_y_soma_gt = []
    all_y_near_spike_gt = []
    all_y_inst_rate_gt = []
    all_y_dend_v_gt = []

    with torch.no_grad():
        for batch_data in dataloader:

            # before passing the input to the model, aplly the scaling on the input that was used for training
            X_spikes = batch_data['X_spikes'].to(device) / model.X_scale
            y_spikes_pred, y_soma_pred, y_near_spike_pred, y_inst_rate_pred, y_dend_v_pred = model(X_spikes)
            
            # Ensure consistent dimensions by removing singleton dimensions appropriately
            y_spikes_pred = y_spikes_pred.squeeze(1).squeeze(1)          # Remove extra dimensions but keep batch
            y_soma_pred = y_soma_pred.squeeze(1).squeeze(1)              # Remove extra dimensions but keep batch
            y_near_spike_pred = y_near_spike_pred.squeeze(1).squeeze(1)  # Remove extra dimensions but keep batch
            y_inst_rate_pred = y_inst_rate_pred.squeeze(1).squeeze(1)    # Remove extra dimensions but keep batch
            y_dend_v_pred = y_dend_v_pred.squeeze(1)                     # Remove one extra dimension but keep batch and channels
            
            # Scale soma V, dendritic V and inst rate predictions back to original scale
            y_soma_pred = model.V_scale_soma * y_soma_pred + model.V_bias_soma
            y_dend_v_pred = model.V_scale_dend * y_dend_v_pred + model.V_bias_dend
            y_inst_rate_pred = y_inst_rate_pred / model.y_inst_rate_multiplier

            # Store predictions and ground truth
            all_y_spikes_pred.append(torch.sigmoid(y_spikes_pred).cpu().numpy())
            all_y_soma_pred.append(y_soma_pred.cpu().numpy())
            all_y_near_spike_pred.append(torch.sigmoid(y_near_spike_pred).cpu().numpy())
            all_y_inst_rate_pred.append(y_inst_rate_pred.cpu().numpy())
            all_y_dend_v_pred.append(y_dend_v_pred.cpu().numpy())
            all_y_spikes_gt.append(batch_data['y_spikes'].numpy())
            all_y_soma_gt.append(batch_data['y_soma'].numpy())
            all_y_near_spike_gt.append(batch_data['y_near_spike'].numpy())
            all_y_inst_rate_gt.append(batch_data['y_inst_rate'].numpy())
            all_y_dend_v_gt.append(batch_data['y_dend_v'].numpy())
    
    # Concatenate all batches along the batch dimension
    y_spikes_pred = np.concatenate(all_y_spikes_pred, axis=0)
    y_soma_pred = np.concatenate(all_y_soma_pred, axis=0)
    y_near_spike_pred = np.concatenate(all_y_near_spike_pred, axis=0)
    y_inst_rate_pred = np.concatenate(all_y_inst_rate_pred, axis=0)
    y_dend_v_pred = np.concatenate(all_y_dend_v_pred, axis=0)
    y_spikes_gt = np.concatenate(all_y_spikes_gt, axis=0)
    y_soma_gt = np.concatenate(all_y_soma_gt, axis=0)
    y_near_spike_gt = np.concatenate(all_y_near_spike_gt, axis=0)
    y_inst_rate_gt = np.concatenate(all_y_inst_rate_gt, axis=0)
    y_dend_v_gt = np.concatenate(all_y_dend_v_gt, axis=0)

    output_dict = {
        'y_spikes_pred': y_spikes_pred,
        'y_soma_pred': y_soma_pred,
        'y_near_spike_pred': y_near_spike_pred,
        'y_inst_rate_pred': y_inst_rate_pred,
        'y_dend_v_pred': y_dend_v_pred,
        'y_spikes_gt': y_spikes_gt,
        'y_soma_gt': y_soma_gt,
        'y_near_spike_gt': y_near_spike_gt,
        'y_inst_rate_gt': y_inst_rate_gt,
        'y_dend_v_gt': y_dend_v_gt,
    }

    return output_dict

def calculate_metrics(y_spikes_gt, y_spikes_pred, 
                        y_soma_gt, y_soma_pred, 
                        y_near_spike_gt, y_near_spike_pred, 
                        y_inst_rate_gt, y_inst_rate_pred, 
                        y_dend_v_gt, y_dend_v_pred, 
                        V_clip_soma_min=-102, V_clip_soma_max=-53, apply_soma_GT_clipping=True,
                        V_clip_dend_min=-50, V_clip_dend_max=50, apply_dend_GT_clipping=True,
                        desired_false_positive_rate=0.002, num_datapoints_in_scatter=20000, print_metrics=False):
    """Calculate various performance metrics for all model outputs."""
    
    if apply_soma_GT_clipping:
        # clip the soma GT signal
        y_soma_gt = np.clip(y_soma_gt, V_clip_soma_min, V_clip_soma_max)
        
    if apply_dend_GT_clipping:
        # clip the dendritic voltage GT signal
        y_dend_v_gt = np.clip(y_dend_v_gt, V_clip_dend_min, V_clip_dend_max)

    # go over all simulations and calculate metrics per simulation (spikes AUC, near_spikes AUC, soma_R2, dend_v_R2, inst_rate_R2)
    spikes_AUC_list = []
    near_spikes_AUC_list = []
    soma_R2_list = []
    dend_v_R2_list = []
    inst_rate_R2_list = []
    total_num_spikes_per_simulation = []
    for i in range(len(y_spikes_gt)):
        curr_sim_total_num_spikes = np.sum(y_spikes_gt[i])
        
        # no need to calculate metrics if there are no spikes in the current simulation
        if curr_sim_total_num_spikes > 0:
            spikes_fpr, spikes_tpr, spikes_thresholds = roc_curve(y_spikes_gt[i], y_spikes_pred[i])
            curr_sim_spikes_AUC = auc(spikes_fpr, spikes_tpr)
            near_spike_fpr, near_spike_tpr, near_spike_thresholds = roc_curve(y_near_spike_gt[i], y_near_spike_pred[i])
            curr_sim_near_spikes_AUC = auc(near_spike_fpr, near_spike_tpr)
            curr_sim_soma_R2 = explained_variance_score(y_soma_gt[i], y_soma_pred[i])
            curr_sim_dend_v_R2 = explained_variance_score(y_dend_v_gt[i].ravel(), y_dend_v_pred[i].ravel())
            curr_sim_inst_rate_R2 = explained_variance_score(y_inst_rate_gt[i].ravel(), y_inst_rate_pred[i].ravel())

            # Append to lists if no NaN or Inf exist in the current simulation
            spikes_AUC_valid = not (np.isnan(curr_sim_spikes_AUC) or np.isinf(curr_sim_spikes_AUC))
            near_spikes_AUC_valid = not (np.isnan(curr_sim_near_spikes_AUC) or np.isinf(curr_sim_near_spikes_AUC))
            soma_R2_valid = not (np.isnan(curr_sim_soma_R2) or np.isinf(curr_sim_soma_R2))
            dend_v_R2_valid = not (np.isnan(curr_sim_dend_v_R2) or np.isinf(curr_sim_dend_v_R2))
            inst_rate_R2_valid = not (np.isnan(curr_sim_inst_rate_R2) or np.isinf(curr_sim_inst_rate_R2))
            
            if spikes_AUC_valid and near_spikes_AUC_valid and soma_R2_valid and dend_v_R2_valid and inst_rate_R2_valid:
                total_num_spikes_per_simulation.append(curr_sim_total_num_spikes)
                spikes_AUC_list.append(curr_sim_spikes_AUC)
                near_spikes_AUC_list.append(curr_sim_near_spikes_AUC)
                soma_R2_list.append(curr_sim_soma_R2)
                dend_v_R2_list.append(curr_sim_dend_v_R2)
                inst_rate_R2_list.append(curr_sim_inst_rate_R2)

    # Binary classification metrics for spikes
    fpr, tpr, thresholds = roc_curve(y_spikes_gt.ravel(), y_spikes_pred.ravel())
    desired_fp_index = max(1, np.argmin(abs(fpr - desired_false_positive_rate)))
    actual_false_positive_rate = fpr[desired_fp_index]
    true_positive_at_FP = tpr[desired_fp_index]
    spikes_AUC_score = auc(fpr, tpr)
    
    # Binary classification metrics for near spikes
    if np.sum(y_near_spike_gt) > 0:  # Check if there are any positive samples
        near_spike_fpr, near_spike_tpr, near_spike_thresholds = roc_curve(y_near_spike_gt.ravel(), y_near_spike_pred.ravel())
        near_spike_AUC_score = auc(near_spike_fpr, near_spike_tpr)
        # Calculate true positive at desired FP rate for near spikes
        near_spike_desired_fp_index = max(1, np.argmin(abs(near_spike_fpr - desired_false_positive_rate)))
        near_spike_actual_false_positive_rate = near_spike_fpr[near_spike_desired_fp_index]
        near_spike_true_positive_at_FP = near_spike_tpr[near_spike_desired_fp_index]
    else:
        near_spike_fpr = np.array([0, 1])
        near_spike_tpr = np.array([0, 0])
        near_spike_thresholds = np.array([1, 0])
        near_spike_AUC_score = 0.0
        near_spike_actual_false_positive_rate = 0.0
        near_spike_true_positive_at_FP = 0.0
    
    # Regression metrics for soma voltage (here we N(0,1) scale the voltage for better numerical stability)
    soma_explained_variance = explained_variance_score(y_soma_gt.ravel(), y_soma_pred.ravel())
    soma_explained_variance_percent = 100.0 * soma_explained_variance
    soma_RMSE = np.sqrt(mean_squared_error(y_soma_gt.ravel(), y_soma_pred.ravel()))
    soma_MAE = mean_absolute_error(y_soma_gt.ravel(), y_soma_pred.ravel())
    
    # Regression metrics for instantaneous rate
    inst_rate_explained_variance = explained_variance_score(y_inst_rate_gt.ravel(), y_inst_rate_pred.ravel())
    inst_rate_explained_variance_percent = 100.0 * inst_rate_explained_variance
    inst_rate_RMSE = np.sqrt(mean_squared_error(y_inst_rate_gt.ravel(), y_inst_rate_pred.ravel()))
    inst_rate_MAE = mean_absolute_error(y_inst_rate_gt.ravel(), y_inst_rate_pred.ravel())
    
    # Regression metrics for dendritic voltage
    # Handle multi-dimensional dendritic voltage (flatten if needed)
    y_dend_v_gt_flat = y_dend_v_gt.ravel() 
    y_dend_v_pred_flat = y_dend_v_pred.ravel()
    dend_v_explained_variance = explained_variance_score(y_dend_v_gt_flat, y_dend_v_pred_flat)
    dend_v_explained_variance_percent = 100.0 * dend_v_explained_variance
    dend_v_RMSE = np.sqrt(mean_squared_error(y_dend_v_gt_flat, y_dend_v_pred_flat))
    dend_v_MAE = mean_absolute_error(y_dend_v_gt_flat, y_dend_v_pred_flat)
            
    # Prepare scatter plot data for all outputs
    max_samples = min(num_datapoints_in_scatter, len(y_soma_gt.ravel()))
    selected_indices = np.random.choice(len(y_soma_gt.ravel()), max_samples, replace=False)
    
    # Soma voltage scatter data
    scatter_soma_gt = y_soma_gt.ravel()[selected_indices]
    scatter_soma_pred = y_soma_pred.ravel()[selected_indices]
    
    # Instantaneous rate scatter data  
    scatter_inst_rate_gt = y_inst_rate_gt.ravel()[selected_indices]
    scatter_inst_rate_pred = y_inst_rate_pred.ravel()[selected_indices]
    
    # Dendritic voltage scatter data (handle multi-dimensional case)
    scatter_dend_v_gt = y_dend_v_gt_flat[selected_indices]
    scatter_dend_v_pred = y_dend_v_pred_flat[selected_indices]
    
    # Store metrics in dictionary
    metrics_dict = {
        # Spikes metrics
        'false_positive_rate': fpr,
        'true_positive_rate': tpr,
        'thresholds': thresholds,
        'requested_false_positive_rate': actual_false_positive_rate,
        'true_positive_at_FP': true_positive_at_FP,
        'AUC_score': spikes_AUC_score,
        
        # Near spike metrics
        'near_spike_false_positive_rate': near_spike_fpr,
        'near_spike_true_positive_rate': near_spike_tpr,
        'near_spike_thresholds': near_spike_thresholds,
        'near_spike_requested_false_positive_rate': near_spike_actual_false_positive_rate,
        'near_spike_true_positive_at_FP': near_spike_true_positive_at_FP,
        'near_spike_AUC_score': near_spike_AUC_score,
        
        # Soma voltage metrics
        'soma_explained_variance_percent': soma_explained_variance_percent,
        'soma_RMSE': soma_RMSE,
        'soma_MAE': soma_MAE,
        'scatter_soma_voltage_GT': scatter_soma_gt,
        'scatter_soma_voltage_pred': scatter_soma_pred,
        
        # Instantaneous rate metrics
        'inst_rate_explained_variance_percent': inst_rate_explained_variance_percent,
        'inst_rate_RMSE': inst_rate_RMSE,
        'inst_rate_MAE': inst_rate_MAE,
        'scatter_inst_rate_GT': scatter_inst_rate_gt,
        'scatter_inst_rate_pred': scatter_inst_rate_pred,
        
        # Dendritic voltage metrics
        'dend_v_explained_variance_percent': dend_v_explained_variance_percent,
        'dend_v_RMSE': dend_v_RMSE,
        'dend_v_MAE': dend_v_MAE,
        'scatter_dend_v_GT': scatter_dend_v_gt,
        'scatter_dend_v_pred': scatter_dend_v_pred,

        # Per simulation metrics
        'total_num_spikes_per_simulation': total_num_spikes_per_simulation,
        'spikes_AUC_list': spikes_AUC_list,
        'near_spikes_AUC_list': near_spikes_AUC_list,
        'soma_R2_list': soma_R2_list,
        'dend_v_R2_list': dend_v_R2_list,
        'inst_rate_R2_list': inst_rate_R2_list,
    }
    
    if print_metrics:
        print(f'Spikes AUC = {spikes_AUC_score:.4f}')
        print(f'at {actual_false_positive_rate:.4f} FP rate, TP = {true_positive_at_FP:.4f}')
        print(f'Near spike AUC = {near_spike_AUC_score:.4f}')
        print(f'at {near_spike_actual_false_positive_rate:.4f} FP rate, TP = {near_spike_true_positive_at_FP:.4f}')
        print(f'soma voltage prediction explained variance = {soma_explained_variance_percent:.2f}%')
        print(f'soma RMSE = {soma_RMSE:.2f} (mV)')
        print(f'soma MAE = {soma_MAE:.2f} (mV)')
        print(f'inst rate explained variance = {inst_rate_explained_variance_percent:.2f}%')
        print(f'inst rate RMSE = {inst_rate_RMSE:.4f}')
        print(f'inst rate MAE = {inst_rate_MAE:.4f}')
        print(f'dend v explained variance = {dend_v_explained_variance_percent:.2f}%')
        print(f'dend v RMSE = {dend_v_RMSE:.2f} (mV)')
        print(f'dend v MAE = {dend_v_MAE:.2f} (mV)')
    
    return metrics_dict

def plot_evaluation_figures(metrics_dict, voltage_granularity=10, dend_voltage_granularity=20):
    """Create extended visualization with all model outputs."""
    fig = plt.figure(figsize=(14, 12))
    
    # Create a more complex grid layout
    gs = gridspec.GridSpec(5, 6)
    gs.update(left=0.08, right=0.95, bottom=0.1, top=0.92, wspace=0.6, hspace=0.95)
    
    # Top row: ROC Curves spanning full width
    
    # Spikes ROC Curve (left half of top row)
    ax1 = plt.subplot(gs[:3, :3])
    ax1.plot(metrics_dict['false_positive_rate'], 
            metrics_dict['true_positive_rate'], 
            color='k', linewidth=2)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title(f"Spikes ROC Curve (AUC = {metrics_dict['AUC_score']:.4f})")
    ax1.grid(True, alpha=0.3)
    
    # Add inset for zoom on low FPR region
    axins = ax1.inset_axes([0.4, 0.15, 0.5, 0.5])
    axins.plot(metrics_dict['false_positive_rate'], metrics_dict['true_positive_rate'], color='k')
    axins.set_xlim(-0.001, 0.031)
    axins.set_ylim(0, 1)
    axins.grid(True)
    ax1.indicate_inset_zoom(axins)
    
    # Near Spike ROC Curve (right half of top row)
    ax2 = plt.subplot(gs[:3, 3:])
    ax2.plot(metrics_dict['near_spike_false_positive_rate'], 
            metrics_dict['near_spike_true_positive_rate'], 
            color='k', linewidth=2)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f"Near Spike ROC Curve (AUC = {metrics_dict['near_spike_AUC_score']:.4f})")
    ax2.grid(True, alpha=0.3)
    
    # Add inset for zoom on low FPR region
    axins2 = ax2.inset_axes([0.4, 0.15, 0.5, 0.5])
    axins2.plot(metrics_dict['near_spike_false_positive_rate'], metrics_dict['near_spike_true_positive_rate'], color='k')
    axins2.set_xlim(-0.001, 0.031)
    axins2.set_ylim(0, 1)
    axins2.grid(True)
    ax2.indicate_inset_zoom(axins2)
    
    # Bottom row: Scatter Plots spanning full width
    
    # Soma Voltage scatter plot (left third)
    ax3 = plt.subplot(gs[3:, :2])
    selected_soma_GT = metrics_dict['scatter_soma_voltage_GT']
    selected_soma_pred = metrics_dict['scatter_soma_voltage_pred']
    
    ax3.scatter(selected_soma_GT, selected_soma_pred, s=1, alpha=0.5, color='blue')
    
    # Set axis limits and ticks (by using 0.1% and 99.9% of the data)
    soma_voltage_lims = [
        np.percentile(selected_soma_GT, 0.1),
        np.percentile(selected_soma_GT, 99.9)
    ]
    # extend the limits by 2.5% on each side for the xylims
    soma_voltage_lims_extended = [
        soma_voltage_lims[0] - 0.025 * (soma_voltage_lims[1] - soma_voltage_lims[0]),
        soma_voltage_lims[1] + 0.025 * (soma_voltage_lims[1] - soma_voltage_lims[0])
    ]
    ax3.plot(soma_voltage_lims_extended, soma_voltage_lims_extended, 'k--', alpha=0.5)
    
    # Customize voltage ticks
    soma_voltage_ticks = np.arange(
        np.floor(soma_voltage_lims[0] / voltage_granularity) * voltage_granularity,
        np.ceil(soma_voltage_lims[1] / voltage_granularity) * voltage_granularity,
        voltage_granularity
    )
    ax3.set_xticks(soma_voltage_ticks)
    ax3.set_yticks(soma_voltage_ticks)
    ax3.set_xlim(soma_voltage_lims_extended)
    ax3.set_ylim(soma_voltage_lims_extended)
    ax3.set_xlabel('Ground Truth Voltage (mV)')
    ax3.set_ylabel('Predicted Voltage (mV)')
    ax3.set_title(f"Soma Voltage\n(R² = {metrics_dict['soma_explained_variance_percent']:.1f}%)")
    ax3.grid(True, alpha=0.3)
    
    # Instantaneous Rate scatter plot (middle third)
    ax4 = plt.subplot(gs[3:, 2:4])
    selected_inst_rate_GT = metrics_dict['scatter_inst_rate_GT']
    selected_inst_rate_pred = metrics_dict['scatter_inst_rate_pred']
    
    ax4.scatter(selected_inst_rate_GT, selected_inst_rate_pred, s=1, alpha=0.5, color='green')
    
    # Set axis limits and ticks (by using 0.1% and 99.9% of the data)
    inst_rate_lims = [
        np.percentile(selected_inst_rate_GT, 0.1),
        np.percentile(selected_inst_rate_GT, 99.9)
    ]
    # extend the limits by 2.5% on each side for the xylims
    inst_rate_lims_extended = [
        inst_rate_lims[0] - 0.025 * (inst_rate_lims[1] - inst_rate_lims[0]),
        inst_rate_lims[1] + 0.025 * (inst_rate_lims[1] - inst_rate_lims[0])
    ]
    ax4.plot(inst_rate_lims, inst_rate_lims, 'k--', alpha=0.5)
    ax4.set_xlim(inst_rate_lims_extended)
    ax4.set_ylim(inst_rate_lims_extended)
    ax4.set_xlabel('Ground Truth Inst. Rate')
    ax4.set_ylabel('Predicted Inst. Rate')
    ax4.set_title(f"Instantaneous Rate\n(R² = {metrics_dict['inst_rate_explained_variance_percent']:.1f}%)")
    ax4.grid(True, alpha=0.3)
    
    # Dendritic Voltage scatter plot (right third)
    ax5 = plt.subplot(gs[3:, 4:])
    selected_dend_v_GT = metrics_dict['scatter_dend_v_GT']
    selected_dend_v_pred = metrics_dict['scatter_dend_v_pred']
    
    # Set axis limits and ticks (by using 0.1% and 99.9% of the data)
    dend_v_lims = [
        np.percentile(selected_dend_v_GT, 0.1),
        np.percentile(selected_dend_v_GT, 99.9)
    ]
    # extend the limits by 2.5% on each side for the xylims
    dend_v_lims_extended = [
        dend_v_lims[0] - 0.025 * (dend_v_lims[1] - dend_v_lims[0]),
        dend_v_lims[1] + 0.025 * (dend_v_lims[1] - dend_v_lims[0])
    ]

    ax5.plot(dend_v_lims, dend_v_lims, 'k--', alpha=0.5)
    ax5.scatter(selected_dend_v_GT, selected_dend_v_pred, s=1, alpha=0.5, color='red')
    
    # Customize voltage ticks (similar to soma voltage)
    dend_v_ticks = np.arange(
        np.floor(dend_v_lims[0] / dend_voltage_granularity) * dend_voltage_granularity,
        np.ceil(dend_v_lims[1] / dend_voltage_granularity) * dend_voltage_granularity,
        dend_voltage_granularity
    )
    ax5.set_xticks(dend_v_ticks)
    ax5.set_yticks(dend_v_ticks)
    ax5.set_xlim(dend_v_lims_extended)
    ax5.set_ylim(dend_v_lims_extended)

    ax5.set_xlabel('Ground Truth Voltage (mV)')
    ax5.set_ylabel('Predicted Voltage (mV)')
    ax5.set_title(f"Dendritic Voltage\n(R² = {metrics_dict['dend_v_explained_variance_percent']:.1f}%)")
    ax5.grid(True, alpha=0.3)

    return fig
    
def plot_per_simulation_metrics(metrics_dict):

    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(5, 6)
    gs.update(left=0.08, right=0.95, bottom=0.1, top=0.92, wspace=0.95, hspace=0.9)

    # top row: spikes AUC, soma R2
    # bottom row: near spikes AUC, dendritic voltage R2, instantaneous rate R2

    ax1 = plt.subplot(gs[:3, :3])
    ax2 = plt.subplot(gs[:3, 3:])
    ax3 = plt.subplot(gs[3:, :2])
    ax4 = plt.subplot(gs[3:, 2:4])
    ax5 = plt.subplot(gs[3:, 4:])

    # spikes AUC vs total number of spikes
    ax1.scatter(metrics_dict['total_num_spikes_per_simulation'], metrics_dict['spikes_AUC_list'], alpha=0.9, s=1, c='red')
    ax1.set_xlabel('Total Number of Spikes')
    ax1.set_ylabel('Spikes AUC')
    ax1.set_title('Spikes AUC vs Total Number of Spikes')
    y_axis_1st_percentile = np.percentile(metrics_dict['spikes_AUC_list'], 2)
    ax1.set_ylim(y_axis_1st_percentile, 1.01)

    # soma R2 vs total number of spikes
    ax2.scatter(metrics_dict['total_num_spikes_per_simulation'], metrics_dict['soma_R2_list'], alpha=0.9, s=1, c='red')
    ax2.set_xlabel('Total Number of Spikes')
    ax2.set_ylabel('Soma R2')
    ax2.set_title('Soma R2 vs Total Number of Spikes')
    y_axis_1st_percentile = np.percentile(metrics_dict['soma_R2_list'], 2)
    ax2.set_ylim(y_axis_1st_percentile, 1.02)

    # near spikes AUC vs total number of spikes
    ax3.scatter(metrics_dict['total_num_spikes_per_simulation'], metrics_dict['near_spikes_AUC_list'], alpha=0.9, s=1, c='orange')
    ax3.set_xlabel('Total Number of Spikes')
    ax3.set_ylabel('Near Spikes AUC')
    ax3.set_title('Near Spikes AUC')
    y_axis_1st_percentile = np.percentile(metrics_dict['near_spikes_AUC_list'], 2)
    ax3.set_ylim(y_axis_1st_percentile, 1.01)

    # dendritic voltage R2 vs total number of spikes
    ax4.scatter(metrics_dict['total_num_spikes_per_simulation'], metrics_dict['dend_v_R2_list'], alpha=0.9, s=1, c='orange')
    ax4.set_xlabel('Total Number of Spikes')
    ax4.set_ylabel('Dendritic Voltage R2')
    ax4.set_title('Dendritic Voltage R2')
    y_axis_1st_percentile = np.percentile(metrics_dict['dend_v_R2_list'], 2)
    ax4.set_ylim(y_axis_1st_percentile, 1.02)

    # instantaneous rate R2 vs total number of spikes
    ax5.scatter(metrics_dict['total_num_spikes_per_simulation'], metrics_dict['inst_rate_R2_list'], alpha=0.9, s=1, c='orange')
    ax5.set_xlabel('Total Number of Spikes')
    ax5.set_ylabel('Instantaneous Rate R2')
    ax5.set_title('Instantaneous Rate R2')
    y_axis_1st_percentile = np.percentile(metrics_dict['inst_rate_R2_list'], 2)
    ax5.set_ylim(y_axis_1st_percentile, 1.02)

    return fig

def evaluate_model_on_dataset(model, dataset, batch_size=8, num_datapoints_in_scatter=20000, verbose=1):
    """Complete model evaluation pipeline."""
    
    # Get predictions
    output_dict = predict_on_all_simulations(model, dataset, batch_size=batch_size)

    y_spikes_pred = output_dict['y_spikes_pred']
    y_soma_pred = output_dict['y_soma_pred']
    y_near_spike_pred = output_dict['y_near_spike_pred']
    y_inst_rate_pred = output_dict['y_inst_rate_pred']
    y_dend_v_pred = output_dict['y_dend_v_pred']
    
    y_spikes_gt = output_dict['y_spikes_gt']
    y_soma_gt = output_dict['y_soma_gt']
    y_near_spike_gt = output_dict['y_near_spike_gt']
    y_inst_rate_gt = output_dict['y_inst_rate_gt']
    y_dend_v_gt = output_dict['y_dend_v_gt']

    # Calculate metrics
    metrics_dict = calculate_metrics(
        y_spikes_gt, y_spikes_pred, 
        y_soma_gt, y_soma_pred,
        y_near_spike_gt, y_near_spike_pred, 
        y_inst_rate_gt, y_inst_rate_pred, 
        y_dend_v_gt, y_dend_v_pred,
        model.V_clip_soma_min, model.V_clip_soma_max, apply_soma_GT_clipping=True,
        V_clip_dend_min=model.V_clip_dend_min, V_clip_dend_max=model.V_clip_dend_max, apply_dend_GT_clipping=True,
        num_datapoints_in_scatter=num_datapoints_in_scatter, print_metrics=(verbose > 0)
    )
    
    return metrics_dict

def calculate_calibration_metrics(y_spikes_pred, y_spikes_gt, num_bins_per_simulation=16):

    num_simulations, num_time_points = y_spikes_gt.shape
    # print(f'y_spikes_gt.shape = {y_spikes_gt.shape}')

    # split the time axis into K bins and calculate the average firing rate in each bin
    K = num_bins_per_simulation
    bin_size = num_time_points // num_bins_per_simulation
    y_spikes_pred = y_spikes_pred.reshape(num_simulations, K, bin_size)
    y_spikes_gt = y_spikes_gt.reshape(num_simulations, K, bin_size)
    # print(f'y_spikes_gt.shape = {y_spikes_gt.shape}')

    # stack the bins along the simulation axis so that we have a 2D array of shape (num_simulations * K, bin_size)
    y_spikes_pred = y_spikes_pred.reshape(-1, bin_size)
    y_spikes_gt = y_spikes_gt.reshape(-1, bin_size)

    # calc correlation and variance explained
    calib_corr = pearsonr(y_spikes_gt.mean(axis=1), y_spikes_pred.mean(axis=1))[0]
    calib_explained_var = explained_variance_score(y_spikes_gt.mean(axis=1), y_spikes_pred.mean(axis=1))

    return calib_corr, calib_explained_var

def display_calibration_figure(y_spikes_pred, y_spikes_gt):

    num_simulations, num_time_points = y_spikes_gt.shape

    # split the time axis into K bins and calculate the average firing rate in each bin
    K = 16
    bin_size = num_time_points // K
    y_spikes_pred = y_spikes_pred.reshape(num_simulations, K, bin_size)
    y_spikes_gt = y_spikes_gt.reshape(num_simulations, K, bin_size)

    # stack the bins along the simulation axis so that we have a 2D array of shape (num_simulations * K, bin_size)
    y_spikes_pred = y_spikes_pred.reshape(-1, bin_size)
    y_spikes_gt = y_spikes_gt.reshape(-1, bin_size)

    y_spikes_pred_flat = y_spikes_pred.ravel()
    y_spikes_gt_flat = y_spikes_gt.ravel()
    bins = np.linspace(0, 1, 100)

    # calc correlation and variance explained
    calib_corr = pearsonr(y_spikes_gt.mean(axis=1), y_spikes_pred.mean(axis=1))[0]
    calib_explained_var = explained_variance_score(y_spikes_gt.mean(axis=1), y_spikes_pred.mean(axis=1))

    # Plot the calibration curve of the average firing rate for each simulation
    title_top_row = f'Calibration Curve of Average Firing Rate per {bin_size}ms bins'
    title_bottom_row = f'Correlation: {calib_corr:.4f}, Explained Variance: {100 * calib_explained_var:.2f}%'

    fig, axs = plt.subplots(2, 1, figsize=(8, 10), height_ratios=[3, 1])     
    fig.subplots_adjust(hspace=0.25)
    axs[0].scatter(y_spikes_gt.mean(axis=1), y_spikes_pred.mean(axis=1), s=1, label='Simulation')
    axs[0].plot([0, 0.2], [0, 0.2], 'k--', label='y=x')
    axs[0].set_xlabel('GT Avg Firing Rate (Spikes per ms = mean(y_spikes(t)))')
    axs[0].set_ylabel('Predicted Avg Firing Rate (Spikes per ms = mean(spike_prob(t)))')
    axs[0].set_title(f'{title_top_row}\n{title_bottom_row}')
    axs[0].legend()

    # 1. All data
    axs[1].hist(y_spikes_pred_flat[y_spikes_gt_flat == 1], bins=bins, alpha=0.5, color='green', label='GT = Spike')
    axs[1].hist(y_spikes_pred_flat[y_spikes_gt_flat == 0], bins=bins, alpha=0.5, color='red', label='GT = No Spike')
    axs[1].set_title('Original Predictions Distribution')
    axs[1].legend()
    axs[1].set_xlabel('Predicted Probability')
    axs[1].set_ylabel('Density')
    axs[1].set_yscale('log')

    return fig

def display_sample_predictions_minimal(model, test_data_folder, neuron_model, sample_basename=None):

    device = next(model.parameters()).device  # Get device from model
    
    # Extract neuron parameters in a way that works for both FF and BS neurons
    if hasattr(neuron_model, 'v_threshold'):
        # Filter and Fire neuron
        neuron_v_threshold = neuron_model.v_threshold
        neuron_v_reset = neuron_model.v_reset
        neuron_v_hard_reset = neuron_model.v_hard_reset
        neuron_num_segments = neuron_model.total_segments
    elif hasattr(neuron_model, 'spike_detection_threshold_mV'):
        # Ball and Stick neuron
        neuron_v_threshold = neuron_model.soma_voltage_cap_mV
        neuron_v_reset = neuron_model.epas_mV
        neuron_v_hard_reset = neuron_model.epas_mV - 5
        neuron_num_segments = neuron_model.num_segments
    else:
        # Fallback for unknown neuron type
        neuron_v_threshold = 0.0
        neuron_v_reset = -70.0
        neuron_v_hard_reset = -75.0
        neuron_num_segments = 4
    
    if sample_basename is None:
        sample_filename = np.random.choice(glob.glob(os.path.join(test_data_folder, 'output_spikes', '*.npy')))
        sample_basename = os.path.basename(sample_filename)

    X_inputs = np.load(os.path.join(test_data_folder, 'inputs', sample_basename))
    y_spike = np.load(os.path.join(test_data_folder, 'output_spikes', sample_basename))
    y_soma = np.load(os.path.join(test_data_folder, 'output_soma_v', sample_basename))

    print(f'Selected sample: {sample_basename}')
    print(f'X_inputs shape: {X_inputs.shape}, dtype: {X_inputs.dtype}')
    print(f'y_spike shape: {y_spike.shape}, dtype: {y_spike.dtype}')
    print(f'y_soma shape: {y_soma.shape}, dtype: {y_soma.dtype}')

    X_exc, X_inh = X_inputs[..., 0], X_inputs[..., 1]

    # Convert input to tensor and prepare for model
    X_spikes = torch.FloatTensor(X_inputs.transpose(2, 0, 1)).unsqueeze(0)  # Add batch dimension
    X_spikes = X_spikes.to(device) / model.X_scale

    # Get model predictions
    model.eval()
    with torch.no_grad():
        y_spikes_pred, y_soma_pred, y_near_spike_pred, y_inst_rate_pred, y_dend_v_pred = model(X_spikes)
        y_spikes_pred = torch.sigmoid(y_spikes_pred.squeeze()).cpu().numpy()
        y_soma_pred = (y_soma_pred.squeeze().cpu().numpy() * model.V_scale_soma) + model.V_bias_soma
        y_inst_rate_pred = y_inst_rate_pred.squeeze().cpu().numpy() / model.y_inst_rate_multiplier
        y_dend_v_pred = y_dend_v_pred.squeeze().cpu().numpy() * model.V_scale_dend + model.V_bias_dend

    spike_threshold_for_plot = 0.5
    spike_times_gt = np.where(y_spike)[0]
    spike_times_pred = np.where(y_spikes_pred > spike_threshold_for_plot)[0]

    # Calculate and display metrics for this sample
    # Spike prediction metrics
    spike_pred_binary = y_spikes_pred > spike_threshold_for_plot
    true_positives = np.sum(spike_pred_binary & y_spike)
    false_positives = np.sum(spike_pred_binary & ~y_spike)
    false_negatives = np.sum(~spike_pred_binary & y_spike)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    total_num_of_GT_spikes = np.sum(y_spike)
    avg_spike_rate = 1000 * total_num_of_GT_spikes / len(y_spike)
    if total_num_of_GT_spikes > 0:
        auc_score = roc_auc_score(y_spike, y_spikes_pred)
    else:
        auc_score = 'N/A'

    # Voltage prediction metrics
    voltage_mae = np.mean(np.abs(y_soma - y_soma_pred))
    voltage_rmse = np.sqrt(np.mean((y_soma - y_soma_pred)**2))
    voltage_r2 = 1 - np.sum((y_soma - y_soma_pred)**2) / np.sum((y_soma - np.mean(y_soma))**2)

    # Create figure for visualization
    fig = plt.figure(figsize=(15, 24))
    gs = gridspec.GridSpec(6, 1, height_ratios=[3, 1, 2, 3, 1, 2])
    plt.subplots_adjust(hspace=0.4)

    # Select random zoom window
    zoomin_duration_ms = 1024
    max_start_time = len(y_spike) - zoomin_duration_ms
    zoom_start_time = np.random.randint(0, max_start_time)
    zoom_end_time = zoom_start_time + zoomin_duration_ms

    # Plot mixed input signals (full)
    ax1 = plt.subplot(gs[0])
    seg_offsets_vec = 5 * np.arange(neuron_num_segments)

    ax1.plot(X_exc.T + seg_offsets_vec, 'r', alpha=0.7)
    ax1.plot(-X_inh.T + seg_offsets_vec, 'b', alpha=0.7)

    # Add zoom rectangle for input signals
    ymin, ymax = ax1.get_ylim()
    rect1 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax1.add_patch(rect1)

    ax1.set_xlim(-5, len(y_spike) + 5)
    ax1.set_title('Mixed Input Signals on Dendritic Segments')
    ax1.set_ylabel('Input Strength')
    ax1.set_xticks([])
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot output spikes (full)
    ax2 = plt.subplot(gs[1])
    ax2.plot(y_spikes_pred, 'cyan', alpha=0.7, label='Spike Probability')
    ax2.scatter(spike_times_gt, [1.3] * len(spike_times_gt), c='purple', marker='|', 
                s=300, label='Ground Truth Spikes')
    ax2.scatter(spike_times_pred, [1.1] * len(spike_times_pred), c='cyan', marker='|', 
                s=300, label='Predicted Spikes')

    # Add zoom rectangle for spikes
    rect2 = plt.Rectangle((zoom_start_time, -0.05), zoomin_duration_ms, 1.6,
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax2.add_patch(rect2)

    ax2.set_ylim(-0.1, 1.7)
    ax2.set_xlim(-5, len(y_spike) + 5)
    spike_title_str = f'Output Spikes: Ground Truth vs Predictions'
    try:
        spike_title_str += f'\nPrecision = {precision:.4f}, Recall = {recall:.4f}, AUC = {auc_score:.4f}'
    except:
        spike_title_str += f'\nPrecision = {precision:.4f}, Recall = {recall:.4f}, AUC = {auc_score}'
    ax2.set_title(spike_title_str)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.set_xticks([])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Plot soma voltage (full)
    ax3 = plt.subplot(gs[2])
    ax3.plot(y_soma, 'purple', label='Ground Truth')
    ax3.plot(y_soma_pred, 'cyan', label='Prediction', alpha=0.7)
    ax3.axhline(y=neuron_v_threshold, color='gray', linestyle='--', alpha=0.8, label='Threshold')
    ax3.axhline(y=neuron_v_reset, color='gray', linestyle=':', alpha=0.6, label='Reset')
    if neuron_v_hard_reset is not None:
        ax3.axhline(y=neuron_v_hard_reset, color='gray', linestyle='-.', alpha=0.78, label='Hard Reset')

    # Add zoom rectangle for voltage
    ymin, ymax = ax3.get_ylim()
    rect3 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax3.add_patch(rect3)

    ax3.set_xlabel('Time (ms)')
    ax3.set_ylabel('Voltage (mV)')
    soma_title_str = f'Soma Voltage: Ground Truth vs Predictions'
    soma_title_str += f'\nMAE = {voltage_mae:.2f} mV, RMSE = {voltage_rmse:.2f} mV, R² = {voltage_r2:.4f}'
    ax3.set_title(soma_title_str)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_xlim(-5, len(y_spike) + 5)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Plot zoomed mixed input signals
    ax4 = plt.subplot(gs[3])
    ax4.plot(X_exc.T + seg_offsets_vec, 'r', alpha=0.7)
    ax4.plot(-X_inh.T + seg_offsets_vec, 'b', alpha=0.7)

    ax4.set_xlim(zoom_start_time - 5, zoom_end_time + 5)
    ax4.set_title(f'Mixed Input Signals (Zoom: {zoom_start_time}ms - {zoom_end_time}ms)')
    ax4.set_ylabel('Input Strength')
    ax4.set_xticks([])
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)

    # Plot zoomed output spikes
    ax5 = plt.subplot(gs[4])
    ax5.plot(y_spikes_pred, 'cyan', alpha=0.7, label='Spike Probability')
    ax5.scatter(spike_times_gt, [1.3] * len(spike_times_gt), c='purple', marker='|', 
                s=300, label='Ground Truth Spikes')
    ax5.scatter(spike_times_pred, [1.1] * len(spike_times_pred), c='cyan', marker='|', 
                s=300, label='Predicted Spikes')

    ax5.set_ylim(-0.1, 1.5)
    ax5.set_xlim(zoom_start_time - 5, zoom_end_time + 5)
    ax5.set_title(f'Output Spikes (Zoom: {zoom_start_time}ms - {zoom_end_time}ms)')
    ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax5.set_xticks([])
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Plot zoomed soma voltage
    ax6 = plt.subplot(gs[5])
    ax6.plot(y_soma, 'purple', label='Ground Truth')
    ax6.plot(y_soma_pred, 'cyan', label='Prediction', alpha=0.7)
    ax6.axhline(y=neuron_v_threshold, color='gray', linestyle='--', alpha=0.8, label='Threshold')
    ax6.axhline(y=neuron_v_reset, color='gray', linestyle=':', alpha=0.6, label='Reset')
    if neuron_v_hard_reset is not None:
        ax6.axhline(y=neuron_v_hard_reset, color='gray', linestyle='-.', alpha=0.78, label='Hard Reset')

    ax6.set_xlabel('Time (ms)')
    ax6.set_ylabel('Voltage (mV)')
    ax6.set_title(f'Soma Voltage (Zoom: {zoom_start_time}ms - {zoom_end_time}ms)')
    ax6.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax6.set_xlim(zoom_start_time - 5, zoom_end_time + 5)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    plt.tight_layout()

    print('---------------------------------------------------------')
    print('Metrics for current sample:')
    print('---------------------------')
    print(f'Total number of GT spikes: {total_num_of_GT_spikes} ({avg_spike_rate:.2f} Hz)')
    print(f'Spike Detection:')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    try:
        print(f'  AUC: {auc_score:.4f}')
    except:
        print(f'  AUC: {auc_score}')

    print(f'Voltage Prediction:')
    print(f'  MAE: {voltage_mae:.2f} mV')
    print(f'  RMSE: {voltage_rmse:.2f} mV')
    print(f'  R²: {voltage_r2:.4f}')
    print('---------------------------------------------------------')

    return fig


def display_sample_predictions_full(model, test_data_folder, neuron_model, sample_basename=None):

    device = next(model.parameters()).device  # Get device from model

    # Extract neuron parameters in a way that works for both FF and BS neurons
    if hasattr(neuron_model, 'v_threshold'):
        # Filter and Fire neuron
        neuron_v_threshold = neuron_model.v_threshold
        neuron_v_reset = neuron_model.v_reset
        neuron_v_hard_reset = neuron_model.v_hard_reset
        neuron_num_segments = neuron_model.total_segments
    elif hasattr(neuron_model, 'spike_detection_threshold_mV'):
        # Ball and Stick neuron
        neuron_v_threshold = neuron_model.soma_voltage_cap_mV
        neuron_v_reset = neuron_model.epas_mV
        neuron_v_hard_reset = neuron_model.epas_mV - 5
        neuron_num_segments = neuron_model.num_segments
    else:
        # Fallback for unknown neuron type
        neuron_v_threshold = 0.0
        neuron_v_reset = -70.0
        neuron_v_hard_reset = -75.0
        neuron_num_segments = 4

    if sample_basename is None:
        sample_filename = np.random.choice(glob.glob(os.path.join(test_data_folder, 'output_spikes', '*.npy')))
        sample_basename = os.path.basename(sample_filename)

    # Load all inputs and ground truth outputs
    X_inputs = np.load(os.path.join(test_data_folder, 'inputs', sample_basename))
    y_spike_gt = np.load(os.path.join(test_data_folder, 'output_spikes', sample_basename))
    y_soma_gt = np.load(os.path.join(test_data_folder, 'output_soma_v', sample_basename))
    y_near_spike_gt = np.load(os.path.join(test_data_folder, 'output_near_spikes', sample_basename))
    y_inst_rate_gt = np.load(os.path.join(test_data_folder, 'output_inst_rate', sample_basename))
    y_dend_v_gt = np.load(os.path.join(test_data_folder, 'intermidiate_branch_v', sample_basename)).astype(np.float32)

    print(f'Selected sample: {sample_basename}')
    print(f'X_inputs shape: {X_inputs.shape}, dtype: {X_inputs.dtype}')
    print(f'y_spike_gt shape: {y_spike_gt.shape}, dtype: {y_spike_gt.dtype}')
    print(f'y_soma_gt shape: {y_soma_gt.shape}, dtype: {y_soma_gt.dtype}')
    print(f'y_near_spike_gt shape: {y_near_spike_gt.shape}, dtype: {y_near_spike_gt.dtype}')
    print(f'y_inst_rate_gt shape: {y_inst_rate_gt.shape}, dtype: {y_inst_rate_gt.dtype}')
    print(f'y_dend_v_gt shape: {y_dend_v_gt.shape}, dtype: {y_dend_v_gt.dtype}')

    X_exc, X_inh = X_inputs[..., 0], X_inputs[..., 1]

    # Convert input to tensor and prepare for model
    X_spikes = torch.FloatTensor(X_inputs.transpose(2, 0, 1)).unsqueeze(0)  # Add batch dimension
    X_spikes = X_spikes.to(device) / model.X_scale

    # Get all model predictions
    model.eval()
    with torch.no_grad():
        y_spikes_pred, y_soma_pred, y_near_spike_pred, y_inst_rate_pred, y_dend_v_pred = model(X_spikes)
        y_spikes_pred = torch.sigmoid(y_spikes_pred.squeeze()).cpu().numpy()
        y_soma_pred = (y_soma_pred.squeeze().cpu().numpy() * model.V_scale_soma) + model.V_bias_soma
        y_near_spike_pred = torch.sigmoid(y_near_spike_pred.squeeze()).cpu().numpy()
        y_inst_rate_pred = y_inst_rate_pred.squeeze().cpu().numpy() / model.y_inst_rate_multiplier
        y_dend_v_pred = y_dend_v_pred.squeeze().cpu().numpy() * model.V_scale_dend + model.V_bias_dend

    # Handle dendritic voltage dimensions if needed
    if len(y_dend_v_gt.shape) == 1:
        # Single branch case - keep as is
        pass
    else:
        # Multi-branch case - if there's only one branch, squeeze the first dimension
        if y_dend_v_gt.shape[0] == 1:
            y_dend_v_gt = y_dend_v_gt.squeeze(0)
            y_dend_v_pred = y_dend_v_pred.squeeze(0) if len(y_dend_v_pred.shape) > 1 else y_dend_v_pred

    spike_threshold_for_plot = 0.5
    spike_times_gt = np.where(y_spike_gt)[0]
    spike_times_pred = np.where(y_spikes_pred > spike_threshold_for_plot)[0]

    # Calculate metrics for titles
    # Spike prediction metrics
    spike_pred_binary = y_spikes_pred > spike_threshold_for_plot
    true_positives = np.sum(spike_pred_binary & y_spike_gt)
    false_positives = np.sum(spike_pred_binary & ~y_spike_gt)
    false_negatives = np.sum(~spike_pred_binary & y_spike_gt)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    total_num_of_GT_spikes = np.sum(y_spike_gt)
    avg_spike_rate = 1000 * total_num_of_GT_spikes / len(y_spike_gt)
    if total_num_of_GT_spikes > 0:
        spike_auc_score = roc_auc_score(y_spike_gt, y_spikes_pred)
    else:
        spike_auc_score = 0.0

    # Soma voltage metrics
    soma_mae = np.mean(np.abs(y_soma_gt - y_soma_pred))
    soma_rmse = np.sqrt(np.mean((y_soma_gt - y_soma_pred)**2))
    soma_r2 = 1 - np.sum((y_soma_gt - y_soma_pred)**2) / np.sum((y_soma_gt - np.mean(y_soma_gt))**2)

    # Near spike metrics
    near_spike_auc = roc_auc_score(y_near_spike_gt, y_near_spike_pred) if np.sum(y_near_spike_gt) > 0 else 0.0
    near_spike_mae = np.mean(np.abs(y_near_spike_gt - y_near_spike_pred))

    # Instantaneous rate metrics
    inst_rate_mae = np.mean(np.abs(y_inst_rate_gt - y_inst_rate_pred))
    inst_rate_rmse = np.sqrt(np.mean((y_inst_rate_gt - y_inst_rate_pred)**2))
    inst_rate_r2 = 1 - np.sum((y_inst_rate_gt - y_inst_rate_pred)**2) / np.sum((y_inst_rate_gt - np.mean(y_inst_rate_gt))**2)

    # Dendritic voltage metrics
    dend_v_mae = np.mean(np.abs(y_dend_v_gt - y_dend_v_pred))
    dend_v_rmse = np.sqrt(np.mean((y_dend_v_gt - y_dend_v_pred)**2))
    dend_v_r2 = 1 - np.sum((y_dend_v_gt - y_dend_v_pred)**2) / np.sum((y_dend_v_gt - np.mean(y_dend_v_gt))**2)

    # Select random zoom window
    zoomin_duration_ms = 512
    max_start_time = len(y_spike_gt) - zoomin_duration_ms
    zoom_start_time = np.random.randint(0, max_start_time)
    zoom_end_time = zoom_start_time + zoomin_duration_ms

    # Create figure with both full and zoomed views
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 2, width_ratios=[3, 2], height_ratios=[2, 1, 2, 1, 1, 1])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    # Get segment offsets for input visualization
    seg_offsets_vec = 5 * np.arange(neuron_num_segments)

    # 1. Plot mixed input signals (full and zoomed)
    ax1_full = plt.subplot(gs[0, 0])
    ax1_full.plot(X_exc.T + seg_offsets_vec, 'r', alpha=0.7, label='Excitatory')
    ax1_full.plot(-X_inh.T + seg_offsets_vec, 'b', alpha=0.7, label='Inhibitory')
    ax1_full.set_xlim(0, len(y_spike_gt))
    ax1_full.set_title('Mixed Input Signals on Dendritic Segments (Full)')
    ax1_full.set_ylabel('Input Strength')
    ax1_full.spines['top'].set_visible(False)
    ax1_full.spines['right'].set_visible(False)

    # Add zoom rectangle
    ymin, ymax = ax1_full.get_ylim()
    rect1 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                    fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax1_full.add_patch(rect1)

    ax1_zoom = plt.subplot(gs[0, 1])
    ax1_zoom.plot(X_exc.T + seg_offsets_vec, 'r', alpha=0.7, label='Excitatory')
    ax1_zoom.plot(-X_inh.T + seg_offsets_vec, 'b', alpha=0.7, label='Inhibitory')
    ax1_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax1_zoom.set_title(f'Zoomed View ({zoom_start_time}-{zoom_end_time} ms)')
    ax1_zoom.spines['top'].set_visible(False)
    ax1_zoom.spines['right'].set_visible(False)

    # 2. Plot output spikes (full and zoomed)
    ax2_full = plt.subplot(gs[1, 0])
    ax2_full.plot(y_spikes_pred, 'cyan', alpha=0.7, label='Spike Probability')
    ax2_full.scatter(spike_times_gt, [1.2] * len(spike_times_gt), c='purple', marker='|', 
            s=200, label='Ground Truth Spikes')
    ax2_full.scatter(spike_times_pred, [1.0] * len(spike_times_pred), c='cyan', marker='|', 
            s=200, label='Predicted Spikes')
    ax2_full.set_ylim(-0.1, 1.4)
    ax2_full.set_xlim(0, len(y_spike_gt))
    ax2_full.set_title(f'Output Spikes (Full) - AUC = {spike_auc_score:.4f}')
    ax2_full.set_ylabel('Spike Probability')
    ax2_full.legend()
    ax2_full.spines['top'].set_visible(False)
    ax2_full.spines['right'].set_visible(False)

    # Add zoom rectangle
    rect2 = plt.Rectangle((zoom_start_time, -0.05), zoomin_duration_ms, 1.4,
                    fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax2_full.add_patch(rect2)

    ax2_zoom = plt.subplot(gs[1, 1])
    ax2_zoom.plot(y_spikes_pred, 'cyan', alpha=0.7, label='Spike Probability')
    ax2_zoom.scatter(spike_times_gt, [1.2] * len(spike_times_gt), c='purple', marker='|', 
            s=200, label='Ground Truth Spikes')
    ax2_zoom.scatter(spike_times_pred, [1.0] * len(spike_times_pred), c='cyan', marker='|', 
            s=200, label='Predicted Spikes')
    ax2_zoom.set_ylim(-0.1, 1.4)
    ax2_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax2_zoom.set_title('Zoomed View')
    ax2_zoom.spines['top'].set_visible(False)
    ax2_zoom.spines['right'].set_visible(False)

    # 3. Plot soma voltage (full and zoomed)
    ax3_full = plt.subplot(gs[2, 0])
    ax3_full.plot(y_soma_gt, 'purple', label='Ground Truth', linewidth=1.5)
    ax3_full.plot(y_soma_pred, 'cyan', label='Prediction', alpha=0.8, linewidth=1.5)
    ax3_full.axhline(y=neuron_v_threshold, color='gray', linestyle='--', alpha=0.8, label='Threshold')
    ax3_full.axhline(y=neuron_v_reset, color='gray', linestyle=':', alpha=0.6, label='Reset')
    if neuron_v_hard_reset is not None:
        ax3_full.axhline(y=neuron_v_hard_reset, color='gray', linestyle='-.', alpha=0.78, label='Hard Reset')
    ax3_full.set_xlim(0, len(y_spike_gt))
    ax3_full.set_title(f'Soma Voltage (Full) - RMSE = {soma_rmse:.2f} mV, R² = {soma_r2:.4f}')
    ax3_full.set_ylabel('Voltage (mV)')
    ax3_full.spines['top'].set_visible(False)
    ax3_full.spines['right'].set_visible(False)

    # Add zoom rectangle
    ymin, ymax = ax3_full.get_ylim()
    rect3 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax3_full.add_patch(rect3)

    ax3_zoom = plt.subplot(gs[2, 1])
    ax3_zoom.plot(y_soma_gt, 'purple', label='Ground Truth', linewidth=1.5)
    ax3_zoom.plot(y_soma_pred, 'cyan', label='Prediction', alpha=0.8, linewidth=1.5)
    ax3_zoom.axhline(y=neuron_v_threshold, color='gray', linestyle='--', alpha=0.8, label='Threshold')
    ax3_zoom.axhline(y=neuron_v_reset, color='gray', linestyle=':', alpha=0.6, label='Reset')
    if neuron_v_hard_reset is not None:
        ax3_zoom.axhline(y=neuron_v_hard_reset, color='gray', linestyle='-.', alpha=0.78, label='Hard Reset')
    ax3_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax3_zoom.set_title('Zoomed View')
    ax3_zoom.spines['top'].set_visible(False)
    ax3_zoom.spines['right'].set_visible(False)

    # 4. Plot near spike predictions (full and zoomed)
    ax4_full = plt.subplot(gs[3, 0])
    ax4_full.plot(y_near_spike_gt, 'purple', label='Ground Truth', linewidth=1.5)
    ax4_full.plot(y_near_spike_pred, 'cyan', label='Prediction', alpha=0.8, linewidth=1.5)
    ax4_full.set_xlim(0, len(y_spike_gt))
    ax4_full.set_title(f'Near Spike Predictions (Full) - AUC = {near_spike_auc:.4f}')
    ax4_full.set_ylabel('Near Spike Probability')
    ax4_full.spines['top'].set_visible(False)
    ax4_full.spines['right'].set_visible(False)

    # Add zoom rectangle
    ymin, ymax = ax4_full.get_ylim()
    rect4 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax4_full.add_patch(rect4)

    ax4_zoom = plt.subplot(gs[3, 1])
    ax4_zoom.plot(y_near_spike_gt, 'purple', label='Ground Truth', linewidth=1.5)
    ax4_zoom.plot(y_near_spike_pred, 'cyan', label='Prediction', alpha=0.8, linewidth=1.5)
    ax4_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax4_zoom.set_title('Zoomed View')
    ax4_zoom.spines['top'].set_visible(False)
    ax4_zoom.spines['right'].set_visible(False)

    # 5. Plot instantaneous rate (full and zoomed)
    ax5_full = plt.subplot(gs[4, 0])
    ax5_full.plot(y_inst_rate_gt, 'purple', label='Ground Truth', linewidth=1.5)
    ax5_full.plot(y_inst_rate_pred, 'cyan', label='Prediction', alpha=0.8, linewidth=1.5)
    ax5_full.set_xlim(0, len(y_spike_gt))
    ax5_full.set_title(f'Instantaneous Rate (Full) - RMSE = {inst_rate_rmse:.4f}, R² = {inst_rate_r2:.4f}')
    ax5_full.set_ylabel('Instantaneous Rate')
    ax5_full.spines['top'].set_visible(False)
    ax5_full.spines['right'].set_visible(False)

    # Add zoom rectangle
    ymin, ymax = ax5_full.get_ylim()
    rect5 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax5_full.add_patch(rect5)

    ax5_zoom = plt.subplot(gs[4, 1])
    ax5_zoom.plot(y_inst_rate_gt, 'purple', label='Ground Truth', linewidth=1.5)
    ax5_zoom.plot(y_inst_rate_pred, 'cyan', label='Prediction', alpha=0.8, linewidth=1.5)
    ax5_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax5_zoom.set_title('Zoomed View')
    ax5_zoom.spines['top'].set_visible(False)
    ax5_zoom.spines['right'].set_visible(False)

    # 6. Plot dendritic voltage (full and zoomed)
    ax6_full = plt.subplot(gs[5, 0])
    if y_dend_v_gt.ndim == 1:
        ax6_full.plot(y_dend_v_gt, 'purple', linewidth=1.5)
        ax6_full.plot(y_dend_v_pred, 'cyan', alpha=0.8, linewidth=1.5)
    else:
        for i in range(y_dend_v_gt.shape[0]):
            ax6_full.plot(y_dend_v_gt[i], 'purple', linewidth=1.5)
            ax6_full.plot(y_dend_v_pred[i], 'cyan', alpha=0.8, linewidth=1.5)
    ax6_full.set_xlim(0, len(y_spike_gt))
    ax6_full.set_xlabel('Time (ms)')
    ax6_full.set_title(f'Dendritic Voltage (Full) - RMSE = {dend_v_rmse:.2f} mV, R² = {dend_v_r2:.4f}')
    ax6_full.set_ylabel('Voltage (mV)')
    ax6_full.spines['top'].set_visible(False)
    ax6_full.spines['right'].set_visible(False)
    
    # Add zoom rectangle
    ymin, ymax = ax6_full.get_ylim()
    rect6 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax6_full.add_patch(rect6)
    
    ax6_zoom = plt.subplot(gs[5, 1])
    if y_dend_v_gt.ndim == 1:
        ax6_zoom.plot(y_dend_v_gt, 'purple', linewidth=1.5)
        ax6_zoom.plot(y_dend_v_pred, 'cyan', alpha=0.8, linewidth=1.5)
    else:
        for i in range(y_dend_v_gt.shape[0]):
            ax6_zoom.plot(y_dend_v_gt[i], 'purple', linewidth=1.5)
            ax6_zoom.plot(y_dend_v_pred[i], 'cyan', alpha=0.8, linewidth=1.5)
            
    ax6_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax6_zoom.set_xlabel('Time (ms)')
    ax6_zoom.set_title('Zoomed View')
    ax6_zoom.spines['top'].set_visible(False)
    ax6_zoom.spines['right'].set_visible(False)
    
    plt.suptitle('Manual Input: FF Neuron vs Twin Model Comparison', fontsize=16, y=0.98)
    plt.tight_layout()

    # Calculate and print metrics for all outputs
    print('---------------------------------------------------------')
    print('Metrics for current sample:')
    print('---------------------------')
    print(f'Total number of GT spikes: {total_num_of_GT_spikes} ({avg_spike_rate:.2f} Hz)')

    print(f'Spike Detection:')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  AUC: {spike_auc_score:.4f}')

    print(f'Soma Voltage Prediction:')
    print(f'  MAE: {soma_mae:.2f} mV')
    print(f'  RMSE: {soma_rmse:.2f} mV')
    print(f'  R²: {soma_r2:.4f}')

    print(f'Near Spike Prediction:')
    print(f'  AUC: {near_spike_auc:.4f}')

    print(f'Instantaneous Rate Prediction:')
    print(f'  MAE: {inst_rate_mae:.4f}')
    print(f'  RMSE: {inst_rate_rmse:.4f}')
    print(f'  R²: {inst_rate_r2:.4f}')

    print(f'Dendritic Voltage Prediction:')
    print(f'  MAE: {dend_v_mae:.2f} mV')
    print(f'  RMSE: {dend_v_rmse:.2f} mV')
    print(f'  R²: {dend_v_r2:.4f}')
    print('---------------------------------------------------------')

    return fig

#%% Main evaluation script

if __name__ == "__main__":

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('----------------------------')
    print(f'Using device: {device}')
    print('----------------------------')

    # Dataset and model parameters
    data_root = config.NEURON_DATA_ROOT
    models_root = config.MODELS_ROOT

    # Look for available model folders
    all_model_folders = glob.glob(os.path.join(models_root, '*'))
    available_model_folders = []

    print('-------------------------------------------------------------------------------------')
    print("all model folders:")
    print('------------------')
    for folder_path in all_model_folders:
        folder_name = os.path.basename(folder_path)
        model_files = glob.glob(os.path.join(folder_path, '*.pt'))
        num_models_in_folder = len(model_files)
        print(f"  - {folder_name} ({num_models_in_folder} model files)")

        if num_models_in_folder > 0:
            available_model_folders.append(folder_path)
    print('-------------------------------------------------------------------------------------')

    # Select the neuron model folder and files
    # 1 branch linear linear-saturating I&F (refractory time constant = 16ms)
    # selected_folder_name = r"MultiBranch_IF_1_branches_of_4_segments__seg_f_linear_branch_f_linear_sat__tau_syn_32ms_tau_refr_16ms"

    # 1 branch linear linear-saturating F&F (refractory time constant = 16ms, synapse time constant = 16ms)
    # selected_folder_name = r"MultiBranch_FF_1_branches_of_4_segments__seg_f_linear_branch_f_linear_sat__tau_syn_16ms_tau_refr_16ms"

    # 1 branch linear linear-saturating F&F (refractory time constant = 16ms)
    selected_folder_name = r"MultiBranch_FF_1_branches_of_4_segments__seg_f_linear_branch_f_linear_sat__tau_syn_32ms_tau_refr_16ms"

    # 1 branch linear linear-saturating F&F (refractory time constant = 40ms)
    # selected_folder_name = r"MultiBranch_FF_1_branches_of_4_segments__seg_f_linear_branch_f_linear_sat__tau_syn_32ms_tau_refr_40ms"

    # 4 branches linear NMDA-saturating F&F (refractory time constant = 16ms)
    # selected_folder_name = r"MultiBranch_FF_4_branches_of_4_segments__seg_f_linear_branch_f_NMDA_sat__tau_syn_32ms_tau_refr_16ms"

    # get the corresponding model and data folders
    selected_model_folder = os.path.join(models_root, selected_folder_name)
    selected_data_folder = os.path.join(data_root, selected_folder_name)

    # get the available twin model files in the selected model folder
    model_files = glob.glob(os.path.join(selected_model_folder, '*.pt'))

    print('------------------------------------------------------------------------------------------------')
    print('all available twin model files:')
    print('--------------------------------')
    for model_file in model_files:
        print(f" - {os.path.basename(model_file)}")
    print('------------------------------------------------------------------------------------------------')

    # select the dnn twin model file
    # 1 branch linear linear-saturating I&F
    if selected_folder_name == r"MultiBranch_IF_1_branches_of_4_segments__seg_f_linear_branch_f_linear_sat__tau_syn_32ms_tau_refr_40ms":
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_7_W_32_T_213_params_221K_AUC_0_9022_somaR2_616__calibR2_9977.pt'
        pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_19_W_16_T_229_params_65K_AUC_0_9046_somaR2_638__calibR2_9983.pt'

    if selected_folder_name == r"MultiBranch_FF_1_branches_of_4_segments__seg_f_linear_branch_f_linear_sat__tau_syn_16ms_tau_refr_16ms":
        pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_27_W_16_T_113_params_39K_AUC_0_9586_somaR2_738__calibR2_9965.pt'

    # 1 branch battery linear-saturating F&F (refractory time constant = 16ms)
    if selected_folder_name == r"MultiBranch_FF_1_branches_of_4_segments__seg_f_linear_branch_f_linear_sat__tau_syn_32ms_tau_refr_16ms":
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_9_W_16_T_225_params_59K_AUC_0_9500_somaR2_699__calibR2_9949.pt'
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_15_W_16_T_211_params_58K_AUC_0_9543_somaR2_725__calibR2_9957.pt'
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_19_W_32_T_229_params_254K_AUC_0_9475_somaR2_822__calibR2_9975.pt'
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_25_W_32_T_201_params_239K_AUC_0_9454_somaR2_809__calibR2_9952.pt'

        pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_31_W_24_T_249_params_168K_AUC_0_9779_somaR2_825__calibR2_9843.pt' # ResNet TCN. 150 epochs, LR 0.0003, WD 0.001, GN 30.0, RMSNorm, leaky gelu 0.15, TC scale 0.15, TC loss 0.61 - 5.8/5 optimizable
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_31_W_48_T_249_params_664K_AUC_0_9884_somaR2_892__calibR2_9936.pt' # ResNet TCN. no TC. 2/5 optimizable (very bad!!!)

    # 1 branch battery linear-saturating F&F (refractory time constant = 40ms)
    if selected_folder_name == r"MultiBranch_FF_1_branches_of_4_segments__seg_f_linear_branch_f_linear_sat__tau_syn_32ms_tau_refr_40ms":
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_9_W_16_T_223_params_59K_AUC_0_9578_somaR2_667__calibR2_9904.pt'
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_9_W_24_T_239_params_140K_AUC_0_9631_somaR2_709__calibR2_9918.pt'
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_9_W_24_T_255_params_149K_AUC_0_9643_somaR2_713__calibR2_9931.pt'
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_9_W_24_T_257_params_149K_AUC_0_9684_somaR2_783__calibR2_9932.pt'
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_9_W_64_T_257_params_1043K_AUC_0_9652_somaR2_791__calibR2_9918.pt'
        pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_7_W_12_T_233_params_34K_AUC_0_9650_somaR2_770__calibR2_9930.pt'

    # 4 branches linear NMDA-saturating F&F (refractory time constant = 16ms)
    if selected_folder_name == r"MultiBranch_FF_4_branches_of_4_segments__seg_f_linear_branch_f_NMDA_sat__tau_syn_32ms_tau_refr_16ms":
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_19_W_16_T_229_params_70K_AUC_0_9407_somaR2_702__calibR2_9959.pt'
        # pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_19_W_24_T_229_params_152K_AUC_0_9433_somaR2_738__calibR2_9948.pt'
        pretrained_model_name = f'{selected_folder_name}_TCN_ResNet_D_23_W_32_T_185_params_226K_AUC_0_9553_somaR2_789__calibR2_9977.pt'

    # Load the GT neuron model
    FF_neuron_file = glob.glob(os.path.join(selected_data_folder, '*.pkl'))[0]
    ff_neuron = MultiBranchFilterAndFireNeuron.load(FF_neuron_file)

    pretrained_model_path = os.path.join(selected_model_folder, pretrained_model_name)

    print('\n\n')
    print('------------------------------------------------------------------------------------------------')
    print(f"Selected folder name: {selected_folder_name}")
    print('------------------------------------------------------------------------------------------------')
    print(f"neuron model filename: {os.path.basename(FF_neuron_file)}")
    print(f"twin model filename  : {os.path.basename(pretrained_model_path)}")
    print('------------------------------------------------------------------------------------------------')
    print(f"Corresponding data folder path:\n   {os.path.dirname(selected_data_folder)}")
    print(f"Selected model folder path:\n   {os.path.dirname(pretrained_model_path)}")
    print(f"Corresponding FF neuron model path:\n   {os.path.dirname(FF_neuron_file)}")
    print('------------------------------------------------------------------------------------------------')

    # Load the model
    print('-------------------------------------------------------------------------------------')
    print("Loading model...")
    model = load_twin_model(pretrained_model_path)
    model.eval()
    model.to(device)
    print("Model loaded successfully!")
    print('-------------------------------------------------------------------------------------')

    # Display model metadata
    model.print_main_metadata()

    # Set up data folders
    valid_data_folder = os.path.join(selected_data_folder, 'valid')
    # valid_data_folder = os.path.join(selected_data_folder, 'valid_aug')
    # valid_data_folder = os.path.join(selected_data_folder, 'valid_orig')
    test_data_folder = os.path.join(selected_data_folder, 'test')
    
    # Create validation dataset
    print("Creating validation dataset...")
    valid_time_window_size = 7168 + 512
    valid_dataset = FFNeuronDataset(valid_data_folder, valid_time_window_size, preload_data=False)
    print(f"Validation dataset created with {len(valid_dataset)} samples")
    
    # Evaluate model on validation set
    print("\nEvaluating model on validation set...")
    valid_metrics_dict = evaluate_model_on_dataset(model, valid_dataset, batch_size=8, verbose=1)
    
    # Display evaluation figures
    print("\nDisplaying evaluation figures...")
    fig1 = plot_evaluation_figures(valid_metrics_dict)
    plt.show()
    
    # Display per-simulation metrics
    print("Displaying per-simulation metrics...")
    fig2 = plot_per_simulation_metrics(valid_metrics_dict)
    plt.show()
    
    # Test calibration
    print("Testing model calibration...")
    output_dict = predict_on_all_simulations(model, valid_dataset, batch_size=8)
    y_spikes_pred = output_dict['y_spikes_pred']
    y_spikes_gt = output_dict['y_spikes_gt']
    
    calib_corr, calib_explained_var = calculate_calibration_metrics(y_spikes_pred, y_spikes_gt)
    print(f'Calibration Correlation: {calib_corr:.4f}')
    print(f'Calibration Explained Variance Percent: {100 * calib_explained_var:.2f}%')
    
    fig3 = display_calibration_figure(y_spikes_pred, y_spikes_gt)
    plt.show()
    
    # Display sample predictions
    print("Displaying sample predictions...")
    
    # Minimal sample display
    fig4 = display_sample_predictions_minimal(model, test_data_folder, ff_neuron)
    plt.show()
    
    # Full sample display
    fig5 = display_sample_predictions_full(model, test_data_folder, ff_neuron)
    plt.show()
    

    #%% Manual Input Generation and Comparison
    
    print('--------------------------------------------------------------------')
    print("Manual standalone input generation and model comparison")
    print('--------------------------------------------------------------------')
    
    # Extract neuron parameters for compatibility with both FF and BS neurons
    if hasattr(ff_neuron, 'v_threshold'):
        # Filter and Fire neuron
        manual_neuron_v_threshold = ff_neuron.v_threshold
        manual_neuron_v_reset = ff_neuron.v_reset
        manual_neuron_v_hard_reset = ff_neuron.v_hard_reset
    elif hasattr(ff_neuron, 'spike_detection_threshold_mV'):
        # Ball and Stick neuron
        manual_neuron_v_threshold = ff_neuron.soma_voltage_cap_mV
        manual_neuron_v_reset = ff_neuron.epas_mV
        manual_neuron_v_hard_reset = ff_neuron.epas_mV - 5
    else:
        # Fallback for unknown neuron type
        manual_neuron_v_threshold = 0.0
        manual_neuron_v_reset = -70.0
        manual_neuron_v_hard_reset = -75.0
    
    # Create manual input for testing
    print("Creating manual input for direct model comparison...")
    
    # Input generation parameters
    T_manual = 2048  # simulation duration in ms
    num_axons = 64
    exc_firing_rate_HZ = 10.0
    inh_firing_rate_HZ = 2.0
    mixing_power = 5.0
    mixing_multiplier = 3.0
    
    print(f"Simulation duration: {T_manual} ms")
    print(f"Number of input axons: {num_axons}")
    print(f"Excitatory firing rate: {exc_firing_rate_HZ} Hz")
    print(f"Inhibitory firing rate: {inh_firing_rate_HZ} Hz")
    
    # Generate axonal spike trains
    exc_spike_prob = exc_firing_rate_HZ / 1000
    inh_spike_prob = inh_firing_rate_HZ / 1000
    X_exc_axons = np.random.rand(num_axons, T_manual) < exc_spike_prob
    X_inh_axons = np.random.rand(num_axons, T_manual) < inh_spike_prob
    
    # Create mixing matrices for all segments
    total_segments = getattr(ff_neuron, 'total_segments', getattr(ff_neuron, 'num_segments', 4))
    exc_mixing_matrix = mixing_multiplier * np.power(np.random.rand(total_segments, num_axons), mixing_power)
    inh_mixing_matrix = mixing_multiplier * np.power(np.random.rand(total_segments, num_axons), mixing_power)
    
    # Generate mixed inputs to segments
    X_exc_manual = np.dot(exc_mixing_matrix, X_exc_axons).astype(np.float32)
    X_inh_manual = np.dot(inh_mixing_matrix, X_inh_axons).astype(np.float32)
    
    print(f"Generated input shapes: X_exc {X_exc_manual.shape}, X_inh {X_inh_manual.shape}")
    
    # Simulate real FF neuron
    print("Simulating real FF neuron...")
    ff_simulation_output = ff_neuron.simulate(X_exc_manual, X_inh_manual)
    
    y_spike_ff = ff_simulation_output['y_spike'].astype(np.float32)
    y_soma_ff = ff_simulation_output['y_soma'].astype(np.float32)
    y_near_spike_ff = ff_simulation_output['y_near_spike'].astype(np.float32)
    y_inst_rate_ff = ff_simulation_output['y_inst_rate'].astype(np.float32)
    y_dend_v_ff = ff_simulation_output['branch_voltages'].astype(np.float32)
    
    # Prepare input for twin model (channels first format)
    X_inputs_manual = np.stack([X_exc_manual, X_inh_manual], axis=-1)  # Shape: (segments, T, 2)
    X_spikes_manual = torch.FloatTensor(X_inputs_manual.transpose(2, 0, 1)).unsqueeze(0).to(device)  # Add batch dim
    X_spikes_manual = X_spikes_manual / model.X_scale
    
    # Simulate twin model
    print("Simulating twin model...")
    model.eval()
    with torch.no_grad():
        y_spikes_pred_manual, y_soma_pred_manual, y_near_spike_pred_manual, y_inst_rate_pred_manual, y_dend_v_pred_manual = model(X_spikes_manual)
        
        # Process predictions
        y_spikes_pred_manual = torch.sigmoid(y_spikes_pred_manual.squeeze()).cpu().numpy()
        y_soma_pred_manual = (y_soma_pred_manual.squeeze().cpu().numpy() * model.V_scale_soma) + model.V_bias_soma
        y_near_spike_pred_manual = torch.sigmoid(y_near_spike_pred_manual.squeeze()).cpu().numpy()
        y_inst_rate_pred_manual = y_inst_rate_pred_manual.squeeze().cpu().numpy() / model.y_inst_rate_multiplier
        y_dend_v_pred_manual = y_dend_v_pred_manual.squeeze().cpu().numpy() * model.V_scale_dend + model.V_bias_dend
    
    # Handle dendritic voltage dimensions
    if len(y_dend_v_ff.shape) == 2 and y_dend_v_ff.shape[0] == 1:
        y_dend_v_ff = y_dend_v_ff.squeeze(0)
        y_dend_v_pred_manual = y_dend_v_pred_manual.squeeze(0) if len(y_dend_v_pred_manual.shape) > 1 else y_dend_v_pred_manual
    
    # Calculate metrics for this manual comparison
    spike_threshold_for_plot = 0.5
    spike_times_ff = np.where(y_spike_ff)[0]
    spike_times_pred = np.where(y_spikes_pred_manual > spike_threshold_for_plot)[0]
    
    # Spike prediction metrics
    spike_pred_binary = y_spikes_pred_manual > spike_threshold_for_plot
    y_spike_ff_bool = y_spike_ff.astype(bool)  # Cast to bool only for bitwise operations
    true_positives = np.sum(spike_pred_binary & y_spike_ff_bool)
    false_positives = np.sum(spike_pred_binary & ~y_spike_ff_bool)
    false_negatives = np.sum(~spike_pred_binary & y_spike_ff_bool)
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    total_num_of_ff_spikes = np.sum(y_spike_ff)
    avg_spike_rate = 1000 * total_num_of_ff_spikes / len(y_spike_ff)
    
    if total_num_of_ff_spikes > 0:
        spike_auc_score = roc_auc_score(y_spike_ff, y_spikes_pred_manual)
    else:
        spike_auc_score = 0.0
    
    # Other metrics
    soma_mae = np.mean(np.abs(y_soma_ff - y_soma_pred_manual))
    soma_rmse = np.sqrt(np.mean((y_soma_ff - y_soma_pred_manual)**2))
    soma_r2 = 1 - np.sum((y_soma_ff - y_soma_pred_manual)**2) / np.sum((y_soma_ff - np.mean(y_soma_ff))**2)
    
    near_spike_auc = roc_auc_score(y_near_spike_ff, y_near_spike_pred_manual) if np.sum(y_near_spike_ff) > 0 else 0.0
    
    inst_rate_mae = np.mean(np.abs(y_inst_rate_ff - y_inst_rate_pred_manual))
    inst_rate_rmse = np.sqrt(np.mean((y_inst_rate_ff - y_inst_rate_pred_manual)**2))
    inst_rate_r2 = 1 - np.sum((y_inst_rate_ff - y_inst_rate_pred_manual)**2) / np.sum((y_inst_rate_ff - np.mean(y_inst_rate_ff))**2)
    
    dend_v_mae = np.mean(np.abs(y_dend_v_ff - y_dend_v_pred_manual))
    dend_v_rmse = np.sqrt(np.mean((y_dend_v_ff - y_dend_v_pred_manual)**2))
    dend_v_r2 = 1 - np.sum((y_dend_v_ff - y_dend_v_pred_manual)**2) / np.sum((y_dend_v_ff - np.mean(y_dend_v_ff))**2)
    
    print(f"Manual comparison metrics:")
    print(f"  Total FF spikes: {total_num_of_ff_spikes} ({avg_spike_rate:.2f} Hz)")
    print(f"  Spike prediction - Precision: {precision:.4f}, Recall: {recall:.4f}, AUC: {spike_auc_score:.4f}")
    print(f"  Soma voltage - RMSE: {soma_rmse:.2f} mV, R²: {soma_r2:.4f}")
    print(f"  Near spike - AUC: {near_spike_auc:.4f}")
    print(f"  Inst rate - RMSE: {inst_rate_rmse:.4f}, R²: {inst_rate_r2:.4f}")
    print(f"  Dend voltage - RMSE: {dend_v_rmse:.2f} mV, R²: {dend_v_r2:.4f}")
    
    # Create comprehensive comparison figure
    print("Creating manual comparison figure...")
    
    # Select random zoom window
    zoomin_duration_ms = 512
    max_start_time = len(y_spike_ff) - zoomin_duration_ms
    zoom_start_time = np.random.randint(0, max_start_time)
    zoom_end_time = zoom_start_time + zoomin_duration_ms
    
    # Create figure with both full and zoomed views
    fig_manual = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(6, 2, width_ratios=[3, 2], height_ratios=[2, 1, 2, 1, 1, 1])
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    
    # Get segment offsets for input visualization
    seg_offsets_vec = 5 * np.arange(getattr(ff_neuron, 'total_segments', getattr(ff_neuron, 'num_segments', 4)))
    
    # 1. Plot mixed input signals (full and zoomed)
    ax1_full = plt.subplot(gs[0, 0])
    ax1_full.plot(X_exc_manual.T + seg_offsets_vec, 'r', alpha=0.7, label='Excitatory')
    ax1_full.plot(-X_inh_manual.T + seg_offsets_vec, 'b', alpha=0.7, label='Inhibitory')
    ax1_full.set_xlim(0, T_manual)
    ax1_full.set_title('Manual Input Signals on Dendritic Segments (Full)')
    ax1_full.set_ylabel('Input Strength')
    ax1_full.spines['top'].set_visible(False)
    ax1_full.spines['right'].set_visible(False)
    
    # Add zoom rectangle
    ymin, ymax = ax1_full.get_ylim()
    rect1 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax1_full.add_patch(rect1)
    
    ax1_zoom = plt.subplot(gs[0, 1])
    ax1_zoom.plot(X_exc_manual.T + seg_offsets_vec, 'r', alpha=0.7, label='Excitatory')
    ax1_zoom.plot(-X_inh_manual.T + seg_offsets_vec, 'b', alpha=0.7, label='Inhibitory')
    ax1_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax1_zoom.set_title(f'Zoomed View ({zoom_start_time}-{zoom_end_time} ms)')
    ax1_zoom.spines['top'].set_visible(False)
    ax1_zoom.spines['right'].set_visible(False)
    
    # 2. Plot output spikes (full and zoomed)
    ax2_full = plt.subplot(gs[1, 0])
    ax2_full.plot(y_spikes_pred_manual, 'cyan', alpha=0.7, label='Twin Model Prob.')
    ax2_full.scatter(spike_times_ff, [1.2] * len(spike_times_ff), c='purple', marker='|', 
                    s=200, label='FF Neuron Spikes')
    ax2_full.scatter(spike_times_pred, [1.0] * len(spike_times_pred), c='cyan', marker='|', 
                    s=200, label='Twin Predicted Spikes')
    ax2_full.set_ylim(-0.1, 1.4)
    ax2_full.set_xlim(0, T_manual)
    ax2_full.set_title(f'Output Spikes Comparison (Full) - AUC = {spike_auc_score:.4f}')
    ax2_full.set_ylabel('Spike Probability')
    ax2_full.legend()
    ax2_full.spines['top'].set_visible(False)
    ax2_full.spines['right'].set_visible(False)
    
    # Add zoom rectangle
    rect2 = plt.Rectangle((zoom_start_time, -0.05), zoomin_duration_ms, 1.4,
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax2_full.add_patch(rect2)
    
    ax2_zoom = plt.subplot(gs[1, 1])
    ax2_zoom.plot(y_spikes_pred_manual, 'cyan', alpha=0.7, label='Twin Model Prob.')
    ax2_zoom.scatter(spike_times_ff, [1.2] * len(spike_times_ff), c='purple', marker='|', 
                    s=200, label='FF Neuron Spikes')
    ax2_zoom.scatter(spike_times_pred, [1.0] * len(spike_times_pred), c='cyan', marker='|', 
                    s=200, label='Twin Predicted Spikes')
    ax2_zoom.set_ylim(-0.1, 1.4)
    ax2_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax2_zoom.set_title('Zoomed View')
    ax2_zoom.spines['top'].set_visible(False)
    ax2_zoom.spines['right'].set_visible(False)
    
    # 3. Plot soma voltage (full and zoomed)
    ax3_full = plt.subplot(gs[2, 0])
    ax3_full.plot(y_soma_ff, 'purple', label='FF Neuron', linewidth=1.5)
    ax3_full.plot(y_soma_pred_manual, 'cyan', label='Twin Model', alpha=0.8, linewidth=1.5)
    ax3_full.axhline(y=manual_neuron_v_threshold, color='gray', linestyle='--', alpha=0.8, label='Threshold')
    ax3_full.axhline(y=manual_neuron_v_reset, color='gray', linestyle=':', alpha=0.6, label='Reset')
    if manual_neuron_v_hard_reset is not None:
        ax3_full.axhline(y=manual_neuron_v_hard_reset, color='gray', linestyle='-.', alpha=0.78, label='Hard Reset')
    ax3_full.set_xlim(0, T_manual)
    ax3_full.set_title(f'Soma Voltage Comparison (Full) - RMSE = {soma_rmse:.2f} mV, R² = {soma_r2:.4f}')
    ax3_full.set_ylabel('Voltage (mV)')
    ax3_full.spines['top'].set_visible(False)
    ax3_full.spines['right'].set_visible(False)
    
    # Add zoom rectangle
    ymin, ymax = ax3_full.get_ylim()
    rect3 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax3_full.add_patch(rect3)
    
    ax3_zoom = plt.subplot(gs[2, 1])
    ax3_zoom.plot(y_soma_ff, 'purple', label='FF Neuron', linewidth=1.5)
    ax3_zoom.plot(y_soma_pred_manual, 'cyan', label='Twin Model', alpha=0.8, linewidth=1.5)
    ax3_zoom.axhline(y=manual_neuron_v_threshold, color='gray', linestyle='--', alpha=0.8, label='Threshold')
    ax3_zoom.axhline(y=manual_neuron_v_reset, color='gray', linestyle=':', alpha=0.6, label='Reset')
    if manual_neuron_v_hard_reset is not None:
        ax3_zoom.axhline(y=manual_neuron_v_hard_reset, color='gray', linestyle='-.', alpha=0.78, label='Hard Reset')
    ax3_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax3_zoom.set_title('Zoomed View')
    ax3_zoom.spines['top'].set_visible(False)
    ax3_zoom.spines['right'].set_visible(False)
    
    # 4. Plot near spike predictions (full and zoomed)
    ax4_full = plt.subplot(gs[3, 0])
    ax4_full.plot(y_near_spike_ff, 'purple', label='FF Neuron', linewidth=1.5)
    ax4_full.plot(y_near_spike_pred_manual, 'cyan', label='Twin Model', alpha=0.8, linewidth=1.5)
    ax4_full.set_xlim(0, T_manual)
    ax4_full.set_title(f'Near Spike Comparison (Full) - AUC = {near_spike_auc:.4f}')
    ax4_full.set_ylabel('Near Spike Probability')
    ax4_full.spines['top'].set_visible(False)
    ax4_full.spines['right'].set_visible(False)
    
    # Add zoom rectangle
    ymin, ymax = ax4_full.get_ylim()
    rect4 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax4_full.add_patch(rect4)
    
    ax4_zoom = plt.subplot(gs[3, 1])
    ax4_zoom.plot(y_near_spike_ff, 'purple', label='FF Neuron', linewidth=1.5)
    ax4_zoom.plot(y_near_spike_pred_manual, 'cyan', label='Twin Model', alpha=0.8, linewidth=1.5)
    ax4_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax4_zoom.set_title('Zoomed View')
    ax4_zoom.spines['top'].set_visible(False)
    ax4_zoom.spines['right'].set_visible(False)
    
    # 5. Plot instantaneous rate (full and zoomed)
    ax5_full = plt.subplot(gs[4, 0])
    ax5_full.plot(y_inst_rate_ff, 'purple', label='FF Neuron', linewidth=1.5)
    ax5_full.plot(y_inst_rate_pred_manual, 'cyan', label='Twin Model', alpha=0.8, linewidth=1.5)
    ax5_full.set_xlim(0, T_manual)
    ax5_full.set_title(f'Instantaneous Rate Comparison (Full) - RMSE = {inst_rate_rmse:.4f}, R² = {inst_rate_r2:.4f}')
    ax5_full.set_ylabel('Instantaneous Rate')
    ax5_full.spines['top'].set_visible(False)
    ax5_full.spines['right'].set_visible(False)
    
    # Add zoom rectangle
    ymin, ymax = ax5_full.get_ylim()
    rect5 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax5_full.add_patch(rect5)
    
    ax5_zoom = plt.subplot(gs[4, 1])
    ax5_zoom.plot(y_inst_rate_ff, 'purple', label='FF Neuron', linewidth=1.5)
    ax5_zoom.plot(y_inst_rate_pred_manual, 'cyan', label='Twin Model', alpha=0.8, linewidth=1.5)
    ax5_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax5_zoom.set_title('Zoomed View')
    ax5_zoom.spines['top'].set_visible(False)
    ax5_zoom.spines['right'].set_visible(False)
    
    # 6. Plot dendritic voltage (full and zoomed)
    ax6_full = plt.subplot(gs[5, 0])
    if y_dend_v_ff.ndim == 1:
        ax6_full.plot(y_dend_v_ff, 'purple', linewidth=1.5, label='FF Neuron')
        ax6_full.plot(y_dend_v_pred_manual, 'cyan', alpha=0.8, linewidth=1.5, label='Twin Model')
    else:
        for i in range(y_dend_v_ff.shape[0]):
            ax6_full.plot(y_dend_v_ff[i], 'purple', linewidth=1.5, alpha=0.7)
            ax6_full.plot(y_dend_v_pred_manual[i], 'cyan', alpha=0.8, linewidth=1.5)
        # Add labels only once
        ax6_full.plot([], [], 'purple', linewidth=1.5, label='FF Neuron')
        ax6_full.plot([], [], 'cyan', linewidth=1.5, label='Twin Model')
    
    ax6_full.set_xlim(0, T_manual)
    ax6_full.set_xlabel('Time (ms)')
    ax6_full.set_title(f'Dendritic Voltage Comparison (Full) - RMSE = {dend_v_rmse:.2f} mV, R² = {dend_v_r2:.4f}')
    ax6_full.set_ylabel('Voltage (mV)')
    ax6_full.legend()
    ax6_full.spines['top'].set_visible(False)
    ax6_full.spines['right'].set_visible(False)
    
    # Add zoom rectangle
    ymin, ymax = ax6_full.get_ylim()
    rect6 = plt.Rectangle((zoom_start_time, ymin + 0.025 * (ymax-ymin)), zoomin_duration_ms, 0.95 * (ymax-ymin),
                        fill=False, linestyle='--', color='k', alpha=0.7, linewidth=3)
    ax6_full.add_patch(rect6)
    
    ax6_zoom = plt.subplot(gs[5, 1])
    if y_dend_v_ff.ndim == 1:
        ax6_zoom.plot(y_dend_v_ff, 'purple', linewidth=1.5)
        ax6_zoom.plot(y_dend_v_pred_manual, 'cyan', alpha=0.8, linewidth=1.5)
    else:
        for i in range(y_dend_v_ff.shape[0]):
            ax6_zoom.plot(y_dend_v_ff[i], 'purple', linewidth=1.5, alpha=0.7)
            ax6_zoom.plot(y_dend_v_pred_manual[i], 'cyan', alpha=0.8, linewidth=1.5)
    
    ax6_zoom.set_xlim(zoom_start_time, zoom_end_time)
    ax6_zoom.set_xlabel('Time (ms)')
    ax6_zoom.set_title('Zoomed View')
    ax6_zoom.spines['top'].set_visible(False)
    ax6_zoom.spines['right'].set_visible(False)
    
    plt.suptitle('Manual Input: FF Neuron vs Twin Model Comparison', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print detailed metrics
    print('---------------------------------------------------------')
    print('Manual comparison metrics for this simulation:')
    print('----------------------------------------------')
    print(f'Total number of FF spikes: {total_num_of_ff_spikes} ({avg_spike_rate:.2f} Hz)')
    
    print(f'Spike Detection:')
    print(f'  Precision: {precision:.4f}')
    print(f'  Recall: {recall:.4f}')
    print(f'  AUC: {spike_auc_score:.4f}')
    
    print(f'Soma Voltage Prediction:')
    print(f'  MAE: {soma_mae:.2f} mV')
    print(f'  RMSE: {soma_rmse:.2f} mV')
    print(f'  R²: {soma_r2:.4f}')
    
    print(f'Near Spike Prediction:')
    print(f'  AUC: {near_spike_auc:.4f}')
    
    print(f'Instantaneous Rate Prediction:')
    print(f'  MAE: {inst_rate_mae:.4f}')
    print(f'  RMSE: {inst_rate_rmse:.4f}')
    print(f'  R²: {inst_rate_r2:.4f}')
    
    print(f'Dendritic Voltage Prediction:')
    print(f'  MAE: {dend_v_mae:.2f} mV')
    print(f'  RMSE: {dend_v_rmse:.2f} mV')
    print(f'  R²: {dend_v_r2:.4f}')
    print('---------------------------------------------------------')
    
