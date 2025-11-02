#%% Imports

import os
import torch
import torch.nn as nn
from torch_module_definitions import TCN_ResNet_Backbone, TCN_Backbone, ELM_Backbone, Transformer_Backbone, SingleNeuronTCN_Heads

#%% various twin model class definitions

# NOTE: make sure any new model class defenition added here has the save_model() and load_model() methods
#       - additional methods are also required for the proper evaluation of the model inside the repo
#       ==> use SingleNeuronTwinModel_ResNetTCN class as a gold standard example for any new model class defenition

# Define a model class that is a DNN twin (of type ResnetTCN) of a single neuron (multi branch F&F neuron) model
class SingleNeuronTwinModel_ResNetTCN(nn.Module):
    def __init__(self, 
                in_channels, in_spatial_dim, first_layer_temporal_kernel_size, 
                num_miniblocks_per_block_list, num_features_per_block_list, 
                temporal_kernel_size_per_block_list, temporal_dilation_per_block_list, 
                bottleneck_dim_per_block_list=None,
                nonlinearity_str='lgelu', leaky_slope=0.2, norm_type='BatchNorm',
                head_prefix_names=['spikes', 'soma', 'near_spike', 'inst_rate', 'dend_v'],
                head_out_channels=[1, 1, 1, 1, 4], 
                head_convert_out_ch_to_sp=[False, False, False, False, False],
                head_spatial_kernel_sizes=None,
                X_scale=20.0, 
                V_bias_soma=-85, V_scale_soma=20, V_clip_soma_min=-102, V_clip_soma_max=-53,
                V_bias_dend=0.0, V_scale_dend=20, V_clip_dend_min=-50, V_clip_dend_max=50,
                y_inst_rate_multiplier=10.0,
                metadata={}):
        super().__init__()

        # backbone parameters
        self.in_channels = in_channels
        self.in_spatial_dim = in_spatial_dim
        self.first_layer_temporal_kernel_size = first_layer_temporal_kernel_size
        self.num_miniblocks_per_block_list = num_miniblocks_per_block_list
        self.num_features_per_block_list = num_features_per_block_list
        self.temporal_kernel_size_per_block_list = temporal_kernel_size_per_block_list
        self.temporal_dilation_per_block_list = temporal_dilation_per_block_list
        self.bottleneck_dim_per_block_list = bottleneck_dim_per_block_list
        self.nonlinearity_str = nonlinearity_str
        self.leaky_slope = leaky_slope
        self.norm_type = norm_type
        # head parameters
        self.head_prefix_names = head_prefix_names
        self.head_out_channels = head_out_channels
        self.head_convert_out_ch_to_sp = head_convert_out_ch_to_sp
        self.head_spatial_kernel_sizes = head_spatial_kernel_sizes
        # various scaling and clipping parameters
        self.X_scale = X_scale
        self.V_bias_soma = V_bias_soma
        self.V_scale_soma = V_scale_soma
        self.V_clip_soma_min = V_clip_soma_min
        self.V_clip_soma_max = V_clip_soma_max
        self.V_bias_dend = V_bias_dend
        self.V_scale_dend = V_scale_dend
        self.V_clip_dend_min = V_clip_dend_min
        self.V_clip_dend_max = V_clip_dend_max
        self.y_inst_rate_multiplier = y_inst_rate_multiplier
        self.metadata = metadata

        # create the backbone
        backbone = TCN_ResNet_Backbone(
            in_channels=in_channels,
            in_spatial_dim=in_spatial_dim,
            first_layer_temporal_kernel_size=first_layer_temporal_kernel_size,
            num_miniblocks_per_block_list=num_miniblocks_per_block_list,
            num_features_per_block_list=num_features_per_block_list,
            temporal_kernel_size_per_block_list=temporal_kernel_size_per_block_list,
            temporal_dilation_per_block_list=temporal_dilation_per_block_list,
            bottleneck_dim_per_block_list=bottleneck_dim_per_block_list,
            nonlinearity_str=nonlinearity_str,
            leaky_slope=leaky_slope,
            norm_type=norm_type,
        )

        # set the output dimensions of the backbone by passing a dummy input through it
        B_dummy, T_dummy = 2, 1024
        X_spikes_dummy = torch.zeros(B_dummy, in_channels, in_spatial_dim, T_dummy)
        print('----------------------------------------------------')
        backbone.set_output_dims(X_spikes_dummy, print_output_dims=True)
        print('----------------------------------------------------')
        print(f'backbone: "{backbone.short_name}"')
        print(f'backbone out channels: {backbone.out_channels}')
        print(f'backbone out spatial dim: {backbone.out_spatial_dim}')
        print('----------------------------------------------------')

        # create the model
        self.backbone_and_heads = SingleNeuronTCN_Heads(
            backbone=backbone,
            head_prefix_names=head_prefix_names,
            head_out_channels=head_out_channels,
            head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
        )

        self.update_short_name()

    def forward(self, x):
        # x.shape = (B, in_C, in_S_dim, T)
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v = self.backbone_and_heads(x)
        return y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v

    def forward_debug(self, x):
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v =  self.backbone_and_heads.forward_debug(x)
        return y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v

    def update_short_name(self):
        # backbone description string
        temporal_description_str = f"T_{self.backbone_and_heads.backbone.total_temporal_kernel_size}"
        depth_description_str = f"D_{self.backbone_and_heads.backbone.total_depth}"
        width_description_str = f"W_{self.backbone_and_heads.backbone.average_width:.0f}"
        backbone_description_str = f"TCN_ResNet_{depth_description_str}_{width_description_str}_{temporal_description_str}"

        # params description string
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_official / 1e3
        self.num_parmas_millions = self.num_params_official / 1e6
        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        # the short name of the model
        self.short_name = f"{backbone_description_str}_{param_description_str}"

    def set_metadata_original_neuron(self, original_neuron_model_metadata_dict):
        self.metadata['original_neuron_model'] = original_neuron_model_metadata_dict

    def set_metadata_training_params(self, training_params_dict):
        self.metadata['training_params'] = training_params_dict

    def set_metadata_learning_curves(self, learning_curves_dict):
        self.metadata['learning_curves'] = learning_curves_dict

    def set_metadata_eval_metrics(self, eval_metrics_dict):
        self.metadata['eval_metrics'] = eval_metrics_dict

    def print_main_metadata(self):

        print('------------------------------------------------------------------------------------------')
        print('DNN Twin Model Metadata')
        print('-----------------------')
        if 'original_neuron_model' in self.metadata:
            print('-----------')
            print('The basics:')
            print('-----------')
            print(f'Original Neuron: "{self.metadata["original_neuron_model"]["original_neuron_model_name"]}"')
            print(f'DNN Twin: "{self.short_name}"')
            print(f'Twin num params: {self.num_parmas_millions:.3f}M')
            print(f'Norm type: {self.norm_type}')
            print(f'Leaky slope: {self.leaky_slope}')
            print(f'Nonlinearity: {self.nonlinearity_str}')
            print('-------------------------------------------------------------------')

        if 'training_params' in self.metadata:
            print('----------------')
            print('Training params:')
            print('----------------')

            print(f'learning rate: {self.metadata["training_params"]["learning_rate"]}')
            print(f'weight decay: {self.metadata["training_params"]["weight_decay"]}')
            print(f'num epochs: {self.metadata["training_params"]["num_epochs"]}')
            print(f'num warmup epochs: {self.metadata["training_params"]["num_warmup_epochs"]}')
            print(f'num cooldown epochs: {self.metadata["training_params"]["num_cooldown_epochs"]}')
            print(f'train time window size: {self.metadata["training_params"]["train_time_window_size"]}')
            print(f'train batch size: {self.metadata["training_params"]["train_batch_size"]}\n')
            print('---------------------------------')

        if 'eval_metrics' in self.metadata:
            print('-------------------')
            print('Evaluation metrics:')
            print('-------------------')

            spikes_AUC_score = self.metadata['eval_metrics']['AUC_score']
            requested_false_positive_rate = self.metadata['eval_metrics']['requested_false_positive_rate']
            true_positive_at_FP = self.metadata['eval_metrics']['true_positive_at_FP']

            near_spike_AUC_score = self.metadata['eval_metrics']['near_spike_AUC_score']
            near_spike_false_positive_rate = self.metadata['eval_metrics']['near_spike_requested_false_positive_rate']
            near_spike_true_positive_at_FP = self.metadata['eval_metrics']['near_spike_true_positive_at_FP']

            soma_explained_variance_percent = self.metadata['eval_metrics']['soma_explained_variance_percent']
            soma_RMSE = self.metadata['eval_metrics']['soma_RMSE']
            soma_MAE = self.metadata['eval_metrics']['soma_MAE']

            inst_rate_explained_variance_percent = self.metadata['eval_metrics']['inst_rate_explained_variance_percent']
            inst_rate_RMSE = self.metadata['eval_metrics']['inst_rate_RMSE']
            inst_rate_MAE = self.metadata['eval_metrics']['inst_rate_MAE']

            dend_v_explained_variance_percent = self.metadata['eval_metrics']['dend_v_explained_variance_percent']
            dend_v_RMSE = self.metadata['eval_metrics']['dend_v_RMSE']
            dend_v_MAE = self.metadata['eval_metrics']['dend_v_MAE']

            print(f'Spikes AUC = {spikes_AUC_score:.4f}')
            print(f'at {requested_false_positive_rate:.4f} FP rate, TP = {true_positive_at_FP:.4f}')
            print(f'Near spike AUC = {near_spike_AUC_score:.4f}')
            print(f'at {near_spike_false_positive_rate:.4f} FP rate, TP = {near_spike_true_positive_at_FP:.4f}')
            print(f'soma voltage prediction explained variance = {soma_explained_variance_percent:.2f}%')
            print(f'soma RMSE = {soma_RMSE:.2f} (mV)')
            print(f'soma MAE = {soma_MAE:.2f} (mV)')
            print(f'inst rate explained variance = {inst_rate_explained_variance_percent:.2f}%')
            print(f'inst rate RMSE = {inst_rate_RMSE:.4f}')
            print(f'inst rate MAE = {inst_rate_MAE:.4f}')
            print(f'dend v explained variance = {dend_v_explained_variance_percent:.2f}%')
            print(f'dend v RMSE = {dend_v_RMSE:.2f} (mV)')
            print(f'dend v MAE = {dend_v_MAE:.2f} (mV)')

            print('---------------------------------------------------------')
        print('------------------------------------------------------------------------------------------')

    def save_model(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        state = {
            'model_state_dict': self.state_dict(),
            'in_channels': self.in_channels,
            'in_spatial_dim': self.in_spatial_dim,
            'first_layer_temporal_kernel_size': self.first_layer_temporal_kernel_size,
            'num_miniblocks_per_block_list': self.num_miniblocks_per_block_list,
            'num_features_per_block_list': self.num_features_per_block_list,
            'temporal_kernel_size_per_block_list': self.temporal_kernel_size_per_block_list,
            'temporal_dilation_per_block_list': self.temporal_dilation_per_block_list,
            'bottleneck_dim_per_block_list': self.bottleneck_dim_per_block_list,
            'nonlinearity_str': self.nonlinearity_str,
            'leaky_slope': self.leaky_slope,
            'norm_type': self.norm_type,
            'head_prefix_names': self.head_prefix_names,
            'head_out_channels': self.head_out_channels,
            'head_convert_out_ch_to_sp': self.head_convert_out_ch_to_sp,
            'head_spatial_kernel_sizes': self.head_spatial_kernel_sizes,
            'X_scale': self.X_scale,
            'V_bias_soma': self.V_bias_soma,
            'V_scale_soma': self.V_scale_soma,
            'V_clip_soma_min': self.V_clip_soma_min,
            'V_clip_soma_max': self.V_clip_soma_max,
            'V_bias_dend': self.V_bias_dend,
            'V_scale_dend': self.V_scale_dend,
            'V_clip_dend_min': self.V_clip_dend_min,
            'V_clip_dend_max': self.V_clip_dend_max,
            'y_inst_rate_multiplier': self.y_inst_rate_multiplier,                
            'metadata': self.metadata,
        }
        torch.save(state, path)
        print(f'Saved model to "{path}"')

    @classmethod
    def load_model(cls, path):
        state = torch.load(path, weights_only=False)
        print(f'Loading model from "{path}"')
        model = cls(
            in_channels=state['in_channels'],
            in_spatial_dim=state['in_spatial_dim'],
            first_layer_temporal_kernel_size=state['first_layer_temporal_kernel_size'],
            num_miniblocks_per_block_list=state['num_miniblocks_per_block_list'],
            num_features_per_block_list=state['num_features_per_block_list'],
            temporal_kernel_size_per_block_list=state['temporal_kernel_size_per_block_list'],
            temporal_dilation_per_block_list=state['temporal_dilation_per_block_list'],
            bottleneck_dim_per_block_list=state['bottleneck_dim_per_block_list'],
            nonlinearity_str=state['nonlinearity_str'],
            leaky_slope=state['leaky_slope'],
            norm_type=state.get('norm_type', 'BatchNorm'),
            head_prefix_names=state['head_prefix_names'],
            head_out_channels=state['head_out_channels'],
            head_convert_out_ch_to_sp=state['head_convert_out_ch_to_sp'],
            head_spatial_kernel_sizes=state['head_spatial_kernel_sizes'],
            X_scale=state['X_scale'],
            V_bias_soma=state['V_bias_soma'],
            V_scale_soma=state['V_scale_soma'],
            V_clip_soma_min=state['V_clip_soma_min'],
            V_clip_soma_max=state['V_clip_soma_max'],
            V_bias_dend=state['V_bias_dend'],
            V_scale_dend=state['V_scale_dend'],
            V_clip_dend_min=state['V_clip_dend_min'],
            V_clip_dend_max=state['V_clip_dend_max'],
            y_inst_rate_multiplier=state['y_inst_rate_multiplier'],
            metadata=state['metadata'],
        )
        model.load_state_dict(state['model_state_dict'])
        return model


# Define a model class that is a DNN twin (of type TCN) of a single neuron (multi branch F&F neuron) model
class SingleNeuronTwinModel_TCN(nn.Module):
    def __init__(self, 
                in_channels, in_spatial_dim, first_layer_temporal_kernel_size, 
                num_layers_per_block_list, num_features_per_block_list, 
                temporal_kernel_size_per_block_list, temporal_dilation_per_block_list, 
                bottleneck_dim_per_block_list=None,
                nonlinearity_str='lgelu', leaky_slope=0.2, norm_type='BatchNorm',
                head_prefix_names=['spikes', 'soma', 'near_spike', 'inst_rate', 'dend_v'],
                head_out_channels=[1, 1, 1, 1, 4], 
                head_convert_out_ch_to_sp=[False, False, False, False, False],
                head_spatial_kernel_sizes=None,
                X_scale=20.0, 
                V_bias_soma=-85, V_scale_soma=20, V_clip_soma_min=-102, V_clip_soma_max=-53,
                V_bias_dend=0.0, V_scale_dend=20, V_clip_dend_min=-50, V_clip_dend_max=50,
                y_inst_rate_multiplier=10.0,
                metadata={}):
        super().__init__()

        # backbone parameters
        self.in_channels = in_channels
        self.in_spatial_dim = in_spatial_dim
        self.first_layer_temporal_kernel_size = first_layer_temporal_kernel_size
        self.num_layers_per_block_list = num_layers_per_block_list
        self.num_features_per_block_list = num_features_per_block_list
        self.temporal_kernel_size_per_block_list = temporal_kernel_size_per_block_list
        self.temporal_dilation_per_block_list = temporal_dilation_per_block_list
        self.bottleneck_dim_per_block_list = bottleneck_dim_per_block_list
        self.nonlinearity_str = nonlinearity_str
        self.leaky_slope = leaky_slope
        self.norm_type = norm_type
        # head parameters
        self.head_prefix_names = head_prefix_names
        self.head_out_channels = head_out_channels
        self.head_convert_out_ch_to_sp = head_convert_out_ch_to_sp
        self.head_spatial_kernel_sizes = head_spatial_kernel_sizes
        # various scaling and clipping parameters
        self.X_scale = X_scale
        self.V_bias_soma = V_bias_soma
        self.V_scale_soma = V_scale_soma
        self.V_clip_soma_min = V_clip_soma_min
        self.V_clip_soma_max = V_clip_soma_max
        self.V_bias_dend = V_bias_dend
        self.V_scale_dend = V_scale_dend
        self.V_clip_dend_min = V_clip_dend_min
        self.V_clip_dend_max = V_clip_dend_max
        self.y_inst_rate_multiplier = y_inst_rate_multiplier
        self.metadata = metadata

        # create the backbone
        backbone = TCN_Backbone(
            in_channels=in_channels,
            in_spatial_dim=in_spatial_dim,
            first_layer_temporal_kernel_size=first_layer_temporal_kernel_size,
            num_layers_per_block_list=num_layers_per_block_list,
            num_features_per_block_list=num_features_per_block_list,
            temporal_kernel_size_per_block_list=temporal_kernel_size_per_block_list,
            temporal_dilation_per_block_list=temporal_dilation_per_block_list,
            bottleneck_dim_per_block_list=bottleneck_dim_per_block_list,
            nonlinearity_str=nonlinearity_str,
            leaky_slope=leaky_slope,
            norm_type=norm_type,
        )

        # set the output dimensions of the backbone by passing a dummy input through it
        B_dummy, T_dummy = 2, 1024
        X_spikes_dummy = torch.zeros(B_dummy, in_channels, in_spatial_dim, T_dummy)
        print('----------------------------------------------------')
        backbone.set_output_dims(X_spikes_dummy, print_output_dims=True)
        print('----------------------------------------------------')
        print(f'backbone: "{backbone.short_name}"')
        print(f'backbone out channels: {backbone.out_channels}')
        print(f'backbone out spatial dim: {backbone.out_spatial_dim}')
        print('----------------------------------------------------')

        # create the model
        self.backbone_and_heads = SingleNeuronTCN_Heads(
            backbone=backbone,
            head_prefix_names=head_prefix_names,
            head_out_channels=head_out_channels,
            head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
        )

        self.update_short_name()

    def forward(self, x):
        # x.shape = (B, in_C, in_S_dim, T)
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v = self.backbone_and_heads(x)
        return y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v

    def forward_debug(self, x):
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v =  self.backbone_and_heads.forward_debug(x)
        return y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v

    def update_short_name(self):
        # backbone description string
        temporal_description_str = f"T_{self.backbone_and_heads.backbone.total_temporal_kernel_size}"
        depth_description_str = f"D_{self.backbone_and_heads.backbone.total_depth}"
        width_description_str = f"W_{self.backbone_and_heads.backbone.average_width:.0f}"
        backbone_description_str = f"TCN_{depth_description_str}_{width_description_str}_{temporal_description_str}"

        # params description string
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_official / 1e3
        self.num_parmas_millions = self.num_params_official / 1e6
        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        # the short name of the model
        self.short_name = f"{backbone_description_str}_{param_description_str}"

    def set_metadata_original_neuron(self, original_neuron_model_metadata_dict):
        self.metadata['original_neuron_model'] = original_neuron_model_metadata_dict

    def set_metadata_training_params(self, training_params_dict):
        self.metadata['training_params'] = training_params_dict

    def set_metadata_learning_curves(self, learning_curves_dict):
        self.metadata['learning_curves'] = learning_curves_dict

    def set_metadata_eval_metrics(self, eval_metrics_dict):
        self.metadata['eval_metrics'] = eval_metrics_dict

    def print_main_metadata(self):

        print('------------------------------------------------------------------------------------------')
        print('DNN Twin Model Metadata')
        print('-----------------------')
        if 'original_neuron_model' in self.metadata:
            print('-----------')
            print('The basics:')
            print('-----------')
            print(f'Original Neuron: "{self.metadata["original_neuron_model"]["original_neuron_model_name"]}"')
            print(f'DNN Twin: "{self.short_name}"')
            print(f'Twin num params: {self.num_parmas_millions:.3f}M')
            print(f'Norm type: {self.norm_type}')
            print(f'Leaky slope: {self.leaky_slope}')
            print(f'Nonlinearity: {self.nonlinearity_str}')
            print('-------------------------------------------------------------------')

        if 'training_params' in self.metadata:
            print('----------------')
            print('Training params:')
            print('----------------')

            print(f'learning rate: {self.metadata["training_params"]["learning_rate"]}')
            print(f'weight decay: {self.metadata["training_params"]["weight_decay"]}')
            print(f'num epochs: {self.metadata["training_params"]["num_epochs"]}')
            print(f'num warmup epochs: {self.metadata["training_params"]["num_warmup_epochs"]}')
            print(f'num cooldown epochs: {self.metadata["training_params"]["num_cooldown_epochs"]}')
            print(f'train time window size: {self.metadata["training_params"]["train_time_window_size"]}')
            print(f'train batch size: {self.metadata["training_params"]["train_batch_size"]}\n')
            print('---------------------------------')

        if 'eval_metrics' in self.metadata:
            print('-------------------')
            print('Evaluation metrics:')
            print('-------------------')

            spikes_AUC_score = self.metadata['eval_metrics']['AUC_score']
            requested_false_positive_rate = self.metadata['eval_metrics']['requested_false_positive_rate']
            true_positive_at_FP = self.metadata['eval_metrics']['true_positive_at_FP']

            near_spike_AUC_score = self.metadata['eval_metrics']['near_spike_AUC_score']
            near_spike_false_positive_rate = self.metadata['eval_metrics']['near_spike_requested_false_positive_rate']
            near_spike_true_positive_at_FP = self.metadata['eval_metrics']['near_spike_true_positive_at_FP']

            soma_explained_variance_percent = self.metadata['eval_metrics']['soma_explained_variance_percent']
            soma_RMSE = self.metadata['eval_metrics']['soma_RMSE']
            soma_MAE = self.metadata['eval_metrics']['soma_MAE']

            inst_rate_explained_variance_percent = self.metadata['eval_metrics']['inst_rate_explained_variance_percent']
            inst_rate_RMSE = self.metadata['eval_metrics']['inst_rate_RMSE']
            inst_rate_MAE = self.metadata['eval_metrics']['inst_rate_MAE']

            dend_v_explained_variance_percent = self.metadata['eval_metrics']['dend_v_explained_variance_percent']
            dend_v_RMSE = self.metadata['eval_metrics']['dend_v_RMSE']
            dend_v_MAE = self.metadata['eval_metrics']['dend_v_MAE']

            print(f'Spikes AUC = {spikes_AUC_score:.4f}')
            print(f'at {requested_false_positive_rate:.4f} FP rate, TP = {true_positive_at_FP:.4f}')
            print(f'Near spike AUC = {near_spike_AUC_score:.4f}')
            print(f'at {near_spike_false_positive_rate:.4f} FP rate, TP = {near_spike_true_positive_at_FP:.4f}')
            print(f'soma voltage prediction explained variance = {soma_explained_variance_percent:.2f}%')
            print(f'soma RMSE = {soma_RMSE:.2f} (mV)')
            print(f'soma MAE = {soma_MAE:.2f} (mV)')
            print(f'inst rate explained variance = {inst_rate_explained_variance_percent:.2f}%')
            print(f'inst rate RMSE = {inst_rate_RMSE:.4f}')
            print(f'inst rate MAE = {inst_rate_MAE:.4f}')
            print(f'dend v explained variance = {dend_v_explained_variance_percent:.2f}%')
            print(f'dend v RMSE = {dend_v_RMSE:.2f} (mV)')
            print(f'dend v MAE = {dend_v_MAE:.2f} (mV)')

            print('---------------------------------------------------------')
        print('------------------------------------------------------------------------------------------')

    def save_model(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        state = {
            'model_state_dict': self.state_dict(),
            'in_channels': self.in_channels,
            'in_spatial_dim': self.in_spatial_dim,
            'first_layer_temporal_kernel_size': self.first_layer_temporal_kernel_size,
            'num_layers_per_block_list': self.num_layers_per_block_list,
            'num_features_per_block_list': self.num_features_per_block_list,
            'temporal_kernel_size_per_block_list': self.temporal_kernel_size_per_block_list,
            'temporal_dilation_per_block_list': self.temporal_dilation_per_block_list,
            'bottleneck_dim_per_block_list': self.bottleneck_dim_per_block_list,
            'nonlinearity_str': self.nonlinearity_str,
            'leaky_slope': self.leaky_slope,
            'norm_type': self.norm_type,
            'head_prefix_names': self.head_prefix_names,
            'head_out_channels': self.head_out_channels,
            'head_convert_out_ch_to_sp': self.head_convert_out_ch_to_sp,
            'head_spatial_kernel_sizes': self.head_spatial_kernel_sizes,
            'X_scale': self.X_scale,
            'V_bias_soma': self.V_bias_soma,
            'V_scale_soma': self.V_scale_soma,
            'V_clip_soma_min': self.V_clip_soma_min,
            'V_clip_soma_max': self.V_clip_soma_max,
            'V_bias_dend': self.V_bias_dend,
            'V_scale_dend': self.V_scale_dend,
            'V_clip_dend_min': self.V_clip_dend_min,
            'V_clip_dend_max': self.V_clip_dend_max,
            'y_inst_rate_multiplier': self.y_inst_rate_multiplier,                
            'metadata': self.metadata,
        }
        torch.save(state, path)
        print(f'Saved model to "{path}"')

    @classmethod
    def load_model(cls, path):
        state = torch.load(path, weights_only=False)
        print(f'Loading model from "{path}"')
        model = cls(
            in_channels=state['in_channels'],
            in_spatial_dim=state['in_spatial_dim'],
            first_layer_temporal_kernel_size=state['first_layer_temporal_kernel_size'],
            num_layers_per_block_list=state['num_layers_per_block_list'],
            num_features_per_block_list=state['num_features_per_block_list'],
            temporal_kernel_size_per_block_list=state['temporal_kernel_size_per_block_list'],
            temporal_dilation_per_block_list=state['temporal_dilation_per_block_list'],
            bottleneck_dim_per_block_list=state['bottleneck_dim_per_block_list'],
            nonlinearity_str=state['nonlinearity_str'],
            leaky_slope=state['leaky_slope'],
            norm_type=state.get('norm_type', 'BatchNorm'),
            head_prefix_names=state['head_prefix_names'],
            head_out_channels=state['head_out_channels'],
            head_convert_out_ch_to_sp=state['head_convert_out_ch_to_sp'],
            head_spatial_kernel_sizes=state['head_spatial_kernel_sizes'],
            X_scale=state['X_scale'],
            V_bias_soma=state['V_bias_soma'],
            V_scale_soma=state['V_scale_soma'],
            V_clip_soma_min=state['V_clip_soma_min'],
            V_clip_soma_max=state['V_clip_soma_max'],
            V_bias_dend=state['V_bias_dend'],
            V_scale_dend=state['V_scale_dend'],
            V_clip_dend_min=state['V_clip_dend_min'],
            V_clip_dend_max=state['V_clip_dend_max'],
            y_inst_rate_multiplier=state['y_inst_rate_multiplier'],
            metadata=state['metadata'],
        )
        model.load_state_dict(state['model_state_dict'])
        return model

# Define a model class that is a DNN twin (of type ELM) of a single neuron (multi branch F&F neuron) model
class SingleNeuronTwinModel_ELM(nn.Module):
    def __init__(self, 
                in_channels, in_spatial_dim, memory_dim=64, 
                mlp_num_hidden_layers=1, mlp_hidden_dim=None, mlp_nonlinearity_str='lsilu', mlp_leaky_slope=0.2, mlp_pre_norm_type='BatchNorm',
                post_mlp_nonlinearity_str='ltanh', post_mlp_leaky_slope=0.1, lambda_value=5.0,
                synapse_tau_value=5.0, memory_tau_min=1.0, memory_tau_max=128.0, learn_memory_tau=True, w_s_value=0.5, delta_t=1.0,
                compile_recurrent_step=False,
                head_prefix_names=['spikes', 'soma', 'near_spike', 'inst_rate', 'dend_v'],
                head_out_channels=[1, 1, 1, 1, 4], 
                head_convert_out_ch_to_sp=[False, False, False, False, False],
                head_spatial_kernel_sizes=None,
                X_scale=20.0, 
                V_bias_soma=-85, V_scale_soma=20, V_clip_soma_min=-102, V_clip_soma_max=-53,
                V_bias_dend=0.0, V_scale_dend=20, V_clip_dend_min=-50, V_clip_dend_max=50,
                y_inst_rate_multiplier=10.0,
                metadata={}):
        super().__init__()

        # backbone parameters
        self.in_channels = in_channels
        self.in_spatial_dim = in_spatial_dim
        self.memory_dim = memory_dim
        self.compile_recurrent_step = compile_recurrent_step
        self.mlp_num_hidden_layers = mlp_num_hidden_layers
        self.mlp_hidden_dim = mlp_hidden_dim if mlp_hidden_dim else 2 * self.memory_dim
        self.mlp_nonlinearity_str = mlp_nonlinearity_str
        self.mlp_leaky_slope = mlp_leaky_slope
        self.mlp_pre_norm_type = mlp_pre_norm_type
        self.post_mlp_nonlinearity_str = post_mlp_nonlinearity_str
        self.post_mlp_leaky_slope = post_mlp_leaky_slope
        self.lambda_value = lambda_value
        self.synapse_tau_value = synapse_tau_value
        self.memory_tau_min = memory_tau_min
        self.memory_tau_max = memory_tau_max
        self.learn_memory_tau = learn_memory_tau
        self.w_s_value = w_s_value
        self.delta_t = delta_t
        # head parameters
        self.head_prefix_names = head_prefix_names
        self.head_out_channels = head_out_channels
        self.head_convert_out_ch_to_sp = head_convert_out_ch_to_sp
        self.head_spatial_kernel_sizes = head_spatial_kernel_sizes
        # various scaling and clipping parameters
        self.X_scale = X_scale
        self.V_bias_soma = V_bias_soma
        self.V_scale_soma = V_scale_soma
        self.V_clip_soma_min = V_clip_soma_min
        self.V_clip_soma_max = V_clip_soma_max
        self.V_bias_dend = V_bias_dend
        self.V_scale_dend = V_scale_dend
        self.V_clip_dend_min = V_clip_dend_min
        self.V_clip_dend_max = V_clip_dend_max
        self.y_inst_rate_multiplier = y_inst_rate_multiplier
        self.metadata = metadata

        # create the backbone
        backbone = ELM_Backbone(
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
        )
        if self.compile_recurrent_step:
            backbone.compile_step_fn()

        # set the output dimensions of the backbone by passing a dummy input through it
        B_dummy, T_dummy = 2, 256
        X_spikes_dummy = torch.zeros(B_dummy, in_channels, in_spatial_dim, T_dummy)
        print('----------------------------------------------------')
        backbone.set_output_dims(X_spikes_dummy, print_output_dims=True)
        print('----------------------------------------------------')
        print(f'backbone: "{backbone.short_name}"')
        print(f'backbone out channels: {backbone.out_channels}')
        print(f'backbone out spatial dim: {backbone.out_spatial_dim}')
        print('----------------------------------------------------')

        # create the model
        self.backbone_and_heads = SingleNeuronTCN_Heads(
            backbone=backbone,
            head_prefix_names=head_prefix_names,
            head_out_channels=head_out_channels,
            head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
        )

        self.update_short_name()

    def forward(self, x):
        # x.shape = (B, in_C, in_S_dim, T)
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v = self.backbone_and_heads(x)
        return y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v

    def forward_debug(self, x):
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v =  self.backbone_and_heads.forward_debug(x)
        return y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v

    def update_short_name(self):
        # backbone description string
        depth_description_str = f"D_{self.backbone_and_heads.backbone.mlp_num_hidden_layers}"
        memory_description_str = f"M_{self.backbone_and_heads.backbone.memory_dim}"
        backbone_description_str = f"ELM_{depth_description_str}_{memory_description_str}"

        # params description string
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_official / 1e3
        self.num_parmas_millions = self.num_params_official / 1e6
        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        # the short name of the model
        self.short_name = f"{backbone_description_str}_{param_description_str}"

    def set_metadata_original_neuron(self, original_neuron_model_metadata_dict):
        self.metadata['original_neuron_model'] = original_neuron_model_metadata_dict

    def set_metadata_training_params(self, training_params_dict):
        self.metadata['training_params'] = training_params_dict

    def set_metadata_learning_curves(self, learning_curves_dict):
        self.metadata['learning_curves'] = learning_curves_dict

    def set_metadata_eval_metrics(self, eval_metrics_dict):
        self.metadata['eval_metrics'] = eval_metrics_dict

    def print_main_metadata(self):

        print('------------------------------------------------------------------------------------------')
        print('DNN Twin Model Metadata')
        print('-----------------------')
        if 'original_neuron_model' in self.metadata:
            print('-----------')
            print('The basics:')
            print('-----------')
            print(f'Original Neuron: "{self.metadata["original_neuron_model"]["original_neuron_model_name"]}"')
            print(f'DNN Twin: "{self.short_name}"')
            print(f'Twin num params: {self.num_parmas_millions:.3f}M')
            print(f'MLP Pre-Norm type: {self.mlp_pre_norm_type}')
            print(f'MLP Leaky slope: {self.mlp_leaky_slope}')
            print(f'MLP Nonlinearity: {self.mlp_nonlinearity_str}')
            print(f'Post-MLP Nonlinearity: {self.post_mlp_nonlinearity_str}')
            print(f'Post-MLP Leaky slope: {self.post_mlp_leaky_slope}')
            print('-------------------------------------------------------------------')

        if 'training_params' in self.metadata:
            print('----------------')
            print('Training params:')
            print('----------------')

            print(f'learning rate: {self.metadata["training_params"]["learning_rate"]}')
            print(f'weight decay: {self.metadata["training_params"]["weight_decay"]}')
            print(f'num epochs: {self.metadata["training_params"]["num_epochs"]}')
            print(f'num warmup epochs: {self.metadata["training_params"]["num_warmup_epochs"]}')
            print(f'num cooldown epochs: {self.metadata["training_params"]["num_cooldown_epochs"]}')
            print(f'train time window size: {self.metadata["training_params"]["train_time_window_size"]}')
            print(f'train batch size: {self.metadata["training_params"]["train_batch_size"]}\n')
            print('---------------------------------')

        if 'eval_metrics' in self.metadata:
            print('-------------------')
            print('Evaluation metrics:')
            print('-------------------')

            spikes_AUC_score = self.metadata['eval_metrics']['AUC_score']
            requested_false_positive_rate = self.metadata['eval_metrics']['requested_false_positive_rate']
            true_positive_at_FP = self.metadata['eval_metrics']['true_positive_at_FP']

            near_spike_AUC_score = self.metadata['eval_metrics']['near_spike_AUC_score']
            near_spike_false_positive_rate = self.metadata['eval_metrics']['near_spike_requested_false_positive_rate']
            near_spike_true_positive_at_FP = self.metadata['eval_metrics']['near_spike_true_positive_at_FP']

            soma_explained_variance_percent = self.metadata['eval_metrics']['soma_explained_variance_percent']
            soma_RMSE = self.metadata['eval_metrics']['soma_RMSE']
            soma_MAE = self.metadata['eval_metrics']['soma_MAE']

            inst_rate_explained_variance_percent = self.metadata['eval_metrics']['inst_rate_explained_variance_percent']
            inst_rate_RMSE = self.metadata['eval_metrics']['inst_rate_RMSE']
            inst_rate_MAE = self.metadata['eval_metrics']['inst_rate_MAE']

            dend_v_explained_variance_percent = self.metadata['eval_metrics']['dend_v_explained_variance_percent']
            dend_v_RMSE = self.metadata['eval_metrics']['dend_v_RMSE']
            dend_v_MAE = self.metadata['eval_metrics']['dend_v_MAE']

            print(f'Spikes AUC = {spikes_AUC_score:.4f}')
            print(f'at {requested_false_positive_rate:.4f} FP rate, TP = {true_positive_at_FP:.4f}')
            print(f'Near spike AUC = {near_spike_AUC_score:.4f}')
            print(f'at {near_spike_false_positive_rate:.4f} FP rate, TP = {near_spike_true_positive_at_FP:.4f}')
            print(f'soma voltage prediction explained variance = {soma_explained_variance_percent:.2f}%')
            print(f'soma RMSE = {soma_RMSE:.2f} (mV)')
            print(f'soma MAE = {soma_MAE:.2f} (mV)')
            print(f'inst rate explained variance = {inst_rate_explained_variance_percent:.2f}%')
            print(f'inst rate RMSE = {inst_rate_RMSE:.4f}')
            print(f'inst rate MAE = {inst_rate_MAE:.4f}')
            print(f'dend v explained variance = {dend_v_explained_variance_percent:.2f}%')
            print(f'dend v RMSE = {dend_v_RMSE:.2f} (mV)')
            print(f'dend v MAE = {dend_v_MAE:.2f} (mV)')

            print('---------------------------------------------------------')
        print('------------------------------------------------------------------------------------------')

    def save_model(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        state = {
            'model_state_dict': self.state_dict(),
            'in_channels': self.in_channels,
            'in_spatial_dim': self.in_spatial_dim,
            'memory_dim': self.memory_dim,
            'mlp_num_hidden_layers': self.mlp_num_hidden_layers,
            'mlp_hidden_dim': self.mlp_hidden_dim,
            'mlp_nonlinearity_str': self.mlp_nonlinearity_str,
            'mlp_leaky_slope': self.mlp_leaky_slope,
            'mlp_pre_norm_type': self.mlp_pre_norm_type,
            'post_mlp_nonlinearity_str': self.post_mlp_nonlinearity_str,
            'post_mlp_leaky_slope': self.post_mlp_leaky_slope,
            'lambda_value': self.lambda_value,
            'synapse_tau_value': self.synapse_tau_value,
            'memory_tau_min': self.memory_tau_min,
            'memory_tau_max': self.memory_tau_max,
            'learn_memory_tau': self.learn_memory_tau,
            'w_s_value': self.w_s_value,
            'delta_t': self.delta_t,
            'head_prefix_names': self.head_prefix_names,
            'head_out_channels': self.head_out_channels,
            'head_convert_out_ch_to_sp': self.head_convert_out_ch_to_sp,
            'head_spatial_kernel_sizes': self.head_spatial_kernel_sizes,
            'X_scale': self.X_scale,
            'V_bias_soma': self.V_bias_soma,
            'V_scale_soma': self.V_scale_soma,
            'V_clip_soma_min': self.V_clip_soma_min,
            'V_clip_soma_max': self.V_clip_soma_max,
            'V_bias_dend': self.V_bias_dend,
            'V_scale_dend': self.V_scale_dend,
            'V_clip_dend_min': self.V_clip_dend_min,
            'V_clip_dend_max': self.V_clip_dend_max,
            'y_inst_rate_multiplier': self.y_inst_rate_multiplier,                
            'metadata': self.metadata,
        }
        torch.save(state, path)
        print(f'Saved model to "{path}"')

    @classmethod
    def load_model(cls, path):
        state = torch.load(path, weights_only=False)
        print(f'Loading model from "{path}"')
        model = cls(
            in_channels=state['in_channels'],
            in_spatial_dim=state['in_spatial_dim'],
            memory_dim=state['memory_dim'],
            mlp_num_hidden_layers=state['mlp_num_hidden_layers'],
            mlp_hidden_dim=state['mlp_hidden_dim'],
            mlp_nonlinearity_str=state['mlp_nonlinearity_str'],
            mlp_leaky_slope=state['mlp_leaky_slope'],
            mlp_pre_norm_type=state['mlp_pre_norm_type'],
            post_mlp_nonlinearity_str=state['post_mlp_nonlinearity_str'],
            post_mlp_leaky_slope=state['post_mlp_leaky_slope'],
            lambda_value=state['lambda_value'],
            synapse_tau_value=state['synapse_tau_value'],
            memory_tau_min=state['memory_tau_min'],
            memory_tau_max=state['memory_tau_max'],
            learn_memory_tau=state['learn_memory_tau'],
            w_s_value=state['w_s_value'],
            delta_t=state['delta_t'],
            head_prefix_names=state['head_prefix_names'],
            head_out_channels=state['head_out_channels'],
            head_convert_out_ch_to_sp=state['head_convert_out_ch_to_sp'],
            head_spatial_kernel_sizes=state['head_spatial_kernel_sizes'],
            X_scale=state['X_scale'],
            V_bias_soma=state['V_bias_soma'],
            V_scale_soma=state['V_scale_soma'],
            V_clip_soma_min=state['V_clip_soma_min'],
            V_clip_soma_max=state['V_clip_soma_max'],
            V_bias_dend=state['V_bias_dend'],
            V_scale_dend=state['V_scale_dend'],
            V_clip_dend_min=state['V_clip_dend_min'],
            V_clip_dend_max=state['V_clip_dend_max'],
            y_inst_rate_multiplier=state['y_inst_rate_multiplier'],
            metadata=state['metadata'],
        )
        model.load_state_dict(state['model_state_dict'])
        return model


# Define a model class that is a DNN twin (of type Transformer) of a single neuron (multi branch F&F neuron) model
class SingleNeuronTwinModel_Transformer(nn.Module):
    def __init__(self, 
                in_channels, in_spatial_dim,
                # Transformer backbone parameters
                first_layer_temporal_kernel_size=1,
                first_norm_type='RMSNorm',
                first_nonlinearity_str='lgelu',
                first_leaky_slope=0.2,
                d_model=64,
                n_heads=8,
                num_layers=4,
                window_size=32,
                max_seq_len=8192,
                dropout_rate=0.1,
                ffn_type='swiglu',
                ffn_hidden_dim=None,
                SWA_type='blocked',
                SWA_block_size=None,
                use_complex_number_rope=False,
                norm_type='rms_norm',
                # Head parameters
                head_prefix_names=['spikes', 'soma', 'near_spike', 'inst_rate', 'dend_v'],
                head_out_channels=[1, 1, 1, 1, 4], 
                head_convert_out_ch_to_sp=[False, False, False, False, False],
                head_spatial_kernel_sizes=None,
                # Scaling parameters
                X_scale=20.0, 
                V_bias_soma=-85, V_scale_soma=20, V_clip_soma_min=-102, V_clip_soma_max=-53,
                V_bias_dend=0.0, V_scale_dend=20, V_clip_dend_min=-50, V_clip_dend_max=50,
                y_inst_rate_multiplier=10.0,
                metadata={}):
        super().__init__()

        # backbone parameters
        self.in_channels = in_channels
        self.in_spatial_dim = in_spatial_dim
        self.first_layer_temporal_kernel_size = first_layer_temporal_kernel_size
        self.first_norm_type = first_norm_type
        self.first_nonlinearity_str = first_nonlinearity_str
        self.first_leaky_slope = first_leaky_slope
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.ffn_hidden_dim = ffn_hidden_dim
        self.SWA_type = SWA_type
        self.SWA_block_size = SWA_block_size
        self.use_complex_number_rope = use_complex_number_rope
        self.norm_type = norm_type
        # head parameters
        self.head_prefix_names = head_prefix_names
        self.head_out_channels = head_out_channels
        self.head_convert_out_ch_to_sp = head_convert_out_ch_to_sp
        self.head_spatial_kernel_sizes = head_spatial_kernel_sizes
        # various scaling and clipping parameters
        self.X_scale = X_scale
        self.V_bias_soma = V_bias_soma
        self.V_scale_soma = V_scale_soma
        self.V_clip_soma_min = V_clip_soma_min
        self.V_clip_soma_max = V_clip_soma_max
        self.V_bias_dend = V_bias_dend
        self.V_scale_dend = V_scale_dend
        self.V_clip_dend_min = V_clip_dend_min
        self.V_clip_dend_max = V_clip_dend_max
        self.y_inst_rate_multiplier = y_inst_rate_multiplier
        self.metadata = metadata

        # create the backbone
        backbone = Transformer_Backbone(
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
        )

        # set the output dimensions of the backbone by passing a dummy input through it
        B_dummy, T_dummy = 2, 1024
        X_spikes_dummy = torch.zeros(B_dummy, in_channels, in_spatial_dim, T_dummy)
        print('----------------------------------------------------')
        backbone.set_output_dims(X_spikes_dummy, print_output_dims=True)
        print('----------------------------------------------------')
        print(f'backbone: "{backbone.short_name}"')
        print(f'backbone out channels: {backbone.out_channels}')
        print(f'backbone out spatial dim: {backbone.out_spatial_dim}')
        print('----------------------------------------------------')

        # create the model
        self.backbone_and_heads = SingleNeuronTCN_Heads(
            backbone=backbone,
            head_prefix_names=head_prefix_names,
            head_out_channels=head_out_channels,
            head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
        )

        # initialize weights
        self._init_weights()

        self.update_short_name()

    def _init_weights(self):
        """Initialize weights Linear layers with normal(0, 0.02)"""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
        
        # apply basic initialization to all modules
        self.apply(_basic_init)
        
    def forward(self, x):
        # x.shape = (B, in_C, in_S_dim, T)
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v = self.backbone_and_heads(x)
        return y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v

    def forward_debug(self, x):
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v =  self.backbone_and_heads.forward_debug(x)
        return y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v

    def update_short_name(self):

        if self.SWA_type == 'full':
            total_temporal_dependence = self.max_seq_len
        else:
            total_temporal_dependence = self.first_layer_temporal_kernel_size - 1
            total_temporal_dependence += self.num_layers * (self.window_size - 1)

        # backbone description string
        temporal_description_str = f"T_{total_temporal_dependence}"
        backbone_description_str = f"Transformer_D_{self.num_layers}_W_{self.d_model}_H_{self.n_heads}_{temporal_description_str}"

        # params description string
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_official / 1e3
        self.num_parmas_millions = self.num_params_official / 1e6
        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        # the short name of the model
        self.short_name = f"{backbone_description_str}_{param_description_str}"

    def set_metadata_original_neuron(self, original_neuron_model_metadata_dict):
        self.metadata['original_neuron_model'] = original_neuron_model_metadata_dict

    def set_metadata_training_params(self, training_params_dict):
        self.metadata['training_params'] = training_params_dict

    def set_metadata_learning_curves(self, learning_curves_dict):
        self.metadata['learning_curves'] = learning_curves_dict

    def set_metadata_eval_metrics(self, eval_metrics_dict):
        self.metadata['eval_metrics'] = eval_metrics_dict

    def print_main_metadata(self):

        print('------------------------------------------------------------------------------------------')
        print('DNN Twin Model Metadata')
        print('-----------------------')
        if 'original_neuron_model' in self.metadata:
            print('-----------')
            print('The basics:')
            print('-----------')
            print(f'Original Neuron: "{self.metadata["original_neuron_model"]["original_neuron_model_name"]}"')
            print(f'DNN Twin: "{self.short_name}"')
            print(f'Twin num params: {self.num_parmas_millions:.3f}M')
            print(f'First norm type: {self.first_norm_type}')
            print(f'First nonlinearity: {self.first_nonlinearity_str}')
            print(f'First leaky slope: {self.first_leaky_slope}')
            print(f'Transformer norm type: {self.norm_type}')
            print(f'FFN type: {self.ffn_type}')
            print(f'SWA type: {self.SWA_type}')
            print('-------------------------------------------------------------------')

        if 'training_params' in self.metadata:
            print('----------------')
            print('Training params:')
            print('----------------')

            print(f'learning rate: {self.metadata["training_params"]["learning_rate"]}')
            print(f'weight decay: {self.metadata["training_params"]["weight_decay"]}')
            print(f'num epochs: {self.metadata["training_params"]["num_epochs"]}')
            print(f'num warmup epochs: {self.metadata["training_params"]["num_warmup_epochs"]}')
            print(f'num cooldown epochs: {self.metadata["training_params"]["num_cooldown_epochs"]}')
            print(f'train time window size: {self.metadata["training_params"]["train_time_window_size"]}')
            print(f'train batch size: {self.metadata["training_params"]["train_batch_size"]}\n')
            print('---------------------------------')

        if 'eval_metrics' in self.metadata:
            print('-------------------')
            print('Evaluation metrics:')
            print('-------------------')

            spikes_AUC_score = self.metadata['eval_metrics']['AUC_score']
            requested_false_positive_rate = self.metadata['eval_metrics']['requested_false_positive_rate']
            true_positive_at_FP = self.metadata['eval_metrics']['true_positive_at_FP']

            near_spike_AUC_score = self.metadata['eval_metrics']['near_spike_AUC_score']
            near_spike_false_positive_rate = self.metadata['eval_metrics']['near_spike_requested_false_positive_rate']
            near_spike_true_positive_at_FP = self.metadata['eval_metrics']['near_spike_true_positive_at_FP']

            soma_explained_variance_percent = self.metadata['eval_metrics']['soma_explained_variance_percent']
            soma_RMSE = self.metadata['eval_metrics']['soma_RMSE']
            soma_MAE = self.metadata['eval_metrics']['soma_MAE']

            inst_rate_explained_variance_percent = self.metadata['eval_metrics']['inst_rate_explained_variance_percent']
            inst_rate_RMSE = self.metadata['eval_metrics']['inst_rate_RMSE']
            inst_rate_MAE = self.metadata['eval_metrics']['inst_rate_MAE']

            dend_v_explained_variance_percent = self.metadata['eval_metrics']['dend_v_explained_variance_percent']
            dend_v_RMSE = self.metadata['eval_metrics']['dend_v_RMSE']
            dend_v_MAE = self.metadata['eval_metrics']['dend_v_MAE']

            print(f'Spikes AUC = {spikes_AUC_score:.4f}')
            print(f'at {requested_false_positive_rate:.4f} FP rate, TP = {true_positive_at_FP:.4f}')
            print(f'Near spike AUC = {near_spike_AUC_score:.4f}')
            print(f'at {near_spike_false_positive_rate:.4f} FP rate, TP = {near_spike_true_positive_at_FP:.4f}')
            print(f'soma voltage prediction explained variance = {soma_explained_variance_percent:.2f}%')
            print(f'soma RMSE = {soma_RMSE:.2f} (mV)')
            print(f'soma MAE = {soma_MAE:.2f} (mV)')
            print(f'inst rate explained variance = {inst_rate_explained_variance_percent:.2f}%')
            print(f'inst rate RMSE = {inst_rate_RMSE:.4f}')
            print(f'inst rate MAE = {inst_rate_MAE:.4f}')
            print(f'dend v explained variance = {dend_v_explained_variance_percent:.2f}%')
            print(f'dend v RMSE = {dend_v_RMSE:.2f} (mV)')
            print(f'dend v MAE = {dend_v_MAE:.2f} (mV)')

            print('---------------------------------------------------------')
        print('------------------------------------------------------------------------------------------')

    def save_model(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        state = {
            'model_state_dict': self.state_dict(),
            'in_channels': self.in_channels,
            'in_spatial_dim': self.in_spatial_dim,
            'first_layer_temporal_kernel_size': self.first_layer_temporal_kernel_size,
            'first_norm_type': self.first_norm_type,
            'first_nonlinearity_str': self.first_nonlinearity_str,
            'first_leaky_slope': self.first_leaky_slope,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'num_layers': self.num_layers,
            'window_size': self.window_size,
            'max_seq_len': self.max_seq_len,
            'dropout_rate': self.dropout_rate,
            'ffn_type': self.ffn_type,
            'ffn_hidden_dim': self.ffn_hidden_dim,
            'SWA_type': self.SWA_type,
            'SWA_block_size': self.SWA_block_size,
            'use_complex_number_rope': self.use_complex_number_rope,
            'norm_type': self.norm_type,
            'head_prefix_names': self.head_prefix_names,
            'head_out_channels': self.head_out_channels,
            'head_convert_out_ch_to_sp': self.head_convert_out_ch_to_sp,
            'head_spatial_kernel_sizes': self.head_spatial_kernel_sizes,
            'X_scale': self.X_scale,
            'V_bias_soma': self.V_bias_soma,
            'V_scale_soma': self.V_scale_soma,
            'V_clip_soma_min': self.V_clip_soma_min,
            'V_clip_soma_max': self.V_clip_soma_max,
            'V_bias_dend': self.V_bias_dend,
            'V_scale_dend': self.V_scale_dend,
            'V_clip_dend_min': self.V_clip_dend_min,
            'V_clip_dend_max': self.V_clip_dend_max,
            'y_inst_rate_multiplier': self.y_inst_rate_multiplier,                
            'metadata': self.metadata,
        }
        torch.save(state, path)
        print(f'Saved model to "{path}"')

    @classmethod
    def load_model(cls, path):
        state = torch.load(path, weights_only=False)
        print(f'Loading model from "{path}"')
        model = cls(
            in_channels=state['in_channels'],
            in_spatial_dim=state['in_spatial_dim'],
            first_layer_temporal_kernel_size=state['first_layer_temporal_kernel_size'],
            first_norm_type=state.get('first_norm_type', 'RMSNorm'),
            first_nonlinearity_str=state.get('first_nonlinearity_str', 'lgelu'),
            first_leaky_slope=state.get('first_leaky_slope', 0.2),
            d_model=state['d_model'],
            n_heads=state['n_heads'],
            num_layers=state['num_layers'],
            window_size=state['window_size'],
            max_seq_len=state['max_seq_len'],
            dropout_rate=state['dropout_rate'],
            ffn_type=state.get('ffn_type', 'swiglu'),
            ffn_hidden_dim=state.get('ffn_hidden_dim', None),
            SWA_type=state.get('SWA_type', 'blocked'),
            SWA_block_size=state.get('SWA_block_size', None),
            use_complex_number_rope=state.get('use_complex_number_rope', False),
            norm_type=state.get('norm_type', 'rms_norm'),
            head_prefix_names=state['head_prefix_names'],
            head_out_channels=state['head_out_channels'],
            head_convert_out_ch_to_sp=state['head_convert_out_ch_to_sp'],
            head_spatial_kernel_sizes=state['head_spatial_kernel_sizes'],
            X_scale=state['X_scale'],
            V_bias_soma=state['V_bias_soma'],
            V_scale_soma=state['V_scale_soma'],
            V_clip_soma_min=state['V_clip_soma_min'],
            V_clip_soma_max=state['V_clip_soma_max'],
            V_bias_dend=state['V_bias_dend'],
            V_scale_dend=state['V_scale_dend'],
            V_clip_dend_min=state['V_clip_dend_min'],
            V_clip_dend_max=state['V_clip_dend_max'],
            y_inst_rate_multiplier=state['y_inst_rate_multiplier'],
            metadata=state['metadata'],
        )
        model.load_state_dict(state['model_state_dict'])
        return model


#%% Universal model loading function

def load_twin_model(model_path):
    """
    Universal function to load twin models based on the filename pattern.
    Automatically detects whether to load a TCN, ResNetTCN, ELM, recurrent_TCN, TCN_ELM, or Transformer model based on the model name.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        model: Loaded model (SingleNeuronTwinModel_TCN, SingleNeuronTwinModel_ResNetTCN, SingleNeuronTwinModel_ELM, 
                SingleNeuronTwinModel_recurrent_TCN, SingleNeuronTwinModel_TCN_ELM, or SingleNeuronTwinModel_Transformer)
    """
    import os
    
    model_filename = os.path.basename(model_path)
    
    # Detect model type based on filename patterns
    if "_Transformer_D" in model_filename:
        print(f"Detected Transformer model from filename: {model_filename}")
        model = SingleNeuronTwinModel_Transformer.load_model(model_path)
    elif "_TCN_ResNet_D" in model_filename:
        print(f"Detected ResNetTCN model from filename: {model_filename}")
        model = SingleNeuronTwinModel_ResNetTCN.load_model(model_path)
    elif "_ELM_D" in model_filename:
        print(f"Detected ELM model from filename: {model_filename}")
        model = SingleNeuronTwinModel_ELM.load_model(model_path)
    elif "_TCN_D" in model_filename:
        print(f"Detected TCN model from filename: {model_filename}")
        model = SingleNeuronTwinModel_TCN.load_model(model_path)
    else:
        # Fallback: try to determine from the saved state if possible
        print(f"Could not determine model type from filename: {model_filename}")
        print("Attempting to determine model type from saved state...")
        
        try:
            state = torch.load(model_path, weights_only=False)
            # Check if the state contains parameters specific to Transformer, ResNetTCN, TCN, or ELM
            if 'd_model' in state and 'n_heads' in state and 'window_size' in state and 'SWA_type' in state:
                print("Found 'd_model', 'n_heads', 'window_size', and 'SWA_type' - loading as Transformer model")
                model = SingleNeuronTwinModel_Transformer.load_model(model_path)
            elif 'num_miniblocks_per_block_list' in state:
                print("Found 'num_miniblocks_per_block_list' - loading as ResNetTCN model")
                model = SingleNeuronTwinModel_ResNetTCN.load_model(model_path)
            elif 'num_memory' in state:
                print("Found 'num_memory' - loading as ELM model")
                model = SingleNeuronTwinModel_ELM.load_model(model_path)
            elif 'num_layers_per_block_list' in state:
                print("Found 'num_layers_per_block_list' - loading as TCN model") 
                model = SingleNeuronTwinModel_TCN.load_model(model_path)
            else:
                raise ValueError("Cannot determine model type from saved state")
        except Exception as e:
            print(f"Error determining model type: {e}")
            print("Defaulting to ResNetTCN model (for backward compatibility)")
            model = SingleNeuronTwinModel_ResNetTCN.load_model(model_path)
    
    return model

#%% Example usage of the classes defined above

if __name__ == "__main__":

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('----------------------------')
    print(f'Using device: {device}')
    print('----------------------------')
    
    # Set tensor precision for better torch.compile() performance
    torch.set_float32_matmul_precision('high')
    print('Set float32_matmul_precision to "high" for better compilation performance (ELM only)')
    
    # Example data parameters
    in_channels = 2      # exc & inh channels
    in_spatial_dim = 16  # number of total dendritic segments (spatial dimension)
    batch_size = 8
    temporal_length = 128
    # Create sample input: (batch, channels, segments, time)
    print('--------------------------------------------------------------')
    print('Creating sample input...')
    X_sample = 5 * torch.rand(batch_size, in_channels, in_spatial_dim, temporal_length).to(device)
    print(f'Input shape: {X_sample.shape}')
    print('--------------------------------------------------------------')

    #%% ResNetTCN model

    # Example model parameters for the ResNetTCN backbone
    first_layer_temporal_kernel_size = 11
    num_miniblocks_per_block_list = [1, 1]
    num_features_per_block_list = [16, 16]
    temporal_kernel_size_per_block_list = [31, 51]
    temporal_dilation_per_block_list = [1, 1]
    nonlinearity_str = 'lgelu'
    leaky_slope = 0.1
    norm_type = 'BatchNorm'
    
    # Head parameters
    head_prefix_names = ['spikes', 'soma', 'near_spike', 'inst_rate', 'dend_v']
    head_out_channels = [1, 1, 1, 1, 4]  # 4 dendritic voltage channels
    head_convert_out_ch_to_sp = [False, False, False, False, False]
    
    # Scaling parameters
    X_scale = 8.0
    V_bias_soma = -85
    V_scale_soma = 20
    V_clip_soma_min = -102
    V_clip_soma_max = -53
    V_bias_dend = 0.0
    V_scale_dend = 20
    V_clip_dend_min = -50
    V_clip_dend_max = 50
    y_inst_rate_multiplier = 10.0
    
    print('-------------------------------------------------------------------------------------')
    print('Creating ResNetTCN model:')
    print('-------------------------')
    twin_model_resnet_TCN = SingleNeuronTwinModel_ResNetTCN(
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
    print('-------------------------------------------------------------------------------------')
    print(f'Model created: "{twin_model_resnet_TCN.short_name}"')
    print(f'Number of parameters: {twin_model_resnet_TCN.num_parmas_millions:.3f}M')
    print('-------------------------------------------------------------------------------------')
    
    # Test forward pass with sample input
    print('-----------------------------------------------')
    print('Testing forward pass...')
    
    # Forward pass
    twin_model_resnet_TCN.eval()
    with torch.no_grad():
        outputs = twin_model_resnet_TCN(X_sample)
        y_spikes, y_soma, y_near_spike, y_inst_rate, y_dend_v = outputs
    
    print('Output shapes:')
    print(f'  y_spikes: {y_spikes.shape}')
    print(f'  y_soma: {y_soma.shape}')
    print(f'  y_near_spike: {y_near_spike.shape}')
    print(f'  y_inst_rate: {y_inst_rate.shape}')
    print(f'  y_dend_v: {y_dend_v.shape}')
    
    print('Output value ranges:')
    print(f'  y_spikes: [{y_spikes.min():.3f}, {y_spikes.max():.3f}]')
    print(f'  y_soma: [{y_soma.min():.3f}, {y_soma.max():.3f}]')
    print(f'  y_near_spike: [{y_near_spike.min():.3f}, {y_near_spike.max():.3f}]')
    print(f'  y_inst_rate: [{y_inst_rate.min():.3f}, {y_inst_rate.max():.3f}]')
    print(f'  y_dend_v: [{y_dend_v.min():.3f}, {y_dend_v.max():.3f}]')
    
    print('-----------------------------------------------')
    
    #%% Test torch.compile() performance for ResNetTCN model
    
    import time
    
    print('=' * 80)
    print('Testing torch.compile() performance for ResNetTCN model')
    print('=' * 80)
    num_runs = 50

    # Warmup runs for fair comparison
    print('Performing warmup runs...')
    for _ in range(5):
        with torch.no_grad():
            _ = twin_model_resnet_TCN(X_sample)
    
    # Benchmark original model
    print('Benchmarking original ResNetTCN model...')
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = twin_model_resnet_TCN(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    original_time = (time.time() - start_time) / num_runs
    
    # Compile model and measure compilation time
    print('Compiling ResNetTCN model with torch.compile()...')
    compile_start = time.time()
    compiled_resnet_model = torch.compile(twin_model_resnet_TCN)
    compile_time = time.time() - compile_start
    print(f'Initial compilation setup time: {compile_time:.4f} seconds')
    
    # First run triggers actual compilation
    print('Running first inference (triggers actual compilation)...')
    actual_compile_start = time.time()
    with torch.no_grad():
        _ = compiled_resnet_model(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    actual_compile_time = time.time() - actual_compile_start
    print(f'Actual compilation time (first run): {actual_compile_time:.4f} seconds')
    
    # Warmup compiled model
    print('Warming up compiled model...')
    for _ in range(5):
        with torch.no_grad():
            _ = compiled_resnet_model(X_sample)
    
    # Benchmark compiled model
    print('Benchmarking compiled ResNetTCN model...')
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = compiled_resnet_model(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    compiled_time = (time.time() - start_time) / num_runs
    
    # Calculate speedup
    speedup = original_time / compiled_time
    
    print(f'\nResNetTCN Performance Results:')
    print(f'  Original model avg time: {original_time*1000:.2f} ms')
    print(f'  Compiled model avg time: {compiled_time*1000:.2f} ms')
    print(f'  Speedup: {speedup:.2f}x')
    print(f'  Compilation overhead: {actual_compile_time:.2f} seconds')
    print('=' * 80)

    #%% TCN model

    # Example model parameters for the TCN backbone
    tcn_first_layer_temporal_kernel_size = 25
    num_layers_per_block_list = [2, 2, 3]  # Note: layers instead of miniblocks
    tcn_num_features_per_block_list = [16, 32, 64]
    tcn_temporal_kernel_size_per_block_list = [3, 5, 7]
    tcn_temporal_dilation_per_block_list = [1, 1, 1]
    
    print('-------------------------------------------------------------------------------------')
    print('Creating TCN model:')
    print('-------------------')
    twin_model_TCN = SingleNeuronTwinModel_TCN(
        in_channels=in_channels,
        in_spatial_dim=in_spatial_dim,
        first_layer_temporal_kernel_size=tcn_first_layer_temporal_kernel_size,
        num_layers_per_block_list=num_layers_per_block_list,
        num_features_per_block_list=tcn_num_features_per_block_list,
        temporal_kernel_size_per_block_list=tcn_temporal_kernel_size_per_block_list,
        temporal_dilation_per_block_list=tcn_temporal_dilation_per_block_list,
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
    print('-------------------------------------------------------------------------------------')
    print(f'TCN Model created: "{twin_model_TCN.short_name}"')
    print(f'TCN Number of parameters: {twin_model_TCN.num_parmas_millions:.3f}M')
    print('-------------------------------------------------------------------------------------')
    
    # Test forward pass with sample input
    print('-----------------------------------------------')
    print('Testing TCN forward pass...')
    
    # Forward pass
    twin_model_TCN.eval()
    with torch.no_grad():
        tcn_outputs = twin_model_TCN(X_sample)
        tcn_y_spikes, tcn_y_soma, tcn_y_near_spike, tcn_y_inst_rate, tcn_y_dend_v = tcn_outputs
    
    print('TCN Output shapes:')
    print(f'  tcn_y_spikes: {tcn_y_spikes.shape}')
    print(f'  tcn_y_soma: {tcn_y_soma.shape}')
    print(f'  tcn_y_near_spike: {tcn_y_near_spike.shape}')
    print(f'  tcn_y_inst_rate: {tcn_y_inst_rate.shape}')
    print(f'  tcn_y_dend_v: {tcn_y_dend_v.shape}')
    
    print('TCN Output value ranges:')
    print(f'  tcn_y_spikes: [{tcn_y_spikes.min():.3f}, {tcn_y_spikes.max():.3f}]')
    print(f'  tcn_y_soma: [{tcn_y_soma.min():.3f}, {tcn_y_soma.max():.3f}]')
    print(f'  tcn_y_near_spike: [{tcn_y_near_spike.min():.3f}, {tcn_y_near_spike.max():.3f}]')
    print(f'  tcn_y_inst_rate: [{tcn_y_inst_rate.min():.3f}, {tcn_y_inst_rate.max():.3f}]')
    print(f'  tcn_y_dend_v: [{tcn_y_dend_v.min():.3f}, {tcn_y_dend_v.max():.3f}]')
    
    print('-----------------------------------------------')
    
    #%% Test torch.compile() performance for TCN model
    
    print('=' * 80)
    print('Testing torch.compile() performance for TCN model')
    print('=' * 80)
    
    # Warmup runs for fair comparison
    print('Performing warmup runs...')
    for _ in range(5):
        with torch.no_grad():
            _ = twin_model_TCN(X_sample)
    
    # Benchmark original model
    print('Benchmarking original TCN model...')
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = twin_model_TCN(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    tcn_original_time = (time.time() - start_time) / num_runs
    
    # Compile model and measure compilation time
    print('Compiling TCN model with torch.compile()...')
    tcn_compile_start = time.time()
    compiled_tcn_model = torch.compile(twin_model_TCN)
    tcn_compile_time = time.time() - tcn_compile_start
    print(f'Initial compilation setup time: {tcn_compile_time:.4f} seconds')
    
    # First run triggers actual compilation
    print('Running first inference (triggers actual compilation)...')
    tcn_actual_compile_start = time.time()
    with torch.no_grad():
        _ = compiled_tcn_model(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    tcn_actual_compile_time = time.time() - tcn_actual_compile_start
    print(f'Actual compilation time (first run): {tcn_actual_compile_time:.4f} seconds')
    
    # Warmup compiled model
    print('Warming up compiled model...')
    for _ in range(5):
        with torch.no_grad():
            _ = compiled_tcn_model(X_sample)
    
    # Benchmark compiled model
    print('Benchmarking compiled TCN model...')
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = compiled_tcn_model(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    tcn_compiled_time = (time.time() - start_time) / num_runs
    
    # Calculate speedup
    tcn_speedup = tcn_original_time / tcn_compiled_time
    
    print(f'\nTCN Performance Results:')
    print(f'  Original model avg time: {tcn_original_time*1000:.2f} ms')
    print(f'  Compiled model avg time: {tcn_compiled_time*1000:.2f} ms')
    print(f'  Speedup: {tcn_speedup:.2f}x')
    print(f'  Compilation overhead: {tcn_actual_compile_time:.2f} seconds')
    print('=' * 80)

    #%% ELM model

    # Example model parameters for the ELM backbone
    elm_memory_dim = 64
    elm_lambda_value = 5.0
    elm_mlp_pre_norm_type = 'BatchNorm'
    elm_post_mlp_nonlinearity_str = 'ltanh'
    elm_post_mlp_leaky_slope = 0.1
    elm_mlp_num_hidden_layers = 1
    elm_mlp_hidden_dim = None
    elm_mlp_nonlinearity_str = 'lsilu'
    elm_mlp_leaky_slope = 0.2
    elm_synapse_tau_value = 5.0
    elm_memory_tau_min = 1.0
    elm_memory_tau_max = 128.0
    elm_learn_memory_tau = True
    elm_w_s_value = 0.5
    elm_delta_t = 1.0
    
    print('-------------------------------------------------------------------------------------')
    print('Creating ELM model:')
    print('-------------------')
    twin_model_ELM = SingleNeuronTwinModel_ELM(
        in_channels=in_channels,
        in_spatial_dim=in_spatial_dim,
        memory_dim=elm_memory_dim,
        mlp_num_hidden_layers=elm_mlp_num_hidden_layers,
        mlp_hidden_dim=elm_mlp_hidden_dim,
        mlp_nonlinearity_str=elm_mlp_nonlinearity_str,
        mlp_leaky_slope=elm_mlp_leaky_slope,
        mlp_pre_norm_type=elm_mlp_pre_norm_type,
        post_mlp_nonlinearity_str=elm_post_mlp_nonlinearity_str,
        post_mlp_leaky_slope=elm_post_mlp_leaky_slope,
        lambda_value=elm_lambda_value,
        synapse_tau_value=elm_synapse_tau_value,
        memory_tau_min=elm_memory_tau_min,
        memory_tau_max=elm_memory_tau_max,
        learn_memory_tau=elm_learn_memory_tau,
        w_s_value=elm_w_s_value,
        delta_t=elm_delta_t,
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
    print('-------------------------------------------------------------------------------------')
    print(f'ELM Model created: "{twin_model_ELM.short_name}"')
    print(f'ELM Number of parameters: {twin_model_ELM.num_parmas_millions:.3f}M')
    print('-------------------------------------------------------------------------------------')
    
    # Test forward pass with sample input
    print('-----------------------------------------------')
    print('Testing ELM forward pass...')
    
    # Forward pass
    twin_model_ELM.eval()
    with torch.no_grad():
        elm_outputs = twin_model_ELM(X_sample)
        elm_y_spikes, elm_y_soma, elm_y_near_spike, elm_y_inst_rate, elm_y_dend_v = elm_outputs
    
    print('ELM Output shapes:')
    print(f'  elm_y_spikes: {elm_y_spikes.shape}')
    print(f'  elm_y_soma: {elm_y_soma.shape}')
    print(f'  elm_y_near_spike: {elm_y_near_spike.shape}')
    print(f'  elm_y_inst_rate: {elm_y_inst_rate.shape}')
    print(f'  elm_y_dend_v: {elm_y_dend_v.shape}')
    
    print('ELM Output value ranges:')
    print(f'  elm_y_spikes: [{elm_y_spikes.min():.3f}, {elm_y_spikes.max():.3f}]')
    print(f'  elm_y_soma: [{elm_y_soma.min():.3f}, {elm_y_soma.max():.3f}]')
    print(f'  elm_y_near_spike: [{elm_y_near_spike.min():.3f}, {elm_y_near_spike.max():.3f}]')
    print(f'  elm_y_inst_rate: [{elm_y_inst_rate.min():.3f}, {elm_y_inst_rate.max():.3f}]')
    print(f'  elm_y_dend_v: [{elm_y_dend_v.min():.3f}, {elm_y_dend_v.max():.3f}]')
    
    print('-----------------------------------------------')
    
    #%% Test torch.compile() performance for ELM model
    
    # test_compile_for_ELM = True
    test_compile_for_ELM = False
    
    if test_compile_for_ELM:

        print('=' * 80)
        print('Testing torch.compile() performance for ELM model')
        print('=' * 80)
        
        # Warmup runs for fair comparison
        print('Performing warmup runs...')
        for _ in range(5):
            with torch.no_grad():
                _ = twin_model_ELM(X_sample)
        
        # Benchmark original model
        print('Benchmarking original ELM model...')
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = twin_model_ELM(X_sample)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elm_original_time = (time.time() - start_time) / num_runs
        
        # Compile model and measure compilation time
        print('Compiling ELM model with torch.compile()...')
        elm_compile_start = time.time()
        compiled_elm_model = torch.compile(twin_model_ELM)
        # compiled_elm_model = torch.compile(twin_model_ELM,
        #     mode='reduce-overhead',  # Better for dynamic shapes
        #     dynamic=True,  # Handle variable sequence lengths
        #     fullgraph=True  # Try to compile everything together
        # )

        elm_compile_time = time.time() - elm_compile_start
        print(f'Initial compilation setup time: {elm_compile_time:.4f} seconds')
        
        # First run triggers actual compilation
        print('Running first inference (triggers actual compilation)...')
        elm_actual_compile_start = time.time()
        with torch.no_grad():
            # _ = compiled_elm_model(X_sample)
            _ = compiled_elm_model(X_sample)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elm_actual_compile_time = time.time() - elm_actual_compile_start
        print(f'Actual compilation time (first run): {elm_actual_compile_time:.4f} seconds')
        
        # Warmup compiled model
        print('Warming up compiled model...')
        for _ in range(5):
            with torch.no_grad():
                _ = compiled_elm_model(X_sample)
        
        # Benchmark compiled model
        print('Benchmarking compiled ELM model...')
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = compiled_elm_model(X_sample)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elm_compiled_time = (time.time() - start_time) / num_runs
        
        # Calculate speedup
        elm_speedup = elm_original_time / elm_compiled_time
        
        print(f'\nELM Performance Results:')
        print(f'  Original model avg time: {elm_original_time*1000:.2f} ms')
        print(f'  Compiled model avg time: {elm_compiled_time*1000:.2f} ms')
        print(f'  Speedup: {elm_speedup:.2f}x')
        print(f'  Compilation overhead: {elm_actual_compile_time:.2f} seconds')
        print('=' * 80)
    
    #%% Test 6: Transformer twin model
    
    print('-------------------------------------------------------------------------------------')
    print('Creating Transformer model:')
    print('---------------------------')
    twin_model_Transformer = SingleNeuronTwinModel_Transformer(
        in_channels=in_channels,
        in_spatial_dim=in_spatial_dim,
        first_layer_temporal_kernel_size=7,
        first_norm_type='RMSNorm',
        first_nonlinearity_str='lgelu',
        first_leaky_slope=0.2,
        d_model=64,
        n_heads=4,
        num_layers=2,
        window_size=32,
        max_seq_len=4096,
        dropout_rate=0.1,
        ffn_type='swiglu',
        ffn_hidden_dim=None,
        SWA_type='blocked',
        SWA_block_size=None,
        use_complex_number_rope=False,
        norm_type='rms_norm',
        head_prefix_names=head_prefix_names,
        head_out_channels=head_out_channels,
        head_convert_out_ch_to_sp=head_convert_out_ch_to_sp,
        head_spatial_kernel_sizes=None,
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
    print('-------------------------------------------------------------------------------------')
    print(f'Transformer Model created: "{twin_model_Transformer.short_name}"')
    print(f'Transformer Number of parameters: {twin_model_Transformer.num_parmas_millions:.3f}M')
    print('-------------------------------------------------------------------------------------')
    
    # Test forward pass with sample input
    print('-----------------------------------------------')
    print('Testing Transformer forward pass...')
    
    # Forward pass
    twin_model_Transformer.eval()
    with torch.no_grad():
        transformer_outputs = twin_model_Transformer(X_sample)
        transformer_y_spikes, transformer_y_soma, transformer_y_near_spike, transformer_y_inst_rate, transformer_y_dend_v = transformer_outputs
    
    print('Transformer Output shapes:')
    print(f'  transformer_y_spikes: {transformer_y_spikes.shape}')
    print(f'  transformer_y_soma: {transformer_y_soma.shape}')
    print(f'  transformer_y_near_spike: {transformer_y_near_spike.shape}')
    print(f'  transformer_y_inst_rate: {transformer_y_inst_rate.shape}')
    print(f'  transformer_y_dend_v: {transformer_y_dend_v.shape}')
    
    print('Transformer Output value ranges:')
    print(f'  transformer_y_spikes: [{transformer_y_spikes.min():.3f}, {transformer_y_spikes.max():.3f}]')
    print(f'  transformer_y_soma: [{transformer_y_soma.min():.3f}, {transformer_y_soma.max():.3f}]')
    print(f'  transformer_y_near_spike: [{transformer_y_near_spike.min():.3f}, {transformer_y_near_spike.max():.3f}]')
    print(f'  transformer_y_inst_rate: [{transformer_y_inst_rate.min():.3f}, {transformer_y_inst_rate.max():.3f}]')
    print(f'  transformer_y_dend_v: [{transformer_y_dend_v.min():.3f}, {transformer_y_dend_v.max():.3f}]')
    
    print('-----------------------------------------------')
    
    #%% Test torch.compile() performance for Transformer model
    
    print('=' * 80)
    print('Testing torch.compile() performance for Transformer model')
    print('=' * 80)
    
    # Warmup runs for fair comparison
    print('Performing warmup runs...')
    for _ in range(5):
        with torch.no_grad():
            _ = twin_model_Transformer(X_sample)
    
    # Benchmark original model
    print('Benchmarking original Transformer model...')
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = twin_model_Transformer(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    transformer_original_time = (time.time() - start_time) / num_runs
    
    # Compile model and measure compilation time
    print('Compiling Transformer model with torch.compile()...')
    transformer_compile_start = time.time()
    compiled_transformer_model = torch.compile(twin_model_Transformer)
    transformer_compile_time = time.time() - transformer_compile_start
    print(f'Initial compilation setup time: {transformer_compile_time:.4f} seconds')
    
    # First run triggers actual compilation
    print('Running first inference (triggers actual compilation)...')
    transformer_actual_compile_start = time.time()
    with torch.no_grad():
        _ = compiled_transformer_model(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    transformer_actual_compile_time = time.time() - transformer_actual_compile_start
    print(f'Actual compilation time (first run): {transformer_actual_compile_time:.4f} seconds')
    
    # Warmup compiled model
    print('Warming up compiled model...')
    for _ in range(5):
        with torch.no_grad():
            _ = compiled_transformer_model(X_sample)
    
    # Benchmark compiled model
    print('Benchmarking compiled Transformer model...')
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = compiled_transformer_model(X_sample)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    transformer_compiled_time = (time.time() - start_time) / num_runs
    
    # Calculate speedup
    transformer_speedup = transformer_original_time / transformer_compiled_time
    
    print(f'\nTransformer Performance Results:')
    print(f'  Original model avg time: {transformer_original_time*1000:.2f} ms')
    print(f'  Compiled model avg time: {transformer_compiled_time*1000:.2f} ms')
    print(f'  Speedup: {transformer_speedup:.2f}x')
    print(f'  Compilation overhead: {transformer_actual_compile_time:.2f} seconds')
    print('=' * 80)
    
    #%% Summary of all torch.compile() results
    
    print('\n' + '=' * 80)
    print('TORCH.COMPILE() PERFORMANCE SUMMARY')
    print('=' * 80)
    print(f'ResNetTCN:')
    print(f'  Original: {original_time*1000:.2f} ms  |  Compiled: {compiled_time*1000:.2f} ms  |  Speedup: {speedup:.2f}x  |  Compile time: {actual_compile_time:.2f}s')
    print(f'TCN:')
    print(f'  Original: {tcn_original_time*1000:.2f} ms  |  Compiled: {tcn_compiled_time*1000:.2f} ms  |  Speedup: {tcn_speedup:.2f}x  |  Compile time: {tcn_actual_compile_time:.2f}s')
    print(f'ELM:')
    if test_compile_for_ELM:
        print(f'  Original: {elm_original_time*1000:.2f} ms  |  Compiled: {elm_compiled_time*1000:.2f} ms  |  Speedup: {elm_speedup:.2f}x  |  Compile time: {elm_actual_compile_time:.2f}s')
    else:
        print(f'  Did not test compile for ELM (because it was too slow)')
    print(f'Transformer:')
    print(f'  Original: {transformer_original_time*1000:.2f} ms  |  Compiled: {transformer_compiled_time*1000:.2f} ms  |  Speedup: {transformer_speedup:.2f}x  |  Compile time: {transformer_actual_compile_time:.2f}s')
    print('=' * 80)
    print('\nNotes:')
    print('- All four model types (ResNetTCN, TCN, ELM, Transformer) now support torch.compile()')
    print('- ELM models benefit from setting float32_matmul_precision to "high" for better compilation')
    print('- Transformer models use sliding window attention with RoPE positional embeddings')
    print('=' * 80)
