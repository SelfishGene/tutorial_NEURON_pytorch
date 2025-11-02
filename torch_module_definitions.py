#%% Imports

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

#%% various helper functions and main classes

# leaky gelu activation function (just the average of gelu and leaky relu)
class LeakyGELU(nn.Module):
    def __init__(self, negative_slope=0.2, x_shift=0.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.x_shift = x_shift
        self.gelu = nn.GELU()
        self.lrelu = nn.LeakyReLU(negative_slope=(2 * self.negative_slope))

    def forward(self, x):
        return 0.5 * self.gelu(x - self.x_shift) + 0.5 * self.lrelu(x - self.x_shift)

    def small_test(self):
        x = torch.linspace(-4.5, 4.5, 1000, requires_grad=True)

        # List of negative slopes to plot
        negative_slopes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        for neg_slope, color in zip(negative_slopes, colors):
            # Create activation function with current negative slope
            activation_function = LeakyGELU(negative_slope=neg_slope)
            
            # Forward pass
            y = activation_function(x)
            ax1.plot(x.detach(), y.detach(), label=f'Negative Slope = {neg_slope}', color=color)
            
            # Compute derivative using PyTorch autograd
            dy_dx = torch.autograd.grad(outputs=y.sum(), inputs=x, retain_graph=True)[0]
            ax2.plot(x.detach(), dy_dx.detach(), label=f'Derivative (Negative Slope = {neg_slope})', color=color)

        # Set labels and titles
        ax1.set_title("'LeakyGELU' Activation Function for Different Negative Slopes")
        ax1.set_xlabel("Input (x)")
        ax1.set_ylabel("Output")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Derivative of 'LeakyGELU' Activation Function for Different Negative Slopes")
        ax2.set_xlabel("Input (x)")
        ax2.set_ylabel("Derivative")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# leaky silu activation function (just the average of silu and leaky relu)
class LeakySiLU(nn.Module):
    def __init__(self, negative_slope=0.2, x_shift=0.0):
        super().__init__()
        self.negative_slope = negative_slope
        self.x_shift = x_shift
        self.silu = nn.SiLU()
        self.lrelu = nn.LeakyReLU(negative_slope=(2 * self.negative_slope))

    def forward(self, x):
        return 0.5 * self.silu(x - self.x_shift) + 0.5 * self.lrelu(x - self.x_shift)

    def small_test(self):
        x = torch.linspace(-4.5, 4.5, 1000, requires_grad=True)

        # List of negative slopes to plot
        negative_slopes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        for neg_slope, color in zip(negative_slopes, colors):
            # Create activation function with current negative slope
            activation_function = LeakySiLU(negative_slope=neg_slope)
            
            # Forward pass
            y = activation_function(x)
            ax1.plot(x.detach(), y.detach(), label=f'Negative Slope = {neg_slope}', color=color)
            
            # Compute derivative using PyTorch autograd
            dy_dx = torch.autograd.grad(outputs=y.sum(), inputs=x, retain_graph=True)[0]
            ax2.plot(x.detach(), dy_dx.detach(), label=f'Derivative (Negative Slope = {neg_slope})', color=color)

        # Set labels and titles
        ax1.set_title("'LeakySiLU' Activation Function for Different Negative Slopes")
        ax1.set_xlabel("Input (x)")
        ax1.set_ylabel("Output")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Derivative of 'LeakySiLU' Activation Function for Different Negative Slopes")
        ax2.set_xlabel("Input (x)")
        ax2.set_ylabel("Derivative")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# leaky tanh activation function (edges_slope * x + (1-edges_slope) * tanh(x))
class LeakyTanh(nn.Module):
    def __init__(self, edges_slope=0.2):
        super().__init__()
        self.edges_slope = edges_slope
        self.tanh = nn.Tanh()

    def forward(self, x):
        return self.edges_slope * x + (1 - self.edges_slope) * self.tanh(x)

    def small_test(self):
        x = torch.linspace(-4.5, 4.5, 1000, requires_grad=True)

        # List of edge slopes to plot
        edges_slopes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        for edge_slope, color in zip(edges_slopes, colors):
            # Create activation function with current edge slope
            activation_function = LeakyTanh(edges_slope=edge_slope)
            
            # Forward pass
            y = activation_function(x)
            ax1.plot(x.detach(), y.detach(), label=f'Edges Slope = {edge_slope}', color=color)
            
            # Compute derivative using PyTorch autograd
            dy_dx = torch.autograd.grad(outputs=y.sum(), inputs=x, retain_graph=True)[0]
            ax2.plot(x.detach(), dy_dx.detach(), label=f'Derivative (Edges Slope = {edge_slope})', color=color)

        # Set labels and titles
        ax1.set_title("'LeakyTanh' Activation Function for Different Edge Slopes")
        ax1.set_xlabel("Input (x)")
        ax1.set_ylabel("Output")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Derivative of 'LeakyTanh' Activation Function for Different Edge Slopes")
        ax2.set_xlabel("Input (x)")
        ax2.set_ylabel("Derivative")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# leaky sigmoid activation function (edges_slope * x + (1-edges_slope) * (2*sigmoid(x) - 1))
class LeakySigmoid(nn.Module):
    def __init__(self, edges_slope=0.2):
        super().__init__()
        self.edges_slope = edges_slope
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 2*sigmoid(x) - 1 gives [-1, 1] range like tanh
        sigmoid_scaled = 2 * self.sigmoid(x) - 1
        return self.edges_slope * x + (1 - self.edges_slope) * sigmoid_scaled

    def small_test(self):
        x = torch.linspace(-4.5, 4.5, 1000, requires_grad=True)

        # List of edge slopes to plot
        edges_slopes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        for edge_slope, color in zip(edges_slopes, colors):
            # Create activation function with current edge slope
            activation_function = LeakySigmoid(edges_slope=edge_slope)
            
            # Forward pass
            y = activation_function(x)
            ax1.plot(x.detach(), y.detach(), label=f'Edges Slope = {edge_slope}', color=color)
            
            # Compute derivative using PyTorch autograd
            dy_dx = torch.autograd.grad(outputs=y.sum(), inputs=x, retain_graph=True)[0]
            ax2.plot(x.detach(), dy_dx.detach(), label=f'Derivative (Edges Slope = {edge_slope})', color=color)

        # Set labels and titles
        ax1.set_title("'LeakySigmoid' Activation Function for Different Edge Slopes")
        ax1.set_xlabel("Input (x)")
        ax1.set_ylabel("Output")
        ax1.legend()
        ax1.grid(True)

        ax2.set_title("Derivative of 'LeakySigmoid' Activation Function for Different Edge Slopes")
        ax2.set_xlabel("Input (x)")
        ax2.set_ylabel("Derivative")
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

# linear at the center, reduced slope at the edges, gelu smooth transition between the three parts
class TwoSidedLeakyGELU(nn.Module):
    def __init__(self, edges_slope=0.2, x_shift=1.0):
        super().__init__()

        self.x_shift = x_shift
        self.edges_slope = edges_slope
        self.curr_nonlin = LeakyGELU(negative_slope=edges_slope)
        self.y_shift = float(1 - self.curr_nonlin(torch.tensor([0.0])))
        self.x_intersection = float(self.find_intersection())

    def forward(self, x):
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        y_avg = (y_left + y_right) / 2
        return torch.where(x < self.x_intersection, y_left, torch.where(x < -self.x_intersection, y_avg, y_right))

    def find_intersection(self):
        x = torch.linspace(-4.5, 4.5, 1000)
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        diff = torch.abs(y_left - y_right)
        return x[torch.where(diff == diff.min())[0][0]]

    def small_test(self):
        name_string = 'TwoSidedLeakyGELU'

        # left column: component breakdown for default slope
        x = torch.linspace(-4.5, 4.5, 1000)
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        y_avg = (y_left + y_right) / 2
        y = torch.where(x < self.x_intersection, y_left, torch.where(x < -self.x_intersection, y_avg, y_right))

        # Second plot: different slopes with activation and derivative
        x_grad = torch.linspace(-4.5, 4.5, 1000, requires_grad=True)
        edges_slopes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Component breakdown (top left)
        axes[0, 0].plot(x, y_left, label='left')
        axes[0, 0].plot(x, y_right, label='right')
        axes[0, 0].plot(x, y_avg, label='avg')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Component Breakdown')
        
        # Final activation (bottom left)
        axes[1, 0].plot(x, y, label='final activation function')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_title('Final Activation Function')
        
        # right column: activation functions for different slopes and derivatives
        # Activation functions for different slopes (top right)
        for slope, color in zip(edges_slopes, colors):
            activation_function = TwoSidedLeakyGELU(edges_slope=slope)
            y_slope = activation_function(x_grad)
            axes[0, 1].plot(x_grad.detach(), y_slope.detach(), label=f'Edges Slope = {slope}', color=color)
        
        axes[0, 1].set_title(f"'{name_string}' for Different Edge Slopes")
        axes[0, 1].set_xlabel("Input (x)")
        axes[0, 1].set_ylabel("Output")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Derivatives for different slopes (bottom right)
        for slope, color in zip(edges_slopes, colors):
            activation_function = TwoSidedLeakyGELU(edges_slope=slope)
            y_slope = activation_function(x_grad)
            dy_dx = torch.autograd.grad(outputs=y_slope.sum(), inputs=x_grad, retain_graph=True)[0]
            axes[1, 1].plot(x_grad.detach(), dy_dx.detach(), label=f'Derivative (Edges Slope = {slope})', color=color)
        
        axes[1, 1].set_title(f"Derivative of '{name_string}' for Different Edge Slopes")
        axes[1, 1].set_xlabel("Input (x)")
        axes[1, 1].set_ylabel("Derivative")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

# linear at the center, reduced slope at the edges, silu smooth transition between the three parts
class TwoSidedLeakySiLU(nn.Module):
    def __init__(self, edges_slope=0.2, x_shift=1.0):
        super().__init__()

        self.x_shift = x_shift
        self.edges_slope = edges_slope
        self.curr_nonlin = LeakySiLU(negative_slope=edges_slope)
        self.y_shift = float(1 - self.curr_nonlin(torch.tensor([0.0])))
        self.x_intersection = float(self.find_intersection())

    def forward(self, x):
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        y_avg = (y_left + y_right) / 2
        return torch.where(x < self.x_intersection, y_left, torch.where(x < -self.x_intersection, y_avg, y_right))

    def find_intersection(self):
        x = torch.linspace(-4.5, 4.5, 1000)
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        diff = torch.abs(y_left - y_right)
        return x[torch.where(diff == diff.min())[0][0]]

    def small_test(self):
        name_string = 'TwoSidedLeakySiLU'

        # left column: component breakdown for default slope
        x = torch.linspace(-4.5, 4.5, 1000)
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        y_avg = (y_left + y_right) / 2
        y = torch.where(x < self.x_intersection, y_left, torch.where(x < -self.x_intersection, y_avg, y_right))

        # Second plot: different slopes with activation and derivative
        x_grad = torch.linspace(-4.5, 4.5, 1000, requires_grad=True)
        edges_slopes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Component breakdown (top left)
        axes[0, 0].plot(x, y_left, label='left')
        axes[0, 0].plot(x, y_right, label='right')
        axes[0, 0].plot(x, y_avg, label='avg')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Component Breakdown')
        
        # Final activation (bottom left)
        axes[1, 0].plot(x, y, label='final activation function')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_title('Final Activation Function')
        
        # right column: activation functions for different slopes and derivatives
        # Activation functions for different slopes (top right)
        for slope, color in zip(edges_slopes, colors):
            activation_function = TwoSidedLeakySiLU(edges_slope=slope)
            y_slope = activation_function(x_grad)
            axes[0, 1].plot(x_grad.detach(), y_slope.detach(), label=f'Edges Slope = {slope}', color=color)
        
        axes[0, 1].set_title(f"'{name_string}' for Different Edge Slopes")
        axes[0, 1].set_xlabel("Input (x)")
        axes[0, 1].set_ylabel("Output")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Derivatives for different slopes (bottom right)
        for slope, color in zip(edges_slopes, colors):
            activation_function = TwoSidedLeakySiLU(edges_slope=slope)
            y_slope = activation_function(x_grad)
            dy_dx = torch.autograd.grad(outputs=y_slope.sum(), inputs=x_grad, retain_graph=True)[0]
            axes[1, 1].plot(x_grad.detach(), dy_dx.detach(), label=f'Derivative (Edges Slope = {slope})', color=color)
        
        axes[1, 1].set_title(f"Derivative of '{name_string}' for Different Edge Slopes")
        axes[1, 1].set_xlabel("Input (x)")
        axes[1, 1].set_ylabel("Derivative")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

# linear at the center, reduced slope at the edges, leaky relu smooth transition between the three parts
class TwoSidedLeakyReLU(nn.Module):
    def __init__(self, edges_slope=0.2, x_shift=1.0):
        super().__init__()

        self.x_shift = x_shift
        self.edges_slope = edges_slope
        self.curr_nonlin = nn.LeakyReLU(negative_slope=edges_slope)
        self.y_shift = float(1 - self.curr_nonlin(torch.tensor([0.0])))
        self.x_intersection = float(self.find_intersection())

    def forward(self, x):
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        y_avg = (y_left + y_right) / 2
        return torch.where(x < self.x_intersection, y_left, torch.where(x < -self.x_intersection, y_avg, y_right))

    def find_intersection(self):
        x = torch.linspace(-4.5, 4.5, 1000)
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        diff = torch.abs(y_left - y_right)
        return x[torch.where(diff == diff.min())[0][0]]

    def small_test(self):
        name_string = 'TwoSidedLeakyReLU'

        # left column: component breakdown for default slope
        x = torch.linspace(-4.5, 4.5, 1000)
        y_left = self.curr_nonlin(x + self.x_shift) - self.y_shift
        y_right = -self.curr_nonlin(-(x - self.x_shift)) + self.y_shift
        y_avg = (y_left + y_right) / 2
        y = torch.where(x < self.x_intersection, y_left, torch.where(x < -self.x_intersection, y_avg, y_right))

        # Second plot: different slopes with activation and derivative
        x_grad = torch.linspace(-4.5, 4.5, 1000, requires_grad=True)
        edges_slopes = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Component breakdown (top left)
        axes[0, 0].plot(x, y_left, label='left')
        axes[0, 0].plot(x, y_right, label='right')
        axes[0, 0].plot(x, y_avg, label='avg')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Component Breakdown')
        
        # Final activation (bottom left)
        axes[1, 0].plot(x, y, label='final activation function')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_title('Final Activation Function')
        
        # right column: activation functions for different slopes and derivatives
        # Activation functions for different slopes (top right)
        for slope, color in zip(edges_slopes, colors):
            activation_function = TwoSidedLeakyReLU(edges_slope=slope)
            y_slope = activation_function(x_grad)
            axes[0, 1].plot(x_grad.detach(), y_slope.detach(), label=f'Edges Slope = {slope}', color=color)
        
        axes[0, 1].set_title(f"'{name_string}' for Different Edge Slopes")
        axes[0, 1].set_xlabel("Input (x)")
        axes[0, 1].set_ylabel("Output")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Derivatives for different slopes (bottom right)
        for slope, color in zip(edges_slopes, colors):
            activation_function = TwoSidedLeakyReLU(edges_slope=slope)
            y_slope = activation_function(x_grad)
            dy_dx = torch.autograd.grad(outputs=y_slope.sum(), inputs=x_grad, retain_graph=True)[0]
            axes[1, 1].plot(x_grad.detach(), dy_dx.detach(), label=f'Derivative (Edges Slope = {slope})', color=color)
        
        axes[1, 1].set_title(f"Derivative of '{name_string}' for Different Edge Slopes")
        axes[1, 1].set_xlabel("Input (x)")
        axes[1, 1].set_ylabel("Derivative")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

# non linearity layer that aceepts a string as an argument and returns the appropriate nonlinearity
class Nonlinearity(nn.Module):
    def __init__(self, nonlinearity_str='lgelu', leaky_negative_slope=0.2):
        super().__init__()
        self.nonlinearity_str = nonlinearity_str.lower()
        self.leaky_negative_slope = leaky_negative_slope

        if self.nonlinearity_str in ['none', 'linear', 'identity', 'lin']:
            self.nonlinearity_layer = nn.Identity()
        elif self.nonlinearity_str == 'relu':
            self.nonlinearity_layer = nn.ReLU()
        elif self.nonlinearity_str in ['lrelu', 'leaky relu', 'leaky_relu']:
            self.nonlinearity_layer = nn.LeakyReLU(negative_slope=self.leaky_negative_slope)
        elif self.nonlinearity_str in ['lrelu_2s', '2s_lrelu', 'two_sided_lrelu', 
                                       'two_sided_leaky_relu', 'two sided lrelu', 'two sided leaky relu']:
            self.nonlinearity_layer = TwoSidedLeakyReLU(edges_slope=self.leaky_negative_slope)
        elif self.nonlinearity_str == 'gelu':
            self.nonlinearity_layer = nn.GELU()
        elif self.nonlinearity_str in ['lgelu', 'leaky gelu', 'leaky_gelu']:
            self.nonlinearity_layer = LeakyGELU(negative_slope=self.leaky_negative_slope)
        elif self.nonlinearity_str in ['lgelu_2s', '2s_lgelu', 'two_sided_lgelu', 
                                       'two_sided_leaky_gelu', 'two sided lgelu', 'two sided leaky gelu']:
            self.nonlinearity_layer = TwoSidedLeakyGELU(edges_slope=self.leaky_negative_slope)
        elif self.nonlinearity_str in ['silu', 'swish']:
            self.nonlinearity_layer = nn.SiLU()
        elif self.nonlinearity_str in ['lsilu', 'leaky silu', 'leaky_silu']:
            self.nonlinearity_layer = LeakySiLU(negative_slope=self.leaky_negative_slope)
        elif self.nonlinearity_str in ['lsilu_2s', '2s_lsilu', 'two_sided_lsilu', 
                                       'two_sided_leaky_silu', 'two sided lsilu', 'two sided leaky silu']:
            self.nonlinearity_layer = TwoSidedLeakySiLU(edges_slope=self.leaky_negative_slope)
        elif self.nonlinearity_str == 'tanh':
            self.nonlinearity_layer = nn.Tanh()
        elif self.nonlinearity_str == 'sigmoid':
            self.nonlinearity_layer = nn.Sigmoid()
        elif self.nonlinearity_str in ['ltanh', 'leaky tanh', 'leaky_tanh']:
            self.nonlinearity_layer = LeakyTanh(edges_slope=self.leaky_negative_slope)
        elif self.nonlinearity_str in ['lsigmoid', 'leaky sigmoid', 'leaky_sigmoid']:
            self.nonlinearity_layer = LeakySigmoid(edges_slope=self.leaky_negative_slope)
        else:
            raise ValueError(f'nonlinearity = {self.nonlinearity_str} is not supported')

    def forward(self, x):
        return self.nonlinearity_layer(x)

# pooling layer that accepts a string as an argument and returns the appropriate pooling layer
class Pooling(nn.Module):
    def __init__(self, pooling_type='max pool', stride=2):
        super().__init__()
        self.pooling_type = pooling_type
        self.stride = stride

        if self.pooling_type == 'max pool':
            self.pooling_layer = nn.MaxPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.pooling_type == 'avg pool':
            self.pooling_layer = nn.AvgPool2d(kernel_size=self.stride, stride=self.stride)
        elif self.pooling_type == 'subsample':
            self.pooling_layer = nn.AvgPool2d(kernel_size=1, stride=self.stride)
        else:
            raise ValueError(f'pooling type = {self.pooling_type} is not supported')

    def forward(self, x):
        return self.pooling_layer(x)

# normalization module for (B, C, S, T) type tensors where T is time dimension
class Normalization(nn.Module):
    """
    Normalizes a 4D tensor of shape (B-batch, C-channels, S-spatial, T-time).
    - norm_type='BatchNorm' -> nn.BatchNorm2d over (B, S, T), per-channel (C)
    - norm_type='BatchNorm_B' -> BatchNorm-like over (B) only, per-location stats; affine per-channel
    - norm_type='LayerNorm' -> per-sample, per-(S,T) LayerNorm over the channel axis only
    - norm_type='RMSnorm'   -> per-sample, per-(S,T) RMSNorm over the channel axis only (no bias)
    """

    def __init__(self, num_channels, norm_type='BatchNorm', eps=1e-5, momentum=0.1):
        super().__init__()
        norm_type_lower = norm_type.lower()
        self.norm_type = norm_type_lower
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum

        if norm_type_lower in ['batchnorm', 'batch_norm']:
            self.norm_layer = nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
            self.weight = None
            self.bias = None
        elif norm_type_lower in ['batchnorm_b', 'batch_norm_b', 'batchnormb', 'batchnorm_bonly', 'batchnorm_b_only', 'batchnorm_only_b']:
            # BatchNorm variant that normalizes across batch only (per-location),
            # with per-channel affine parameters. Running stats are tracked per-channel
            # by aggregating the per-location batch stats over (S,T) to keep memory small.
            self.norm_layer = None
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias   = nn.Parameter(torch.zeros(num_channels))
            # Channel-wise running stats (aggregated over spatial/time to avoid huge memory use)
            self.register_buffer('running_mean', torch.zeros(num_channels))
            self.register_buffer('running_var',  torch.ones(num_channels))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        elif norm_type_lower in ['layernorm', 'layer_norm']:
            # affine params over channels only
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias   = nn.Parameter(torch.zeros(num_channels))
            self.norm_layer = None
        elif norm_type_lower in ['rmsnorm', 'rms_norm']:
            # scale only (no bias), over channels
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias   = None
            self.norm_layer = None
        else:
            raise ValueError(f"Unknown norm_type '{norm_type}'. Use 'BatchNorm' | 'LayerNorm' | 'RMSnorm'.")

    def forward(self, x):
        # x: (B, C, S, T)
        if self.norm_type == 'batchnorm':
            return self.norm_layer(x)
        
        if self.norm_type in ['batchnorm_b', 'batch_norm_b', 'batchnormb', 'batchnorm_bonly', 'batchnorm_b_only', 'batchnorm_only_b']:
            # Compute per-location stats across batch only
            # Keep math in fp32 for stability
            x_float = x.float()
            batch_mean = x_float.mean(dim=0, keepdim=True)                # (1, C, S, T)
            batch_var  = x_float.var(dim=0, keepdim=True, unbiased=False) # (1, C, S, T)

            if self.training:
                # Aggregate location stats to channel level for lightweight running stats
                mean_ch = batch_mean.view(1, x.shape[1], -1).mean(dim=-1).view(-1)  # (C,)
                # Use second-moment aggregation to better match BN2d channel-wise running var
                ex2_loc = batch_var + batch_mean.pow(2)                              # (1, C, S, T)
                ex2_ch  = ex2_loc.view(1, x.shape[1], -1).mean(dim=-1).view(-1)     # (C,)
                var_ch  = ex2_ch - mean_ch.pow(2)                                    # (C,)
                with torch.no_grad():
                    self.norm_layer.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean_ch)
                    self.norm_layer.running_var.mul_(1 - self.momentum).add_(self.momentum * var_ch)
                    self.num_batches_tracked.add_(1)

                mean_to_use = batch_mean
                var_to_use  = batch_var
            else:
                # Use channel-wise running stats broadcast over (S,T)
                mean_to_use = self.norm_layer.running_mean.view(1, -1, 1, 1)
                var_to_use  = self.norm_layer.running_var.view(1, -1, 1, 1)

            xhat = (x_float - mean_to_use) / torch.sqrt(var_to_use + self.eps)
            y = xhat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            return y.to(x.dtype)
        
        # LN/RMS: compute stats across channel axis only, per (B, S, T)
        # do stats in fp32 for stability in mixed precision
        x_float = x.float()
        if self.norm_type == 'layernorm':
            mean = x_float.mean(dim=1, keepdim=True)
            var  = x_float.var(dim=1, keepdim=True, unbiased=False)
            xhat = (x_float - mean) / torch.sqrt(var + self.eps)
            y = xhat * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
            return y.to(x.dtype)
        elif self.norm_type == 'rmsnorm':
            rms = torch.sqrt(x_float.pow(2).mean(dim=1, keepdim=True) + self.eps)
            y = (x_float / rms) * self.weight.view(1, -1, 1, 1)
            return y.to(x.dtype)
    
    def _assert_allclose(self, a, b, name, rtol=1e-6, atol=1e-6):
        """Helper method for testing."""
        if not torch.allclose(a, b, rtol=rtol, atol=atol):
            diff = (a - b).abs().max().item()
            raise AssertionError(f"{name} mismatch: max_abs_diff={diff}")

    def test_batchnorm_identical(self, B=8, C=None, S=4, T=32):
        """Test that BatchNorm matches PyTorch's BatchNorm2d."""
        if C is None:
            C = self.num_channels
        print("Testing BatchNorm2d...")
        x = torch.randn(B, C, S, T)

        custom_norm = Normalization(C, norm_type='BatchNorm', eps=self.eps, momentum=self.momentum)
        batch_norm = nn.BatchNorm2d(C, eps=self.eps, momentum=self.momentum, affine=True, track_running_stats=True)

        # Ensure identical initial state
        with torch.no_grad():
            batch_norm.weight.copy_(custom_norm.norm_layer.weight)
            batch_norm.bias.copy_(custom_norm.norm_layer.bias)
            batch_norm.running_mean.copy_(custom_norm.norm_layer.running_mean)
            batch_norm.running_var.copy_(custom_norm.norm_layer.running_var)

        # Train mode: outputs should match; running stats should update identically
        custom_norm.train(); batch_norm.train()
        y_cn_tr = custom_norm(x)
        y_bn_tr = batch_norm(x)
        self._assert_allclose(y_cn_tr, y_bn_tr, "BN train forward")
        self._assert_allclose(custom_norm.norm_layer.running_mean, batch_norm.running_mean, "BN running_mean after train forward")
        self._assert_allclose(custom_norm.norm_layer.running_var,  batch_norm.running_var,  "BN running_var after train forward")

        # Eval mode: use running stats; outputs must match
        custom_norm.eval(); batch_norm.eval()
        y_cn_ev = custom_norm(x)
        y_bn_ev = batch_norm(x)
        self._assert_allclose(y_cn_ev, y_bn_ev, "BN eval forward")

        print("  ✓ BatchNorm2d matches Normalization(BatchNorm)")

    def test_layernorm_identical(self, B=8, C=None, S=4, T=32):
        """Test that LayerNorm matches PyTorch's LayerNorm applied channelwise."""
        if C is None:
            C = self.num_channels
        print("Testing LayerNorm...")
        x = torch.randn(B, C, S, T)

        custom_norm = Normalization(C, norm_type='LayerNorm', eps=self.eps)
        layer_norm = nn.LayerNorm(C, eps=self.eps, elementwise_affine=True)

        # Align affine params
        with torch.no_grad():
            layer_norm.weight.copy_(custom_norm.weight)
            layer_norm.bias.copy_(custom_norm.bias)

        # Reference: LN over channels requires channels-last
        def ln_ref_apply(x):
            x_perm = x.permute(0, 2, 3, 1)               # (B, S, T, C)
            y_perm = layer_norm(x_perm)                  # normalize last dim (C)
            return y_perm.permute(0, 3, 1, 2)            # back to (B, C, S, T)

        custom_norm.eval(); layer_norm.eval()
        y_cn = custom_norm(x)
        y_ln = ln_ref_apply(x)
        self._assert_allclose(y_cn, y_ln, "LayerNorm (channelwise)")

        print("  ✓ LayerNorm(channelwise) matches Normalization(LayerNorm)")

    def test_rmsnorm_identical(self, B=8, C=None, S=4, T=32):
        """Test that RMSNorm matches PyTorch's RMSNorm applied channelwise."""
        if C is None:
            C = self.num_channels
        print("Testing RMSNorm...")
        x = torch.randn(B, C, S, T)

        custom_norm = Normalization(C, norm_type='RMSnorm', eps=self.eps)

        rms_norm = nn.RMSNorm(C, eps=self.eps)
        with torch.no_grad():
            rms_norm.weight.copy_(custom_norm.weight)
        
        # reference function
        def rms_ref_apply(x):
            x_perm = x.permute(0, 2, 3, 1)            # (B, S, T, C)
            y_perm = rms_norm(x_perm)                 # normalize last dim (C)
            return y_perm.permute(0, 3, 1, 2)         # (B, C, S, T)

        custom_norm.eval(); rms_norm.eval()
        y_cn = custom_norm(x)
        y_rms = rms_ref_apply(x)
        self._assert_allclose(y_cn, y_rms, "RMSNorm (channelwise)")

        print("  ✓ RMSNorm(channelwise) matches Normalization(RMSnorm)")

    def test_batchnorm_b_only(self, B=8, C=None, S=4, T=32):
        """Test BatchNorm_B: batch-only normalization per location with channel-wise affine and running stats."""
        if C is None:
            C = self.num_channels
        print("Testing BatchNorm_B (batch-only stats)...")
        x = torch.randn(B, C, S, T)

        custom_norm = Normalization(C, norm_type='BatchNorm_B', eps=self.eps, momentum=self.momentum)

        # Give non-trivial affine params to verify scaling/shift
        with torch.no_grad():
            custom_norm.weight.copy_(torch.randn(C) * 0.1 + 1.0)
            custom_norm.bias.copy_(torch.randn(C) * 0.05)

        # Train mode: output must match manual batch-only per-location normalization
        custom_norm.train()
        y_tr = custom_norm(x)

        x_float = x.float()
        batch_mean = x_float.mean(dim=0, keepdim=True)                # (1, C, S, T)
        batch_var  = x_float.var(dim=0, keepdim=True, unbiased=False) # (1, C, S, T)
        xhat = (x_float - batch_mean) / torch.sqrt(batch_var + self.eps)
        y_ref_tr = xhat * custom_norm.weight.view(1, -1, 1, 1) + custom_norm.bias.view(1, -1, 1, 1)
        self._assert_allclose(y_tr, y_ref_tr, "BatchNorm_B train forward")

        # Running stats should update to momentum*channel_agg + (1-momentum)*init
        mean_ch = batch_mean.view(1, C, -1).mean(dim=-1).view(-1)  # (C,)
        ex2_loc = batch_var + batch_mean.pow(2)                    # (1, C, S, T)
        ex2_ch  = ex2_loc.view(1, C, -1).mean(dim=-1).view(-1)     # (C,)
        var_ch  = ex2_ch - mean_ch.pow(2)                          # (C,)

        rm_expected = (1 - self.momentum) * torch.zeros_like(mean_ch) + self.momentum * mean_ch
        rv_expected = (1 - self.momentum) * torch.ones_like(var_ch) + self.momentum * var_ch
        self._assert_allclose(custom_norm.running_mean, rm_expected, "BatchNorm_B running_mean after train forward")
        self._assert_allclose(custom_norm.running_var,  rv_expected, "BatchNorm_B running_var after train forward")

        # Eval mode: output must match manual eval using running (channel) stats broadcast spatially
        custom_norm.eval()
        y_ev = custom_norm(x)
        mean_eval = custom_norm.running_mean.view(1, -1, 1, 1)
        var_eval  = custom_norm.running_var.view(1, -1, 1, 1)
        y_ref_ev = (x_float - mean_eval) / torch.sqrt(var_eval + self.eps)
        y_ref_ev = y_ref_ev * custom_norm.weight.view(1, -1, 1, 1) + custom_norm.bias.view(1, -1, 1, 1)
        self._assert_allclose(y_ev, y_ref_ev, "BatchNorm_B eval forward")

        print("  ✓ BatchNorm_B matches manual batch-only normalization (train/eval)")

    def small_test(self, B=8, C=None, S=4, T=32):
        """Run all tests for this normalization type."""
        if C is None:
            C = self.num_channels
        
        print('-------------------------------------------------------------------------------')
        print(f"Running small_test for Normalization(norm_type='{self.norm_type}', num_channels={C})")
        
        if self.norm_type in ['batchnorm', 'batch_norm']:
            self.test_batchnorm_identical(B, C, S, T)
        elif self.norm_type in ['batchnorm_b', 'batch_norm_b', 'batchnormb', 'batchnorm_bonly']:
            self.test_batchnorm_b_only(B, C, S, T)
        elif self.norm_type in ['layernorm', 'layer_norm']:
            self.test_layernorm_identical(B, C, S, T)
        elif self.norm_type in ['rmsnorm', 'rms_norm']:
            self.test_rmsnorm_identical(B, C, S, T)
        
        print(f"✅ small_test passed for {self.norm_type}")
        print('-------------------------------------------------------------------------------')
    
    @staticmethod
    def run_all_tests(B=8, C=16, S=4, T=32, eps=1e-5, momentum=0.1):
        """Static method to run tests for all normalization types."""
        print('----------------------------------------------------------------------------------------------------')
        print("Running all normalization tests")
        print('-------------------------------')
        
        # Test BatchNorm
        bn_norm = Normalization(C, norm_type='BatchNorm', eps=eps, momentum=momentum)
        bn_norm.small_test(B, C, S, T)
        
        # Test BatchNorm_B (batch-only)
        bnb_norm = Normalization(C, norm_type='BatchNorm_B', eps=eps, momentum=momentum)
        bnb_norm.small_test(B, C, S, T)
        
        # Test LayerNorm
        ln_norm = Normalization(C, norm_type='LayerNorm', eps=eps)
        ln_norm.small_test(B, C, S, T)
        
        # Test RMSNorm
        rms_norm = Normalization(C, norm_type='RMSnorm', eps=eps)
        rms_norm.small_test(B, C, S, T)
        
        print('---------------------------------')
        print("All normalization tests passed ✅")
        print('----------------------------------------------------------------------------------------------------')

# TCN block - assume input is (B, C, 1, T)
class TCN_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, temporal_kernel_size, 
                 temporal_dilation=1, bottleneck_dim=None, nonlinearity_str='lgelu', leaky_slope=0.3, norm_type='BatchNorm'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_dilation = temporal_dilation
        self.bottleneck_dim = bottleneck_dim
        self.nonlinearity_str = nonlinearity_str
        self.leaky_slope = leaky_slope
        self.norm_type = norm_type

        # instantiate the layers of SpatioTemporalCausalConv2D blocks
        for i in range(num_layers):
            self.add_module(f"conv{i}", SpatioTemporalCausalConv2D(in_channels, out_channels, spatial_kernel_size=1, 
                                                                   temporal_kernel_size=self.temporal_kernel_size, 
                                                                   bottleneck_dim=self.bottleneck_dim, 
                                                                   temporal_dilation=self.temporal_dilation,
                                                                   temp_conv_first=True))
            # self.add_module(f"bn{i}", nn.BatchNorm2d(out_channels))
            self.add_module(f"norm{i}", Normalization(out_channels, norm_type=self.norm_type))
            self.add_module(f"nlin{i}", Nonlinearity(self.nonlinearity_str, self.leaky_slope))
            in_channels = out_channels

        self.update_short_name()

    def forward(self, x):
        # assume x.shape = (B, C, 1, T)
        for i in range(self.num_layers):
            x = self.__getattr__(f"conv{i}")(x) # equivalent to "x = self.conv{i}}(x)"
            x = self.__getattr__(f"norm{i}")(x)   # equivalent to "x = self.norm{i}}(x)"
            x = self.__getattr__(f"nlin{i}")(x) # equivalent to "x = self.nlin{i}}(x)"
        return x

    def update_short_name(self):

        self.effective_params_per_layer = [self.__getattr__(f"conv{i}").get_num_effective_params() for i in range(self.num_layers)]
        self.num_params_effective = sum(self.effective_params_per_layer)
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_effective / 1e3
        self.num_parmas_millions = self.num_params_effective / 1e6
        if self.temporal_dilation == 1:
            self.total_temporal_kernel_size = (self.temporal_kernel_size - 1) * self.num_layers + 1
        else:
            # NOTE: verify this calculation
            self.total_temporal_kernel_size = self.temporal_kernel_size * (self.temporal_dilation ** (self.num_layers - 1)) - (self.num_layers - 1)

        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        temporal_description_str = f"T_{self.total_temporal_kernel_size}"
        depth_description_str = f"D_{self.num_layers}"

        self.short_name = f"TCN_block_{depth_description_str}_{temporal_description_str}_{param_description_str}"


# TCN backbone - assume input is (B, C, in_spatial_dim, T). 
# first layer is using spatial_kernel_size=in_spatial_dim and temporal_kernel_size=first_layer_temporal_kernel_size
class TCN_Backbone(nn.Module):
    def __init__(self, in_channels, in_spatial_dim, first_layer_temporal_kernel_size, num_layers_per_block_list, num_features_per_block_list, 
                 temporal_kernel_size_per_block_list, temporal_dilation_per_block_list, bottleneck_dim_per_block_list=None, 
                 nonlinearity_str='lgelu', leaky_slope=0.3, norm_type='BatchNorm'):
        super().__init__()

        self.in_channels = in_channels
        self.in_spatial_dim = in_spatial_dim
        self.out_channels = num_features_per_block_list[-1]
        self.num_blocks = len(num_layers_per_block_list)
        self.num_layers_per_block_list = num_layers_per_block_list
        self.num_features_per_block_list = num_features_per_block_list
        self.temporal_kernel_size_per_block_list = temporal_kernel_size_per_block_list
        self.temporal_dilation_per_block_list = temporal_dilation_per_block_list
        self.bottleneck_dim_per_block_list = [None] * self.num_blocks if bottleneck_dim_per_block_list is None else bottleneck_dim_per_block_list
        self.nonlinearity_str = nonlinearity_str
        self.leaky_slope = leaky_slope
        self.norm_type = norm_type

        assert self.num_blocks == len(self.bottleneck_dim_per_block_list) == len(self.num_features_per_block_list), 'lists must have the same length'
        assert self.num_blocks == len(self.temporal_kernel_size_per_block_list) == len(self.temporal_dilation_per_block_list), 'lists must have the same length'

        # first layer is using spatial_kernel_size=S and temporal_kernel_size=first_layer_temporal_kernel_size
        self.first_conv = SpatioTemporalCausalConv2D(in_channels, out_channels=self.num_features_per_block_list[0],
                                                     spatial_kernel_size=self.in_spatial_dim, spatial_padding='valid',
                                                     temporal_kernel_size=first_layer_temporal_kernel_size,
                                                     bottleneck_dim=None, temp_conv_first=False,
                                                     convert_out_channels_to_spatial=True, 
                                                     post_conversion_out_channels=self.num_features_per_block_list[0])
        # self.first_bn = nn.BatchNorm2d(self.num_features_per_block_list[0])
        self.first_norm = Normalization(self.num_features_per_block_list[0], norm_type=self.norm_type)
        self.first_nlin = Nonlinearity(self.nonlinearity_str, self.leaky_slope)

        # blocks
        in_channels = self.num_features_per_block_list[0]
        for i in range(self.num_blocks):
            out_channels = self.num_features_per_block_list[i]
            self.add_module(f"block{i}", TCN_Block(in_channels, out_channels, self.num_layers_per_block_list[i],
                                                   self.temporal_kernel_size_per_block_list[i],
                                                   temporal_dilation=self.temporal_dilation_per_block_list[i],
                                                   bottleneck_dim=self.bottleneck_dim_per_block_list[i],
                                                   nonlinearity_str=self.nonlinearity_str, leaky_slope=self.leaky_slope, norm_type=self.norm_type))
            in_channels = out_channels

        self.update_short_name()

    def forward(self, x):
        # assume x.shape = (B, C_in, S, T)
        x = self.first_nlin(self.first_norm(self.first_conv(x))) # (B, C_b0, 1, T)
        for i in range(self.num_blocks):
            x = self.__getattr__(f"block{i}")(x) # equivalent to "x = self.block{i}}(x)"
        return x
    
    def forward_debug(self, x):

        print(f'TCN backbone:            input shape: {x.shape}')
        x = self.first_nlin(self.first_norm(self.first_conv(x)))
        print(f'TCN backbone: 1st layer output shape: {x.shape}')
        for i in range(self.num_blocks):
            x = self.__getattr__(f"block{i}")(x)
            print(f'TCN backbone:   block {i} output shape: {x.shape}')

        return x
    
    def update_short_name(self):
            
        self.effective_params_per_block = [self.__getattr__(f"block{i}").num_params_effective for i in range(self.num_blocks)]
        self.num_params_effective = sum(self.effective_params_per_block) + self.first_conv.get_num_effective_params()
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_effective / 1e3
        self.num_parmas_millions = self.num_params_effective / 1e6
        self.total_temporal_kernel_size = self.first_conv.temporal_kernel_size
        self.total_temporal_kernel_size += sum([(self.__getattr__(f"block{i}").total_temporal_kernel_size - 1) for i in range(self.num_blocks)])
        self.total_depth = sum(self.num_layers_per_block_list) + 1

        self.widths_list = [self.first_conv.out_channels]
        self.widths_list += [self.num_features_per_block_list[i] for i in range(self.num_blocks)]
        self.average_width = np.mean(self.widths_list)

        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        temporal_description_str = f"T_{self.total_temporal_kernel_size}"
        depth_description_str = f"D_{self.total_depth}"
        width_description_str = f"W_{self.average_width:.0f}"

        self.short_name = f"TCN_backbone_{depth_description_str}_{width_description_str}_{temporal_description_str}_{param_description_str}"

    def set_output_dims(self, x, print_output_dims=False):

        y = self.forward(x)
        B, C, S, T = y.shape

        self.out_channels = C
        self.out_spatial_dim = S

        if print_output_dims:
            print(f'TCN backbone: just set output dims')
            print(f'TCN backbone: input shape: {x.shape}')
            print(f'TCN backbone: output shape: {y.shape}')


# ResNet TCN miniblock - assume input is (B, C, 1, T). have skip connection with 1x1 conv to match dimensions
# only supports "basic" (2 SpatioTemporalCausalConv2D convs with spatial_kernel_size=1
class TCN_ResNet_MiniBlock(nn.Module):
    def __init__(self, in_channels, out_channels, temporal_kernel_size, 
                 temporal_dilation=1, bottleneck_dim=None, nonlinearity_str='lgelu', leaky_slope=0.3, norm_type='BatchNorm'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_dilation = temporal_dilation
        self.bottleneck_dim = bottleneck_dim
        self.nonlinearity_str = nonlinearity_str
        self.leaky_slope = leaky_slope
        self.norm_type = norm_type

        # two layers of SpatioTemporalCausalConv2D
        self.conv1 = SpatioTemporalCausalConv2D(self.in_channels, self.out_channels, spatial_kernel_size=1,
                                                temporal_kernel_size=self.temporal_kernel_size,
                                                bottleneck_dim=self.bottleneck_dim,
                                                temporal_dilation=self.temporal_dilation,
                                                temp_conv_first=True)
        # self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.norm1 = Normalization(self.out_channels, norm_type=self.norm_type)
        self.nlin1 = Nonlinearity(self.nonlinearity_str, self.leaky_slope)

        self.conv2 = SpatioTemporalCausalConv2D(self.out_channels, self.out_channels, spatial_kernel_size=1,
                                                temporal_kernel_size=self.temporal_kernel_size,
                                                bottleneck_dim=self.bottleneck_dim,
                                                temporal_dilation=self.temporal_dilation,
                                                temp_conv_first=True)
        # self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.norm2 = Normalization(self.out_channels, norm_type=self.norm_type)
        self.nlin2 = Nonlinearity(self.nonlinearity_str, self.leaky_slope)

        # skip connection
        self.shortcut_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

        self.num_params_effective = self.conv1.get_num_effective_params() + self.conv2.get_num_effective_params()
        self.num_params_effective += sum(p.numel() for p in self.shortcut_conv.parameters())
        self.num_params_official = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        # assume x.shape = (B, C, 1, T)
        res = self.nlin1(self.norm1(self.conv1(x)))
        res = self.nlin2(self.norm2(self.conv2(res)))
        return self.shortcut_conv(x) + res


# TCN_ResNet_Block contains several 'TCN_ResNet_MiniBlock's
class TCN_ResNet_Block(nn.Module):
    def __init__(self, in_channels, out_channels, num_miniblocks, temporal_kernel_size,
                 temporal_dilation=1, bottleneck_dim=None, nonlinearity_str='lgelu', leaky_slope=0.3, norm_type='BatchNorm'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_miniblocks = num_miniblocks
        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_dilation = temporal_dilation
        self.bottleneck_dim = bottleneck_dim
        self.nonlinearity_str = nonlinearity_str
        self.leaky_slope = leaky_slope
        self.norm_type = norm_type
        
        # instantiate TCN_ResNet_MiniBlock blocks
        for i in range(num_miniblocks):
            self.add_module(f"block{i}", TCN_ResNet_MiniBlock(in_channels, out_channels, temporal_kernel_size=self.temporal_kernel_size,
                                                              temporal_dilation=self.temporal_dilation, bottleneck_dim=self.bottleneck_dim,
                                                              nonlinearity_str=self.nonlinearity_str, leaky_slope=self.leaky_slope, norm_type=self.norm_type))
            in_channels = out_channels

        self.update_short_name()

    def forward(self, x):
        # assume x.shape = (B, C, 1, T)
        for i in range(self.num_miniblocks):
            x = self.__getattr__(f"block{i}")(x) # equivalent to "x = self.block{i}}(x)"
        return x
    
    def update_short_name(self):

        self.effective_params_per_block = [self.__getattr__(f"block{i}").num_params_effective for i in range(self.num_miniblocks)]
        self.num_params_effective = sum(self.effective_params_per_block)
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_effective / 1e3
        self.num_parmas_millions = self.num_params_effective / 1e6
        if self.temporal_dilation == 1:
            self.total_temporal_kernel_size = (self.temporal_kernel_size - 1) * (2 * self.num_miniblocks) + 1
        else:
            # NOTE: verify this calculation
            self.total_temporal_kernel_size = self.temporal_kernel_size * (self.temporal_dilation ** (2 * self.num_miniblocks - 1)) - (self.num_miniblocks - 1)

        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        temporal_description_str = f"T_{self.total_temporal_kernel_size}"
        depth_description_str = f"D_{self.num_miniblocks}"

        self.short_name = f"TCN_ResNet_block_{depth_description_str}_{temporal_description_str}_{param_description_str}"


# TCN ResNet backbone - assume input is (B, C, in_spatial_dim, T). 
# first layer is using spatial_kernel_size=in_spatial_dim and temporal_kernel_size=first_layer_temporal_kernel_size
class TCN_ResNet_Backbone(nn.Module):
    def __init__(self, in_channels, in_spatial_dim, first_layer_temporal_kernel_size, num_miniblocks_per_block_list, num_features_per_block_list, 
                 temporal_kernel_size_per_block_list, temporal_dilation_per_block_list, bottleneck_dim_per_block_list=None, 
                 nonlinearity_str='lgelu', leaky_slope=0.3, norm_type='BatchNorm'):
        super().__init__()

        self.in_channels = in_channels
        self.in_spatial_dim = in_spatial_dim
        self.out_channels = num_features_per_block_list[-1]
        self.num_blocks = len(num_miniblocks_per_block_list)
        self.num_miniblocks_per_block_list = num_miniblocks_per_block_list
        self.num_features_per_block_list = num_features_per_block_list
        self.temporal_kernel_size_per_block_list = temporal_kernel_size_per_block_list
        self.temporal_dilation_per_block_list = temporal_dilation_per_block_list
        self.bottleneck_dim_per_block_list = [None] * self.num_blocks if bottleneck_dim_per_block_list is None else bottleneck_dim_per_block_list
        self.nonlinearity_str = nonlinearity_str
        self.leaky_slope = leaky_slope
        self.norm_type = norm_type
        
        assert self.num_blocks == len(self.num_miniblocks_per_block_list) == len(self.num_features_per_block_list), 'lists must have the same length'
        assert self.num_blocks == len(self.temporal_kernel_size_per_block_list) == len(self.temporal_dilation_per_block_list), 'lists must have the same length'

        # first layer is using spatial_kernel_size=S and temporal_kernel_size=first_layer_temporal_kernel_size
        self.first_conv = SpatioTemporalCausalConv2D(in_channels, out_channels=self.num_features_per_block_list[0],
                                                     spatial_kernel_size=self.in_spatial_dim, spatial_padding='valid',
                                                     temporal_kernel_size=first_layer_temporal_kernel_size,
                                                     bottleneck_dim=None, temp_conv_first=False,
                                                     convert_out_channels_to_spatial=True, 
                                                     post_conversion_out_channels=self.num_features_per_block_list[0])
        # self.first_bn = nn.BatchNorm2d(self.num_features_per_block_list[0])
        self.first_norm = Normalization(self.num_features_per_block_list[0], norm_type=self.norm_type)
        self.first_nlin = Nonlinearity(self.nonlinearity_str, self.leaky_slope)

        # blocks
        in_channels = self.num_features_per_block_list[0]
        for i in range(self.num_blocks):
            out_channels = self.num_features_per_block_list[i]
            self.add_module(f"block{i}", TCN_ResNet_Block(in_channels, out_channels, self.num_miniblocks_per_block_list[i],
                                                          self.temporal_kernel_size_per_block_list[i],
                                                          temporal_dilation=self.temporal_dilation_per_block_list[i],
                                                          bottleneck_dim=self.bottleneck_dim_per_block_list[i],
                                                          nonlinearity_str=self.nonlinearity_str, leaky_slope=self.leaky_slope, norm_type=self.norm_type))
            in_channels = out_channels

        self.update_short_name()

    def forward(self, x):
        # assume x.shape = (B, C_in, S, T)
        x = self.first_nlin(self.first_norm(self.first_conv(x))) # (B, C_b0, 1, T)
        for i in range(self.num_blocks):
            x = self.__getattr__(f"block{i}")(x) # equivalent to "x = self.block{i}}(x)"
        return x
    
    def forward_debug(self, x):
            
        print(f'TCN ResNet backbone:            input shape: {x.shape}')
        x = self.first_nlin(self.first_norm(self.first_conv(x)))
        print(f'TCN ResNet backbone: 1st layer output shape: {x.shape}')
        for i in range(self.num_blocks):
            x = self.__getattr__(f"block{i}")(x)
            print(f'TCN ResNet backbone:   block {i} output shape: {x.shape}')

        return x
    
    def update_short_name(self):
        
        self.effective_params_per_block = [self.__getattr__(f"block{i}").num_params_effective for i in range(self.num_blocks)]
        self.num_params_effective = sum(self.effective_params_per_block) + self.first_conv.get_num_effective_params()
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_effective / 1e3
        self.num_parmas_millions = self.num_params_effective / 1e6
        self.total_temporal_kernel_size = self.first_conv.temporal_kernel_size
        self.total_temporal_kernel_size += sum([(self.__getattr__(f"block{i}").total_temporal_kernel_size - 1) for i in range(self.num_blocks)])
        self.total_depth = 2 * sum(self.num_miniblocks_per_block_list) + 1

        self.widths_list = [self.first_conv.out_channels]
        self.widths_list += [self.num_features_per_block_list[i] for i in range(self.num_blocks)]
        self.widths_list += [self.num_features_per_block_list[i] for i in range(self.num_blocks)]
        self.average_width = np.mean(self.widths_list)

        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        temporal_description_str = f"T_{self.total_temporal_kernel_size}"
        depth_description_str = f"D_{self.total_depth}"
        width_description_str = f"W_{self.average_width:.0f}"

        self.short_name = f"TCN_ResNet_backbone_{depth_description_str}_{width_description_str}_{temporal_description_str}_{param_description_str}"

    def set_output_dims(self, x, print_output_dims=False):

        y = self.forward(x)
        B, C, S, T = y.shape

        self.out_channels = C
        self.out_spatial_dim = S

        if print_output_dims:
            print(f'TCN ResNet backbone: just set output dims')
            print(f'TCN ResNet backbone: input shape: {x.shape}')
            print(f'TCN ResNet backbone: output shape: {y.shape}')

# some helper functions for the ELM model
def scaled_sigmoid(x, lower_bound: float, upper_bound: float):
    return (upper_bound - lower_bound) * torch.sigmoid(x) + lower_bound

def inverse_scaled_sigmoid(x, lower_bound: float, upper_bound: float):
    x = torch.clamp(x, lower_bound + 1e-6, upper_bound - 1e-6)
    return torch.log((x - lower_bound) / (upper_bound - x))

# ShallowMLP block - assume input is (B, C_in) and output is (B, C_out)
class ShallowMLP(nn.Module):
    """Shallow Multi-layer perceptron (MLP) model. uses only single pre-norm layer, so cant really go too deep."""

    def __init__(self, input_dim, output_dim, num_hidden_layers=1, hidden_dim=32, nonlinearity_str='lgelu', leaky_slope=0.2, pre_norm_type='RMSNorm'):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_hidden_layers = num_hidden_layers 
        self.hidden_dim = hidden_dim
        self.nonlinearity_str = nonlinearity_str
        self.leaky_slope = leaky_slope
        self.pre_norm_type = pre_norm_type

        if self.pre_norm_type == 'RMSNorm':
            self.pre_norm = nn.RMSNorm(self.input_dim)
        elif self.pre_norm_type == 'LayerNorm':
            self.pre_norm = nn.LayerNorm(self.input_dim)
        elif self.pre_norm_type == 'BatchNorm':
            self.pre_norm = nn.BatchNorm1d(self.input_dim)
        elif self.pre_norm_type in ['none', 'None', 'identity', 'Identity', 'linear', 'Linear']:
            self.pre_norm = nn.Identity()
        else:
            raise ValueError(f"Unknown pre_norm_type: {self.pre_norm_type}")

        current_input_dim = self.input_dim
        for i in range(num_hidden_layers):
            self.add_module(f"fc{i}", nn.Linear(current_input_dim, self.hidden_dim))
            self.add_module(f"nlin{i}", Nonlinearity(self.nonlinearity_str, self.leaky_slope))
            current_input_dim = self.hidden_dim

        self.add_module(f"fc{self.num_hidden_layers}", nn.Linear(current_input_dim, self.output_dim))

    def forward(self, x):
        # assume x.shape = (B, input_dim)
        x = self.pre_norm(x)                                   # x.shape = (B, input_dim)
        for i in range(self.num_hidden_layers):
            x = self.__getattr__(f"fc{i}")(x)                  # x.shape = (B, num_units_per_layer)
            x = self.__getattr__(f"nlin{i}")(x)                # x.shape = (B, num_units_per_layer)
        x = self.__getattr__(f"fc{self.num_hidden_layers}")(x) # x.shape = (B, output_dim)

        return x

# ELM backbone - assume input is (B, C, in_spatial_dim, T). 
# spatially flattened to (B, C*in_spatial_dim, 1, T) then processed through ELM dynamics
class ELM_Backbone(nn.Module):
    def __init__(self, in_channels, in_spatial_dim, memory_dim=64, 
                 mlp_num_hidden_layers=1, mlp_hidden_dim=None, mlp_nonlinearity_str='lsilu', mlp_leaky_slope=0.2, mlp_pre_norm_type='BatchNorm',
                 post_mlp_nonlinearity_str='ltanh', post_mlp_leaky_slope=0.1, lambda_value=5.0, 
                 synapse_tau_value=5.0, memory_tau_min=1.0, memory_tau_max=128.0, learn_memory_tau=True, w_s_value=0.5, delta_t=1.0,
                 recurrent_w_init_method='normal', init_std=0.02):
        super().__init__()

        self.in_channels = in_channels
        self.in_spatial_dim = in_spatial_dim
        self.memory_dim = memory_dim
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
        self.recurrent_w_init_method = recurrent_w_init_method
        self.init_std = init_std

        # derived properties
        self.syn_input_dim = in_channels * in_spatial_dim  # flattened spatial input
        self.mlp_input_dim = self.syn_input_dim + self.memory_dim
        self.mlp_output_dim = self.memory_dim

        # initialization of model weights
        # input processing related (synapse time constants and decay factors)
        self._proto_w_s = nn.parameter.Parameter(torch.full((self.syn_input_dim,), w_s_value))
        tau_s = torch.full((self.syn_input_dim,), synapse_tau_value)
        self.tau_s = nn.parameter.Parameter(tau_s, requires_grad=False)

        # recurrent backbone related (memory time constants and decay factors)
        _proto_tau_m = torch.logspace(math.log10(memory_tau_min + 1e-6), math.log10(memory_tau_max - 1e-6), memory_dim)
        _proto_tau_m = inverse_scaled_sigmoid(_proto_tau_m, memory_tau_min, memory_tau_max)
        self._proto_tau_m = nn.parameter.Parameter(_proto_tau_m, requires_grad=learn_memory_tau)

        # recurrent backbone related (MLP & post-MLP nonlinearity)
        self.mlp = ShallowMLP(
            input_dim=self.mlp_input_dim,
            output_dim=self.mlp_output_dim,
            num_hidden_layers=self.mlp_num_hidden_layers,
            hidden_dim=self.mlp_hidden_dim,
            nonlinearity_str=self.mlp_nonlinearity_str,
            leaky_slope=self.mlp_leaky_slope,
            pre_norm_type=self.mlp_pre_norm_type,
        )
        self.post_mlp_nlin = Nonlinearity(self.post_mlp_nonlinearity_str, self.post_mlp_leaky_slope)

        self._initialize_mlp_weights()
        self.update_short_name()

    def _initialize_mlp_weights(self):
        """Initialize the weights of the MLP linear layers based on the specified initialization method."""
        for i in range(self.mlp_num_hidden_layers + 1):
            linear_layer = getattr(self.mlp, f"fc{i}")
            
            if self.recurrent_w_init_method == 'kaiming':
                nn.init.kaiming_normal_(linear_layer.weight, nonlinearity='leaky_relu', a=self.mlp_leaky_slope)
            elif self.recurrent_w_init_method == 'orthogonal':
                nn.init.orthogonal_(linear_layer.weight, gain=1.0)
            elif self.recurrent_w_init_method == 'normal':
                nn.init.normal_(linear_layer.weight, mean=0.0, std=self.init_std)
            else:
                raise ValueError(f"Unknown recurrent_w_init_method: {self.recurrent_w_init_method}")

    @property
    def tau_m(self):
        return scaled_sigmoid(self._proto_tau_m, self.memory_tau_min, self.memory_tau_max)

    @property
    def kappa_m(self):
        return torch.exp(-self.delta_t / torch.clamp(self.tau_m, min=1e-6))

    @property
    def kappa_s(self):
        return torch.exp(-self.delta_t / torch.clamp(self.tau_s, min=1e-6))

    @property
    def w_s(self):
        return self._proto_w_s

    def step_fn(self, x_t, s_prev, m_prev, kappa_s, kappa_m, w_s):
        """Single step dynamics - can be compiled nicely with torch.compile to help a bit with timing performance."""
        s_t = kappa_s * s_prev + w_s * x_t # input dynamics
        delta_m_t = self.mlp(torch.cat([s_t, kappa_m * m_prev], dim=-1)) # recurrent memory dynamics
        delta_m_t = self.post_mlp_nlin(delta_m_t) # post-MLP nonlinearity
        # delta_m_t = self.post_mlp_nlin(0.6666667 * delta_m_t) * 1.7159 # post-MLP nonlinearity (with original scaling before and after tanh)
        m_t = kappa_m * m_prev + self.lambda_value * (1 - kappa_m) * delta_m_t
        return s_t, m_t

    def forward(self, x):
        # assume x.shape = (B, C_in, S, T)
        B, C, S, T = x.shape
        
        # Reshape (B, C, S, T) to (B, T, C*S) for ELM processing
        x_reshaped = x.permute(0, 3, 1, 2).contiguous()  # (B, T, C, S)
        x_reshaped = x_reshaped.view(B, T, C * S)        # (B, T, C*S)
        
        # ELM processing: X.shape = (batch_size, T, num_input)
        batch_size, T, _ = x_reshaped.shape

        w_s = self.w_s          # w_s.shape     = (num_input,)
        kappa_s = self.kappa_s  # kappa_s.shape = (num_input,)
        kappa_m = self.kappa_m  # kappa_m.shape = (num_memory,)
        
        s_prev = torch.zeros(batch_size, len(kappa_s), device=x.device) # s_prev.shape = (batch_size, num_input)
        m_prev = torch.zeros(batch_size, len(kappa_m), device=x.device) # m_prev.shape = (batch_size, num_memory)
        
        # NOTE: we cannot pre-allocate a single M_matrix of dimensions (M, T) here in which M[:,t] is m_t
        # because torch will not be able to calculate the gradients this way, 
        # torch needs the explicit variables to create the computational graph
        # the M_t_list is of length T, and the computational graph is enormously large
        # compilation times increases as T increases and this is a big mess
        M_t_list = []
        inputs = x_reshaped
        
        # use the compiled step function if it exists
        if hasattr(self, '_compiled_step_fn'):
            step_fn = self._compiled_step_fn
        else:
            step_fn = self.step_fn

        # loop through time steps and compute the M matrix
        for t in range(T):
            s_t, m_t = step_fn(inputs[:, t], s_prev, m_prev, kappa_s, kappa_m, w_s)
            s_prev = s_t
            m_prev = m_t
            M_t_list.append(m_t)
            
        M_matrix = torch.stack(M_t_list, dim=-2)  # M_matrix.shape = (batch_size, T, num_memory)
        
        # Reshape to TCN-style output
        output = M_matrix.permute(0, 2, 1).contiguous()  # (B, num_memory, T)
        output = output.unsqueeze(2)                     # (B, num_memory, 1, T)
        
        return output
    
    def forward_debug(self, x):

        print(f'ELM backbone:         input shape: {x.shape}')
        output = self.forward(x)
        print(f'ELM backbone: memory output shape: {output.shape}')

        return output
    
    def update_short_name(self):
            
        self.num_params_effective = sum(p.numel() for p in self.parameters())
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_effective / 1e3
        self.num_parmas_millions = self.num_params_effective / 1e6

        self.widths_list = [self.syn_input_dim, self.memory_dim]
        self.average_width = np.mean(self.widths_list)

        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        depth_description_str = f"D_{self.mlp_num_hidden_layers}"

        self.short_name = f"ELM_backbone_{depth_description_str}_M_{self.memory_dim}_{param_description_str}"

    def set_output_dims(self, x, print_output_dims=False):

        y = self.forward(x)
        B, C, S, T = y.shape

        self.out_channels = C
        self.out_spatial_dim = S

        if print_output_dims:
            print(f'ELM backbone: just set output dims')
            print(f'ELM backbone: input shape: {x.shape}')
            print(f'ELM backbone: output shape: {y.shape}')

    def compile_step_fn(self):
        self._compiled_step_fn = torch.compile(self.step_fn)
        return self._compiled_step_fn or self.step_fn


# RoPE using explicit sin/cos rotation (easier to understand)
class RoPE_SinCos(nn.Module):
    """RoPE using explicit sin/cos rotation"""
    def __init__(self, dim, max_seq_len, theta=10000.0):
        super().__init__()
        self.dim = dim
        # Compute frequency for each dimension pair
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len)
        angles = torch.outer(t, freqs)
        # Precompute cos and sin
        self.register_buffer('freqs_cos', torch.cos(angles))
        self.register_buffer('freqs_sin', torch.sin(angles))
    
    def forward(self, xq, xk):
        # Input: xq, xk have shape (B, T, n_heads, d_head)
        B, T, n_heads, d_head = xq.shape
        seq_len = T
        
        # Get frequencies for this sequence length: (T, d_head//2)
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]
        
        # Reshape x for rotation: split into pairs
        # (B, T, n_heads, d_head) -> (B, T, n_heads, d_head//2, 2)
        xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
        
        # Extract even and odd elements: each has shape (B, T, n_heads, d_head//2)
        xq0, xq1 = xq_reshaped[..., 0], xq_reshaped[..., 1]
        xk0, xk1 = xk_reshaped[..., 0], xk_reshaped[..., 1]
        
        # Reshape freqs for broadcasting: (T, d_head//2) -> (1, T, 1, d_head//2)
        freqs_cos = freqs_cos.view(1, seq_len, 1, -1)
        freqs_sin = freqs_sin.view(1, seq_len, 1, -1)
        
        # Apply rotation: [cos -sin; sin cos] @ [x0; x1]
        # All outputs have shape (B, T, n_heads, d_head//2)
        xq_out0 = xq0 * freqs_cos - xq1 * freqs_sin
        xq_out1 = xq0 * freqs_sin + xq1 * freqs_cos
        xk_out0 = xk0 * freqs_cos - xk1 * freqs_sin
        xk_out1 = xk0 * freqs_sin + xk1 * freqs_cos
        
        # Combine back and flatten: (B, T, n_heads, d_head//2, 2) -> (B, T, n_heads, d_head)
        xq_out = torch.stack([xq_out0, xq_out1], dim=-1).flatten(-2)
        xk_out = torch.stack([xk_out0, xk_out1], dim=-1).flatten(-2)
        
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    def forward_debug(self, xq, xk):
        # Input: xq, xk have shape (B, T, n_heads, d_head)
        B, T, n_heads, d_head = xq.shape
        seq_len = T
        
        print(f'  RoPE_SinCos input: xq={xq.shape}, xk={xk.shape}')
        
        # Get frequencies for this sequence length: (T, d_head//2)
        freqs_cos = self.freqs_cos[:seq_len]
        freqs_sin = self.freqs_sin[:seq_len]
        
        print(f'  RoPE_SinCos freqs: cos={freqs_cos.shape}, sin={freqs_sin.shape}')
        
        # Reshape x for rotation: split into pairs
        # (B, T, n_heads, d_head) -> (B, T, n_heads, d_head//2, 2)
        xq_reshaped = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_reshaped = xk.float().reshape(*xk.shape[:-1], -1, 2)
        
        print(f'  RoPE_SinCos reshaped: xq={xq_reshaped.shape}, xk={xk_reshaped.shape}')
        
        # Extract even and odd elements: each has shape (B, T, n_heads, d_head//2)
        xq0, xq1 = xq_reshaped[..., 0], xq_reshaped[..., 1]
        xk0, xk1 = xk_reshaped[..., 0], xk_reshaped[..., 1]
        
        # Reshape freqs for broadcasting: (T, d_head//2) -> (1, T, 1, d_head//2)
        freqs_cos = freqs_cos.view(1, seq_len, 1, -1)
        freqs_sin = freqs_sin.view(1, seq_len, 1, -1)
        
        print(f'  RoPE_SinCos freqs broadcast: cos={freqs_cos.shape}, sin={freqs_sin.shape}')
        
        # Apply rotation: [cos -sin; sin cos] @ [x0; x1]
        # All outputs have shape (B, T, n_heads, d_head//2)
        xq_out0 = xq0 * freqs_cos - xq1 * freqs_sin
        xq_out1 = xq0 * freqs_sin + xq1 * freqs_cos
        xk_out0 = xk0 * freqs_cos - xk1 * freqs_sin
        xk_out1 = xk0 * freqs_sin + xk1 * freqs_cos
        
        # Combine back and flatten: (B, T, n_heads, d_head//2, 2) -> (B, T, n_heads, d_head)
        xq_out = torch.stack([xq_out0, xq_out1], dim=-1).flatten(-2)
        xk_out = torch.stack([xk_out0, xk_out1], dim=-1).flatten(-2)
        
        print(f'  RoPE_SinCos output: xq={xq_out.shape}, xk={xk_out.shape}')
        
        return xq_out.type_as(xq), xk_out.type_as(xk)

# RoPE using complex number multiplication (faster than sin/cos)
class RoPE_Complex(nn.Module):
    """RoPE using complex number multiplication"""
    def __init__(self, dim, max_seq_len, theta=10000.0):
        super().__init__()
        self.dim = dim
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(max_seq_len)
        freqs = torch.outer(t, freqs).float()
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer('freqs_cis', freqs_cis)
    
    def forward(self, xq, xk):
        # Input: xq, xk have shape (B, T, n_heads, d_head)
        B, T, n_heads, d_head = xq.shape
        seq_len = T
        
        # Convert to complex: (B, T, n_heads, d_head) -> (B, T, n_heads, d_head//2, 2) -> (B, T, n_heads, d_head//2) complex
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        # Get frequencies for this sequence length and reshape for broadcasting
        # freqs_cis: (T, d_head//2) complex
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Reshape for broadcasting to (1, T, 1, d_head//2)
        shape = [d if i == 1 or i == xq_.ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
        freqs_cis = freqs_cis.view(*shape)
        
        # Apply rotation: multiply complex numbers
        # xq_ * freqs_cis: (B, T, n_heads, d_head//2) complex
        # view_as_real: (B, T, n_heads, d_head//2, 2)
        # flatten(3): (B, T, n_heads, d_head)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        
        return xq_out.type_as(xq), xk_out.type_as(xk)
    
    def forward_debug(self, xq, xk):
        # Input: xq, xk have shape (B, T, n_heads, d_head)
        B, T, n_heads, d_head = xq.shape
        seq_len = T
        
        print(f'  RoPE_Complex input: xq={xq.shape}, xk={xk.shape}')
        
        # Convert to complex: (B, T, n_heads, d_head) -> (B, T, n_heads, d_head//2, 2) -> (B, T, n_heads, d_head//2) complex
        xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
        xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
        
        print(f'  RoPE_Complex as complex: xq={xq_.shape}, xk={xk_.shape}')
        
        # Get frequencies for this sequence length and reshape for broadcasting
        # freqs_cis: (T, d_head//2) complex
        freqs_cis = self.freqs_cis[:seq_len]
        
        # Reshape for broadcasting to (1, T, 1, d_head//2)
        shape = [d if i == 1 or i == xq_.ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
        freqs_cis = freqs_cis.view(*shape)
        
        print(f'  RoPE_Complex freqs_cis broadcast: {freqs_cis.shape}')
        
        # Apply rotation: multiply complex numbers
        # xq_ * freqs_cis: (B, T, n_heads, d_head//2) complex
        # view_as_real: (B, T, n_heads, d_head//2, 2)
        # flatten(3): (B, T, n_heads, d_head)
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
        
        print(f'  RoPE_Complex output: xq={xq_out.shape}, xk={xk_out.shape}')
        
        return xq_out.type_as(xq), xk_out.type_as(xk)

# Sliding window attention layer (with full masked attention)
class SlidingWindowAttention_Full(nn.Module):
    """Sliding window attention with full masked attention"""
    def __init__(self, d_model, n_heads, window_size, max_seq_len, dropout_rate=0.1, use_complex_number_rope=False):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.use_complex_number_rope = use_complex_number_rope

        assert self.d_head % 2 == 0, f'd_head must be even for RoPE, but got {self.d_head}'
        assert self.max_seq_len > window_size, f'max_seq_len must be greater than window_size, but got {self.max_seq_len} <= {window_size}'
        assert self.d_head * n_heads == d_model, f'd_head * n_heads must be equal to d_model, but got {self.d_head * n_heads} != {d_model}'
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

        # Use RoPE_SinCos for better torch.compile compatibility (avoids complex ops)
        if self.use_complex_number_rope:
            self.rope = RoPE_Complex(self.d_head, self.max_seq_len)
        else:
            self.rope = RoPE_SinCos(self.d_head, self.max_seq_len)

    def forward(self, x):
        # Input: x has shape (B, T, C) where C = d_model
        B, T, C = x.shape
        
        # QKV projections and split into heads: (B, T, C) -> (B, T, n_heads, d_head)
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head)
        
        # Apply RoPE: (B, T, n_heads, d_head) -> (B, T, n_heads, d_head)
        q, k = self.rope(q, k)
        
        # Transpose for attention: (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Create sliding window mask: (T, T)
        # True means masked out (will be filled with -inf)
        mask = torch.ones((T, T), dtype=torch.bool, device=x.device)
        mask = torch.triu(mask, diagonal=1)  # Causal mask: can't attend to future
        for i in range(T):
            if i >= self.window_size:
                # Can't attend to positions before (i - window_size + 1)
                mask[i, :i-self.window_size+1] = True
        
        # Compute attention scores: (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        # Apply mask and softmax: (B, n_heads, T, T)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values: (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        out = attn @ v
        
        # Transpose and reshape: (B, n_heads, T, d_head) -> (B, T, n_heads, d_head) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection: (B, T, C) -> (B, T, C)
        out = self.wo(out)
        
        return out
    
    def forward_debug(self, x):
        # Input: x has shape (B, T, C) where C = d_model
        B, T, C = x.shape
        
        print(f'  SlidingWindowAttention_Full input: x={x.shape}')
        
        # QKV projections and split into heads: (B, T, C) -> (B, T, n_heads, d_head)
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head)
        
        print(f'  SlidingWindowAttention_Full after projection: q={q.shape}, k={k.shape}, v={v.shape}')
        
        # Apply RoPE: (B, T, n_heads, d_head) -> (B, T, n_heads, d_head)
        q, k = self.rope.forward_debug(q, k)
        
        # Transpose for attention: (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        print(f'  SlidingWindowAttention_Full after transpose: q={q.shape}, k={k.shape}, v={v.shape}')
        
        # Create sliding window mask: (T, T)
        # True means masked out (will be filled with -inf)
        mask = torch.ones((T, T), dtype=torch.bool, device=x.device)
        mask = torch.triu(mask, diagonal=1)  # Causal mask: can't attend to future
        for i in range(T):
            if i >= self.window_size:
                # Can't attend to positions before (i - window_size + 1)
                mask[i, :i-self.window_size+1] = True
        
        print(f'  SlidingWindowAttention_Full mask: {mask.shape}')
        
        # Compute attention scores: (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        
        print(f'  SlidingWindowAttention_Full scores: {scores.shape}')
        
        # Apply mask and softmax: (B, n_heads, T, T)
        scores = scores.masked_fill(mask, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        print(f'  SlidingWindowAttention_Full attn: {attn.shape}')
        
        # Apply attention to values: (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        out = attn @ v
        
        print(f'  SlidingWindowAttention_Full out after attn: {out.shape}')
        
        # Transpose and reshape: (B, n_heads, T, d_head) -> (B, T, n_heads, d_head) -> (B, T, C)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection: (B, T, C) -> (B, T, C)
        out = self.wo(out)
        
        print(f'  SlidingWindowAttention_Full final output: {out.shape}')
        
        return out

# Sliding window attention layer (with blocked processing for memory efficiency)
class SlidingWindowAttention_Blocked(nn.Module):
    """Sliding window attention with blocked processing for memory efficiency"""
    def __init__(self, d_model, n_heads, window_size, max_seq_len, block_size=None, dropout_rate=0.1, use_complex_number_rope=False):
        super().__init__()
        assert d_model % n_heads == 0, f'd_model must be divisible by n_heads, but got {d_model} % {n_heads} != 0'
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.block_size = block_size or window_size * 4  # Default: 4x window size
        self.dropout_rate = dropout_rate
        self.use_complex_number_rope = use_complex_number_rope

        assert self.d_head % 2 == 0, f'd_head must be even for RoPE, but got {self.d_head} % 2 != 0'
        
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(self.dropout_rate)

        # Use RoPE_SinCos for better torch.compile compatibility (avoids complex ops)
        if self.use_complex_number_rope:
            self.rope = RoPE_Complex(self.d_head, self.max_seq_len)
        else:
            self.rope = RoPE_SinCos(self.d_head, self.max_seq_len)

    
    def forward(self, x):
        # Input: x has shape (B, T, C) where C = d_model
        B, T, C = x.shape
        
        # QKV projections and split into heads: (B, T, C) -> (B, T, n_heads, d_head)
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head)
        
        # Apply RoPE: (B, T, n_heads, d_head) -> (B, T, n_heads, d_head)
        q, k = self.rope(q, k)
        
        # Transpose for attention: (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Determine effective block size
        block_size = min(self.block_size, T)
        
        # Output accumulator: (B, n_heads, T, d_head)
        output = torch.zeros_like(q)
        
        # Process sequence in blocks
        for block_start in range(0, T, block_size):
            block_end = min(block_start + block_size, T)
            
            # Context window: queries need to see up to window_size positions back
            context_start = max(0, block_start - self.window_size + 1)
            context_end = block_end
            
            # Extract query block and context K,V
            # q_block: (B, n_heads, block_len, d_head)
            # k_context, v_context: (B, n_heads, context_len, d_head)
            q_block = q[:, :, block_start:block_end, :]
            k_context = k[:, :, context_start:context_end, :]
            v_context = v[:, :, context_start:context_end, :]
            
            # Compute attention scores: (B, n_heads, block_len, d_head) @ (B, n_heads, d_head, context_len) -> (B, n_heads, block_len, context_len)
            scores = (q_block @ k_context.transpose(-2, -1)) / math.sqrt(self.d_head)
            
            # Create per-position mask for this block: (block_len, context_len)
            block_len = block_end - block_start
            context_len = context_end - context_start
            mask = torch.zeros(block_len, context_len, device=x.device, dtype=torch.bool)
            
            for i in range(block_len):
                query_pos = block_start + i
                # Position i can attend to [max(0, query_pos-window_size+1), query_pos]
                attend_start = max(0, query_pos - self.window_size + 1) - context_start
                attend_end = query_pos + 1 - context_start
                mask[i, attend_start:attend_end] = True
            
            # Apply mask and softmax: (B, n_heads, block_len, context_len)
            scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention: (B, n_heads, block_len, context_len) @ (B, n_heads, context_len, d_head) -> (B, n_heads, block_len, d_head)
            output[:, :, block_start:block_end, :] = attn @ v_context
        
        # Transpose back and reshape: (B, n_heads, T, d_head) -> (B, T, n_heads, d_head) -> (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection: (B, T, C) -> (B, T, C)
        output = self.wo(output)
        
        return output
    
    def forward_debug(self, x):
        # Input: x has shape (B, T, C) where C = d_model
        B, T, C = x.shape
        
        print(f'  SlidingWindowAttention_Blocked input: x={x.shape}')
        
        # QKV projections and split into heads: (B, T, C) -> (B, T, n_heads, d_head)
        q = self.wq(x).view(B, T, self.n_heads, self.d_head)
        k = self.wk(x).view(B, T, self.n_heads, self.d_head)
        v = self.wv(x).view(B, T, self.n_heads, self.d_head)
        
        print(f'  SlidingWindowAttention_Blocked after projection: q={q.shape}, k={k.shape}, v={v.shape}')
        
        # Apply RoPE: (B, T, n_heads, d_head) -> (B, T, n_heads, d_head)
        q, k = self.rope.forward_debug(q, k)
        
        # Transpose for attention: (B, T, n_heads, d_head) -> (B, n_heads, T, d_head)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        print(f'  SlidingWindowAttention_Blocked after transpose: q={q.shape}, k={k.shape}, v={v.shape}')
        
        # Determine effective block size
        block_size = min(self.block_size, T)
        
        print(f'  SlidingWindowAttention_Blocked block_size={block_size}')
        
        # Output accumulator: (B, n_heads, T, d_head)
        output = torch.zeros_like(q)
        
        # Process sequence in blocks
        for block_start in range(0, T, block_size):
            block_end = min(block_start + block_size, T)
            
            # Context window: queries need to see up to window_size positions back
            context_start = max(0, block_start - self.window_size + 1)
            context_end = block_end
            
            # Extract query block and context K,V
            # q_block: (B, n_heads, block_len, d_head)
            # k_context, v_context: (B, n_heads, context_len, d_head)
            q_block = q[:, :, block_start:block_end, :]
            k_context = k[:, :, context_start:context_end, :]
            v_context = v[:, :, context_start:context_end, :]
            
            if block_start == 0:  # Only print for first block
                print(f'  SlidingWindowAttention_Blocked block[{block_start}:{block_end}]: q_block={q_block.shape}')
                print(f'  SlidingWindowAttention_Blocked context[{context_start}:{context_end}]: k_context={k_context.shape}, v_context={v_context.shape}')
            
            # Compute attention scores: (B, n_heads, block_len, d_head) @ (B, n_heads, d_head, context_len) -> (B, n_heads, block_len, context_len)
            scores = (q_block @ k_context.transpose(-2, -1)) / math.sqrt(self.d_head)
            
            # Create per-position mask for this block: (block_len, context_len)
            block_len = block_end - block_start
            context_len = context_end - context_start
            mask = torch.zeros(block_len, context_len, device=x.device, dtype=torch.bool)
            
            for i in range(block_len):
                query_pos = block_start + i
                # Position i can attend to [max(0, query_pos-window_size+1), query_pos]
                attend_start = max(0, query_pos - self.window_size + 1) - context_start
                attend_end = query_pos + 1 - context_start
                mask[i, attend_start:attend_end] = True
            
            if block_start == 0:  # Only print for first block
                print(f'  SlidingWindowAttention_Blocked mask: {mask.shape}')
            
            # Apply mask and softmax: (B, n_heads, block_len, context_len)
            scores = scores.masked_fill(~mask[None, None, :, :], float('-inf'))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            # Apply attention: (B, n_heads, block_len, context_len) @ (B, n_heads, context_len, d_head) -> (B, n_heads, block_len, d_head)
            output[:, :, block_start:block_end, :] = attn @ v_context
        
        print(f'  SlidingWindowAttention_Blocked output before transpose: {output.shape}')
        
        # Transpose back and reshape: (B, n_heads, T, d_head) -> (B, T, n_heads, d_head) -> (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        
        # Output projection: (B, T, C) -> (B, T, C)
        output = self.wo(output)
        
        print(f'  SlidingWindowAttention_Blocked final output: {output.shape}')
        
        return output

# SwiGLU FFN implementation
class SwiGLU_FFN(nn.Module):
    """two layer FFN with SwiGLU activation function: swish(W1(x)) * W2(x)"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8/3) # Standard scaling for SwiGLU (to keep the number of parameters the same as GELU)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, dim, bias=False)
        
    def forward(self, x):
        # Input: x has shape (B, T, dim)
        
        # w1(x) and w2(x): (B, T, dim) -> (B, T, hidden_dim)
        w1_out = self.w1(x)
        w2_out = self.w2(x)
        
        # SiLU activation and element-wise multiplication: (B, T, hidden_dim)
        activated = F.silu(w1_out) * w2_out
        
        # w3: (B, T, hidden_dim) -> (B, T, dim)
        output = self.w3(activated)
        
        return output
    
    def forward_debug(self, x):
        # Input: x has shape (B, T, dim)
        print(f'  SwiGLU_FFN input: x={x.shape}')
        
        # w1(x) and w2(x): (B, T, dim) -> (B, T, hidden_dim)
        w1_out = self.w1(x)
        w2_out = self.w2(x)
        
        print(f'  SwiGLU_FFN w1/w2 output: {w1_out.shape}')
        
        # SiLU activation and element-wise multiplication: (B, T, hidden_dim)
        activated = F.silu(w1_out) * w2_out
        
        # w3: (B, T, hidden_dim) -> (B, T, dim)
        output = self.w3(activated)
        
        print(f'  SwiGLU_FFN output: {output.shape}')
        
        return output

# GELU FFN implementation for transformer block
class GELU_FFN(nn.Module):
    """two layer FFN with GELU activation function"""
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 4) # Standard scaling for GELU
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.gelu = nn.GELU()
        
    def forward(self, x):
        # Input: x has shape (B, T, dim)
        
        # w1: (B, T, dim) -> (B, T, hidden_dim)
        w1_out = self.w1(x)
        
        # GELU activation: (B, T, hidden_dim)
        activated = self.gelu(w1_out)
        
        # w2: (B, T, hidden_dim) -> (B, T, dim)
        output = self.w2(activated)
        
        return output
    
    def forward_debug(self, x):
        # Input: x has shape (B, T, dim)
        print(f'  GELU_FFN input: x={x.shape}')
        
        # w1: (B, T, dim) -> (B, T, hidden_dim)
        w1_out = self.w1(x)
        
        print(f'  GELU_FFN w1 output: {w1_out.shape}')
        
        # GELU activation: (B, T, hidden_dim)
        activated = self.gelu(w1_out)
        
        # w2: (B, T, hidden_dim) -> (B, T, dim)
        output = self.w2(activated)
        
        print(f'  GELU_FFN output: {output.shape}')
        
        return output

# Transformer block with blocked sliding window attention
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, window_size, max_seq_len=4096, dropout_rate=0.1, 
                 ffn_type='gelu', ffn_hidden_dim=None, SWA_type='full', blocked_block_size=None, 
                 use_complex_number_rope=False, norm_type='layer_norm'):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        self.ffn_type = ffn_type
        self.ffn_hidden_dim = ffn_hidden_dim
        self.SWA_type = SWA_type
        self.blocked_block_size = blocked_block_size
        self.use_complex_number_rope = use_complex_number_rope
        self.norm_type = norm_type

        if self.SWA_type == 'full':
            self.attn = SlidingWindowAttention_Full(self.d_model, self.n_heads, self.window_size, self.max_seq_len, 
                                                    dropout_rate=self.dropout_rate, use_complex_number_rope=self.use_complex_number_rope)
        elif self.SWA_type == 'blocked':
            self.attn = SlidingWindowAttention_Blocked(self.d_model, self.n_heads, self.window_size, self.max_seq_len, 
                                                       block_size=self.blocked_block_size, dropout_rate=self.dropout_rate, 
                                                       use_complex_number_rope=self.use_complex_number_rope)
        else:
            raise ValueError(f"Unsupported SWA_type: {self.SWA_type}. Choose 'full' or 'blocked'.")

        # Choose normalization type
        if self.norm_type in ['layernorm', 'layer_norm']:
            self.ln1 = nn.LayerNorm(self.d_model)
            self.ln2 = nn.LayerNorm(self.d_model)
        elif self.norm_type in ['rmsnorm', 'rms_norm']:
            self.ln1 = nn.RMSNorm(self.d_model)
            self.ln2 = nn.RMSNorm(self.d_model)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}. Choose 'layer_norm' or 'rms_norm'.")
        
        # Choose FFN type
        if self.ffn_type == 'swiglu':
            self.mlp = SwiGLU_FFN(self.d_model, self.ffn_hidden_dim)
        elif self.ffn_type == 'gelu':
            self.mlp = GELU_FFN(self.d_model, self.ffn_hidden_dim)
        else:
            raise ValueError(f"Unsupported ffn_type: {self.ffn_type}. Choose 'gelu' or 'swiglu'.")
            
        self.dropout = nn.Dropout(self.dropout_rate)
        
    def forward(self, x):
        # Input: x has shape (B, T, d_model)
        
        # Self-attention with pre-norm and residual connection
        # ln1: (B, T, d_model) -> (B, T, d_model)
        x_norm = self.ln1(x)
        
        # attn: (B, T, d_model) -> (B, T, d_model)
        attn_out = self.attn(x_norm)
        
        # Residual connection: (B, T, d_model)
        x = x + attn_out
        
        # FFN with pre-norm and residual connection
        # ln2: (B, T, d_model) -> (B, T, d_model)
        x_norm = self.ln2(x)
        
        # mlp: (B, T, d_model) -> (B, T, d_model)
        mlp_out = self.mlp(x_norm)
        
        # Dropout and residual connection: (B, T, d_model)
        x = x + self.dropout(mlp_out)
        
        return x
    
    def forward_debug(self, x):
        # Input: x has shape (B, T, d_model)
        print(f'TransformerBlock input: x={x.shape}')
        
        # Self-attention with pre-norm and residual connection
        # ln1: (B, T, d_model) -> (B, T, d_model)
        x_norm = self.ln1(x)
        
        print(f'TransformerBlock after ln1: x_norm={x_norm.shape}')
        
        # attn: (B, T, d_model) -> (B, T, d_model)
        attn_out = self.attn.forward_debug(x_norm)
        
        print(f'TransformerBlock after attn: attn_out={attn_out.shape}')
        
        # Residual connection: (B, T, d_model)
        x = x + attn_out
        
        print(f'TransformerBlock after attn residual: x={x.shape}')
        
        # FFN with pre-norm and residual connection
        # ln2: (B, T, d_model) -> (B, T, d_model)
        x_norm = self.ln2(x)
        
        print(f'TransformerBlock after ln2: x_norm={x_norm.shape}')
        
        # mlp: (B, T, d_model) -> (B, T, d_model)
        mlp_out = self.mlp.forward_debug(x_norm)
        
        print(f'TransformerBlock after mlp: mlp_out={mlp_out.shape}')
        
        # Dropout and residual connection: (B, T, d_model)
        x = x + self.dropout(mlp_out)
        
        print(f'TransformerBlock output: x={x.shape}')
        
        return x

# Transformer Backbone for processing (B, C_in, S, T) inputs
class Transformer_Backbone(nn.Module):
    def __init__(self, in_channels, in_spatial_dim, first_layer_temporal_kernel_size=1, first_norm_type='RMSNorm', first_nonlinearity_str='lgelu', first_leaky_slope=0.2,
                 d_model=64, n_heads=8, num_layers=4, window_size=32, max_seq_len=8192, dropout_rate=0.1, ffn_type='swiglu', ffn_hidden_dim=None, 
                 SWA_type='blocked', SWA_block_size=None, use_complex_number_rope=False, norm_type='rms_norm'):
        super().__init__()
        
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
        self.out_channels = d_model
        
        # First layer: spatio-temporal convolution
        # (B, C_in, in_spatial_dim, T) -> (B, d_model, 1, T)
        self.first_conv = SpatioTemporalCausalConv2D(in_channels, out_channels=self.d_model, spatial_kernel_size=self.in_spatial_dim, 
                                                     spatial_padding='valid', temporal_kernel_size=first_layer_temporal_kernel_size,
                                                     bottleneck_dim=None, temp_conv_first=False, convert_out_channels_to_spatial=True, 
                                                     post_conversion_out_channels=self.d_model)
        self.first_norm = Normalization(self.d_model, norm_type=self.first_norm_type)
        self.first_nlin = Nonlinearity(self.first_nonlinearity_str, self.first_leaky_slope)
        
        # Transformer blocks
        for i in range(self.num_layers):
            self.add_module(f"block{i}", TransformerBlock(d_model=self.d_model, n_heads=self.n_heads, window_size=self.window_size,
                                                          max_seq_len=self.max_seq_len, dropout_rate=self.dropout_rate, ffn_type=self.ffn_type,
                                                          ffn_hidden_dim=self.ffn_hidden_dim, SWA_type=self.SWA_type, blocked_block_size=self.SWA_block_size,
                                                          use_complex_number_rope=self.use_complex_number_rope, norm_type=self.norm_type))
        
        # Final normalization (pre-norm architecture usually has final norm)
        if self.norm_type in ['layernorm', 'layer_norm']:
            self.final_norm = nn.LayerNorm(self.d_model)
        elif self.norm_type in ['rmsnorm', 'rms_norm']:
            self.final_norm = nn.RMSNorm(self.d_model)
        else:
            raise ValueError(f"Unsupported norm_type: {self.norm_type}")
        
        # Output dimensions (will be set properly after first forward pass or explicit call)
        self.out_spatial_dim = 1
        self.out_temporal_dim = None
        
        self.update_short_name()
    
    def forward(self, x):
        # Input: x has shape (B, C_in, in_spatial_dim, T)
        B, C_in, S, T = x.shape
        
        # First conv layer with norm and nonlinearity: (B, C_in, S, T) -> (B, d_model, 1, T)
        x = self.first_nlin(self.first_norm(self.first_conv(x)))
        
        # Reshape for transformer: (B, d_model, 1, T) -> (B, T, d_model)
        x = x.squeeze(2).transpose(1, 2)
        
        # Apply transformer blocks
        for i in range(self.num_layers):
            x = self.__getattr__(f"block{i}")(x)
        
        # Final normalization
        x = self.final_norm(x)
        
        # Reshape back: (B, T, d_model) -> (B, d_model, 1, T)
        x = x.transpose(1, 2).unsqueeze(2)
        
        return x
    
    def forward_debug(self, x):
        # Input: x has shape (B, C_in, in_spatial_dim, T)
        B, C_in, S, T = x.shape
        print(f'Transformer_Backbone:            input shape: {x.shape}')
        
        # First conv layer with norm and nonlinearity: (B, C_in, S, T) -> (B, d_model, 1, T)
        x = self.first_nlin(self.first_norm(self.first_conv(x)))
        print(f'Transformer_Backbone: 1st layer output shape: {x.shape}')
        
        # Reshape for transformer: (B, d_model, 1, T) -> (B, T, d_model)
        x = x.squeeze(2).transpose(1, 2)
        print(f'Transformer_Backbone:  after reshape for TF: {x.shape}')
        
        # Apply transformer blocks
        for i in range(self.num_layers):
            x = self.__getattr__(f"block{i}")(x)
            print(f'Transformer_Backbone:         block {i} output: {x.shape}')
        
        # Final normalization
        x = self.final_norm(x)
        print(f'Transformer_Backbone:      after final_norm: {x.shape}')
        
        # Reshape back: (B, T, d_model) -> (B, d_model, 1, T)
        x = x.transpose(1, 2).unsqueeze(2)
        print(f'Transformer_Backbone:           final output: {x.shape}')
        
        return x
    
    def update_short_name(self):
        if self.SWA_type == 'full':
            total_temporal_dependence = self.max_seq_len
        else:
            total_temporal_dependence = self.first_layer_temporal_kernel_size - 1
            total_temporal_dependence += self.num_layers * (self.window_size - 1)
        
        # Calculate number of effective parameters
        num_params_first_conv = self.first_conv.get_num_effective_params()
        num_params_transformer = sum(p.numel() for i in range(self.num_layers) for p in self.__getattr__(f"block{i}").parameters())
        num_params_final_norm = sum(p.numel() for p in self.final_norm.parameters())
        
        self.num_params_effective = num_params_first_conv + num_params_transformer + num_params_final_norm
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_params_thousands = self.num_params_effective / 1e3
        self.num_params_millions = self.num_params_effective / 1e6
        
        # Create parameter description
        if self.num_params_millions < 10:
            param_description_str = f"params_{self.num_params_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_params_millions:.0f}M"
        
        self.short_name = f"Transformer_backbone_D_{self.num_layers}_W_{self.d_model}_H_{self.n_heads}_T_{total_temporal_dependence}_{param_description_str}"
    
    def set_output_dims(self, x, print_output_dims=False):
        # Set output dimensions based on input
        # Note: Temporal dimension is preserved in causal convolution
        y = self.forward(x)
        B, C, S, T = y.shape
        
        self.out_channels = C
        self.out_spatial_dim = S
        self.out_temporal_dim = T
        
        if print_output_dims:
            print(f'Transformer_Backbone: just set output dims')
            print(f'Transformer_Backbone: input shape: {x.shape}')
            print(f'Transformer_Backbone: output shape: {y.shape}')


class SingleNeuronTCN_Heads(nn.Module):
    def __init__(self, backbone, 
                 head_prefix_names=['spikes', 'soma', 'nexus', 'DVTs'],
                 head_out_channels=[1, 1, 1, 639], 
                 head_convert_out_ch_to_sp=[False, False, False, True],
                 head_spatial_kernel_sizes=None):
        super().__init__()

        self.backbone = backbone
        in_C = self.backbone.out_channels
        in_S_dim = self.backbone.out_spatial_dim
        self.head_names = [name_prefix + '_head' for name_prefix in head_prefix_names]
        self.n_heads = len(self.head_names)
        self.head_out_channels = head_out_channels
        self.head_convert_out_ch_to_sp = head_convert_out_ch_to_sp
        self.head_spatial_kernel_sizes = [in_S_dim] * self.n_heads if head_spatial_kernel_sizes is None else head_spatial_kernel_sizes
        self.head_in_channels = in_C
        self.head_in_S_dim = in_S_dim

        # self.pre_head_bn = nn.BatchNorm2d(in_C)
        for i, head_name in enumerate(self.head_names):
            self.add_module(head_name, SpatioTemporalCausalConv2D(in_C, out_channels=self.head_out_channels[i], 
                                                                  spatial_kernel_size=self.head_spatial_kernel_sizes[i], 
                                                                  temporal_kernel_size=1, bottleneck_dim=None,
                                                                  spatial_padding='valid', spatial_dilation=1, temporal_dilation=1, temp_conv_first=False,
                                                                  convert_out_channels_to_spatial=self.head_convert_out_ch_to_sp[i],
                                                                  post_conversion_out_channels=self.head_out_channels[i]))
            
            # for the spikes head, set the bias to be negative 2.0 to start with
            if 'spike' in head_name:
                print(f'setting "{head_name}" bias to -2.0')
                # because we've set temp_conv_first=False, the bias is the temporal conv bias (the last layer)
                self.__getattr__(head_name).temporal_conv.bias.data = -2.0 * torch.ones_like(self.__getattr__(head_name).temporal_conv.bias.data)

        self.update_short_name()
        self.metadata = {} # dictionary to hold metadata about the model's training and performance

    # the forward function only output the heads that are specified in the heads_to_use list
    def forward(self, x):
        x = self.backbone(x) # (B, in_C, in_S_dim, T)
        # x = self.pre_head_bn(x)

        head_outputs = []
        for head_name in self.head_names:
            head_outputs.append(self.__getattr__(head_name)(x))

        return head_outputs

    def forward_debug(self, x):
        
        print(f'TCN: input shape: {x.shape}')
        x = self.backbone.forward_debug(x)
        # x = self.pre_head_bn(x)
        print(f'TCN: backbone output shape: {x.shape}')

        head_outputs = []
        for head_name in self.head_names:
            head_outputs.append(self.__getattr__(head_name)(x))
            print(f'TCN: {head_name} output shape: {head_outputs[-1].shape}')

        return head_outputs

    def update_short_name(self):

        # assess number of effective params per head
        self.effective_params_per_head = {}
        for head_name in self.head_names:
            self.effective_params_per_head[head_name] = self.__getattr__(head_name).get_num_effective_params()

        # params description string
        self.num_params_effective = sum(self.effective_params_per_head.values()) + self.backbone.num_params_effective
        self.num_params_official = sum(p.numel() for p in self.parameters())
        self.num_parmas_thousands = self.num_params_effective / 1e3
        self.num_parmas_millions = self.num_params_effective / 1e6
        if self.num_parmas_millions < 10:
            param_description_str = f"params_{self.num_parmas_thousands:.0f}K"
        else:
            param_description_str = f"params_{self.num_parmas_millions:.0f}M"

        # the short name of the model
        self.short_name = f"backbone_{self.backbone.short_name}_heads_{self.n_heads}_{param_description_str}"

    def save_model(self, path):
        directory = os.path.dirname(path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        state = {
            'model_state_dict': self.state_dict(),
            'head_names': self.head_names,
            'n_heads': self.n_heads,
            'head_out_channels': self.head_out_channels,
            'head_convert_out_ch_to_sp': self.head_convert_out_ch_to_sp,
            'head_spatial_kernel_sizes': self.head_spatial_kernel_sizes,
            'head_in_channels': self.head_in_channels,
            'head_in_S_dim': self.head_in_S_dim,
            'metadata': self.metadata,
        }
        torch.save(state, path)
        print(f'Saved model to "{path}"')

    @classmethod
    def load_model(cls, path):
        state = torch.load(path, weights_only=False)
        print(f'Loading model from "{path}"')
        model = cls(
            # num_keypoints=state['num_keypoints'],
            # num_classes_per_coordinate=state['num_classes_per_coordinate'],
            # efficientnet_variant=state['efficientnet_variant'],
            # num_hidden_dims=state['num_hidden_dims'],
            # lrelu_negative_slope=state['lrelu_negative_slope'],
            # dropout_prob=state['dropout_prob']
        )
        model.load_state_dict(state['model_state_dict'])
        return model


class SpatioTemporalCausalConv2D(nn.Module):
    '''SpatioTemporalCausalConv2D performs spatiotemporal convolution with a single conv2d layer. 
    Maintains the same interface as SpatioTemporalCausalConv2D_low_rank for backward compatibility. 
    Spatial dimension is "height", time dimension is "width", i.e. x.shape = (B, C, S, T)'''

    def __init__(self, in_channels, out_channels, spatial_kernel_size=3, temporal_kernel_size=7, bottleneck_dim=None, 
                 spatial_padding='valid', spatial_dilation=1, temporal_dilation=1, spatial_stride=1,
                 temp_conv_first=False, convert_out_channels_to_spatial=False, post_conversion_out_channels=1):
        super().__init__()

        self.in_channels = in_channels  
        self.out_channels = out_channels
        self.spatial_kernel_size = spatial_kernel_size
        self.temporal_kernel_size = temporal_kernel_size
        self.spatial_dilation = spatial_dilation
        self.temporal_dilation = temporal_dilation
        self.spatial_stride = spatial_stride
        self.temp_conv_first = temp_conv_first  # kept for interface compatibility but not used
        self.convert_out_channels_to_spatial = convert_out_channels_to_spatial
        self.post_conversion_out_channels = post_conversion_out_channels
        self.bottleneck_dim = bottleneck_dim  # kept for interface compatibility but not used

        self.temporal_padding = (self.temporal_kernel_size - 1) * self.temporal_dilation

        if spatial_padding == 'same':
            self.spatial_padding = (self.spatial_kernel_size - 1) * self.spatial_dilation // 2
            if self.spatial_kernel_size % 2 == 0:
                print("Warning: 'spatial_kernel_size' is even, so 'spatial_padding' is not exactly same. please use odd kernel sizes!")
        elif spatial_padding == 'valid':
            self.spatial_padding = 0
        else:
            self.spatial_padding = spatial_padding

        # Single conv2d that handles both spatial and temporal dimensions
        self.temporal_conv = nn.Conv2d(self.in_channels, self.out_channels, 
                                       kernel_size=(self.spatial_kernel_size, self.temporal_kernel_size),
                                       padding=(self.spatial_padding, self.temporal_padding),
                                       dilation=(self.spatial_dilation, self.temporal_dilation),
                                       stride=(self.spatial_stride, 1))

    def forward(self, x):
        x = self.temporal_conv(x)

        if self.temporal_padding > 0:
            x = x[:, :, :, :-self.temporal_padding]

        if self.convert_out_channels_to_spatial:
            B, C, S, T = x.shape
            C_new = self.post_conversion_out_channels
            total_features = C * S
            if total_features % C_new != 0:
                raise ValueError(f"Cannot convert channels to spatial: total features ({total_features}) "
                                 f"must be divisible by post_conversion_out_channels ({C_new}). "
                                 f"Current shape: {x.shape}")
            S_new = total_features // C_new
            x = x.view(B, C_new, S_new, T)

        return x
    
    def get_equivalent_kernel(self):
        # For single conv, the equivalent kernel is just the conv weights
        equivalent_kernel = self.temporal_conv.weight.data.cpu().numpy()
        
        print(f'------------------------------------------')
        print(f'single conv kernel shape: {equivalent_kernel.shape}')
        print(f'------------------------------------------')
        
        return equivalent_kernel

    def get_num_effective_params(self, verbose=0):
        spatial_kernel_size = self.spatial_kernel_size
        temporal_kernel_size = self.temporal_kernel_size
        in_ch = self.in_channels
        out_ch = self.out_channels

        # For single conv, effective params is just the kernel params
        effective_num_params = spatial_kernel_size * temporal_kernel_size * in_ch * out_ch
        official_num_params = sum(p.numel() for p in self.parameters())

        if verbose > 0:
            print(f'num params (official, effective) = ({official_num_params}, {effective_num_params})')

        return effective_num_params

    # a small test to make sure the layer is working as expected
    @staticmethod
    def small_test():

        # first part of the test - test the causal convolution
        spatial_kernel_size = 1
        temporal_kernel_size = 8
        spatial_dilation = 1
        temporal_dilation = 1
        spatial_padding = 'same'
        temp_conv_first = True  # ignored in single conv version

        # Create the SpatioTemporalCausalConv2D layer
        spatio_temporal_conv_layer = SpatioTemporalCausalConv2D(
            in_channels          = 1,
            out_channels         = 1,
            spatial_kernel_size  = spatial_kernel_size, 
            temporal_kernel_size = temporal_kernel_size,
            spatial_dilation     = spatial_dilation, 
            temporal_dilation    = temporal_dilation,
            spatial_padding      = spatial_padding,
            temp_conv_first      = temp_conv_first
        )

        # edit the weights so that it's easier to see what's going on in the plots
        spatio_temporal_conv_layer.temporal_conv.weight.data = 2 * torch.ones_like(spatio_temporal_conv_layer.temporal_conv.weight.data) / temporal_kernel_size
        spatio_temporal_conv_layer.temporal_conv.bias.data = torch.zeros_like(spatio_temporal_conv_layer.temporal_conv.bias.data) + 0.1

        # Create the specified input signal
        S = 4
        T = 512
        x = torch.zeros(1, 1, S, T)

        start_ind = 100
        x[0, 0, :, start_ind] = 1

        start_ind = 150
        x[0, 0, :, start_ind] = 1
        x[0, 0, :, start_ind + 1 * (temporal_kernel_size + 4)] = 0.2 + 1.3 * np.random.rand()
        x[0, 0, :, start_ind + 2 * (temporal_kernel_size + 4)] = 0.2 + 1.3 * np.random.rand()
        x[0, 0, :, start_ind + 3 * (temporal_kernel_size + 4)] = 0.2 + 1.3 * np.random.rand()

        start_ind = 250
        x[0, 0, :, start_ind:(start_ind + temporal_kernel_size)] = torch.rand(1, 1, S, temporal_kernel_size)

        start_ind = 300
        x[0, 0, :, start_ind:(start_ind + 3 * temporal_kernel_size)] = torch.rand(1, 1, S, 3 * temporal_kernel_size)

        # Pass the input signal through the layer
        y = spatio_temporal_conv_layer(x)

        print(f'Input shape: {x.shape}')
        print(f'Output shape: {y.shape}')

        # Convert the tensors to numpy arrays for plotting
        x_np = x.detach().numpy().squeeze()
        y_np = y.detach().numpy().squeeze()

        # Plot the input signals and the corresponding output signals
        plt.figure(figsize=(14, 8))
        for i in range(x_np.shape[0]):
            plt.subplot(x_np.shape[0], 1, i + 1)
            plt.plot(x_np[i], color='blue', label='Input')
            plt.plot(y_np[i], color='orange', label='Output')
            plt.xlim(50,400)
            plt.legend()
        plt.tight_layout()
        plt.show()


        # second part of the test:
        num_reps = 5
        for i in range(num_reps):
            # create a random input signal of random dimensions S,T
            S = 8 * np.random.randint(8, 32)
            T = 32 * np.random.randint(4, 32)
            in_ch = 2 * np.random.randint(1, 8)
            x = torch.rand(1, in_ch, S, T)

            # create random spatial and temporal kernel sizes and dilations and padding
            out_ch = 16 * np.random.randint(1, 4)
            spatial_kernel_size = 2 * np.random.randint(1, 7) + 1
            temporal_kernel_size = np.random.randint(1, 9)
            spatial_dilation = np.random.randint(1, 3)
            temporal_dilation = np.random.randint(1, 3)
            spatial_padding = np.random.choice(['same', 'valid'])
            temp_conv_first = np.random.choice([True, False])
            bottleneck_dim = np.random.randint(4, 32)
            convert_out_channels_to_spatial = np.random.choice([True, False])
            post_conversion_out_channels = 4 * np.random.randint(1, 4)            

            print('--------------------------------------------------')
            print(f'test {i + 1}')
            # create the SpatioTemporalCausalConv2D layer
            spatio_temporal_conv_layer = SpatioTemporalCausalConv2D(
                in_channels                     = in_ch,
                out_channels                    = out_ch, 
                spatial_kernel_size             = spatial_kernel_size, 
                temporal_kernel_size            = temporal_kernel_size,
                bottleneck_dim                  = bottleneck_dim,
                spatial_dilation                = spatial_dilation, 
                temporal_dilation               = temporal_dilation,
                spatial_padding                 = spatial_padding,
                temp_conv_first                 = temp_conv_first,
                convert_out_channels_to_spatial = convert_out_channels_to_spatial,
                post_conversion_out_channels    = post_conversion_out_channels
            )
            
            # pass the input signal through the layer
            y = spatio_temporal_conv_layer(x)

            # print the input and output shapes as well as the kernel sizes and dilations and padding
            print('--------------------------------------------------')
            print(f'Input  shape: {x.shape}')
            print(f'Output shape: {y.shape}')
            print(f'(in_ch -> out_ch) = ({in_ch}, {out_ch})')
            print(f'(spatial ,temporal) kernel size: {spatial_kernel_size, temporal_kernel_size}')
            print(f'(spatial ,temporal) dilations: {spatial_dilation, temporal_dilation}')
            print(f'spatial padding: {spatial_padding}')
            print(f'temp_conv_first: {temp_conv_first} (ignored in single conv)')
            print(f'bottleneck_dim: {bottleneck_dim} (ignored in single conv)')
            print(f'convert_out_channels_to_spatial: {convert_out_channels_to_spatial}')
            print(f'post_conversion_out_channels: {post_conversion_out_channels}')
            spatio_temporal_conv_layer.get_num_effective_params(verbose=1)
            print('--------------------------------------------------')


class SoftPotentialWell(nn.Module):
    def __init__(self, center=0.0, well_half_width=2.0, center_height=0.0, 
                 well_edge_height=0.1, barrier_width=0.3, barrier_height=1.0, 
                 outside_slope=0.1):
        super().__init__()
        self.center = center
        self.well_half_width = well_half_width
        self.center_height = center_height
        self.well_edge_height = well_edge_height
        self.barrier_width = barrier_width
        self.barrier_height = barrier_height
        self.outside_slope = outside_slope
        
        # Precompute constants
        self.barrier_end = self.well_half_width + barrier_width
        
        # Slopes
        self.well_slope = (well_edge_height - center_height) / self.well_half_width if self.well_half_width > 0 else 0
        self.barrier_slope = (barrier_height - well_edge_height) / barrier_width if barrier_width > 0 else 0
    
    def forward(self, x):
        # Distance from center
        d = torch.abs(x - self.center)
        
        # Initialize output
        out = torch.zeros_like(x)
        
        # Region 1: Inside well (|x - center| <= well_width/2)
        well_mask = d <= self.well_half_width
        out[well_mask] = self.center_height + self.well_slope * d[well_mask]
        
        # Region 2: Barrier (well_width/2 < |x - center| <= well_width/2 + barrier_width)
        barrier_mask = (d > self.well_half_width) & (d <= self.barrier_end)
        barrier_dist = d[barrier_mask] - self.well_half_width
        out[barrier_mask] = self.well_edge_height + self.barrier_slope * barrier_dist
        
        # Region 3: Outside (|x - center| > well_width/2 + barrier_width)
        outside_mask = d > self.barrier_end
        outside_dist = d[outside_mask] - self.barrier_end
        out[outside_mask] = self.barrier_height + self.outside_slope * outside_dist
        
        return out

    @staticmethod
    def small_test():

        # Create potential well
        potential_well = SoftPotentialWell(
            center=0.0,
            well_half_width=1.0,
            center_height=0.0,
            well_edge_height=0.1,
            barrier_width=0.5,
            barrier_height=1.0,
            outside_slope=0.1,
        )
        
        # Create input range: -1.5*width to +1.5*width
        x_range = 2.0 * (potential_well.well_half_width + potential_well.barrier_width)
        left_limit = -x_range + potential_well.center
        right_limit = x_range + potential_well.center
        x = torch.linspace(left_limit, right_limit, 1000, requires_grad=True)
        
        # Compute function values
        y = potential_well(x)
        
        # Compute derivative
        dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        
        print(f'x.shape: {x.shape}, y.shape: {y.shape}')

        # Convert to numpy for plotting
        x_np = x.detach().numpy()
        y_np = y.detach().numpy()
        dy_dx_np = dy_dx.detach().numpy()
        
        # Create 2x1 figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5))
        
        # Plot function
        ax1.plot(x_np, y_np, 'b-', linewidth=1)
        ax1.set_title('Soft Potential Well Function')
        ax1.set_ylabel('f(x)')
        ax1.grid(True, alpha=0.3)
        
        # Plot derivative
        ax2.plot(x_np, dy_dx_np, 'r-', linewidth=1)
        ax2.set_title('Derivative')
        ax2.set_xlabel('x')
        ax2.set_ylabel("f'(x)")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


def get_mean_abs_pairwise_correlations(x):
    """
    Compute mean absolute correlation between all pairs of activity traces.
    
    Args:
        x: tensor of shape (batch_size, 1, num_neurons, time_steps)
        
    Returns:
        mean absolute pairwise correlations (scalar tensor)
    """
    # Remove singleton channel dimension
    x = x.squeeze(1)
    batch_size, num_neurons, time_steps = x.shape
    
    # Center the activity traces across time
    x_centered = x - x.mean(dim=-1, keepdim=True)
    
    # Compute standard deviations across time
    std = torch.std(x_centered, dim=-1, keepdim=True)
    
    # Normalize to get correlation components
    x_normalized = x_centered / (std + 1e-8)
    
    # Compute correlation matrix for each batch
    corr_matrix = torch.bmm(x_normalized, x_normalized.transpose(-1, -2)) / time_steps
    # corr_matrix.shape: (batch_size, num_neurons, num_neurons)

    # Zero out diagonal
    diag_mask = torch.eye(num_neurons, device=x.device).unsqueeze(0) # diag_mask.shape: (1, num_neurons, num_neurons)
    corr_matrix = corr_matrix * (1 - diag_mask) 
    
    # Compute mean per batch first (reduces the sum magnitude)
    num_pairs = num_neurons * (num_neurons - 1)
    mean_per_batch = torch.abs(corr_matrix).sum(dim=[1, 2]) / num_pairs
    
    # Then average across batches
    mean_abs_corr = mean_per_batch.mean()
    
    return mean_abs_corr


#%% main to serve as standalone test of the SpatioTemporalCausalConv2D and SpatioTemporalCausalConv2D_low_rank layers and the SingleNeuronTCN model
if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    #%% Basic test of the TwoSidedLeakyGELU layer

    print("----- Testing TwoSidedLeakyGELU -----")

    twosided_leaky_gelu_layer = TwoSidedLeakyGELU()
    twosided_leaky_gelu_layer.small_test()

    #%% Basic test of the SoftPotentialWell layer

    print("----- Testing SoftPotentialWell -----")

    soft_potential_well_layer = SoftPotentialWell()
    soft_potential_well_layer.small_test()

    #%% Basic test of the SpatioTemporalCausalConv2D layer

    print("----- Testing SpatioTemporalCausalConv2D -----")

    spatial_kernel_size   = 1
    temporal_kernel_size  = 8
    spatial_dilation      = 1
    temporal_dilation     = 1
    spatial_padding       = 'same'
    temp_conv_first       = True

    spatio_temporal_conv_layer = SpatioTemporalCausalConv2D(
        in_channels           = 1,
        out_channels          = 1,
        spatial_kernel_size   = spatial_kernel_size,
        temporal_kernel_size  = temporal_kernel_size,
        spatial_dilation      = spatial_dilation,
        temporal_dilation     = temporal_dilation,
        spatial_padding       = spatial_padding,
        temp_conv_first       = temp_conv_first
    )
    
    spatio_temporal_conv_layer.small_test()
    print('--------------------------------------------------')

    #%% generate inputs for the testing of the backbones

    input_channels = 2
    B = 8
    C = input_channels
    S = 639
    T = np.random.choice([512, 768, 1024, 1280])
    x = torch.rand(B, C, S, T)
    print(f'x.shape: {x.shape}')

    #%% basic test for TCN_Backbone

    print("----- Testing TCN_Backbone -----")

    in_spatial_dim                      = S
    first_layer_temporal_kernel_size    = 25
    num_layers_per_block_list           = [   2,    2,    3]
    num_features_per_block_list         = [  16,   32,   64]
    temporal_kernel_size_per_block_list = [   3,    5,    7]
    temporal_dilation_per_block_list    = [   1,    1,    1]
    bottleneck_dim_per_block_list       = [None, None, None]
    nonlinearity_str                    = 'lgelu'
    leaky_slope                         = 0.3

    backbone_tcn = TCN_Backbone(
        in_channels                         = input_channels,
        in_spatial_dim                      = in_spatial_dim,
        first_layer_temporal_kernel_size    = first_layer_temporal_kernel_size,
        num_layers_per_block_list           = num_layers_per_block_list,
        num_features_per_block_list         = num_features_per_block_list,
        temporal_kernel_size_per_block_list = temporal_kernel_size_per_block_list,
        temporal_dilation_per_block_list    = temporal_dilation_per_block_list,
        bottleneck_dim_per_block_list       = bottleneck_dim_per_block_list,
        nonlinearity_str                    = nonlinearity_str,
        leaky_slope                         = leaky_slope
    )

    print('--------------------------------------------------')
    backbone_tcn.set_output_dims(x, print_output_dims=True)
    print('--------------------------------------------------')
    print(f'TCN_Backbone: "{backbone_tcn.short_name}"')
    print('--------------------------------------------------')

    x_tcn = backbone_tcn(x)

    print('--------------------------------------------------')
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {x_tcn.shape}')
    print('--------------------------------------------------')

    model = SingleNeuronTCN_Heads(backbone_tcn)
    print('--------------------------------------------------')
    print(f'full model: "{model.short_name}"')
    print('--------------------------------------------------')
    y_spikes, y_soma, y_nexus, y_DVTs = model.forward_debug(x)
    print('--------------------------------------------------')

    print(f'x shape: {x.shape}')
    print(f'y_spikes shape: {y_spikes.shape}')
    print(f'y_soma shape: {y_soma.shape}')
    print(f'y_nexus shape: {y_nexus.shape}')
    print(f'y_DVTs shape: {y_DVTs.shape}')

    print('--------------------------------------------------')

    #%% basic test for TCN_ResNet_Backbone

    print("----- Testing TCN_ResNet_Backbone -----")

    in_spatial_dim                      = S
    first_layer_temporal_kernel_size    = 35
    num_miniblocks_per_block_list       = [   2,    2,    1]
    num_features_per_block_list         = [  16,   32,   64]
    temporal_kernel_size_per_block_list = [   3,    3,    3]
    temporal_dilation_per_block_list    = [   1,    2,    4]
    bottleneck_dim_per_block_list       = [None, None, None]
    nonlinearity_str                    = 'lgelu'
    leaky_slope                         = 0.3

    backbone_tcn_resnet = TCN_ResNet_Backbone(
        in_channels                         = input_channels,
        in_spatial_dim                      = in_spatial_dim,
        first_layer_temporal_kernel_size    = first_layer_temporal_kernel_size,
        num_miniblocks_per_block_list       = num_miniblocks_per_block_list,
        num_features_per_block_list         = num_features_per_block_list,
        temporal_kernel_size_per_block_list = temporal_kernel_size_per_block_list,
        temporal_dilation_per_block_list    = temporal_dilation_per_block_list,
        bottleneck_dim_per_block_list       = bottleneck_dim_per_block_list,
        nonlinearity_str                    = nonlinearity_str,
        leaky_slope                         = leaky_slope
    )

    backbone_tcn_resnet.set_output_dims(x, print_output_dims=True)
    print(f'TCN_ResNet_Backbone: "{backbone_tcn_resnet.short_name}"')
    x_tcn_resnet = backbone_tcn_resnet(x)
    print('--------------------------------------------------')
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {x_tcn_resnet.shape}')
    print('--------------------------------------------------')

    model = SingleNeuronTCN_Heads(backbone_tcn_resnet)
    print('--------------------------------------------------')
    print(f'full model: "{model.short_name}"')
    print('--------------------------------------------------')
    y_spikes, y_soma, y_nexus, y_DVTs = model.forward_debug(x)
    print('--------------------------------------------------')

    print(f'x shape: {x.shape}')
    print(f'y_spikes shape: {y_spikes.shape}')
    print(f'y_soma shape: {y_soma.shape}')
    print(f'y_nexus shape: {y_nexus.shape}')
    print(f'y_DVTs shape: {y_DVTs.shape}')

    print('--------------------------------------------------')


    #%% 