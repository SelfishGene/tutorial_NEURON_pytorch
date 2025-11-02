#%% Imports

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import platform
import subprocess
import sys

#%% Functions

def get_system_utilization_stats(printable_strings=True):

    # Initialize CPU, GPU stats dictionaries
    gpu_stats = {}
    cpu_stats = {}

    # Get GPU stats using nvidia-smi
    try:
        # Run nvidia-smi command
        output = subprocess.check_output([
            'nvidia-smi', 
            '--query-gpu=name,memory.total,memory.used,utilization.gpu,power.draw,power.limit', 
            '--format=csv,noheader,nounits'
        ], encoding='utf-8')
        
        # Parse the output
        lines = output.strip().split('\n')
        if lines:
            # Assuming we're using the first GPU
            line = lines[0]
            parts = [part.strip() for part in line.split(',')]
            gpu_stats['GPU_name'] = parts[0]

            # Helper function to parse float values
            def parse_float(value):
                return float(value) if value not in ('[N/A]', 'N/A') else None

            gpu_stats['GPU_available_vRAM'] = parse_float(parts[1])  # MiB
            gpu_stats['GPU_vRAM_usage'] = parse_float(parts[2])      # MiB
            gpu_stats['GPU_utilization'] = parse_float(parts[3])     # Percentage
            gpu_stats['GPU_watt_usage'] = parse_float(parts[4])      # Watts
            gpu_stats['GPU_max_watts'] = parse_float(parts[5])       # Watts
        else:
            gpu_stats['error'] = 'No GPU devices found.'
    except Exception as e:
        gpu_stats['error'] = str(e)

    # Get CPU stats
    mem = psutil.virtual_memory()
    uname = platform.uname()

    cpu_stats['CPU_name'] = uname.processor
    cpu_stats['CPU_RAM_usage'] = mem.used / (1024 ** 3)    # GB
    cpu_stats['CPU_available_RAM'] = mem.total / (1024 ** 3)  # GB
    cpu_stats['CPU_utilization'] = psutil.cpu_percent(interval=1)  # Percentage

    # Attempt to get CPU watt usage
    cpu_watt_usage = None
    try:
        if sys.platform.startswith('win'):
            # Windows-specific method using wmi
            import wmi
            c = wmi.WMI(namespace="root\OpenHardwareMonitor")
            sensors = c.Sensor()
            for sensor in sensors:
                if sensor.SensorType == 'Power' and 'cpu package' in sensor.Name.lower():
                    cpu_watt_usage = sensor.Value
                    break
        elif sys.platform.startswith('linux'):
            # Linux-specific method
            try:
                with open('/sys/class/powercap/intel-rapl:0/energy_uj', 'r') as f:
                    energy_uj = int(f.read())
                cpu_watt_usage = energy_uj / 1e6  # Convert microjoules to Watts
            except Exception:
                pass
        # You can add more platform-specific methods here
    except Exception:
        pass
    cpu_stats['CPU_watt_usage'] = cpu_watt_usage

    # Convert to printable strings if requested
    if printable_strings:
        # Convert GPU stats
        new_gpu_stats = {}
        for key, value in gpu_stats.items():
            if key == 'error':
                new_gpu_stats[key] = value
            elif key == 'GPU_name':
                new_gpu_stats[key] = value
            elif key == 'GPU_vRAM_usage':
                if value is not None:
                    new_gpu_stats[key] = f"{value / 1024:.2f} GB"  # Convert MiB to GB
                else:
                    new_gpu_stats[key] = 'N/A'
            elif key == 'GPU_available_vRAM':
                if value is not None:
                    new_gpu_stats[key] = f"{value / 1024:.2f} GB"
                else:
                    new_gpu_stats[key] = 'N/A'
            elif key == 'GPU_watt_usage':
                if value is not None:
                    new_gpu_stats[key] = f"{value:.2f} W"
                else:
                    new_gpu_stats[key] = 'N/A'
            elif key == 'GPU_max_watts':
                if value is not None:
                    new_gpu_stats[key] = f"{value:.2f} W"
                else:
                    new_gpu_stats[key] = 'N/A'
            elif key == 'GPU_utilization':
                if value is not None:
                    new_gpu_stats[key] = f"{value:.1f}%"
                else:
                    new_gpu_stats[key] = 'N/A'
            else:
                new_gpu_stats[key] = str(value)
        gpu_stats = new_gpu_stats

        # Convert CPU stats
        new_cpu_stats = {}
        for key, value in cpu_stats.items():
            if key == 'CPU_name':
                new_cpu_stats[key] = value
            elif key == 'CPU_RAM_usage':
                new_cpu_stats[key] = f"{value:.2f} GB"
            elif key == 'CPU_available_RAM':
                new_cpu_stats[key] = f"{value:.2f} GB"
            elif key == 'CPU_utilization':
                new_cpu_stats[key] = f"{value:.1f}%"
            elif key == 'CPU_watt_usage':
                if value is not None:
                    new_cpu_stats[key] = f"{value:.2f} W"
                else:
                    new_cpu_stats[key] = 'N/A'
            else:
                new_cpu_stats[key] = str(value)
        cpu_stats = new_cpu_stats

    return gpu_stats, cpu_stats


#%% Main

if __name__ == '__main__':

    gpu_stats, cpu_stats = get_system_utilization_stats()
    print('GPU stats:')
    print('----------')
    for key, value in gpu_stats.items():
        print(f'  {key}: {value}')
    print('---------------------------------------------')

    print('CPU stats:')
    print('----------')
    for key, value in cpu_stats.items():
        print(f'  {key}: {value}')
    print('---------------------------------------------')



# %%
