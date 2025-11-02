import os

DESKTOP_PATH = r"/home/davidbeniaguev/Desktop"

# Data and models root directories
NEURON_DATA_ROOT = os.path.join(DESKTOP_PATH, "Data", "BS_neuron_data")
MODELS_ROOT = os.path.join(DESKTOP_PATH, "Models", "BS_neuron_twin_models")

# Ensure directories exist
os.makedirs(NEURON_DATA_ROOT, exist_ok=True)
os.makedirs(MODELS_ROOT, exist_ok=True)

