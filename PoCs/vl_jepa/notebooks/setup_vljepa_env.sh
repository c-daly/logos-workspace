#!/bin/bash
# Setup script for VL-JEPA environment

# Create conda environment with Python 3.10 and PyTorch with CUDA
conda create -n vlj python=3.10 pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Activate environment
conda activate vlj 

# Install core ML libraries
pip install transformers>=4.36.0
pip install datasets>=2.16.0
pip install accelerate>=0.25.0

# Install LoRA/QLoRA support
pip install peft>=0.7.0
pip install bitsandbytes>=0.41.0

# Install UnslothAI (optional, for 2-5x speedup)
pip install "unsloth[cu121] @ git+https://github.com/unslothai/unsloth.git"

# Install video loading
pip install decord>=0.6.0

# Install Jupyter for notebook support
pip install jupyter ipykernel ipywidgets

# Register kernel with Jupyter
python -m ipykernel install --user --name vlj --display-name "VL-JEPA (Python 3.10)"

# Install additional utilities
pip install tqdm
pip install matplotlib
pip install pillow

echo ""
echo "✓ Environment 'vljepa' created successfully!"
echo ""
echo "To activate: conda activate vljepa"
echo "To launch notebook: jupyter notebook vljepa_poc.ipynb"
echo ""
echo "GPU Check:"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
