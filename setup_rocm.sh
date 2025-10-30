#!/bin/bash
# Minimal ROCm and PyTorch installation for AMD GPUs

echo "Installing ROCm (AMD GPU support)..."

# Add AMD repo and install ROCm (adjust for your distro)
wget https://repo.radeon.com/amdgpu-install/6.2/ubuntu/jammy/amdgpu-install_6.2.60200-1_all.deb
sudo dpkg -i amdgpu-install_6.2.60200-1_all.deb
sudo apt update
sudo apt install -y rocm-dev

# Add user to video/render groups
sudo usermod -a -G video,render $USER

# Install PyTorch for ROCm (via pip; use nightly for latest)
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm6.2/

echo "Setup complete. Reboot recommended. Verify with: rocm-smi"
echo "Then: python -c 'import torch; print(torch.cuda.is_available())'"
