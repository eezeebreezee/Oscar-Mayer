<img width="450" height="450" alt="image" src="https://github.com/user-attachments/assets/3d280e8f-2b80-43e8-a445-5bfbdf405c07" />

# GPU-Accelerated Wiener and DDM Animations

A minimal PyTorch project for real-time animations of Wiener Processes (Brownian Motion) and Drift Diffusion Models, leveraging CUDA/ROCm for AMD GPUs.

## Prerequisites
- Linux (Ubuntu 22.04+ recommended for ROCm)
- AMD GPU (e.g., Radeon RX 7900 series) with ROCm support
- Python 3.10+

## Setup
1. Install ROCm:
./setup_rocm.sh

2. Install Python dependencies:
pip install -r requirements.txt

3. Run animations:
python main.py --simulation wiener # Or 'ddm' for Drift Diffusion Model


## Usage
- `--simulation wiener`: Runs Wiener Process animation.
- `--simulation ddm`: Runs Drift Diffusion Model animation.
- Animations use PyTorch tensors on GPU for simulation state and computations.

## Extending
- Add multi-path simulations by batching tensors in `src/` scripts.
- For Windows, install PyTorch via conda with ROCm preview (experimental).

## Troubleshooting
- Verify GPU: `rocm-smi`
- If no GPU: Falls back to CPU.

