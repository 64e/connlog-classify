# connlog-classify
Neural net binary malware classifier for zeek conn.log files

## Installation
This code was written running NVIDIA CUDA running on a WSL2 VM, with NVIDIA drivers running on host machine. The setup is as follows:
- Install WSL2 distro: `wsl --install -d Ubuntu`
- Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads
- Create python environment: `python -m venv .venv`
- Install python dependencies: `pip install -r requirements.txt`

## Dataset
The dataset used is sourced from: `https://www.stratosphereips.org/datasets-iot23`
