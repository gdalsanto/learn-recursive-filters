# Learning Recursive Attenuation Filters Under Noisy Conditions 

[JAES-DAFx paper (arXiv)](https://arxiv.org/abs/2512.16318) 

### 🛠️ Getting started
When cloning this repository make sure to clone all the submodules by running 

```shell
git clone --recurse-submodules https://github.com/gdalsanto/learn-recursive-filters
```

To install it via pip, on a new Python virtual environment `venv` 
```shell
python3.10 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 📄 Script overview
- `src/config.py`: Pydantic configuration models for FDNs, losses, training, and optimization experiments.
- `src/fdn.py`: Builds Feedback Delay Network instances.
- `src/losses.py`: Differentiable loss functions (multi-scale spectral, EDR/EDC, sparsity penality) used for training.
- `src/target.py`: Helpers to generate noise terms and target RT60 profiles.
- `src/utils.py`: Utility functions to parse configs, instantiate loss functions, and measure echo density.
- `src/test_loss_profile.py`: Script to sweep/plot loss profiles or surfaces for FDN parameters and export reference IRs and minima.
- `src/test_optimization.py`: Script to optimize FDN parameters against targets with FLAMO’s trainer, saving checkpoints and results.
