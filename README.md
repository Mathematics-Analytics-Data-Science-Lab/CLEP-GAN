# CLEP-GAN: An Innovative Approach to Subject-Independent ECG Reconstruction from PPG Signals

## Requirements

- **Python**: >= 3.10
- See `requirements.txt` for Python package dependencies
  
## Installation
Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing
Preprocess raw physiological signals before training or evaluation:

```bash
python data_processing.py --config preprocessing_config.yaml
```
- Customize `preprocessing_config.yaml` to set data paths and hyperparameters.

### 2. Synthetic PPG-ECG Generation:
Generate synthetic paired data using the proposed ODE model

```bash
python synthetic_data_generation.py
```
### 3. PPG-to-ECG Transformation with CLEP-GAN
Train or evaluate the CLEP-GAN model on the processed data:

```bash
python CLEP-GAN.py
```
