# ORACLE: Time and Feature Specialist Transformer

[!Oracle Architecture](architecture.pdf)

This repository contains the PyTorch implementation of **ORACLE: Time and Feature Specialist Transformer with Test-Time Adaptation for Time Series Forecasting**.

ORACLE is a dual-stream Transformer architecture designed for highly accurate multivariate time series forecasting. It leverages Time and Feature Specialist Encoders alongside Test-Time Adaptation to dynamically capture both temporal dependencies and cross-channel feature relationships.

## Repository Structure

- `train.py`: Main entry point for training the ORACLE model.
- `config.json`: Configuration file for hyperparameters, model dimensions, and dataset paths.
- `src/transformer.py`: Core PyTorch implementation of the dual-stream Transformer and Specialist Encoders.
- `src/test.py`: Evaluation script, including the implementation of Test-Time Adaptation.
- `src/utils.py`: Helper functions for data loading, preprocessing, and metric calculations.
- `src/oracle_forecast_plot.png`: Visualization of forecasting outputs and architecture diagrams.

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed along with PyTorch. 

### Usage

1. **Configure Hyperparameters:**
   Adjust the settings in `config.json` to define your lookback windows, forecasting horizons, batch sizes, and dataset paths.

2. **Train the Model:**
   To train the dual-stream architecture from scratch, run:
   ```bash
   python train.py
