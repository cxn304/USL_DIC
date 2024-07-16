# Unsupervised neural networks DIC
Author: Cheng Xn.
## Overview
This document provides an overview and explanation of the Python script designed for Unsupervised neural networks DIC. The script includes the following key components:

- Model definition and setup
- Data loading and preprocessing
- Training, validation, and testing procedures
- Loss and accuracy tracking
- Checkpoint saving and loading
- Visualization and result saving

## Contents

### 1. Import Statements
The script starts with the necessary import statements for various Python packages like `time`, `math`, `random`, `shutil`, `pdb`, `scipy`, `numpy`, `torch`, `matplotlib`, `pandas`, and specific utility modules for data reading and model definition.

### 2. Args Class
The `Args` class defines the parameters and configurations for the training process, including paths for data and checkpoints, hyperparameters like learning rate and batch size, and options for GPU/CUDA usage.

### 3. imagesc Function
The `imagesc` function is used for visualizing the results of the DL-DIC model. It plots the predicted and real displacement fields for a subset of the data.

### 4. adjust_learning_rate Function
This function adjusts the learning rate for the optimizer based on the current epoch and the specified learning rate schedule.

### 5. cls_train Function
The `cls_train` function handles the training of the model for one epoch. It computes the loss, performs backpropagation, and updates the model's weights.

### 6. cls_validate Function
This function performs validation of the model on a validation dataset and computes the average loss.

### 7. cls_test Function
Similar to `cls_validate`, the `cls_test` function tests the model on a test dataset and computes the average loss.

### 8. seed_everything Function
This function sets the random seeds for reproducibility of the results.

### 9. Main Execution Block
The main execution block of the script initializes the model, criterion, optimizer, and data loaders for training, validation, and testing datasets. It also handles checkpoint loading if resuming from a previous training session.

### 10. Training Loop
The training loop iterates over the specified number of epochs, calling `cls_train`, `cls_validate`, and `cls_test` functions in each iteration. It also saves checkpoints at regular intervals and records the losses.

### 11. Dataframe and CSV Saving
The script creates a pandas DataFrame with the recorded losses and saves it to a CSV file.

## Usage Instructions

To use this script, ensure that the required Python packages are installed, and the data is organized according to the paths specified in the `Args_cxn` class. Adjust the parameters in the `Args_cxn` class as needed for your specific use case. Run the script in an environment where PyTorch and the other dependencies are available.

## Notes

- The visualization part of the script (`imagesc` function) is commented out and should be enabled if visualization is required.
- Ensure that the paths for data and checkpoints are correctly set up before running the script.
- The script includes error handling for CUDA availability, which allows it to run on both CPU and GPU environments.
- The code and project, as well as the initial draft of the article, were completed in August 2022.


@article{CHENG2025111414,
title = {Using unsupervised learning based convolutional neural networks to solve Digital Image Correlation},
journal = {Optics & Laser Technology},
volume = {180},
pages = {111414},
year = {2025},
issn = {0030-3992},
doi = {https://doi.org/10.1016/j.optlastec.2024.111414}
}
