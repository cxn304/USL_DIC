# Unsupervised neural networks with U-Net

This code implements an unsupervised learning approach using the U-Net architecture for image segmentation. The goal is to train a model to predict deformations in images without any labeled training data. The paper has submitted to optics and lasers in engineering.

## Dependencies

The code relies on several Python libraries:

- `time`: Used for measuring the execution time.
- `math, random`: General mathematical operations and random number generation.
- `shutil`: Provides functions for file and directory operations.
- `pdb`: Python debugger for debugging purposes.
- `scipy.io`: Library for reading and writing MATLAB files.
- `numpy`: Fundamental package for scientific computing with Python.
- `torch`: PyTorch library for deep learning.
- `torch.nn`: Neural network module of PyTorch.
- `torch.optim`: Optimization algorithms for PyTorch.
- `torch.utils.data`: Tools for creating data loaders.
- `matplotlib.pyplot`: Plotting library for visualization.
- `pandas`: Library for data manipulation and analysis.

## Configuration

The code includes a configuration class `Args_cxn` that holds various parameters for training the model. These parameters include:

- `print_freq`: Frequency of printing training progress.
- `checkpoint_path`: Path to save the model checkpoint.
- `loss_path`: Path to save the training loss.
- `train`: Path to the training data.
- `val`: Path to the validation data.
- `epochs`: Number of training epochs.
- `warmup`: Number of warm-up epochs.
- `batch_size`: Batch size for training.
- `lr`: Learning rate for optimization.
- `weight_decay`: Weight decay for regularization.
- `clip_grad_norm`: Maximum norm of gradients for gradient clipping.
- `gpu_id`: ID of the GPU to use.
- `disable_cos`: Flag to disable cosine learning rate decay.
- `disable_aug`: Flag to disable data augmentation.
- `no_cuda`: Flag to disable CUDA acceleration.
- `add_all_features`: Flag to add all features.
- `RESUME`: Flag to resume training from a checkpoint.

## Functions

The code defines several functions for different purposes:

- `imagesc`: Visualizes the predicted u, v, and deformation fields.
- `adjust_learning_rate`: Adjusts the learning rate based on the current epoch.
- `cls_train`: Performs the training of the U-Net model.
- `seed_everything`: Sets the random seed for reproducibility.
