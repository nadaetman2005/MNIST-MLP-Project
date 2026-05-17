# Handwritten Digit Recognition using PyTorch (MNIST)

This project implements a Deep Learning model using PyTorch to recognize handwritten digits from the famous MNIST dataset.

## Project Idea
The model takes a 28×28 grayscale image of a handwritten digit and predicts the correct number from 0 to 9 using a Multilayer Perceptron (MLP) neural network.

## Project Features
- Loading and preprocessing the MNIST dataset
- Flattening image data into vectors
- Building an MLP model with PyTorch
- Training using SGD optimizer and CrossEntropyLoss
- Creating a validation set for evaluation
- Performing different experiments:
  - Changing hidden layer sizes
  - Changing learning rate
- Comparing experiment results
- Visualizing Loss and Accuracy curves using Matplotlib

## Baseline Model Architecture
- Input Layer: 784 Neurons
- Hidden Layer 1: 128 Neurons
- Hidden Layer 2: 64 Neurons
- Output Layer: 10 Neurons

## Technologies Used
- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn

## Project Results
The model achieved high accuracy in handwritten digit classification while demonstrating how architectural changes and learning rate adjustments affect performance.

## Project Goals
Understanding the fundamentals of:
- Neural Networks
- Deep Learning with PyTorch
- Data preprocessing
- Model evaluation
- Experiment comparison and performance analysis
