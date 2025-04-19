[![Project Report](https://img.shields.io/badge/Project%20Report-Click%20Here-blue?style=for-the-badge)](https://wandb.ai/cs24m035-indian-institute-of-technology-madras/CS6910_Assignment2_final2/reports/DA6401-Assignment-2---VmlldzoxMjM2MTg1Nw?accessToken=069sem3p535rxsxdu7t6j3lzj1njtbgxtcxgss3xrusg0amhip5thk4fnwh38ucn)
[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-black?style=for-the-badge&logo=github)](https://github.com/Priyanshu999/Deep-Learning-Assignment-2)

# Project Repository 📂

Welcome to this repository! 

## Table of Contents  
1. [Project Title & Description](#project-title--description)
2. [Project Structure](#project-structure)
3. [Features](#features)  
4. [Installation & Dependencies](#installation--dependencies)  
5. [Usage](#usage)  
6. [Dataset](#dataset)    
7. [Acknowledgments](#acknowledgements)
    

# Convolutional Neural Network

## Project Title & Description  

In this assignment, the task is to build and experiment with Convolutional Neural Network (CNN) based image classifiers using a subset of the iNaturalist dataset. The assignment is divided into two parts:
Part A focuses on training a CNN model from scratch. This involves designing the architecture, tuning hyperparameters such as learning rate and batch size, and visualizing learned filters to understand the model's internal representations.

Part B involves fine-tuning a pre-trained CNN model, which mirrors many real-world applications where transfer learning is used to leverage prior knowledge from large-scale datasets. Fine-tuning enables faster convergence and often leads to better performance, especially when the available training data is limited.


The main objective is to understand how different training strategies affect model performance and how visualization and experimentation help in interpreting and debugging deep learning models. The entire implementation is to be done in Python, using only PyTorch, Torchvision, or optionally PyTorch Lightning for training, logging, and visualization. 

### 🔹 Key Objectives:  
- ✅ **Train a CNN from Scratch**
  - Understand architecture design and implement a convolutional neural network (CNN) from the ground up.
  - Learn to tune hyperparameters and visualize learned filters.

- ✅ **Fine-tune a Pre-trained Model**
  - Apply transfer learning techniques to adapt a pre-trained model for a new task.
  - Simulate real-world deep learning workflows using fine-tuning.

- ✅ **Tooling & Frameworks**
  - Use **Python** as the primary programming language.
  - Only use libraries from **PyTorch**, **Torchvision**, or **PyTorch-Lightning**.
  - Avoid TensorFlow, Keras, or other DL libraries unless explicitly approved by TAs.
  - Leverage PyTorch-Lightning for features like FP16 mixed-precision training, Weights & Biases (wandb) integration, and reduced boilerplate code.

- ✅ **Reporting & Visualization**
  - Use **wandb.ai** to log metrics, visualize training curves, filter activations, and generate the final report.
  - Start by cloning the provided wandb report template and populate it throughout the assignment.


## Project Structure

This repository contains two parts — **PartA** and **PartB** — of the Deep Learning assignment/project. Each part includes:

- A Jupyter notebook showcasing model runs, training logs, and results
- A Python script to train the network from scratch
- Folder-level README files with detailed training instructions

### 🔧 How to Use

1. Navigate to either `PartA` or `PartB` depending on which part of the assignment you're interested in.
2. Open the Jupyter notebook (`.ipynb`) to explore pre-recorded model runs and visualizations.
3. Follow the instructions in the corresponding `README.md` file within each folder to train the model using the provided Python script.

---

### 📌 Requirements

Make sure you have the required Python packages installed. You can typically install them using:




## Features  

This project implements a fully customizable **Convolutional Neural network** with various tunable hyperparameters and optimization techniques.  

### 🔹 Supported Functionalities  

- **Number of epochs:** `{10, 20}`  
- **Filter Size:** `{3, 5}`  
- **Dense Layer Size:** `{128, 512}`  
- **Dropout:** `{0, 0.2, 0.5}`  
- **Data Augmentation:** `{True, False}`  
- **Batch Normalization:** `{True, False}`  
- **Filter Organization:** `{'same', 'double', 'half'}`  
- **Activation functions:** `{ReLU, GELU, Leaky ReLU}`  

### 🔹 Hyperparameter Tuning  
To efficiently search for optimal hyperparameters, we utilized **Weights & Biases (WandB) Sweep functionality**, allowing automated experimentation with different configurations for improved performance.  

This flexibility makes the model **highly configurable and scalable**, enabling experimentation with various architectures and optimization strategies! 🚀  

## Installation & Dependencies  

### 🔹 Prerequisites  
Ensure you have **Python 3.x** installed on your system. You can check your Python version using:  

```bash
python --version
```
or  
```bash
python3 --version
```

### 🔹 Cloning the Repository  
To download this project, open a terminal and run:  

```bash
git clone https://github.com/Priyanshu999/Deep-Learning-Assignment-1.git
cd Deep-Learning-Assignment-1
```

### 🔹 Installing Dependencies  
Install all necessary dependencies first.

After installation, you're all set to run the project! 🚀


## Usage

### 🔹 Running Training Script
See the individual Parts' Readme files.

## Dataset
## 🐾 iNaturalist Dataset

The **iNaturalist** dataset is a large-scale, real-world image dataset designed for fine-grained visual classification, especially in biodiversity and wildlife domains.

### 🌿 Dataset Overview

- Collected from the **iNaturalist** platform — a citizen science project and social network for sharing biodiversity observations.
- Contains **hundreds of thousands of images** spanning **thousands of species**.
- Includes images of **plants, animals, fungi, and other organisms** with species-level labels.
- Highly imbalanced and noisy, reflecting real-world data distributions.


### 🔹 Dataset Integration with Weights & Biases
Both datasets can be used with Weights & Biases (WandB) for tracking model performance, visualizing accuracy, and analyzing results across different hyperparameter settings.


## Acknowledgements
I would like to acknowledge the following resources that helped shape this project:

- [Mitesh Khapra Deep Learning Course](https://www.youtube.com/playlist?list=PLZ2ps__7DhBZVxMrSkTIcG6zZBDKUXCnM)
- [Deep Learning - Stanford CS231N](https://www.youtube.com/playlist?list=PLSVEhWrZWDHQTBmWZufjxpw3s8sveJtnJ)


🔗 Happy Coding! 😊
