# MNIST Digit Recognizer Neural Network

This project demonstrates the implementation of a simple neural network from scratch in Python to classify handwritten digits using the MNIST dataset. The notebook includes all the essential steps, from data preprocessing to model training and evaluation, providing an intuitive understanding of how neural networks work.

## Overview
The goal of this project is to classify handwritten digits (0-9) using a neural network implemented from scratch, without relying on deep learning libraries like TensorFlow or PyTorch. The MNIST dataset is used as the input data for training and evaluation.

The neural network architecture:
- **Input Layer:** 784 neurons (28x28 pixels).
- **Hidden Layer:** 10 neurons using ReLU activation.
- **Output Layer:** 10 neurons using softmax activation.

---

## Features
- Data preprocessing and normalization.
- Neural network initialization with random weights and biases.
- Forward propagation with ReLU and softmax activations.
- Backward propagation and gradient descent for optimization.
- Evaluation on training and development datasets.
- Visualization of predictions for individual samples.

---

## Setup
### Prerequisites
- Python 3.7 or higher
- Required libraries: `numpy`, `pandas`, `matplotlib`, `kagglehub`

### Installation
Clone this repository:
   ```bash
   git clone https://github.com/your-repo/digit-recognizer.git
   cd digit-recognizer
```
   
Install the required libraries:
```bash
pip install -r requirements.txt
```
(Ensure kagglehub is set up to access Kaggle datasets.)

Download the "Digit Recognizer" dataset from Kaggle:
```bash
import kagglehub
kagglehub.login()
kagglehub.competition_download('digit-recognizer')
```
## Acknowledgments
This project uses the MNIST dataset and was inspired by the Kaggle "Digit Recognizer" competition. Special thanks to Kaggle for providing the dataset and a collaborative environment for machine learning projects.

## License
This project is licensed under the MIT License.
