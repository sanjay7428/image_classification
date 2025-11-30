COMPANY. CODTECH IT SOLUTIONS

NAME: SANJAY KUMAR SAINI

INTERN ID : CT04DR1297

DOMAIN : MACHINE LEARNING

DURATION: 4 WEEKS

MENTOR : NEELA SANTOSH

# Image Classification using CNN (CIFAR-10)

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. It includes data loading, preprocessing, model building, training, and evaluation.

## Features
- CIFAR-10 dataset loading and normalization  
- Deep CNN with Batch Normalization and Dropout  
- Training with Adam optimizer  
- Evaluation using accuracy, classification report, and confusion matrix  

## Project Structure
```
main.py       # training, evaluation, metrics  
model.py      # CNN model architecture  
requirements.txt
```

## How to Run
```
pip install -r requirements.txt
python main.py
```

## Model Summary
The CNN consists of:
- 3 convolutional blocks (Conv2D, BatchNorm, MaxPooling, Dropout)  
- Dense layers with BatchNorm and Dropout  
- Output layer with softmax activation  

## Output
The program prints:
- Test accuracy  
- Classification report  
- Confusion matrix  
