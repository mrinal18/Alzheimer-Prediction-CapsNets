# Alzheimers using Capsule network 
This code is used for project related to predicting Alzheimers using Capsule Network

## Preprocessing
We preprocessed brain MRI scans with segmentation, registration, normalization, noise addition and skull-stripping.
Processing is done using MRI scan with segmentation, normalisation, augmentation and noise addition along with stripping. 

## Training
1. Trained using 3D autoencoder first on few patches extracted from scans to generate feature whiich are inputed to 3D CNN for predition
2. This preprocessed image is fed as input in Capsule network for better prediction task. 

## Results
Validation accuracy with pre-trained model is 91.3%
