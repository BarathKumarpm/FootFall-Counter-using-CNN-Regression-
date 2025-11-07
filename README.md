# FootFall Counting with Deep Learning CNN Regression Model (AI Footfall Assignment)

## Overview
This repository demonstrates a deep learning approach for crowd counting using a regression-based Convolutional Neural Network (CNN). The model is trained, validated, and visualized on the [Crowduit](https://www.kaggle.com/datasets/khitthanhnguynphan/crowduit) and [Crowd Counting](https://www.kaggle.com/datasets/fmena14/crowd-counting) Kaggle image datasets.

## Features
- Data pipeline for .npy images and .csv labels
- Modern CNN with dropout and adaptive learning rate
- Training and validation splits and error visualization
- MAE and MSE metrics after training
- Annotated video demo with predicted counts overlayed

## Dataset
- [Crowduit](https://www.kaggle.com/datasets/khitthanhnguynphan/crowduit)
- [fmena14/crowd-counting](https://www.kaggle.com/datasets/fmena14/crowd-counting)

## Usage
1. Upload datasets to Kaggle or your Colab environment.
2. Run the notebook cells in order to train the CNN.
3. The notebook saves a trained model (`my_model.keras`) and generates a demo video displaying count predictions.
4. Visualization and MAE/MSE statistics are presented for final review.

## Model Architecture
- Stacked Conv2D layers for feature extraction
- MaxPooling and Dropout for regularization
- Flatten + Dense for regression output
- Trains with Huber (Smooth L1) loss

## Dependencies
- Python 3.8+
- TensorFlow/Keras
- OpenCV
- Numpy, Pandas, Seaborn, Matplotlib

## Example Results
- MAE on validation: (fill in your value, e.g., 3-5)
- Video demo: See `crowd_count_demo.mp4` in notebook output.

## Contact
- [Your Name], [Your Email]
