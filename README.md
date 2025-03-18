# Food Image Classification using Transfer Learning

This repository contains a deep learning project for classifying food images into **101 categories** using transfer learning with PyTorch. The model leverages pre-trained weights (e.g., ResNet, VGG, or EfficientNet) to achieve high accuracy on the Food-101 dataset.

## Features

- **Transfer Learning**: Uses pre-trained models for efficient training and high accuracy.
- **101 Food Categories**: Classifies food images into 101 distinct categories.
- **PyTorch Implementation**: Built with PyTorch for flexibility and performance.
- **Easy to Use**: Includes scripts for training, evaluation, and prediction.

## Usage

1. Clone the repository:
  

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Train the model:
   

4. Evaluate the model:
  

5. Make predictions:
 

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- NumPy
- OpenCV
- Matplotlib

Example `requirements.txt`:
```plaintext
torch>=1.8.0
torchvision>=0.9.0
numpy>=1.19.0
opencv-python>=4.5.0
matplotlib>=3.3.0
```

## Model Details

- **Transfer Learning**: Uses pre-trained weights from models like ResNet, VGG, or EfficientNet.
- **Food-101 Dataset**: Trained on the Food-101 dataset containing 101 food categories.
- **Custom Layers**: Added custom layers on top of the pre-trained model for fine-tuning.

## Example

Input:
```bash
python predict.py --image data/test_image.jpg
```

Output:
```
Predicted Class: Pizza
