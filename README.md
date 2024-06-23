# CIFAR-10 Image Classification with Transfer Learning
This repository explores image classification on the CIFAR-10 dataset using transfer learning with pre-trained models (AlexNet & ResNet) in PyTorch. The notebooks demonstrate two approaches:

**1. Fine-tuning ResNet-50 (Notebook 1)**
- Leverages the pre-trained ResNet-50 model, freezing its weights and retraining only the final classification layer.

**2. Fine-tuning AlexNet (Notebook 2)**
- Utilizes the pre-trained AlexNet model, similar to ResNet-50, with fine-tuning of the final layer.


## Code Structure:

### Notebook 1 (ResNet-50):
  - Imports necessary libraries.
  - Loads the pre-trained ResNet-50 model with frozen weights.
  - Modifies the final layer for 10-class classification.
  - Configures device (CPU or GPU).
  - Defines data transformations for image pre-processing (resizing, conversion to tensors, normalization).
  - Loads the CIFAR-10 dataset for training and testing with data augmentation.
  - Creates data loaders for training and testing.
  - Defines the loss function (cross-entropy) and optimizer (Adam).
  - Implements a training loop with epoch iterations, loss calculation, backpropagation, and parameter updates.
  - Evaluates the model on the test set and prints accuracy.


### Notebook 2 (AlexNet):
  - Similar structure to Notebook 1, but using the pre-trained AlexNet model downloaded with `alexnet_pytorch`.

## Key Points:

- Transfer learning leverages pre-trained models on large datasets for improved performance on new tasks with less training data.
- Fine-tuning adjusts only the final layers of the pre-trained model for the specific classification problem.
- Data augmentation is an optional technique to increase training data variety, potentially improving model robustness.
- Hyperparameter tuning (learning rate, epochs) can be explored to optimize performance.
