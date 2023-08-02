# Pytorch-CNN-Cifar10-100
This repository contains code for training and evaluating Convolutional Neural Networks (CNN) models using PyTorch on the CIFAR-10 and CIFAR-100 datasets.

## Table of Contents
- [Introduction](#Introduction)
- [Experiment Result](Experiment_Result)
- [Installation](#Installation)
- [Usage](#Usage)
- [Models](#Models)
- [Project Structure](#Project_Structure)
- [RelativeProject](#Relative_Project)
- [Contributing](#Contributing)
- [License](#License)

<a id="Introduction"></a>

## Introduction
This project provides a framework for training and evaluating CNN models on the CIFAR-10 and CIFAR-100 datasets. It includes various models such as ResNet, VGG, and MobileNetV2, and utilizes PyTorch's deep learning library for efficient training and evaluation.

<a id="Experiment_Result"></a>

## Experiment Result

| Dataset  | Network     | Params (M) | Top1 Acc (%) | Epoch |
|----------|-------------|------------|--------------|-------|
| Cifar100 | [MobilenetV2](https://arxiv.org/pdf/1801.04381) | 2.412      | 79.32        | 200   |
| Cifar100 | [VGG16](https://arxiv.org/pdf/1409.1556)       | 34.015     | 76.41        | 200   |
| Cifar100 | [Resnet101](https://arxiv.org/pdf/1512.03385)   | 42.697     | 83.41        | 200   |
| Cifar10  | [MobilenetV2](https://arxiv.org/pdf/1801.04381) | 2.297      | 95.82        | 200   |
| Cifar10  | [VGG16](https://arxiv.org/pdf/1409.1556)       | 33.647     | 95.26        | 200   |
| Cifar10  | [Resnet101](https://arxiv.org/pdf/1512.03385)   | 42.513     | -            | 200   |

<a id="Installation"></a>

## Installation
To install the required dependencies, follow these steps: 
1. Clone this repository:

```bash
git clone https://github.com/your_username/Pytorch-CNN-Cifar10-100.git
```
 
2. Navigate to the project directory:
```bash
cd Pytorch-CNN-Cifar10-100
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

<a id="Usage"></a>

## Usage
To use this project, follow these steps:

1. Configure the training parameters: 
    - Open the config.yaml file and set the desired hyperparameters such as batch size, learning rate, and number of epochs. 
    - Modify other settings such as model type, dataset path, and output directory if needed.

2. Train the model: 
    - Run the training script: 
        ```bash
        python Training.py
        ``` 
    - Monitor the training progress and observe the log output.
        
<a id="Models"></a>

## Models
This project includes the following pre-defined models:

- [Resnet101](https://arxiv.org/pdf/1512.03385)
- [VGG16](https://arxiv.org/pdf/1409.1556)
- [MobilenetV2](https://arxiv.org/pdf/1801.04381)

These models are defined in separate Python files in the Models directory.

<a id="Project_Structure"></a>

## Project Structure
The structure of the project is as follows:

Pytorch-CNN-Cifar10-100
├── config.yaml
├── LICENSE
├── Models
│   ├── Mobilenetv2.py
│   ├── Resnet.py
│   └── Vgg.py
├── README.md
├── requirements.txt
├── Training.py
│── Weapon
│   └── WarmUpLR.py
 
- config.yaml: Configuration file containing hyperparameters and settings for training the models.
- Training.py: Script for training and testing the CNN models using PyTorch.
- Models directory: Contains implementation files for different CNN models.
    - Resnet.py: Contains code for the ResNet model.
    - Vgg.py: Contains code for the VGG model.
    - Mobilenetv2.py: Contains code for the MobileNetV2 model.
Weapon directory: Contains code related to learning rate scheduling.
    - WarmUpLR.py: Defines the WarmUpLR class for implementing learning rate warm-up.
README.md: Documentation file providing an overview of the project and instructions for usage.
This structure tree provides an overview of the organization of the project files.

<a id="Relative_Project"> </a>

## Relative Project
[Toward Robust High-Density EMG Pattern Recognition using Generative Adversarial Network and Convolutional Neural Network Implementation](https://github.com/MIC-Laboratory/IEEE-NER-2023-RoHDE) (IEEE NER'23)


[EffiE: Efficient Convolutional Neural Network for Real-Time EMG Pattern Recognition System on Edge Devices](https://github.com/MIC-Laboratory/IEEE-NER-2023-EffiE) (IEEE NER'23)


[Real-time Bionic Arm Control Via CNN-based sEMG Recognition](https://github.com/MIC-Laboratory/Real-time-Bionic-Arm-Control-via-CNN-on-Sony-Spresense) (Sony Spresense Developer Challenge 2022 Grand Price)


## Contributing
We welcome contributions from the community to improve this project. If you encounter any issues or have suggestions for enhancements, please feel free to submit a pull request or open an issue on the GitHub repository.

## License
This project is licensed under the MIT License.