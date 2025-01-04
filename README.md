# Hippocampus Segmentation for Alzheimer's Induced Dementia Prediction from MRI Images

This repository contains the implementation of a project focused on segmenting the hippocampus from MRI images to predict Alzheimer's-induced dementia. By leveraging advanced segmentation techniques and improving existing architectures, this project aims to enhance the accuracy of hippocampus segmentation, an important step in understanding Alzheimer's disease progression.

---

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Modifications and Enhancements](#modifications-and-enhancements)
- [Performance Metrics](#performance-metrics)
- [Resources Used](#resources-used)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)

---

## Introduction

Alzheimer's disease is a progressive neurodegenerative disorder, and early diagnosis can help improve management strategies. Hippocampus segmentation from MRI images plays a critical role in assessing the extent of dementia. This project proposes improvements to segmentation architectures to achieve higher performance in this domain.

---

## Dataset
Dataset used: [MRI Hippocampus Segmentation](https://www.kaggle.com/datasets/sabermalek/mrihs)

The dataset consists of:
- MRI images of the brain
  - 18900 MRI images of 100 patients in the training set
  - 6615 MRI images of 35 patients in the test set 
- Corresponding segmentation masks for the hippocampus.

The dataset was preprocessed to normalize pixel intensities and resize images for input into the neural network. (128 x 128 x 3)

---

## Model Architecture

The architecture for segmentation was inspired by existing state-of-the-art convolutional neural network (CNN)-based designs for image segmentation. Key layers include:
- Encoder-Decoder structure.
- U-Net-like skip connections for feature preservation.
- Batch normalization to stabilize training.

Base models and architectures were taken from:
- [MultiResUNet](https://doi.org/10.1016/j.neunet.2019.08.025)
- [Modified U-Net](https://pubmed.ncbi.nlm.nih.gov/35304675/)

---

## Modifications and Enhancements

Several architectural modifications were introduced to improve segmentation performance:

1. **Swish Activation Function**:
   - Replaced ReLU with Swish for smoother gradients and better convergence.

2. **SpatialDropout2D**:
   - Added spatial dropout to prevent overfitting and improve generalization.

3. **Bilinear Interpolation**:
   - Used bilinear interpolation for upsampling, ensuring smoother feature maps during decoding.

4. **Combined Loss Function**:
   - Designed a combined loss function incorporating Binary Cross-Entropy and Dice Loss to optimize pixel-wise accuracy and segmentation overlap:
     - Binary Cross-Entropy ensures precise classification.
     - Dice Loss emphasizes accurate segmentation by accounting for overlap between predicted and true masks.

---

## Performance Metrics

The modifications resulted in significant performance improvements, particularly in:

- **Dice Coefficient**: Measures the overlap between predicted and ground truth masks.
- **Jaccard Index**: Evaluates similarity between predicted and true segmentation regions.

|**Model**      |**Dice Coefficient**|**Jaccard Index**|
|---------------|--------------------|-----------------|
|MultiResUNet   |0.897474389         |0.87823319367    |
|Modified U-Net |0.8914032548        |0.872448485      |
|Our Model      |0.9057411749        |0.88360496       |

---

## Resources Used

- Platform: Kaggle
- RAM: 29 GB
- GPU: Tesla P100
- VRAM: 16 GB

[Kaggle Notebook Link](https://www.kaggle.com/code/ishrak26/cse472-ml-sessional-project) 

---

<!--
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ishrak26/CSE-472-Machine-Learning-Sessional-Project.git
   cd CSE-472-Machine-Learning-Sessional-Project
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the dataset is placed in the `data/` directory, with MRI images and segmentation masks organized appropriately.

---

## Usage

1. Train the model:
   ```bash
   python train.py
   ```

2. Evaluate the model:
   ```bash
   python evaluate.py
   ```

3. Visualize predictions:
   ```bash
   python visualize.py
   ```

---

## Results

Here are sample outputs from the modified model:

- Input MRI Image:
  ![Input MRI Image](path/to/input_image.png)

- Ground Truth Mask:
  ![Ground Truth Mask](path/to/ground_truth_mask.png)

- Predicted Mask:
  ![Predicted Mask](path/to/predicted_mask.png)

---
-->
## Future Work

- Use the predicted masks for AD classification tasks.
- Extend the model to multi-class segmentation tasks.
- Experiment with other activation functions and loss functions.
- Integrate clinical data to improve prediction of Alzheimer's progression.
- Deploy the model in a user-friendly interface for clinical usage.

---

## Acknowledgements

- We would like to thank the contributors of the dataset and the open-source libraries that made this project possible.
- We would also like to express our gratitude for the direction and guidelines from our project supervisor [Dr. Muhammad Masroor Ali](https://cse.buet.ac.bd/faculty/faculty_detail/mmasroorali) sir.

---

## Contributors

- [Md. Ishrak Ahsan](https://github.com/ishrak26)
- [Farhan Tahmidul Karim](https://github.com/farhanitrate35)

Feel free to contribute, raise issues, or suggest enhancements! Reach out via [ishrak26@gmail.com](mailto:ishrak26@gmail.com).

