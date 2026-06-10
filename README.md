# Semantic Segmentation of Fluorescent Neuronal Cells

A deep learning project for semantic segmentation of fluorescent neuronal cell images using a custom **U-Net** model.

This project was originally developed in a Kaggle Notebook environment and demonstrates a biomedical image segmentation workflow for identifying neuronal cell regions in fluorescence microscopy images.

---

## Overview

Semantic segmentation is a computer vision task where the model assigns a class label to every pixel in an image.

In this project, the goal is to segment fluorescent neuronal cells from microscopy images.

Instead of predicting one label for the whole image, the model predicts a segmentation mask.

Example:

```text id="8v13cs"
Input:
    fluorescence microscopy image

Output:
    pixel-level segmentation mask
```

The output mask shows which pixels belong to neuronal cell structures and which pixels belong to the background.

---

## What This Project Does

This project demonstrates how to:

* Load fluorescence microscopy images
* Load or prepare corresponding segmentation masks
* Preprocess biomedical images
* Resize images and masks for neural network training
* Build a custom U-Net architecture
* Train a segmentation model
* Predict pixel-level masks
* Visualize original images, ground-truth masks, and predicted masks

---

## Problem Type

This is a **semantic segmentation** problem.

For each pixel, the model predicts whether it belongs to:

```text id="twfthc"
cell / neuron region
background
```

In binary segmentation, the output mask usually has values like:

```text id="47yquo"
0 → background
1 → cell / neuron
```

---

## Why Semantic Segmentation?

In biomedical image analysis, segmentation is very important because researchers often need to measure or analyze:

* Cell shape
* Cell area
* Neurite structures
* Fluorescent regions
* Cell density
* Morphological patterns
* Spatial distribution of cells

Manual segmentation can be slow and inconsistent, so deep learning models can help automate the process.

---

## Dataset

The project focuses on fluorescent neuronal cell images.

A typical dataset for this task contains:

```text id="t52831"
input microscopy images
corresponding segmentation masks
```

The input image shows fluorescent neuronal cells, while the mask marks the target cell regions.

A common structure is:

```text id="3u6kca"
dataset/
│
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
│
└── masks/
    ├── mask_001.png
    ├── mask_002.png
    └── ...
```

The exact dataset structure may depend on the Kaggle notebook environment.

---

## Input and Output

### Input

The input is a fluorescence microscopy image.

Example:

```text id="6wkfr6"
neuronal_cell_image.png
```

### Output

The output is a segmentation mask.

Example:

```text id="enzmwv"
predicted_mask.png
```

The predicted mask identifies neuronal cell pixels.

Conceptually:

```text id="mt7kwx"
Input image:
    grayscale or RGB fluorescent image

Output mask:
    same spatial size or resized mask
    each pixel = background or cell
```

---

## What is U-Net?

U-Net is a convolutional neural network architecture designed for image segmentation.

It is especially popular in biomedical image segmentation because it performs well even with relatively small datasets.

U-Net has two main parts:

```text id="o6qr84"
Encoder path
Decoder path
```

The encoder extracts high-level visual features.

The decoder upsamples those features back to the original image resolution.

Skip connections connect encoder layers to decoder layers, helping the model recover fine details.

---

## U-Net Architecture

A typical U-Net structure looks like this:

```text id="8qfjhn"
Input Image
    ↓
Encoder Block 1 ───────────────┐
    ↓                           │
Encoder Block 2 ───────────┐    │
    ↓                       │    │
Encoder Block 3 ───────┐    │    │
    ↓                   │    │    │
Bottleneck              │    │    │
    ↓                   │    │    │
Decoder Block 3 ←───────┘    │    │
    ↓                        │    │
Decoder Block 2 ←────────────┘    │
    ↓                             │
Decoder Block 1 ←─────────────────┘
    ↓
Segmentation Mask
```

---

## Why Use U-Net for Neuronal Cell Segmentation?

U-Net is a strong choice for this project because:

* It works well for biomedical images
* It can learn pixel-level boundaries
* Skip connections help preserve fine spatial details
* It can be trained on relatively small image datasets
* It is effective for binary segmentation tasks
* It can segment irregular biological structures

Neuronal cells and fluorescent structures can have thin, complex shapes, so preserving spatial detail is important.

---

## General Pipeline

The full project pipeline can be described as:

```text id="0trroj"
Fluorescence Images
        ↓
Segmentation Masks
        ↓
Image and Mask Loading
        ↓
Image Resizing / Normalization
        ↓
Train / Validation Split
        ↓
Custom U-Net Model
        ↓
Model Training
        ↓
Mask Prediction
        ↓
Segmentation Evaluation
        ↓
Visualization
```

---

## Image Preprocessing

Common preprocessing steps for this project include:

```text id="79vstr"
read image
resize image
normalize pixel values
read mask
resize mask
binarize mask
expand dimensions if needed
```

Example:

```python id="zusgsh"
image = image / 255.0
mask = mask / 255.0
mask = mask > 0.5
```

For binary segmentation, masks are usually converted into values between `0` and `1`.

---

## Example U-Net Model

A simplified custom U-Net model can look like this:

```python id="jhfcoi"
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Model

def conv_block(x, filters):
    x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    return x

def build_unet(input_shape=(256, 256, 1)):
    inputs = Input(input_shape)

    c1 = conv_block(inputs, 64)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = conv_block(p1, 128)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = conv_block(p2, 256)
    p3 = MaxPooling2D((2, 2))(c3)

    bottleneck = conv_block(p3, 512)

    u3 = Conv2DTranspose(256, 2, strides=(2, 2), padding="same")(bottleneck)
    u3 = Concatenate()([u3, c3])
    c4 = conv_block(u3, 256)

    u2 = Conv2DTranspose(128, 2, strides=(2, 2), padding="same")(c4)
    u2 = Concatenate()([u2, c2])
    c5 = conv_block(u2, 128)

    u1 = Conv2DTranspose(64, 2, strides=(2, 2), padding="same")(c5)
    u1 = Concatenate()([u1, c1])
    c6 = conv_block(u1, 64)

    outputs = Conv2D(1, 1, activation="sigmoid")(c6)

    model = Model(inputs, outputs)
    return model
```

---

## Training Objective

For binary segmentation, common loss functions include:

```text id="g6t3g5"
Binary Crossentropy
Dice Loss
Binary Crossentropy + Dice Loss
Focal Loss
Tversky Loss
```

Common metrics include:

```text id="dbenhe"
Dice coefficient
IoU / Jaccard index
Pixel accuracy
Precision
Recall
```

For biomedical segmentation, Dice and IoU are usually more informative than pixel accuracy.

---

## Evaluation

The current README does not include final numeric evaluation results.

Recommended metrics:

| Metric              | Description                                               |
| ------------------- | --------------------------------------------------------- |
| Dice Coefficient    | Measures overlap between predicted and ground-truth masks |
| IoU / Jaccard Index | Measures intersection over union                          |
| Pixel Accuracy      | Percentage of correctly classified pixels                 |
| Precision           | How many predicted cell pixels are correct                |
| Recall              | How many true cell pixels are detected                    |
| Boundary F1         | Useful for boundary quality                               |

Example evaluation code:

```python id="7c82ld"
import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)

    return (2.0 * intersection + smooth) / (denominator + smooth)
```

IoU example:

```python id="73m3c7"
def iou_score(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > 0.5, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)
```

---

## Suggested Result Table

After running evaluation, the README can be updated with:

| Metric              | Score |
| ------------------- | ----: |
| Dice Coefficient    |   TBD |
| IoU / Jaccard Index |   TBD |
| Pixel Accuracy      |   TBD |
| Precision           |   TBD |
| Recall              |   TBD |

---

## Visualization

Segmentation projects should include visual examples.

Recommended visual output:

```text id="81i4gf"
Input image
Ground-truth mask
Predicted mask
Overlay result
```

Example layout:

```text id="6kybvf"
┌──────────────┬──────────────┬──────────────┬──────────────┐
│ Input Image  │ True Mask    │ Pred Mask    │ Overlay      │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

This helps users quickly understand whether the model is segmenting the right structures.

---

## Repository Structure

Current repository structure:

```text id="4p4bkg"
semantic-segmentation-of-fluorescent-neuronal-cells/
│
├── README.md
└── semantic-segmentation-of-cells-using-custom-u-net.ipynb
```

Suggested future structure:

```text id="n9mgnn"
semantic-segmentation-of-fluorescent-neuronal-cells/
│
├── README.md
├── requirements.txt
├── notebooks/
│   └── semantic-segmentation-of-cells-using-custom-u-net.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── models/
│   └── unet_neuronal_cells.h5
├── assets/
│   ├── sample_input.png
│   ├── sample_mask.png
│   ├── sample_prediction.png
│   └── overlay_result.png
└── examples/
    └── sample_predictions.md
```

---

## Installation

Clone the repository:

```bash id="u5jstv"
git clone https://github.com/VafaKnm/semantic-segmentation-of-fluorescent-neuronal-cells.git
cd semantic-segmentation-of-fluorescent-neuronal-cells
```

Create a virtual environment:

```bash id="xsl9o2"
python -m venv .venv
source .venv/bin/activate
```

Install common dependencies:

```bash id="66jj8t"
pip install numpy pandas matplotlib opencv-python scikit-learn tensorflow keras jupyter
```

Start Jupyter Notebook:

```bash id="t9am1o"
jupyter notebook
```

Open:

```text id="il9c2p"
semantic-segmentation-of-cells-using-custom-u-net.ipynb
```

---

## Suggested `requirements.txt`

```txt id="bp099e"
numpy
pandas
matplotlib
opencv-python
scikit-learn
tensorflow
keras
jupyter
```

If additional biomedical image processing tools are added later:

```txt id="i0zl58"
scikit-image
albumentations
tifffile
```

---

## Example Inference Function

A future reusable inference function could look like this:

```python id="iknvie"
import cv2
import numpy as np

def predict_mask(image_path, model, image_size=(256, 256), threshold=0.5):
    """
    Predict a binary segmentation mask for a fluorescence microscopy image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, image_size)
    image = image / 255.0

    input_image = np.expand_dims(image, axis=(0, -1))

    predicted_mask = model.predict(input_image)[0, :, :, 0]
    binary_mask = (predicted_mask >= threshold).astype(np.uint8)

    return binary_mask
```

---

## Suggested Improvements

### 1. Add Sample Segmentation Results

The README should include visual examples:

```text id="ditdnp"
input image
ground truth mask
predicted mask
overlay
```

Recommended files:

```text id="7s86rt"
assets/sample_input.png
assets/sample_ground_truth.png
assets/sample_prediction.png
assets/sample_overlay.png
```

---

### 2. Add Dice and IoU Scores

Segmentation quality should be measured with overlap-based metrics.

Recommended metrics:

```text id="3mqcv5"
Dice coefficient
IoU / Jaccard index
Precision
Recall
```

---

### 3. Add Data Augmentation

Biomedical image segmentation often benefits from augmentation.

Useful augmentations:

```text id="nkbmbt"
horizontal flip
vertical flip
rotation
zoom
brightness adjustment
contrast adjustment
elastic deformation
Gaussian noise
```

Libraries such as `albumentations` can help apply the same transformations to both images and masks.

---

### 4. Add Better Loss Functions

For imbalanced segmentation masks, better losses can help.

Recommended losses:

```text id="7hwovs"
Dice Loss
Focal Loss
Tversky Loss
BCE + Dice Loss
Focal Tversky Loss
```

These are useful when cell pixels occupy a small portion of the image.

---

### 5. Add Post-processing

Predicted masks can be improved using post-processing.

Possible methods:

```text id="3kgyrr"
threshold tuning
remove small objects
fill holes
morphological opening
morphological closing
connected component analysis
watershed segmentation
```

---

### 6. Add Cross-Validation

If the dataset is small, cross-validation provides more reliable performance estimates.

Recommended setup:

```text id="knmpvb"
K-Fold validation
image-level split
experiment-level split
source-level split
```

---

### 7. Compare More Segmentation Models

U-Net is a strong baseline, but future versions can compare:

| Model           | Notes                                                    |
| --------------- | -------------------------------------------------------- |
| U-Net           | Current baseline                                         |
| U-Net++         | Improved skip connection design                          |
| Attention U-Net | Helps focus on relevant regions                          |
| DeepLabV3+      | Strong semantic segmentation model                       |
| SegNet          | Encoder-decoder segmentation model                       |
| Mask R-CNN      | Instance segmentation, useful if individual cells matter |
| Cellpose        | Specialized cell segmentation model                      |
| StarDist        | Specialized cell/nuclei segmentation model               |

---

### 8. Add a Simple Demo

A demo would make the project easier to use.

Recommended tools:

* Streamlit
* Gradio
* FastAPI

Example UI:

```text id="zyf0wh"
Upload fluorescence image
      ↓
Run U-Net segmentation
      ↓
Show predicted mask
      ↓
Show overlay result
      ↓
Download mask
```

---

## Limitations

The current project has several limitations:

* It is notebook-based and not yet structured as reusable Python code.
* The README does not currently report Dice or IoU results.
* The trained model checkpoint is not included in the repository.
* The dataset must be accessed through Kaggle or downloaded separately.
* Semantic segmentation marks cell regions but does not separate individual cell instances.
* Thin neuronal structures may be difficult to segment accurately.
* Fluorescence intensity, noise, staining quality, and microscope settings can affect performance.
* Model performance should be validated on images from different experiments or acquisition conditions.

---

## Use Cases

This project can be useful for:

* Biomedical image segmentation learning
* Fluorescence microscopy image analysis
* Neuronal cell segmentation experiments
* Learning U-Net architecture
* Pixel-level image classification
* Building segmentation baselines
* Preparing masks for downstream cell analysis
* Educational computer vision projects in biology

---

## Ethical and Scientific Notes

This project is intended for educational and research use.

For scientific use, segmentation results should be validated carefully because errors can affect downstream measurements such as cell area, morphology, density, or fluorescence intensity.

Automated segmentation should support expert review rather than replace careful biological analysis.

---

## Possible Future Work

Recommended future work:

* Add `requirements.txt`
* Add clean Python scripts
* Add saved model checkpoint
* Add inference script
* Add sample input/mask/prediction images
* Add Dice and IoU metrics
* Add overlay visualizations
* Add data augmentation
* Add post-processing pipeline
* Add U-Net++ or Attention U-Net comparison
* Add Cellpose or StarDist baseline
* Add Streamlit or Gradio demo
* Add Dockerfile

---

## Conclusion

This repository demonstrates a custom U-Net approach for semantic segmentation of fluorescent neuronal cell images.

The project is useful for learning:

```text id="86y2d8"
biomedical image segmentation
fluorescence microscopy preprocessing
custom U-Net architecture
pixel-level prediction
binary segmentation masks
segmentation evaluation workflow
Kaggle-based deep learning experimentation
```

With Dice/IoU evaluation, sample visual results, saved model artifacts, data augmentation, and a clean inference interface, this repository can become a much stronger biomedical image segmentation portfolio project.
