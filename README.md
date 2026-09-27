# Multiclass Image Segmentation with U-Net

This repository contains a **U-Net-based semantic segmentation
pipeline** implemented with TensorFlow/Keras for grayscale image
segmentation. The workflow includes image and mask preprocessing,
multiclass label encoding, U-Net training, model evaluation, inference,
and segmentation post-processing.

## Overview

The model performs pixel-wise multiclass semantic segmentation. Each
input image is resized to **640 × 640 pixels** and provided to a U-Net
with a single grayscale input channel.

The current notebook is configured for:

-   Input size: **640 × 640 × 1**
-   Number of classes: **9**
-   Model: **Multiclass U-Net**
-   Optimizer: **Adam**
-   Loss / cost function: **Categorical cross-entropy**
-   Batch size: **16**
-   Training epochs: **200**
-   Primary training metric: **Accuracy**
-   Additional evaluation: **Mean IoU and Dice coefficient**

## Files

The main workflow is contained in:

``` text
multiclass_segmentation.ipynb
```

The notebook imports the U-Net architecture from:

``` text
simple_unet.py
```

The model is created using:

``` python
multi_unet_model(
    n_classes=n_classes,
    IMG_HEIGHT=SIZE_Y,
    IMG_WIDTH=SIZE_X,
    IMG_CHANNELS=1
)
```

## Dataset Structure

Training images and their corresponding segmentation masks are stored in
separate directories:

``` text
dataset/
├── X/      # Raw grayscale images
└── Y/      # Ground-truth segmentation masks
```

The paths should be updated before training:

``` python
TRAIN_PATH_X = "path/to/X"
TRAIN_PATH_Y = "path/to/Y"
```

The notebook currently loads `.jpeg` files from these directories.

## Image Preprocessing

Raw images are:

1.  Loaded as grayscale images.
2.  Resized to **640 × 640** pixels.
3.  Converted to NumPy arrays.
4.  Expanded to include a single-channel dimension.
5.  Normalized using `keras.utils.normalize`.

Example:

``` python
img = cv2.imread(img_path, 0)
img = cv2.resize(img, (640, 640))
```

After preprocessing, the model input has the form:

``` text
(number_of_images, 640, 640, 1)
```

## Mask Preprocessing

Ground-truth masks are also resized to **640 × 640** pixels.
Nearest-neighbor interpolation is used to prevent interpolation from
creating artificial class values:

``` python
mask = cv2.resize(
    mask,
    (640, 640),
    interpolation=cv2.INTER_NEAREST
)
```

Several mask intensity values are remapped to a common value before
class encoding.

The mask values are then converted to consecutive integer class labels
using `LabelEncoder`.

Finally, the labels are converted to one-hot categorical representations
using:

``` python
to_categorical(y_train, num_classes=n_classes)
```

The resulting target shape is:

``` text
(number_of_images, 640, 640, 9)
```

## Train/Test Split

The notebook first reserves **10% of the dataset for testing**:

``` python
X1, X_test, y1, y_test = train_test_split(
    train_images,
    train_masks_input,
    test_size=0.10,
    random_state=0
)
```

The remaining data are split again, with **20% of that subset placed in
`X_do_not_use` / `y_do_not_use`**. Therefore, the effective
model-training subset is approximately **72% of the complete dataset**,
while 10% is used as the test/validation set in the current training
code.

## Model Training

The model is compiled using the Adam optimizer:

``` python
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
```

The **cost function is categorical cross-entropy**, which compares the
predicted class-probability distribution at each pixel with the one-hot
encoded ground-truth class.

Training is performed with:

``` python
history = model.fit(
    X_train,
    y_train_cat,
    batch_size=16,
    epochs=200,
    validation_data=(X_test, y_test_cat),
    shuffle=False
)
```

### Training Parameters

  Parameter       Value
  --------------- ---------------------------
  Image size      640 × 640
  Channels        1
  Classes         9
  Optimizer       Adam
  Loss function   Categorical cross-entropy
  Batch size      16
  Epochs          200
  Shuffle         False

## Training Curves

The notebook plots:

-   Training loss
-   Validation loss
-   Training accuracy
-   Validation accuracy

These curves can be used to monitor convergence and identify potential
overfitting.

## Prediction

The trained network outputs a probability map for every class:

``` python
y_pred = model.predict(X_test)
```

For a 9-class model, each pixel contains 9 predicted class
probabilities.

The final segmentation label is selected using:

``` python
y_pred_argmax = np.argmax(y_pred, axis=3)
```

Thus, the class with the highest predicted probability is assigned to
each pixel.

## Evaluation

### Accuracy

The model is evaluated using:

``` python
_, acc = model.evaluate(X_test, y_test_cat)
```

### Mean Intersection over Union

Mean IoU is calculated using:

``` python
MeanIoU(num_classes=n_classes)
```

This measures the overlap between predicted and ground-truth
segmentation regions.

### Dice Coefficient

The notebook also includes Dice coefficient calculation for individual
classes:

\[ Dice = `\frac{2|A \cap B|}{|A|+|B|}`{=tex} \]

where (A) is the ground-truth region and (B) is the predicted region.

## Model Saving and Loading

A trained Keras model can be saved as:

``` python
save_model(model, "RPE_segmentation.keras")
```

and loaded using:

``` python
loaded_model = load_model("RPE_segmentation.keras")
```

## Segmentation Post-processing

The notebook contains additional post-processing for segmentation
results. It uses OpenCV contour detection to retain the largest
connected segmentation region and, when sufficiently large, the
second-largest region.

The general post-processing workflow is:

``` text
Predicted probability maps
        ↓
Argmax classification
        ↓
Class mask
        ↓
Binary threshold
        ↓
Contour detection
        ↓
Largest-region filtering
        ↓
Final segmentation mask
```

The notebook also contains batch-processing code for applying the
trained model to folders of images and saving generated segmentation
outputs.

## Requirements

The code uses the following Python packages:

``` text
tensorflow
keras
numpy
opencv-python
matplotlib
scikit-learn
```

Install the required packages with, for example:

``` bash
pip install tensorflow keras numpy opencv-python matplotlib scikit-learn
```

## Notes

-   Update all dataset and inference paths before running the notebook.
-   `simple_unet.py` must be available for `multi_unet_model` to be
    imported.
-   The notebook is currently configured for **9 classes**.
-   Ground-truth masks should contain consistent discrete class values.
-   Nearest-neighbor interpolation should be retained when resizing
    segmentation masks.
-   The notebook's per-class IoU example explicitly calculates only four
    classes and should be extended if per-class IoU is required for all
    nine classes.
