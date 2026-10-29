# Multimodal Respiratory Disease Classification

This project explores deep-learning methods for classifying four respiratory-health categories (**Healthy, Lung Cancer, Pneumonia, and COVID-19**) using three types of image data: CT, chest X-ray (CXR), and cough-sound representations.

The notebook compares models trained on each modality individually with two multimodal fusion strategies. The goal is to examine how classification behavior changes when the model uses one source of information versus information from multiple modalities. This is an experimental machine-learning project, not a clinical diagnostic tool.

## Project workflow

1. **Prepare the dataset.** Organize examples by split, modality, and class. The notebook's preparation cell shuffles each modality/class collection and assigns 400 examples to training, 50 to validation, and the remaining examples to testing (50 test examples when a collection contains exactly 500).
2. **Train modality-specific models.** Train separate CT-only, CXR-only, and cough-representation-only image classifiers using transfer learning with ImageNet-pretrained ResNet50.
3. **Compare early fusion.** Resize each modality's input to a common image shape, concatenate the three inputs along the channel axis into a nine-channel tensor, then classify the fused tensor with a convolutional neural network.
4. **Compare late fusion.** Process CT, CXR, and cough inputs through separate frozen ResNet50 feature extractors, concatenate their feature vectors, and use a dense classification head to predict the class.
5. **Evaluate predictions.** The notebook uses accuracy during training and generates classification reports and confusion matrices for the test data.

## Dataset layout

The notebook expects a dataset arranged like this:

```text
processed_dataset/
|-- CT/
|   |-- Healthy/
|   |-- LungsCancer/
|   |-- Pneumonia/
|   `-- covid/
|-- CXR/
|   `-- ... (same class folders)
`-- Cough sound/
    `-- ... (same class folders)
```

The notebook initially uses a Kaggle input path:

```text
/kaggle/input/cxr-ct-cough/processed_dataset/processed_dataset
```

Update the input and output paths in the notebook to match your environment. The dataset itself is not included in this repository; obtain it from its authorized source and follow its license and usage conditions.

## Running the notebook

The notebook was developed for a Kaggle-style environment with TensorFlow/Keras and access to the dataset. Open `deep-learning-task-ct-cough.ipynb` in Kaggle or Jupyter, make the dataset available, and update the paths before running the cells.

Key libraries used include:

- Python
- TensorFlow and Keras
- ResNet50 with ImageNet weights
- NumPy and Pillow
- scikit-learn
- Matplotlib and Seaborn

The notebook creates model checkpoints during training. GPU acceleration is recommended for practical training times.

## Important notes and limitations

- The notebook loads each modality using image-based pipelines. In particular, its cough branch treats the available cough data as image files; it does not demonstrate raw-waveform audio processing.
- The fusion pipeline expects CT, CXR, and cough examples to correspond to the same underlying case. Verify that the files are correctly paired before interpreting fusion results. The dataset-preparation cell shuffles each modality/class collection separately, so matching filenames created by that cell alone do not prove patient-level alignment.
- The split code uses unseeded random shuffling and does not demonstrate patient-level or source-level separation. Results may not be reproducible or free of leakage unless the split strategy is adapted to the dataset's case identities.
- Review the dataset license, privacy requirements, and permissions before sharing data or results. Do not use this experimental notebook to make clinical decisions.

## Repository contents

- `deep-learning-task-ct-cough.ipynb` - dataset preparation, single-modality transfer-learning models, early- and late-fusion experiments, and evaluation.
