# Breast Cancer Classification using Neural Networks

An advanced deep learning solution for automated breast cancer detection and binary classification from clinical diagnostic features using Multi-Layer Perceptrons (MLPs) built with TensorFlow and Keras.

## Overview

This project leverages deep learning techniques to classify breast mass anomalies as either **Malignant** or **Benign** based on digitized images of fine needle aspirates (FNA). The model maps 30 distinct cellular and geometric features to assist healthcare professionals in early diagnosis and definitive risk profiling.

**Clinical Application**: Automated, rapid identification of malignancy can drastically improve patient outcomes through early therapeutic intervention and optimized clinical decision pipelines.

## Problem Statement

Malignant breast tumors represent critical oncological challenges requiring objective, urgent evaluation. Manual evaluation of cytological features across multi-dimensional feature spaces is:
* Prone to human interpretation variability and observer bias.
* Highly dependent on specialized, readily available laboratory expertise.
* A common source of processing bottlenecks within clinical and diagnostic pathology workflows.

This project addresses these challenges by delivering an automated, reproducible, and highly accurate deep learning pipeline for binary classification.

## Key Features

### 🎯 Tumor Classification & Profiling
**Binary Classification**: Precise Malignant vs. Benign identification.
**Structured Feature Mapping**: Direct processing of 30 physical and cytological attributes without relying on image-domain feature extraction.
**High Performance Evaluation**: Comprehensive verification tracking loss curves, precise confusion matrices, and detailed metrics.

### 🧠 Advanced Multilayer Perceptron (MLP)
**Sequential API Modeling**: Clean, layered structure implementing Keras Sequential architecture.
**Dense Interconnections**: Deeply integrated hidden representation layers utilizing Rectified Linear Unit (ReLU) activations.
**Feature Standardization**: Fully operational z-score normalization pipeline built with `StandardScaler` to handle extreme feature variance.

### 📊 Data Engineering & Processing
**Standardized Diagnostic Profiles**: Robust support for standard multivariate benchmark datasets (`load_breast_cancer`).
**Automated Verification**: Programmatic data validation mapping dimensional checks, structural properties (`df.info()`), and class balance tracking.

---

## Dataset Information

### Breast Cancer Wisconsin (Diagnostic) Dataset
**Total Samples**: 569 patient profiles (Malignant: 212, Benign: 357).
**Dimensionality**: 30 numerical features representing geometric properties of cell nuclei.
**Features Overview**:
**Core Measurements**: Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, Concave points, Symmetry, Fractal dimension.
**Error/Variance Profiling**: Standard errors calculated for all core measurements.
**Worst Case Bounds**: "Worst" or largest values (mean of the three largest values) for all core attributes.
**Data Characteristics**:
**Class Imbalance**: ~62.7% Benign vs. ~37.3% Malignant distribution.
**Feature Scale Invariance**: Addressed globally via standalone scaling algorithms before training.

---

## Technical Architecture

### Model Architecture Breakdown
The network is built as a highly structured, dense Artificial Neural Network (ANN / MLP) processing feature tensors via sequential layers:

**Input Layer**: Expects a flattened array matching the 30 standardized feature dimensions.

**Hidden Layer 1**: Fully connected layer (Dense) mapping features to local abstractions using ReLU.

**Hidden Layer 2**: Secondary deep processing layer reinforcing feature weights via structural combinations (ReLU).

**Output Layer**: Single neural termination using a `Sigmoid` activation function to output a calibrated probability distribution between 0 (Malignant) and 1 (Benign).

---

## Methodology

### Data Pipeline

```
[Raw Features (569, 30)] ──> [StandardScaler (Z-Score)] ──> [Train-Test Stratified Split]
                                             │
        ┌────────────────────────────────────┴───────────────────────────────────┐
        ▼                                                                        ▼
[Training Set (80%)]                                                     [Testing Set (20%)]
```

**Data Loading**:
* Import data profiles programmatically through `sklearn.datasets`.
* Encapsulate arrays within clean `pandas.DataFrame` structures for analysis.

**Preprocessing**:
* Analyze feature metrics (`mean area` vs `smoothness error`) for extreme scale variations.
* Apply fit-transform functions using `StandardScaler` to align data points into unified normal spaces.

**Train-Test Partitioning**:
* Execute an 80/20 train-test split configuration.
* Isolate target variables (`diagnosis`) to maintain uncompromised held-out test groups.

### Model Training Configuration

```python
# Training Configuration Specifications
batch_size = 32               # Feature blocks evaluated per gradient step
epochs = 50 - 100             # Iterative cycles over the entire training set
optimizer = 'Adam'            # Stochastic gradient descent via adaptive moment estimation
loss_function = 'binary_crossentropy' # Optimal loss metric for calibrated binary labels
metrics = ['accuracy']        # Primary operational baseline tracking metric
```

---
## Evaluation Metrics

The classification network evaluates model safety, precision, and error rates using foundational metrics:

* **Binary Accuracy:** Measurement of correctly pinpointed overall classes.
* **Confusion Matrix:** In-depth verification parsing:
    * *True Positives (TP)* & *True Negatives (TN)*
    * *False Positives (FP)* (Type I Errors)
    * *False Negatives (FN)* (Type II Errors) - **Highest clinical criticality to minimize.**
* **Normalized Confusion Matrix:** Conversion of absolute errors into proportions along the operational rows (`true` boundaries).

---

## Technical Stack

* **Deep Learning Framework:** TensorFlow 2.x, Keras Sequential API
* **Data Analytics:** NumPy, Pandas
* **Machine Learning Suite:** Scikit-Learn (`StandardScaler`, `train_test_split`, `metrics`)
* **Data Visualization:** Matplotlib, Seaborn (`sns.heatmap`, `sns.countplot`)

---

## Installation & Setup

### Requirements
* Python 3.8+
* Pip environment manager

### Dependency Installation
```bash
pip install tensorflow scikit-learn numpy pandas matplotlib seaborn jupyter
```

### Repository Deployment

```bash
# Clone the repository
git clone https://github.com/0-Ahmed-Tamer-0/breast_cancer_classification.git
cd breast_cancer_classification

# Initialize environment mapping
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Launch environment
jupyter notebook
```

---

## Usage Guide

### Running the Notebook Workflow

Launch `Breast Cancer Binary Classification.ipynb` within your Jupyter instance.

**Execute sections in chronological sequence**:
* Cell 1: Pipeline imports (TensorFlow, Sklearn, Visualization toolkits).
* Cell 2: Raw loading, dictionary mapping, shape audits, and `df.info()` configurations.
* Cell 3: Graphical distribution tracing via custom countplots.
* Subsequent Cells: Target scaling, network definitions, compiler configuration, training execution loops, loss charts, and confusion matrix profiling (`draw_confusion_matrix`).

### Expected Core Visual Output Scripts

```python
# Call to generate confusion metrics
print("Confusion matrix : \n")
draw_confusion_matrix(y_test, y_pred)

print("Normalized confusion matrix : \n")
draw_confusion_matrix(y_test, y_pred, "true")
```

Outputs dynamically reveal exact true-positive rates along with visual viridis heatmaps highlighting predictive overlap configurations.

---

## Key Insights & Best Practices

* Scalability Mandate: Always utilize standard scaling metrics before injecting feature instances into a sequential network to prevent heavy variables (like `area`) from drowning out delicate measurements (like `smoothness`).
* Clinical Priorities: Prioritize minimizing Type II errors (False Negatives), as misclassifying a malignant case as benign poses a far greater clinical risk than a false positive.
* Deterministic Splitting: Utilize stratified partitioning to ensure target distributions match precisely across both testing and validation arrays.

---

## Disclaimer

> ⚠️ **CRITICAL MEDICAL DISCLAIMER**: This project is developed exclusively for educational, research, and technical prototyping purposes. It is **NOT** validated for active diagnostic or clinical deployments. Real-world decisions must always rely on accredited diagnostic procedures and certified healthcare practitioners.

---

## Author

Ahmed Tamer Mohamed Ezzat
