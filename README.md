# InternPe_AI_ML_Project_4
Breast Cancer Detection


## Project 4: Breast Cancer Classification with a simple Neural Network (NN)
<br>

**--Dataset Overview--**
<br>

**Source:** Scikit-learn Breast Cancer Dataset
<br>
**Features:** Mean radius, mean texture, mean perimeter, mean area, mean smoothness, etc.
<br>
A total of 30 features extracted from digitized images of breast mass.
<br>
**Target Variable:** 0 (Malignant) and 1 (Benign)
<br>


**--Project Overview--** 
<br>

This project aims to detect breast cancer using deep learning techniques. The dataset used is sourced from Scikit-learn's breast cancer dataset, which includes various medical features to classify whether a tumor is malignant or benign. The model is trained using a Neural Network to assist in early detection and diagnosis.
<br>


**--Tools Used--** 
<br>

**Python** : For scripting and implementation.
<br>
**Google Colab** : For writing and running the code in a Jupyter Notebook environment.
<br>

**--Libraries Used--**
<br>

**pandas (pd)** – For data manipulation and analysis
<br>
**numpy (np)** – For numerical computations
<br>
**Matplotlib** – Data visualization
<br>
**Scikit-learn** – Dataset loading, model training, and evaluation.
<br>


**--Implementation Steps--** 
<br>

(1) **Data Preprocessing**:
<br>

:) Loading the dataset from Scikit-learn
<br>
:) Converting the dataset into a Pandas DataFrame
<br>
:) Checking for missing values and handling them if necessary
<br>
:) Splitting data into features (X) and target variable (Y)
<br>
:) Normalizing feature values for better neural network performance
<br>

(2) **Exploratory Data Analysis (EDA)**:
<br>

:) Checking the distribution of benign and malignant cases
<br>
:) Computing statistical summaries of features
<br>
:) Visualizing data distributions
<br>

(3) **Feature Engineering**:
<br>

:) Selecting relevant features for classification
<br>
:) Normalizing feature values if necessary
<br>

(4) **Model Selection & Training**:
<br>

:) Splitting the dataset into training and testing sets (80-20 ratio)
<br>
:) Implementing a Deep Neural Network (DNN) using TensorFlow/Keras
<br>

(5) **Model architecture**:
<br>

**Input Layer** (30 neurons)
<br>
**Hidden Layers** (Dense layers with ReLU activation)
<br>
**Output Layer** (1 neuron with Sigmoid activation)
<br>

(6) **Compiling the model** using Binary Crossentropy Loss and Adam Optimizer
<br> 

(7) **Training the model** with multiple epochs and batch size tuning
<br> 
 
**--Results--**
<br>

**Model Accuracy:** 98.26% (Highly accurate in predicting breast cancer).
<br>
**Prediction Approach:**
<br>
:) Model predicts whether a tumor is Malignant (0) or Benign (1) based on 30 medical features.
<br>
:) Single patient prediction example shows a well-standardized pipeline for real-world usage.
<br>
