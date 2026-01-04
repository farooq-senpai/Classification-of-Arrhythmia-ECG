# Project-Arrhythmia

## Live Demo

🔗 **Live Application:**  
[https://your-replit-app-link-here  ](https://classification-of-arrhythmia-ecg--farooq-senpai.replit.app/)

---

## Introduction

This project focuses on predicting and classifying cardiac arrhythmias using various machine learning algorithms. The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Arrhythmia). It contains clinical and ECG-related attributes used to identify normal and abnormal heart rhythms.

The dataset consists of 452 patient records spanning 16 different classes. Among these, 245 instances represent normal heart rhythms, while the remaining samples correspond to different arrhythmia conditions such as coronary artery disease and right bundle branch block.

### Dataset Overview

- **Number of samples:** 452  
- **Number of features:** 279  
- **Number of classes:** 16 (1 normal + multiple arrhythmia types)

### Objective

The objective of this project is to:
- Predict whether a patient is suffering from arrhythmia  
- Classify the specific type of arrhythmia when present  

---

## Algorithms Used

The following machine learning algorithms were implemented and evaluated:

1. K-Nearest Neighbors (KNN)
2. Logistic Regression
3. Decision Tree Classifier
4. Linear Support Vector Classifier (SVC)
5. Kernelized Support Vector Classifier (SVC)
6. Random Forest Classifier
7. Principal Component Analysis (PCA) for dimensionality reduction

---

## Project Workflow

### Step 1: Exploratory Data Analysis
- Analyzed relationships and distributions across 279 features
- Identified high dimensionality and feature correlation issues

### Step 2: Data Preprocessing
- Handled missing values and standardized numerical features
- Prepared the dataset for machine learning models

### Step 3: Dimensionality Reduction with PCA
- Applied Principal Component Analysis to reduce feature dimensionality
- Addressed collinearity and improved computational efficiency

### Step 4: Model Training and Evaluation
- Trained multiple classifiers on both original and PCA-transformed data
- Evaluated performance using metrics such as accuracy and recall

---

## Results

![Results](https://raw.githubusercontent.com/shsarv/Project-Arrhythmia/master/Image/result.png)

---

## Conclusion

Applying Principal Component Analysis (PCA) significantly improved model performance by reducing feature dimensionality and eliminating multicollinearity. PCA enhanced both execution speed and prediction quality across models.

The **best-performing model** was the **Kernelized Support Vector Machine (SVM) with PCA**, achieving an accuracy of **80.21%**, along with strong recall performance. This demonstrates the effectiveness of combining dimensionality reduction techniques with advanced classification algorithms for high-dimensional medical datasets.

---

## Future Work

- Experiment with advanced models such as XGBoost and Neural Networks  
- Perform extensive hyperparameter tuning  
- Combine PCA with feature selection techniques  
- Expand the application into a real-time clinical decision support tool  

---

## Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)  
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)  
- [Understanding PCA](https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60)  

---

This project demonstrates the application of machine learning techniques for medical data analysis and highlights the importance of dimensionality reduction when working with high-dimensional clinical datasets.
