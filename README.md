# Machine Learning Portfolio Project

A comprehensive collection of machine learning projects covering classical ML, deep learning, and NLP with ethical considerations and debugging challenges.

## 📋 Project Overview

This portfolio demonstrates end-to-end machine learning capabilities across three key domains:

1. **Classical Machine Learning** with Scikit-learn (Iris Dataset)
2. **Deep Learning** with TensorFlow (MNIST Dataset) 
3. **Natural Language Processing** with spaCy (Amazon Reviews)

## 🎯 Task 1: Classical ML - Iris Species Classification

### Goal
Preprocess data and train a decision tree classifier to predict iris species with comprehensive evaluation.

### Key Features
- **Data Preprocessing**: Handling missing values, label encoding, train-test splitting
- **Model Training**: Decision Tree classifier with hyperparameter tuning
- **Evaluation**: Accuracy, precision, recall, confusion matrix, feature importance
- **Visualization**: Decision tree structure, prediction results, feature analysis

### Results
- Achieved high accuracy (>95%) on Iris dataset
- Comprehensive model interpretation and visualization
- Feature importance analysis for botanical insights

## 🧠 Task 2: Deep Learning - MNIST Digit Classification

### Goal
Build a CNN model to classify handwritten digits with >95% test accuracy and visualize predictions.

### Model Architecture

**3 Convolutional Blocks with:**

1. Conv2D + BatchNorm + MaxPooling + Dropout

2. Dense layers with regularization

3. Softmax output for 10-class classification


### Key Features
- **Data Preprocessing**: Normalization, reshaping, one-hot encoding
- **Advanced Training**: Early stopping, learning rate reduction
- **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score
- **Visualization**: Training history, confusion matrix, sample predictions

### Results
- **98%+ test accuracy** (exceeding 95% target)
- Detailed error analysis and misclassification review
- Interactive prediction visualization

## 📝 Task 3: NLP - Amazon Reviews Analysis

### Goal
Perform named entity recognition and rule-based sentiment analysis on product reviews.

### Key Features
- **Named Entity Recognition**: Custom patterns for products and brands
- **Rule-Based Sentiment**: Context-aware analysis with negations and intensifiers
- **Entity-Sentiment Correlation**: Link extracted entities with sentiment scores
- **Visualization**: Entity frequency, sentiment distribution, performance metrics

### Results
- Successfully extracted product names, brands, and companies
- Accurate sentiment classification matching star ratings
- Comprehensive entity-sentiment correlation analysis

## 🛡️ Ethics & Optimization

### Ethical Considerations

#### Identified Biases:
- **MNIST**: Cultural/demographic representation, accessibility limitations
- **Amazon Reviews**: Language, cultural, product category biases

#### Mitigation Strategies:
- **TensorFlow Fairness Indicators**: Slice-based evaluation, disparity metrics
- **spaCy Custom Rules**: Cultural-aware patterns, expanded sentiment lexicons
- **Comprehensive Framework**: Diverse data collection, continuous monitoring

### Troubleshooting Challenge

#### Fixed Critical TensorFlow Errors:
1. **Dimension Mismatches**: Input shapes, layer sequencing
2. **Incorrect Loss Functions**: Binary vs categorical crossentropy
3. **Data Preprocessing**: Normalization, reshaping, one-hot encoding
4. **Architecture Issues**: Missing flatten layers, wrong activations

#### Debugging Solutions:
- Proper model summary and shape validation
- Comprehensive error handling
- Step-by-step data pipeline verification

## 🚀 Quick Start

### Prerequisites
```bash
pip install tensorflow scikit-learn spacy matplotlib seaborn pandas numpy
python -m spacy download en_core_web_sm
