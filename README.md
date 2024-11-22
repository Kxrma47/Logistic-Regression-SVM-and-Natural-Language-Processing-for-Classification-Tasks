# Logistic Regression, SVM, and Natural Language Processing for Classification Tasks

This repository contains implementations and analyses for binary and multiclass classification tasks using logistic regression, support vector machines (SVMs), and natural language processing (NLP). The project includes data preprocessing, model training, hyperparameter tuning, evaluation, and visualization of results.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Tasks and Features](#tasks-and-features)
3. [Data Preparation](#data-preparation)
4. [Binary Classification](#binary-classification)
5. [Multiclass Classification](#multiclass-classification)
6. [Results and Outputs](#results-and-outputs)
7. [Dependencies and Installation](#dependencies-and-installation)
8. [Usage](#usage)
9. [Examples](#examples)
10. [Future Work](#future-work)
11. [License](#license)

---

## **Overview**
This project demonstrates various machine learning techniques applied to text data, focusing on:
- Logistic regression with Elastic Net regularization.
- Support vector machines with multiple kernel types.
- Natural language processing tasks, including preprocessing, tokenization, and feature vectorization.

The dataset consists of text sentences authored by famous writers, and the goal is to classify these sentences into binary or multiclass categories.

---

## **Tasks and Features**
1. **Gradient Descent for Logistic Regression**:
   - Implemented with Elastic Net regularization.
   - Visualized decision boundaries and loss trends.

2. **Support Vector Machines**:
   - Explored the impact of different kernels (linear, polynomial, RBF) and hyperparameters.

3. **Natural Language Processing**:
   - Preprocessing: Tokenization, stopword removal, lemmatization, stemming.
   - Feature extraction: Bag of Words (BoW) and TF-IDF vectorization.

4. **Binary Classification**:
   - Compared logistic regression and SVM models for a two-class problem.

5. **Multiclass Classification**:
   - Extended logistic regression to multiclass classification using the One-vs-One strategy.

6. **Visualization**:
   - Confusion matrices, ROC curves, and decision boundaries.

---

## **Data Preparation**
### Steps:
1. **Dataset Creation**:
   - Selected six authors: Pushkin, Dostoevsky, Tolstoy, Chekhov, Gogol, and Turgenev.
   - Extracted sentences from their works.
   - Dropped sentences shorter than 15 characters.
   - Created a balanced dataset with specified sample sizes per author.

2. **Preprocessing**:
   - Tokenized sentences into words.
   - Removed stopwords, punctuation, and numbers.
   - Applied stemming or lemmatization for normalization.

3. **Vectorization**:
   - Bag of Words (BoW): Encodes the frequency of words.
   - TF-IDF: Assigns weights to words based on their importance in the text.

---

## **Binary Classification**
### Objective:
Classify sentences written by two authors (e.g., Pushkin and Dostoevsky).

### Steps:
1. Preprocess and vectorize the data.
2. Split the dataset into training (70%) and testing (30%).
3. Train models:
   - Logistic Regression
   - SVM with a linear kernel
4. Use `GridSearchCV` to tune hyperparameters based on F1-score.
5. Evaluate:
   - Metrics: Accuracy, precision, recall, F1-score.
   - Visualizations: Confusion matrices, ROC curves.

---

## **Multiclass Classification**
### Objective:
Classify sentences into one of six classes (authors).

### Steps:
1. Preprocess and vectorize the data.
2. Split the dataset into training (70%) and testing (30%).
3. Train a multiclass logistic regression model using the One-vs-One strategy.
4. Use `GridSearchCV` for hyperparameter optimization.
5. Evaluate:
   - Metrics: Weighted accuracy, precision, recall, F1-score.
   - Visualizations: Confusion matrices.

---

## **Results and Outputs**
### **Binary Classification**:
- **Models**: Logistic Regression and SVM
- **Metrics**:
  - AUC for both models: ~0.90.
  - Slight edge in performance for SVM in terms of precision and recall.
- **Visualization**:
  - ROC curves with thresholds for 30% false positive rate.

### **Multiclass Classification**:
- **Model**: Logistic Regression with One-vs-One.
- **Metrics**:
  - Accuracy: ~57.6%.
  - Highest classification accuracy for Gogol; lowest for Turgenev.
- **Confusion Matrix**:
  - Gogol's sentences were easier to classify, while Pushkin's and Turgenev's were often misclassified.

### **Preprocessing Examples**:
- **Original Sentence**:
  - "Владимир отпер комоды и ящики, занялся разбором бумаг."
- **Processed Sentence**:
  - "владимир отпер комод ящик заня разбор бумаг"

---

## **Future Work**
- Experiment with deep learning models for text classification.
- Incorporate embeddings like Word2Vec or BERT for feature extraction.
- Expand the dataset to include more authors and balanced samples.
