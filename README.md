# Movies Reviews Classification Using BERT and RoBERTa README

This README provides an overview of the Movies Reviews Classification project, including objectives, problem statement, lab session details, and reporting requirements.

## 1. Objectives
The objectives of this project are as follows:
1. Apply state-of-the-art language models BERT and RoBERTa to solve NLP classification problems.
2. Gain experience in using pretrained language models for natural language understanding and classification tasks.

## 2. Problem Statement
The project involves classifying movie reviews as positive or negative. This classification can be useful for various purposes, such as rating films or conducting statistical analysis on user preferences based on different factors like country and age. The dataset consists of 50k reviews labeled as positive or negative. The goal is to train BERT and RoBERTa based classifiers for movie reviews.

## 3. Lab Session
### 3.1 Data Split
- Randomly split the dataset into three sets: training (70%), validation (10%), and testing (20%). Ensure that all splits are balanced.

### 3.2 Text Pre-processing
Perform the following text pre-processing steps on the data before classification:
- Remove punctuation.
- Remove stop words.
- Convert all characters to lowercase.
- Lemmatization of words.

### 3.3 Classification using BERT and RoBERTa
Build a classifier model based on BERT and RoBERTa using the transformers library supplied by Hugging Face. Tokenize the input sentences in the required format for BERT and RoBERTa, and pad or trim sentences accordingly. Use the CLS token embedding outputs of BERT as input to the hidden dense classification layers. Add four hidden layers with 512, 256, 128, and 64 units respectively before the output layer. Use binary cross-entropy loss and the Adam optimizer.

### 3.4 Validation and Hyperparameter Tuning
- Use the validation split to evaluate the model's performance after each training epoch.
- Save model checkpoints and choose the best-performing model as the final model.
- Implement dropout between dense layers to prevent overfitting.
- Tune the learning rate hyperparameter of the Adam optimizer using validation performance.

### 3.5 Checking Pre-processing Importance
- Train two models: one using the pre-processed data and the other using the original data.
- Test both models using the testing set and compare the results.
- Repeat the validation and hyperparameter tuning steps for both models.
- Ensure that the model trained on pre-processed data is validated and tested with pre-processed data and vice versa.

### 3.6 Classifier Head Layers Tune
- Finetune the number of hidden dense layers and the number of units in each layer using the validation set.
- Test the best model using the testing set and report all required metrics.

### 3.7 RNN Model
- Build an RNN model from scratch as an alternative to using BERT and RoBERTa.
- Perform the same steps and evaluations as done for the BERT and RoBERTa models.


