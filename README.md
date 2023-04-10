Movies Reviews Classification Using BERT and RoBERTa
1 Objectives
    • Applying state of the art language model BERT and RoBERTa to solve NLP classification
    problem.
2 Problem Statement
    Classifying the positive reviews and the negative ones can be useful for several purposes such
    as giving an overall rating for the film or making statistical analysis about the preferences of
    people from different countries, age levels, etc... So Movie Review dataset is released which
    composed of 50k reviews labeled as positive or negative to enable training movie reviews clas￾sifiers. Moreover, NLP tasks are currently solved based on pretrained     language models such
    as BERT and RoBERTa. These models provide a deep understanding of both semantic and
    contextual aspects of language words, sentences or even large paragraphs due to their training
    on huge corpus for very long time. i train BERT and RoBERTa based classifier for
    movie reviews
3 Lab session
    3.1 Data Split
        Split the dataset randomly so that the training set would form 70% of the dataset, the vali￾dation set would form 10% and the testing set would form 20% of it.         I keep all the
        splits balanced.
    3.2 Text Pre-processing
        Text pre-processing is essential for NLP tasks. So, apply the following steps on our
        data before used for classification:
        • Remove punctuation.
        • Remove stop words.
        • Lowercase all characters.
        • Lemmatization of words.

    3.3 Classification using BERT and RoBERTa
        I need to build a classifier model based on BERT and RoBERTa. You can use transformers
        library supplied by hugging face to get a pretrained and ready version of BERT and RoBERTa
        models. It will also help me to tokenize the input sentence in the BERT required form and to
        pad the short sentences or trim the long ones. We will use the CLS token embedding outputs
        of BERT as input to the hidden dense classification layers we need to add after BERT. This
        embedding is of size 768. I add 4 hidden layers of 512, 256, 128, 64 units respectively
        before the output layer. I used binary cross entropy loss and adam optimizer.
    3.4 Validation and Hyperparameter Tuning
        Use the validation split to evaluate the model performance after each training epoch then save
        the model checkpoint to choose the one with the best performance as the final model. You can
        use dropout between dense layers to avoid overfitting if it arises. Also, you need to tune the
        learning rate hyperparameter of Adam optimizer using the performance on the validation set.
    3.5 Checking Pre-processing Importance
        BERT is assumed to capture the semantic and contextual aspects of the language. So, some￾times it is better to input the text to it without pre-processing. To check the pre- processing
        importance on our task we will train the model twice one using the pre- processed version of
        data and the other using the original version then test both models using the testing set and
        compare between the results. Note that I repeat the validation and hyperparameter
        tuning steps in both cases. Also, note that the model trained on pre-processed data must be
        validated and tested using pre-processed data and vice versa.
    3.6 Classifier Head Layers Tune
        Finetune the number of hidden dense layers we need to add for classification and the number
        of units in each layer using the validation set. Then test the best model using the testing set
        and report all the above required metrics.
    3.7 Bonus - RNN Model
        Build RNN model from scratch instead of using and (BERT and RoBERTa) and redo the same
        steps I did for both models.

    3.8 Report Requirements
        • report graphs representing the change of training and validation accura- cies
        with the number of training epochs for my experiments.
        • report a graph comparing between the best validation accuracies for the
        different values of learning rate.
        • I report the model accuracy, precision, recall, specificity and F-score as well as
        the resultant confusion matrix using the testing set for the best model with pre-processing
        and without.
        • My comments on all results and comparisons.

    4 Notes
        • code written  in python.
        • use nltk, transformers and pytorch libraries.
