
# Abusive Email Classification Using NLP

Abusive email classification using NLP employs language analysis to automatically detect offensive content in emails. This task enhances digital communication safety by enabling the identification of harmful language patterns and addressing potential biases.

## Business Problem 

Inappropriate emails would demotivates and spoil the positive environment that would lead to more attrition rate and low productivity and Inappropriate emails could be on form of bullying, racism, sexual favoritism and hate in the gender or culture, in today's world so dominated by email no organization is immune to these hate emails.

## Objective

The goal of this project is to build a text classification model to find whether the given email is abusive or non abusive on the given day based on the above inappropriate content 

## Project Architecture

Text Documents ---> EDA ---> Text Preprocessing ---> Feature Engineering ---> Model Building ---> Deployment

## Python Libraries 

![NumPy](https://img.shields.io/badge/-NumPy-05122A?style=flat&logo=NumPy)&nbsp;
![Pandas](https://img.shields.io/badge/-Pandas-05122A?style=flat&logo=Pandas)&nbsp;
![Matplotlib](https://img.shields.io/badge/-Matplotlib-05122A?style=flat&logo=Matplotlib)&nbsp;
![Seaborn](https://img.shields.io/badge/-Seaborn-05122A?style=flat&logo=Seaborn)&nbsp;
![NLTK](https://img.shields.io/badge/-NLTK-05122A?style=flat&logo=NLTK)&nbsp;
![Scikit-Learn](https://img.shields.io/badge/-ScikitLearn-05122A?style=flat&logo=scikit-learn)\
![Imblearn](https://img.shields.io/badge/-Imblearn-05122A?style=flat&logo=Imblearn)&nbsp;

## Dataset

We are using Emails.csv as the dataset which contains 5 columns and 48076 rows.
![image](https://github.com/nithu24/Abusive-Email-Classification/assets/92521223/ae6dff77-f127-4ec3-b353-8a2954cfd0c8)

## Data Insight

Target class is imbalanced with 93% non abusive email and 7% abusive email 
![image](https://github.com/nithu24/Abusive-Email-Classification/assets/92521223/e222a2bc-cdc6-4def-99b3-f60ee18ffffe)


## Preprocessing

To preprocess the text, follow these steps: 
convert to lowercase, remove punctuation, digits, and special characters, eliminate stop words, tokenize the text, and perform lemmatization.
We can use Word cloud is another to find the most frequent word.

## Feature Engineering

We leverage TF-IDF (Term Frequency-Inverse Document Frequency), a powerful NLP technique, to enhance our abusive email classification task. 

## Model Building 

We build models on different algorithms using parameters such as accuracy, precision and recall.
While comparing, random forest got high accuracy.
so we used random forest for final model building.
![image](https://github.com/nithu24/Abusive-Email-Classification/assets/92521223/43b605ec-d83c-4d08-8cd0-7a7df0337a1c)

## Deployment

We opted for Streamlit to deploy our solution, recognizing its capability to rapidly transform data scripts into accessible web applications. By simply providing the running script to the tool, it seamlessly converts our work into a user-friendly web app, offering an efficient way to showcase and share our results.
![image](https://github.com/nithu24/Abusive-Email-Classification/assets/92521223/31f690fb-cd49-444c-afcc-8d3608324bf5)

## Challenges 

1) Target class is imbalanced in nature 
2) We used random forest for model building and got 97% accuracy and in deployment also we are  predicting correctly but if we can get a out of vocabulary word at the time of testing which is not trained during training dataset.

## Solution to overcome challenges


1) We used SMOTE to remove class imbalance 
2) We can use hugging face library which allows us to easily add new tokens to the vocabulary of an existing tokenizer like BERT, DISTILBERT etc. it is a pertained model which contains Wikipedia and book corpus words. Here we used DistilBERT because it is smaller faster cheaper and light transformer model trained by distilling BERT base.

