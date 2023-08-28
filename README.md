
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
![Transformers](https://img.shields.io/badge/-Transformers-05122A?style=flat&logo=Transformers)&nbsp;

## Dataset

The dataset utilized for this project is 'Emails.csv,' comprising 48,076 rows and featuring 5 columns.
![image](https://github.com/nithu24/Abusive-Email-Classification/assets/92521223/ae6dff77-f127-4ec3-b353-8a2954cfd0c8)

## Data Insight

Target class is imbalanced with 93% non abusive email and 7% abusive email 
![image](https://github.com/nithu24/Abusive-Email-Classification/assets/92521223/e222a2bc-cdc6-4def-99b3-f60ee18ffffe)


## Preprocessing

To preprocess the text, follow these steps: 
convert to lowercase, remove punctuation, digits, and special characters, eliminate stop words, tokenize the text, and perform lemmatization.
We can use Word cloud is another to find the most frequent word.

## Feature Engineering

This project leverages the powerful NLP technique of TF-IDF (Term Frequency-Inverse Document Frequency) to enhance our abusive email classification task.

## Model Building 

I constructed models using different algorithms, considering performance metrics such as accuracy, precision, and recall. After comparing the results, I observed that the random forest algorithm achieved the highest accuracy. Consequently, I opted to utilize the random forest algorithm for the final model building.
![image](https://github.com/nithu24/Abusive-Email-Classification/assets/92521223/43b605ec-d83c-4d08-8cd0-7a7df0337a1c)

## Deployment

I opted for Streamlit to deploy the solution, recognizing its capability to rapidly transform data scripts into accessible web applications. By simply providing the running script to the tool, it seamlessly converted my work into a user-friendly web app, offering an efficient way to showcase and share the results.
![image](https://github.com/nithu24/Abusive-Email-Classification/assets/92521223/31f690fb-cd49-444c-afcc-8d3608324bf5)

## Challenges 

1) Target class is imbalanced in nature. 
2) I employed the random forest for model building, achieving a commendable accuracy of 97%. During deployment, my predictions remained accurate. However, a challenge arose during testing: the potential for encountering out-of-vocabulary words not present in the training dataset. This scenario prompted me to explore strategies for improving the model's ability to handle previously unseen words.

## Solution to overcome challenges

1) We used SMOTE to remove class imbalance 
2) To tackle the above mentioned challenge, I utilized the Hugging Face library, which offers a convenient method for introducing new tokens to the vocabulary of existing tokenizers such as BERT and DistilBERT. These pretrained models encompass a wide range of vocabulary from sources like Wikipedia and book corpora. In my approach, I specifically employed DistilBERT due to its merits: it's a lightweight and efficient transformer model derived from BERT base through distillation, aligning well with the goals of achieving efficient enhancements to model performance.






