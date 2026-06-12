import streamlit as st
import pickle 
import re
import string
import nltk
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from textblob import TextBlob
import tensorflow as tf
import json
import transformers

def clean_text(text):
    text = text.lower()                                                        # lower case
    text = re.sub(r"\S*https?:\S*", " ", text)                                 # removing links
    text = re.sub('[^a-zA-Z]',' ', text) 
    text = re.sub('\[.*?\]', ' ', text)                                        # removing text in square brackets
    text = re.sub(r"[’…]", " ", text)                                          # removing special chracters
    text = re.sub("[0-9" "]+"," ",text)                                        # removing numbers
    text = text.translate(str.maketrans(' ', ' ', string.punctuation))         # remove punctuations
    text_tokens = word_tokenize(text)                                          # Tokenization
    stop_words = stopwords.words('english')
    new_stop_words = ['cc','pm','subject','also','td','hou','ect','br','tr','com','excelr','www','image','font']
    stop_words.extend(new_stop_words)
    stop_tokens = [word for word in text_tokens if not word in stop_words]     # Remove stop words
    stop_text = ' '.join(stop_tokens)
    lemmas = TextBlob(stop_text)         
    lemmas_token = [w.lemmatize() for w in lemmas.words]                       # Lemmatization
    lemmas_text = ' '.join(lemmas_token)  
    return lemmas_text
clean = lambda x: clean_text(x)


from transformers import TFDistilBertForSequenceClassification
from transformers import DistilBertTokenizer
loaded_tokenizer = DistilBertTokenizer.from_pretrained("save_models")
loaded_model = TFDistilBertForSequenceClassification.from_pretrained("save_models")

html_temp = """
    <div style="background-color:DodgerBlue;padding:10px">
    <h1 style="color:red;text-align:center;">ABUSIVE EMAIL DETECTOR USING DISTILBERT MODEL</h1>
    </div>
    """
    
st.markdown(html_temp, unsafe_allow_html=True)


st.header("ENTER THE EMAIL HERE")
email = st.text_area("")



if st.button('PREDICT'):
    data = clean_text(email)
    predict_input = loaded_tokenizer.encode(data,
                                 truncation=True,
                                 padding=True,
                                 return_tensors="tf")

    output = loaded_model(predict_input)[0]

    prediction_value = tf.argmax(output, axis=1).numpy()[0]
    if prediction_value == 1:
        st.header("The given email is ---> Non Abusive")
    else:
        st.header("The given email is ---> Abusive")

