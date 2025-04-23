from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from keras.models import load_model

word_index=imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

model = load_model('simple_rnn_imdb_3.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

def predict_sent(review):
   prepr_input = preprocess_text(review)
   pred = model.predict(prepr_input)
   sentiment = 'Positive' if pred[0][0] > 0.5 else 'Negative'
   return sentiment,pred[0][0]

##
import streamlit as st
st.title('IMDB senitment analisys')
st.write('Enter a movie review to classify it')

user_input = st.text_area('Review')

if st.button('Classify!'):
    sentiment, score = predict_sent(user_input)
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Score: {score}')
else:
    st.write('Enter a review')
