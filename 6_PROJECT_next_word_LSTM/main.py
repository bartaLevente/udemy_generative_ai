from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import streamlit as st
import numpy as np
import pickle

max_seq_len = 16

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = load_model('lstm_shakesp_next_word.h5')

def generate_text(seed_text, next_words=4):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        
        # If no known words in input, exit early
        if not token_list:
            return f"Cannot predict further. No known tokens in: '{seed_text}'"

        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        predicted_word_index = np.argmax(predicted)

        next_word = next((word for word, index in tokenizer.word_index.items() if index == predicted_word_index), None)

        if next_word:
            seed_text += " " + next_word
        else:
            return f"Prediction failed: couldn't map predicted index {predicted_word_index} to a word."

    return seed_text


st.title('Next word prediction with small lstm')
st.write('trained on tiny_shakespeare')

user_input = st.text_area('Type something')

if st.button('Predict!'):
    result = generate_text(user_input)
    st.write(f'Sentiment: {result}')
else:
    st.write('Start typing...')