import os
from numpy.core.defchararray import encode

# limit tensorflow's various levels of messages(warnings, etc)
from tensorflow.python.eager.monitoring import Metric
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled=True

import keras
from keras.datasets import imdb
from keras.preprocessing import sequence
import tensorflow as tf 
import numpy as np

VOCAB_SIZE = 8854
MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

# Make each review the same length
# if the review is greater than 250 words then trim off the extra words
# if the review is less than 250 words add the necessary amount of 0's to equal 250
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

# Create the model
model = tf.keras.Sequential([tf.keras.layers.Embedding(VOCAB_SIZE,  32),
                             tf.keras.layers.LSTM(32),
                             tf.keras.layers.Dense(1, activation='sigmoid')])

model.summary()

#  Train the model
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=1, validation_split=0.2)

# Evaluate the model
results = model.evaluate(test_data, test_labels)
print(results)

# Making predictions 
word_index = imdb.get_word_index()

def encode_text(text):
    tokens = keras.preprocessing.text.text_to_word_sequence(text)
    tokens = [word_index[word] if word in word_index else 0 for word in tokens]
    return sequence.pad_sequences([tokens], MAXLEN)[0]

# Decode function
reverse_word_index = {value:key for (key, value) in word_index.items()}

def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
        if num != PAD:
            text += reverse_word_index[num] + " "
        
        return text[:-1]

# let's make an actual prediction on a review
def predict(text):
    encoded_text =encode_text(text)
    pred = np.zeros((1, 250))
    pred[0] = encoded_text
    result = model.predict(pred)
    print(result[0])
    
positive_review = "This movie is really interesting. I like it"
predict(positive_review)

negative_review = "That movie sucks. I hate it"
predict(negative_review)
