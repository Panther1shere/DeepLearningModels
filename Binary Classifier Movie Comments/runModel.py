from keras.models import load_model
from keras.datasets import imdb
import numpy as np
model = load_model("model.keras")

# get the one hot encoded data
def multi_hot_encode(sequences, dimension):
    results= np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence]= 1.0
    return results
# change to words index
def changeToIndexFromString(string):
    word_index = imdb.get_word_index()
    encoded = [1]
    for word in string.lower().split():
        index = word_index.get(word)
        if index is not None and index < 10000:
            encoded.append(index + 3)  # OFFSET by 3 for Keras preprocessing
        else:
            encoded.append(2)  # 2 is the "unknown" token
    return encoded



x=True
while x:
    a=input("Enter a comment ")
    if a=="exit":
        x=False
    else:
        indxedString = np.array([changeToIndexFromString(a)])
        encoded=multi_hot_encode(indxedString, 10000)
        prediction = model.predict(encoded)
        print(f"probability of being positive: {prediction[0][0]} and negative: {1-prediction[0][0]}")
