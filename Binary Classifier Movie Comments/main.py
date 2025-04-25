from keras.datasets import imdb
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt



# function for multi hot encoding
def multi_hot_encode(sequences, dimension):
    results= np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence]= 1.0
    return results


def drawTrainingPlot(history, epochs):
    plt.clf()
    plt.plot(epochs,history['loss'], "r--", label="Training loss")
    plt.plot(epochs,history['val_loss'], "b--", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.xticks(epochs)
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("lossGraph.png", dpi=300)
    plt.show()






(train_data, train_labels), (testData, testLabels)= imdb.load_data(num_words=10000)
print("Train data shape: ", len(train_data))
print("Test data shape: ", type(testData))

# fetchig the word index
word_index = imdb.get_word_index()

reverse_word_index= dict([(value, key) for (key, value) in word_index.items()])
firstReview=" ".join([reverse_word_index.get(i-3, "?") for i in train_data[0]])
print(firstReview)

# getting the training data
x_train= multi_hot_encode(train_data, 10000)
x_test= multi_hot_encode(testData, 10000)
y_train= np.asarray(train_labels).astype("float32")
y_test= np.asarray(testLabels).astype("float32")

# creating a three layer model with two intermediate layers of 16 units and one last layer
model = Sequential([
    Dense(16, activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# compiling the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# training the model

history = model.fit(x_train, y_train, epochs=4, batch_size=512, validation_split=0.2)
history_dict = history.history
print(history_dict.keys())
# plotting the training and validation loss
epochs = range(1, len(history_dict['loss']) + 1)
drawTrainingPlot(history_dict, epochs)

results=model.evaluate(x_test, y_test)
print(results)
# making predictions
predictions = model.predict(x_test[0:5])
print (["positive" if pred > 0.5 else "negative" for pred in predictions])
# saving the model
model.save("model.keras")

