from keras.datasets import reuters
from keras.src.datasets.reuters import get_label_names
from keras.models import Sequential
import keras
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

def multiHotEncode(sequences, dimension):
    result= np.zeros((len(sequences), dimension))
    for i,sequence in enumerate(sequences):
        result[i, sequence] = 1.0
    return result

def oneHotEncode(labels, dimension=46):
    result = np.zeros((len(labels), dimension))
    for i,label in enumerate(labels):
        result[i, label] = 1.0
    return result


def drawTrainingPlot(history, epochs):
    plt.clf()
    epochsToDraw = range(1, epochs + 1)
    plt.plot(epochsToDraw,history['loss'], "r--", label="Training loss")
    plt.plot(epochsToDraw,history['val_loss'], "b--", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")

    plt.xticks(epochsToDraw)
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("lossGraph.png", dpi=300)
    plt.show()







(trainData,trainLabels), (testdata,testLabels) = reuters.load_data(num_words=10000)
print(testLabels)
word_index = reuters.get_word_index()
label_names = reuters.get_label_names()
print(label_names)
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
x_train = multiHotEncode(trainData, 10000)
x_test = multiHotEncode(testdata, 10000)
y_train = oneHotEncode(trainLabels, 46)
y_test = oneHotEncode(testLabels, 46)

model = Sequential([
    Dense(32, activation="relu"),
    Dense(32, activation="relu"),
    Dense(46, activation="softmax")
])

# better for these kind of measurement
top_3_accuracy = keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy")
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy", top_3_accuracy])


model.fit(x_train, y_train, epochs=20, batch_size=512, validation_split=0.4)

history= model.history.history
print("history is this ", history.keys())
drawTrainingPlot(history,20)

# evaluate the model
results = model.evaluate(x_test, y_test)
print("Test loss, test acc, top3Accuracy:", results)


