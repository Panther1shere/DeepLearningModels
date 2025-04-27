from keras.datasets import california_housing
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

""" loading the data
# training date and test data points are basically the districted with each district has
# different variables and the labels corrospond to the median house price
"""


def RegressionModel():
    model = Sequential([
        Dense(64, activation="relu"),
        Dense(64, activation="relu"),
        Dense(1)
    ])
    # compiling the model
    model.compile(optimizer="adam",
                  loss="mean_squared_error",
                  metrics=["mean_absolute_error"])
    return model

(train_data, train_labels), (test_data, test_labels) = california_housing.load_data(version = "small")
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
x_train = (train_data - mean) / std
x_test = (test_data - mean) / std

# scaling the targets
y_train= train_labels/100000.
y_test= test_labels/100000.



# implementing the manual k-fold cross validation
k=4
epochs=200
num_val_samples = len(x_train) // k
all_mae_scores = []

for i in range(k):
    x_val_fold = x_train[i*num_val_samples:(i+1)*num_val_samples]
    y_val_fold = y_train[i*num_val_samples:(i+1)*num_val_samples]
    partial_x_train = np.concatenate([x_train[:i*num_val_samples], x_train[(i+1)*num_val_samples:]], axis=0)
    partial_y_train= np.concatenate([y_train[:i*num_val_samples], y_train[(i+1)*num_val_samples:]])

    model=RegressionModel()
    history = model.fit(partial_x_train,
                        partial_y_train,
                        epochs=epochs,
                        batch_size=16,
                        validation_data=(x_val_fold, y_val_fold),
                        verbose=0)

    mae_history = history.history["val_mean_absolute_error"]
    all_mae_scores.append(mae_history)


average_mae_history = [
    np.mean(
        [x[i] for x in all_mae_scores]
    )
    for i in range(epochs)
]


epochs = range(1, len(average_mae_history) + 1)
plt.plot(epochs, average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.savefig("california_housing_validation_mae_plot.png", dpi=300)






























