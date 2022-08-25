# LSTM for stock exchange forecasting
import ipdb
import numpy
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pmdarima.metrics import smape
from common import load_model, save_model, get_data, plot_forecast
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import train_test_split

# False - the model will be computed from scratch and saved, 
# True - the mode will be loaded from the file
SERIALISED = True

# convert an array of values into a dataset matrix
def to_windows(ts, window):
    x, y = [], []
    for i in range(len(ts)-window-1):
        a = ts[i:(i+window)]
        x.append(a)
        y.append(ts[i + window])
    
    return numpy.array(x), numpy.array(y)


def build_model(train_X, train_y, validate_X, validate_y):
    model = Sequential()
    model.add(LSTM(12, input_shape=(1, window)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    history = model.fit(train_X, train_y, epochs=20, batch_size=1, verbose=2,
            validation_data=(validate_X, validate_y))
    return(model, history)


def plot_performance(history):
    history_dict = history.history
    loss_values = history_dict["loss"]
    val_loss_values = history_dict["val_loss"]
    epochs = range(1, len(loss_values) + 1)
    plt.plot(epochs, loss_values, "bo", label="Training loss")
    plt.plot(epochs, val_loss_values, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("perf_history.png")
    plt.show()


# load the dataset
dataset = get_data("data/swig80.txt")
# reshape into X=t and Y=t+1
window = 6
X, y = to_windows(dataset, window)
# add a dimension to match the template: [samples, time steps, features]
X = numpy.expand_dims(X, axis=1)

train_X, test_X, train_y, test_y = train_test_split(
    X, y, test_size=0.10, shuffle=False)
train_X, validate_X, train_y, validate_y = train_test_split(
    train_X, train_y, test_size=0.20, shuffle=False)

print("Size: train={}, validate={}, test={}".format(len(train_X),
      len(validate_X), len(test_X)))

if SERIALISED is False:
    # create and fit the LSTM network
    model, history = build_model(train_X, train_y, validate_X, validate_y)
    # serialize model to JSON
    save_model('model.json', model)
    plot_performance(history)
else:
    # load json and create model
    model = load_model('model.json')

plot_model(model, to_file='model_plot.png', show_shapes=True,
           show_layer_names=True)
# make predictions
trainPredict = model.predict(train_X)
testPredict = model.predict(test_X)
naive = numpy.roll(test_y, 2)
# calculate root mean squared error
print(f"SMAPE for train: {smape(train_y, trainPredict[:, 0])})")
print(f"SMAPE for test: {smape(test_y, testPredict[:, 0])})")
print(f"SMAPE for naive: {smape(test_y, naive)})")

# Get the real values from test data
test_X = test_X.squeeze()
# We shift by 1 to get value x+1
Y = test_X[1:, window-1]
# Plot the forecast vs the real values
plot_forecast(testPredict, Y)
