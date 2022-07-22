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


# load the dataset
d = get_data("data/swig80.txt")
# fix random seed for reproducibility

numpy.random.seed(7)
dataset = d.astype('float32')
# dataset = dataset.reshape(len(dataset), 1)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]
# reshape into X=t and Y=t+1
window = 6
trainX, trainY = to_windows(train, window)
testX, testY = to_windows(test, window)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

#ipdb.set_trace()

if SERIALISED is False:
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, window)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)
    # serialize model to JSON
    save_model('model.json', model)
else:
    # load json and create model
    model = load_model('model.json')

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
naive = numpy.roll(testY, 2)
# calculate root mean squared error
print(f"SMAPE for train: {smape(trainY, trainPredict[:, 0])})")
print(f"SMAPE for test: {smape(testY, testPredict[:, 0])})")
print(f"SMAPE for naive: {smape(testY, naive)})")

trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:] = numpy.nan
trainPredictPlot[window:len(trainPredict)+window] = trainPredict[:, 0]

testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:] = numpy.nan
testPredictPlot[len(trainPredict)+(window*2)+1:len(dataset)-1] = testPredict[:, 0]

plot_forecast(testPredictPlot, dataset)
