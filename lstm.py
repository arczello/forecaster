# LSTM for international airline passengers problem with regression framing
import ipdb
import numpy
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential, model_from_json
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import common
from pmdarima.metrics import smape

SERIALISED = True


# convert an array of values into a dataset matrix
def to_windows(ts, window):
    x, y = [], []
    for i in range(len(ts)-window-1):
        a = ts[i:(i+window), 0]
        x.append(a)
        y.append(ts[i + window, 0])
    
    return numpy.array(x), numpy.array(y)


# load the dataset
d = common.get_data("data/swig80.txt")
# fix random seed for reproducibility

numpy.random.seed(7)
dataset = d.astype('float32')
dataset = dataset.reshape(len(dataset), 1)
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+1
look_back = 6
trainX, trainY = to_windows(train, look_back)
testX, testY = to_windows(test, look_back)
# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

if SERIALISED is False:
    # create and fit the LSTM network
    model = Sequential()
    model.add(LSTM(4, input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
else:
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    model = loaded_model

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

naive = numpy.roll(testY[0], 2)
# calculate root mean squared error
print(f"SMAPE for train: {smape(trainY[0], trainPredict[:, 0])})")
print(f"SMAPE for test: {smape(testY[0], testPredict[:, 0])})")
print(f"SMAPE for naive: {smape(testY[0], naive)})")

ipdb.sset_trace()
# trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
# print('Train Score: %.2f RMSE' % (trainScore))
# testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
# print('Test Score: %.2f RMSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:] = numpy.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1] = testPredict

common.plot_forecasts(testPredictPlot, scaler.inverse_transform(dataset))
