import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import os
from termcolor import colored
import matplotlib.dates as mdates
import ipdb
import pmdarima as pm
from sklearn.metrics import mean_squared_error
from pmdarima.metrics import smape
from statsmodels.tsa.arima_model import ARIMA
import statsmodels as sm 
import common

# For serialization:
import joblib
import pickle
import numpy as np

EQ = r'wigtech.txt'
DB_PATH = r'Dropbox/projects/forecast/data/'
SERIALIZED_MODEL = 'arima.model'
COL_DATE = '<DATE>'
COL_OPEN = '<OPEN>'
COL_CLOSE = '<CLOSE>'
COL_LOW = '<LOW>'
COL_HIGH = '<HIGH>'

TEST_RATIO = 0.90
TRAIN_SIZE = 50



def check_corr():
    DB_PATH_FULL=os.path.join(os.environ ['HOME'], DB_PATH)
    for root, dirs, files in os.walk(DB_PATH_FULL, topdown=False):
        print("1st for {}".format(root))
        for name in files:
            fp=os.path.join(root, name)
            df = pd.read_csv(fp)
            #Convert data in string to date
            df[COL_DATE] = df[COL_DATE].apply(pd.to_datetime,format="%Y%m%d")
            #Restrict the data which is to be plotted to several months
            df=df[df[COL_DATE]>='2019-01-01']
            #Make a date an index of the frame
            df.set_index(COL_DATE, inplace=True)
            df_w=df.resample('1w').agg({COL_OPEN: 'first', 
                                     COL_HIGH: 'max', 
                                     COL_LOW: 'min', 
                                     COL_CLOSE: 'last'})
            df_w['change'] = df_w[COL_CLOSE].pct_change()*100
            # df['change'] = df[COL_CLOSE].pct_change()*100
            # ipdb.set_trace()
            print('{eq}, corr = {corr}'.format(eq=name, corr=round(df_w['change'].autocorr(2),2)))


def forecast_with_update(y_train, y_test):
    if os.path.exists(SERIALIZED_MODEL):
        print("File with model exists")
        with open(SERIALIZED_MODEL, 'rb') as pkl:
            auto_arima = pickle.load(pkl)
    else:
        print("File doesn't exists")
        auto_arima = pm.auto_arima(y_train, seasonal=False, stepwise=False, 
                                   approximation=False, n_jobs=-1)
        with open(SERIALIZED_MODEL, 'wb') as pkl:
            pickle.dump(auto_arima, pkl)
    
    forecasts = []
    confidence_intervals = []

    for new_ob in y_test:
        fc, conf = auto_arima.predict(n_periods=1, return_conf_int=True)
        forecasts.append(fc)
        confidence_intervals.append(conf)
        # Updates the existing model with a small number of MLE steps
        auto_arima.update(new_ob)

    print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
    print(f"SMAPE: {smape(y_test, forecasts)}")
    return(forecasts, confidence_intervals)


def forecast_with_new(y_train, y_test):
    forecasts = []
    confidence_intervals = []
    print("Iterations {}".format(y_test.size))
    for new_ob in y_test:
        auto_arima = pm.auto_arima(y_train, seasonal=False, stepwise=False,
                                   approximation=False, n_jobs=-1)
        fc, conf = auto_arima.predict(n_periods=1, return_conf_int=True)
        forecasts.append(fc)
        confidence_intervals.append(conf)
        # Updates the existing model with a small number of MLE steps
        print("Iteration")
        y_train = np.append(y_train[1:], new_ob)

    print(f"Mean squared error: {mean_squared_error(y_test, forecasts)}")
    print(f"SMAPE: {smape(y_test, forecasts)}")
    return(forecasts, confidence_intervals)


def plot_forecasts(forecasts, y_test):
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    axes[0].plot(forecasts, color='green', marker='o', label='Predicted')
    axes[0].plot(y_test, color='red', label='Actual', marker='x')

    axes[0].legend()   
    plt.show()

    return

 
def forecast_arima(df):
    n_forecasts = len(df)

    arima_pred = auto_arima.forecast(n_forecasts)

    arima_pred = [pd.DataFrame(arima_pred[0], columns=['prediction']),
                  pd.DataFrame(arima_pred[2], columns=['ci_lower', 
                                                       'ci_upper'])]
    arima_pred = pd.concat(arima_pred, axis=1).set_index(test.index)
    print(arima_pred)

def simulate():
    np.random.seed(12345)
    arparams = np.array([.75, -.25])
    maparams = np.array([.65, .35])
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag
    y = sm.tsa.arima_process.arma_generate_sample(ar, ma, 250)
    y_sum = np.cumsum(y)
    # fig = plt.figure(figsize=(6,6))
    # ax = fig.add_subplot(111)
    # ax.plot(y_sum, color='green', marker='o', label='Arima')
    # plt.show()
    
    return(y_sum)


# check_corr()
# load the dataset
y = common.get_data("data/swig80.txt")
# forecast_arima()
# y=simulate()
train_len = int(y.size * TEST_RATIO)
# y_train = y[train_len-TRAIN_SIZE:train_len]
y_train = y[:train_len]
y_test = y[train_len:]

print(f"{y_train.size} train samples")
print(f"{y.shape[0] - train_len} test samples")

forecasts, confidence_intervals = forecast_with_update(y_train, y_test)
# forecasts, confidence_intervals=forecast_with_new(y_train,y_test)
plot_forecasts(forecasts, y_test)
